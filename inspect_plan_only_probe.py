"""
inspect_plan_only_probe.py — 在完全没有真实执行的情况下(只看plan, CURRENT
STEP用占位符填), 探测plan_score能不能提前判断哪个plan更好。

占位符设计(claimed/code/exec三个字段, 跟训练数据里真实步骤的字段结构完全
一致, 只是内容换成"没有真实工作"的样子):
  claimed = "(none)"                       不写成P1这种真实阶段编号
  code    = "# (plan not yet executed)"    纯注释, 没有可执行语句
  exec    = "[no output]"                  沿用训练数据里"没有输出"的既有写法
这跟模型训练时见过的任何真实输入都不完全一样(模型从没在"零执行"的情况下被
要求打分), 是个明确的分布外探测, 不是模型熟悉的场景 —— 这正是为什么必须
同时统计"占位符自己被判成step_score=0的比例"作为健全性检查: 如果模型连这个
都判不对, 说明这次测出来的plan_score也不能直接信。

复用eval_bon.py里已经验证过的INSTRUCTION_DUAL_VARIANTS/build_input_dual,
保证prompt格式跟训练/评测时完全一致, 不是另起一套自己拼的格式。

用法:
  python inspect_plan_only_probe.py --prm /path/to/checkpoint --score_scheme full01 \
      --traj runs/eval/trajectories.jsonl \
      --problems /public/home/ljt/lzm/code/Qwen2.5-Math/evaluation/data/math/test.jsonl
"""
import argparse, json
from collections import defaultdict

PLACEHOLDER_STEP = {
    "claimed": "(none)",
    "code": "# (plan not yet executed)",
    "exec": "[no output]",
}

def build_plan_groups(trajs):
    """按(prob_idx, plan_idx)分组, 同一组内的plan内容理论上一致(取第一条代表),
    real_acc是这组所有真实执行的正确率(用于跟probe出来的选择结果对照)。"""
    groups = defaultdict(list)
    for t in trajs:
        groups[(t["prob_idx"], t["plan_idx"])].append(t)
    plan_info = {}
    for (p, pl), ts in groups.items():
        decided = [t for t in ts if t.get("correct") is not None]
        real_acc = sum(1 for t in decided if t["correct"]) / len(decided) if decided else None
        plan_info[(p, pl)] = {"plan": ts[0]["plan"], "real_acc": real_acc, "n": len(ts)}
    return plan_info

def analyze_probe_results(plan_info, probe_scores):
    """probe_scores: {(prob_idx, plan_idx): (plan_score, step_score)}
    跟build_plan_groups的输出对照, 算: (1) 占位符被判成step_score=0的比例
    (2) 按probe出的plan_score选出的'赢家plan', 它的真实正确率分布
    (3) 跟真实更好的plan比, probe选对了多少 (对照inspect_plan_pruning_backtest.py
        那次用'全程平均plan_score'选出来的82.1%选对率)"""
    by_prob = defaultdict(list)
    for (p, pl) in plan_info:
        by_prob[p].append(pl)

    n_step0, n_total_probe = 0, 0
    for k, (psc, ssc) in probe_scores.items():
        if ssc is not None:
            n_total_probe += 1
            if ssc == 0.0:
                n_step0 += 1

    winner_accs = []
    n_correct_pick, n_total_pick = 0, 0
    for p, pls in by_prob.items():
        if len(pls) < 2:
            continue
        scored = [(probe_scores.get((p, pl), (None, None))[0], pl) for pl in pls]
        scored = [(s, pl) for s, pl in scored if s is not None]
        if len(scored) < 2:
            continue
        winner_pl = max(scored, key=lambda x: x[0])[1]
        winner_acc = plan_info[(p, winner_pl)]["real_acc"]
        if winner_acc is not None:
            winner_accs.append(winner_acc)
        real_accs = {pl: plan_info[(p, pl)]["real_acc"] for pl in pls
                    if plan_info[(p, pl)]["real_acc"] is not None}
        if len(real_accs) >= 2:
            best_real_pl = max(real_accs, key=real_accs.get)
            n_total_pick += 1
            if winner_pl == best_real_pl:
                n_correct_pick += 1

    avg_random_acc = sum(v["real_acc"] for v in plan_info.values()
                         if v["real_acc"] is not None) / sum(
                         1 for v in plan_info.values() if v["real_acc"] is not None)

    return {
        "占位符step_score=0的比例": n_step0 / n_total_probe if n_total_probe else None,
        "占位符探测覆盖的plan数": n_total_probe,
        "按probe选出的赢家plan_平均真实正确率": sum(winner_accs) / len(winner_accs) if winner_accs else None,
        "不做任何选择的基线(所有plan平均真实正确率)": avg_random_acc,
        "probe选对率(选中的是不是真实更好的那个)": n_correct_pick / n_total_pick if n_total_pick else None,
        "n_problems_evaluated": len(winner_accs),
    }

def score_probe(plan_info, args):
    """实际调用模型打分, 需要vLLM, 在集群上跑。"""
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    import eval_bon as eb

    tok = AutoTokenizer.from_pretrained(args.prm, trust_remote_code=True)
    llm = LLM(model=args.prm, tensor_parallel_size=args.tp,
              max_model_len=args.max_model_len, gpu_memory_utilization=0.90,
              trust_remote_code=True)
    instruction = eb.INSTRUCTION_DUAL_VARIANTS[args.score_scheme]
    problems = [json.loads(l) for l in open(args.problems)] if args.problems else None

    def chat(inp):
        return tok.apply_chat_template(
            [{"role": "user", "content": instruction + "\n\n" + inp}],
            tokenize=False, add_generation_prompt=True)

    keys, prompts = [], []
    for (p, pl), info in plan_info.items():
        ptext = problems[p]["problem"] if problems else "(unknown problem text)"
        inp = eb.build_input_dual(ptext, info["plan"], [], PLACEHOLDER_STEP)
        prompts.append(chat(inp))
        keys.append((p, pl))

    sp = SamplingParams(temperature=0.0, max_tokens=args.max_gen,
                        repetition_penalty=args.repetition_penalty)
    tag = f"[shard {args.shard_id}] " if args.num_shards > 1 else ""
    print(f"{tag}[plan_only_probe] 探测 {len(prompts)} 个plan...")
    outs = llm.generate(prompts, sp)
    probe_scores = {}
    for k, o in zip(keys, outs):
        plan_sc, step_sc = eb.parse_dual_score(o.outputs[0].text)
        probe_scores[k] = (plan_sc, step_sc)
    return probe_scores

def save_probe_shard(path, probe_scores, score_scheme):
    data = {"score_scheme": score_scheme,
           "probe_scores": {f"{p}:{pl}": s for (p, pl), s in probe_scores.items()}}
    json.dump(data, open(path, "w"))
    print(f"  分片已保存 -> {path} (共{len(probe_scores)}个plan)")

def load_probe_shard(path):
    data = json.load(open(path))
    probe_scores = {}
    for k, v in data["probe_scores"].items():
        p_str, pl_str = k.split(":")
        probe_scores[(int(p_str), int(pl_str))] = tuple(v)
    return probe_scores, data.get("score_scheme")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["probe", "merge"], default="probe",
                    help="probe=实际打分(单卡或本分片), merge=合并多个分片出最终分析")
    ap.add_argument("--prm", default=None)
    ap.add_argument("--score_scheme", default=None,
                    help="必须跟--prm这个checkpoint训练时用的方案一致, "
                         "见eval_bon.py的INSTRUCTION_DUAL_VARIANTS。--mode probe时必填")
    ap.add_argument("--traj", default="runs/eval/trajectories.jsonl")
    ap.add_argument("--problems", default=None)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--max_model_len", type=int, default=8192)
    ap.add_argument("--max_gen", type=int, default=400)
    ap.add_argument("--repetition_penalty", type=float, default=1.15)
    ap.add_argument("--out", default="runs/eval/plan_only_probe.json")
    # ---- 数据并行分片 (跟eval_bon.py同一套用法) ----
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--shard_glob", default=None)
    args = ap.parse_args()
    if args.mode == "probe" and not args.prm:
        ap.error("--mode probe 需要 --prm")
    if args.mode == "probe" and args.score_scheme is None:
        ap.error("--mode probe 需要显式指定 --score_scheme, 不允许用默认值")
    if args.mode == "merge" and not args.shard_glob:
        ap.error("--mode merge 需要 --shard_glob")

    trajs = [json.loads(l) for l in open(args.traj)]
    plan_info_full = build_plan_groups(trajs)

    if args.mode == "merge":
        import glob
        files = sorted(glob.glob(args.shard_glob))
        if not files:
            raise SystemExit(f"未匹配到任何分片文件: {args.shard_glob}")
        probe_scores = {}
        schemes_seen = set()
        for f in files:
            ps, scheme = load_probe_shard(f)
            probe_scores.update(ps)
            schemes_seen.add(scheme)
        if len(schemes_seen) > 1:
            print(f"  ⚠⚠ 严重警告: 合并的分片里混入了不止一种score_scheme: {schemes_seen}")
        missing = set(plan_info_full.keys()) - set(probe_scores.keys())
        if missing:
            print(f"  ⚠ 警告: {len(missing)} 个plan未被任何分片覆盖")
        result = analyze_probe_results(plan_info_full, probe_scores)
        json.dump({"result": result,
                  "probe_scores": {f"{p}:{pl}": s for (p, pl), s in probe_scores.items()}},
                 open(args.out, "w"), ensure_ascii=False, indent=2)
        print(f"\n  已合并 {len(files)} 个分片, 覆盖 {len(probe_scores)} 个plan")
        print("=" * 60)
        for k, v in result.items():
            print(f"  {k}: {v}")
        print(f"\n  saved -> {args.out}")
        return

    # ---- probe模式: 按需分片 ----
    if args.num_shards > 1:
        plan_info = {k: v for k, v in plan_info_full.items() if k[0] % args.num_shards == args.shard_id}
        if not plan_info:
            raise SystemExit(f"分片 {args.shard_id}/{args.num_shards} 没分到任何题")
    else:
        plan_info = plan_info_full

    probe_scores = score_probe(plan_info, args)

    if args.num_shards > 1:
        shard_path = f"{args.out}.shard{args.shard_id}.json"
        save_probe_shard(shard_path, probe_scores, args.score_scheme)
        return

    result = analyze_probe_results(plan_info_full, probe_scores)
    json.dump({"result": result,
              "probe_scores": {f"{p}:{pl}": s for (p, pl), s in probe_scores.items()}},
             open(args.out, "w"), ensure_ascii=False, indent=2)
    print("\n" + "=" * 60)
    for k, v in result.items():
        print(f"  {k}: {v}")
    print(f"\n  saved -> {args.out}")

if __name__ == "__main__":
    main()
