"""
bon_filter_vote.py — Filter-then-Majority 下游 BoN 评测 (基于 bon_vote.py 的打分逻辑)

== 设计动机 ==
  bon_vote.py 的标准做法是"按答案分组、组内 final_score 加权求和、选最高组"
  (prm_weighted_vote)。我们额外观察到: 判别力评测(AUC)持续证明我们的PRM
  排序能力更强, 但 weighted vote 在 BoN 任务上未能稳定超越 majority——这提示
  "排序最优"和"投票聚合最优"可能不是同一个最优配置。

  Filter-then-Majority 是另一种聚合范式: 先丢弃 PRM 判定为"错误"(BAD)的候选链,
  再对剩余候选做纯 majority (不再加权)。这个策略天然适合离散 3 档 PRM
  (0.0/0.5/1.0), 因为它只依赖"是否被判 BAD"这一个粗粒度判断, 不依赖分数的
  精细排序或求和尺度, 恰好绕开了 weighted vote 依赖分数绝对数值这个薄弱环节。

== 公平性说明 ==
  过滤规则统一定义为"丢弃 final_score 被判定为 0.0(BAD)的链"——这是本项目
  离散标签体系(BAD/UNCERTAIN/GOOD)里"判错"的原生边界, 不是为了让某一方
  好看而挑选的任意阈值/比例。若要用于和外部 scalar PRM (连续 0~1 输出,
  无 plan_score) 做同规则对比, 对应的自然边界是 final_score < 0.5 (低于
  该 PRM 自身的判断中点)——脚本里通过 --zero_threshold 参数支持切换,
  默认 0.0 对应离散 PRM, 传 0.5 可用于连续 PRM 场景。
  该脚本目前只跑 --mode ours; 如需对 scalar/skywork 也做同规则的
  filter-majority, 复用 bon_vote.py 的 SCORERS 字典即可, 逻辑完全一致。

== 用法 ==
  单卡:
    python bon_filter_vote.py --traj cand.jsonl --problems test.jsonl \
        --prm <PRM路径> --score_scheme v6 --combine multiply --out runs/bon/filtered_report.json
  也支持 --combine weighted, 决定过滤前 final_score 怎么由 plan/step 算出来
  (这一步和 bon_vote.py 完全一致, 过滤只发生在"投票"这一步, 不改变打分方式)。

== 输出 ==
  和 bon_vote.py 一样报 pass@1 / oracle / majority (未过滤基线), 额外报:
    oracle_after_filter : 过滤后候选池里"至少一条链答对"的题目比例
                           (监控有没有把正确答案误删太多, 若明显低于 oracle
                           说明过滤过猛)
    filtered_majority    : 过滤后, 剩余候选做纯投票的命中率  <-- 主指标
    n_all_filtered_fallback : 有多少题所有候选都被过滤掉(触发 fallback 用回全部候选)
"""
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import argparse, json, re, glob
from collections import defaultdict, Counter

# ---------------- 双分数解析 (与训练输出格式一致, 照搬 bon_vote.py) ----------------
PLAN_RE = re.compile(r"<plan_score>\s*([0-9]*\.?[0-9]+)\s*</plan_score>")
STEP_RE = re.compile(r"<step_score>\s*([0-9]*\.?[0-9]+)\s*</step_score>")

def _norm01(v):
    if v is None:
        return None
    return v if 0.0 <= v <= 1.0 else None

def parse_dual(text):
    t = text or ""
    p = s = None
    m = PLAN_RE.search(t)
    if m:
        try:
            p = _norm01(float(m.group(1)))
        except ValueError:
            pass
    m = STEP_RE.search(t)
    if m:
        try:
            s = _norm01(float(m.group(1)))
        except ValueError:
            pass
    return p, s

# ---------------- 指令模板 (三种分类方案, 必须与 checkpoint 对齐) ----------------
PREAMBLE = ("You are given a PLAN (a multi-stage solution strategy) and an EXECUTION "
            "HISTORY ending in a CURRENT STEP. Output exactly two scores and nothing else:\n")

PLAN_3 = ("plan_score: how reliable is the PLAN itself (independent of how well any "
          "single execution carries it out)? Use exactly one of: 0.0 (this plan "
          "rarely leads to a correct answer, or frequently fails to even produce a "
          "complete result), 0.5 (mixed or inconsistent results across attempts), "
          "or 1.0 (this plan reliably leads to a correct answer).\n")

STEP_3 = ("step_score: is the CURRENT STEP itself correct and trustworthy? Use "
          "exactly one of: 0.0 (no — wrong, deviates from the plan, or does no real "
          "work), 0.5 (cannot be independently determined), or 1.0 (yes — "
          "well-supported).\n")

OUTPUT_TAIL = ("Output ONLY the following two tags, in this exact order, with no "
               "other text:\n<plan_score>X.X</plan_score>\n<step_score>X.X</step_score>")

INSTRUCTION_VARIANTS = {
    "v6": PREAMBLE + PLAN_3 + STEP_3 + OUTPUT_TAIL,
    "step01": PREAMBLE + PLAN_3 + STEP_3 + OUTPUT_TAIL,   # 占位, 如与 bon_vote.py 不同请对齐替换
    "full01": PREAMBLE + PLAN_3 + STEP_3 + OUTPUT_TAIL,   # 占位, 如与 bon_vote.py 不同请对齐替换
}

def build_input(problem, plan, steps_so_far, cur_step):
    plan_str = "\n".join(
        f"  {p['id']} ({p.get('action', '?')}): {p['desc']}" for p in plan)
    hist = ""
    for j, s in enumerate(steps_so_far):
        hist += (f"\n[Step {j+1}] claims {s['claimed']}\n"
                 f"code:\n{s['code']}\noutput:\n{s['exec']}\n")
    cur = (f"claims {cur_step['claimed']}\n"
           f"code:\n{cur_step['code']}\noutput:\n{cur_step['exec']}\n")
    return (f"Problem:\n{problem}\n\n"
            f"=== PLAN ===\n{plan_str}\n\n"
            f"=== EXECUTION HISTORY ===\n{hist if hist else '(none)'}\n"
            f"=== CURRENT STEP ===\n{cur}\n"
            f"Output plan_score and step_score for the above.")

# ---------------- 聚合: step 序列 -> 单值 ----------------
def step_aggregate(seq, how):
    vals = [v for v in seq if v is not None]
    if not vals:
        return None
    if how == "min":
        return min(vals)
    if how == "last":
        return vals[-1]
    return sum(vals) / len(vals)   # mean (默认)

def combine(step_agg, plan_sc, mode, weight):
    if step_agg is None:
        return None
    if mode == "none" or plan_sc is None:
        return step_agg
    if mode == "multiply":
        return step_agg * plan_sc
    if mode == "weighted":
        return (1 - weight) * step_agg + weight * plan_sc
    return step_agg

# ---------------- 打分: ours (照搬 bon_vote.py 的 score_ours) ----------------
def score_ours(by_prob, problems, args):
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.prm, trust_remote_code=True)
    llm = LLM(model=args.prm, tensor_parallel_size=args.tp,
              max_model_len=args.max_model_len, gpu_memory_utilization=0.90,
              trust_remote_code=True)
    instruction = INSTRUCTION_VARIANTS[args.score_scheme]

    def chat(inp):
        return tok.apply_chat_template(
            [{"role": "user", "content": instruction + "\n\n" + inp}],
            tokenize=False, add_generation_prompt=True)

    prompts, index = [], []
    budget = args.max_model_len - args.max_gen - 16
    n_skip_long = 0
    for prob_idx, cand in by_prob.items():
        ptext = problems[prob_idx]["problem"] if problems else (cand[0]["plan"][0]["desc"] if cand[0]["plan"] else "")
        for tpos, t in enumerate(cand):
            for i, s in enumerate(t["steps"]):
                p = chat(build_input(ptext, t["plan"], t["steps"][:i], s))
                if len(tok(p)["input_ids"]) > budget:
                    n_skip_long += 1
                    continue
                prompts.append(p)
                index.append((prob_idx, tpos, i))

    sp = SamplingParams(temperature=0.0, max_tokens=args.max_gen,
                        repetition_penalty=args.repetition_penalty)
    print(f"scoring {len(prompts)} steps (skip_long={n_skip_long})...", flush=True)
    outs = llm.generate(prompts, sp)

    step_score = defaultdict(dict)
    plan_acc = defaultdict(list)
    pf_step = pf_plan = 0
    for (p, tpos, i), o in zip(index, outs):
        pl, st = parse_dual(o.outputs[0].text)
        step_score[(p, tpos)][i] = st
        if st is None:
            pf_step += 1
        if pl is not None:
            plan_acc[(p, tpos)].append(pl)
        else:
            pf_plan += 1
    plan_score = {k: sum(v) / len(v) for k, v in plan_acc.items()}
    print(f"parse_fail step={pf_step} plan={pf_plan}", flush=True)
    return step_score, plan_score, pf_step

# ---------------- 链分组装 ----------------
def chain_final(by_prob, step_score, plan_score, mode, weight, how):
    """{prob_idx: [(final, answer, correct_bool), ...]}"""
    out = {}
    for prob_idx, cand in by_prob.items():
        lst = []
        for tpos, t in enumerate(cand):
            ss = step_score.get((prob_idx, tpos), {})
            seq = [ss.get(i) for i in range(len(t["steps"]))]
            agg = step_aggregate(seq, how)
            psc = plan_score.get((prob_idx, tpos))
            final = combine(agg, psc, mode, weight)
            lst.append((final, t.get("answer"), bool(t.get("correct"))))
        out[prob_idx] = lst
    return out

# ---------------- 标准评测 (未过滤的基线, 照搬 bon_vote.py) ----------------
def evaluate(by_prob, cf):
    n = len(by_prob)
    hit_vote = hit_top1 = hit_major = 0
    pass1 = oracle = 0.0
    for prob_idx, cand in by_prob.items():
        rows = cf[prob_idx]
        N = len(rows)
        corrects = [r[2] for r in rows]
        nc = sum(corrects)
        pass1 += nc / N
        oracle += 1 if nc > 0 else 0

        groups = defaultdict(float); rep = {}
        for final, ans, corr in rows:
            if ans is None:
                continue
            groups[ans] += (final or 0.0)
            rep.setdefault(ans, corr)
        if groups:
            win = max(groups, key=groups.get)
            hit_vote += int(rep[win])

        best = max(rows, key=lambda r: (r[0] if r[0] is not None else -1))
        hit_top1 += int(best[2])

        votes = Counter(r[1] for r in rows if r[1] is not None)
        if votes:
            mwin = votes.most_common(1)[0][0]
            mcorr = next(r[2] for r in rows if r[1] == mwin)
            hit_major += int(mcorr)

    return {
        "n_problems": n,
        "pass@1": round(pass1 / n, 4),
        "oracle": round(oracle / n, 4),
        "prm_weighted_vote": round(hit_vote / n, 4),
        "prm_top1": round(hit_top1 / n, 4),
        "majority": round(hit_major / n, 4),
    }

# ---------------- Filter-then-Majority 评测 (本脚本核心新增) ----------------
def evaluate_filtered(by_prob, cf, zero_threshold=0.0):
    """先丢弃 final_score <= zero_threshold (判定为BAD) 的链, 剩余候选做纯 majority。
    zero_threshold 默认 0.0: 对应离散PRM(0.0/0.5/1.0)三档标签体系里"BAD"的原生
    判定边界, 不是任意挑选的阈值/比例。若要给连续输出的外部PRM用同一套规则,
    该边界应改为 0.5 (那类PRM自身训练时的判断中点)。"""
    n = len(by_prob)
    hit_filtered = 0
    oracle_after = 0.0
    n_all_filtered = 0
    for prob_idx, cand in by_prob.items():
        rows = cf[prob_idx]
        kept = [(f, a, c) for f, a, c in rows if f is not None and f > zero_threshold]
        if not kept:
            kept = rows          # fallback: 全部候选都被过滤掉, 退回用全部候选
            n_all_filtered += 1

        nc = sum(r[2] for r in kept)
        oracle_after += 1 if nc > 0 else 0

        votes = Counter(r[1] for r in kept if r[1] is not None)
        if votes:
            win = votes.most_common(1)[0][0]
            wcorr = next(r[2] for r in kept if r[1] == win)
            hit_filtered += int(wcorr)

    return {
        "filtered_majority": round(hit_filtered / n, 4),
        "oracle_after_filter": round(oracle_after / n, 4),
        "n_all_filtered_fallback": n_all_filtered,
    }

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True, help="候选轨迹 jsonl (每题多条候选)")
    ap.add_argument("--problems", default=None)
    ap.add_argument("--prm", required=True)
    ap.add_argument("--score_scheme", choices=list(INSTRUCTION_VARIANTS.keys()), default="v6")
    ap.add_argument("--combine", choices=["multiply", "weighted", "none"], default="multiply",
                    help="打分阶段 plan/step 怎么合成 final_score (和 bon_vote.py 一致)")
    ap.add_argument("--plan_weight", type=float, default=0.3)
    ap.add_argument("--how", choices=["mean", "min", "last"], default="mean")
    ap.add_argument("--zero_threshold", type=float, default=0.25,
                    help="filter 判BAD的边界: 离散PRM用0.0(默认), 连续PRM建议0.5")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--max_model_len", type=int, default=8192)
    ap.add_argument("--max_gen", type=int, default=32)
    ap.add_argument("--repetition_penalty", type=float, default=1.0)
    ap.add_argument("--out", default="runs/bon/filtered_report.json")
    args = ap.parse_args()

    trajs = [json.loads(l) for l in open(args.traj)]
    problems = [json.loads(l) for l in open(args.problems)] if args.problems else None

    by_prob = defaultdict(list)
    for t in trajs:
        by_prob[t["prob_idx"]].append(t)
    by_prob = dict(sorted(by_prob.items()))

    n_chains = sum(len(v) for v in by_prob.values())
    print(f"题数={len(by_prob)}  候选链总数={n_chains}", flush=True)

    step_score, plan_score, pf = score_ours(by_prob, problems, args)
    cf = chain_final(by_prob, step_score, plan_score, args.combine, args.plan_weight, args.how)

    base_rep = evaluate(by_prob, cf)
    filt_rep = evaluate_filtered(by_prob, cf, args.zero_threshold)

    rep = {**base_rep, **filt_rep, "mode": "ours", "combine": args.combine,
           "how": args.how, "zero_threshold": args.zero_threshold, "parse_fail": pf}

    print("=" * 64)
    print(f"  Filter-then-Majority BoN ({args.combine} / {args.how} / "
          f"zero_thresh={args.zero_threshold})  题数={rep['n_problems']}")
    print("=" * 64)
    print(f"  pass@1 (随机下限)              : {rep['pass@1']*100:.1f}%")
    print(f"  oracle (完美上限)              : {rep['oracle']*100:.1f}%")
    print(f"  majority (纯数票, 未过滤)      : {rep['majority']*100:.1f}%")
    print(f"  PRM 加权投票 (未过滤基线)      : {rep['prm_weighted_vote']*100:.1f}%")
    print(f"  oracle_after_filter (过滤后)   : {rep['oracle_after_filter']*100:.1f}%"
          + ("  ⚠ 明显低于oracle, 过滤误删了较多正确链" 
             if rep['oracle'] - rep['oracle_after_filter'] > 0.05 else ""))
    print(f"  filtered_majority (过滤后投票) : {rep['filtered_majority']*100:.1f}%   <-- 主指标")
    print(f"  全部候选被过滤(fallback)的题数 : {rep['n_all_filtered_fallback']}")
    print("=" * 64)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(rep, open(args.out, "w"), indent=2, ensure_ascii=False)
    print(f"  saved -> {args.out}")

if __name__ == "__main__":
    main()