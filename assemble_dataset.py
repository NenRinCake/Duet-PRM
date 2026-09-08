"""
assemble_dataset.py — 三维标签总装 (v6: plan_score 也压成3档, 不再做连续回归)

v6 改动背景:
  v5训完用 analyze_plan_score.py 实测发现: 模型预测的plan_score(连续值)在真实
  plan_quality从0.0到0.7这一大段(占样本82%)全部挤在0.70-0.74之间, 完全分不开;
  只有到0.8/1.0才开始分得开 —— 模型实际只学会了"是否几乎注定成功"这个粗粒度
  判断, 强行让它回归一个连续值是在拟合它学不动的精度。BoN矩阵也印证了这一点:
  把这个学不准的连续plan_score混进final_score(无论weighted还是multiply), 每一
  行都比完全不用它(none)更差 —— 不是噪声, 是"在该打低分的地方系统性地打高了",
  注入的是误导信号而非中性噪声。

  这正是 step_score 那次 7档->3档压缩的同一个教训, 换了个目标重演了一次:
  与其强求模型做不动的精细回归, 不如把目标也压成它证明学得动的粗粒度判断。

v6 核心改动: compute_plan_score(success_rate, trunc_rate) 把 plan_score 也变成
  BAD(0.0)/UNCERTAIN(0.5)/GOOD(1.0) 三档, 和 step_score 共享同一套量纲, 规则:
    - success_rate(该plan下兄弟执行成功率) <= 0.25 (4次里最多对1次) -> BAD
    - success_rate >= 0.75 (4次里至少对3次) -> GOOD (模型已证明能分出这一段)
    - 否则 (中间这段模型分不出来) -> UNCERTAIN, 老实给中性, 不臆造精度
    - 新增: trunc_rate(该plan下兄弟执行的截断/烂尾率) 一票否决: 若 >= 0.5, 直接
      判BAD, 不管"跑完的那几次"成功率多高 —— 因为那个成功率只统计了跑完的样本,
      有幸存者偏差, 一个经常烧光预算说不出答案的plan, 实际上是个差策略。
      (该项目实测分布: trunc_rate在0.2和0.5之间有个天然断层, 87%的plan聚集在
      0.0, 病态尾部聚集在0.5+, 0.5是个有数据支撑的硬切分点, 不是拍脑袋)

  这两档的分界阈值(0.25/0.75)选在 N=4 兄弟执行的天然刻度上(1/4, 3/4), 不是
  任意切的等分点 —— plan_quality本身就只能取{0,0.25,0.5,0.75,1.0}这几个值
  (N不固定时会有零碎分数), 不是真正连续的, 压成3档是"重新分组离散标签",
  跟v4压缩step_score是同一类操作, 不是新发明。

  step_score 的3类压缩逻辑(v4验证过)完全不变, 这版只动 plan_score。
"""
import argparse, json, random
from collections import defaultdict

# ============================================================ step_score 压缩表 (v4验证过, 不变)
BAD_PROV  = {"localized_fail", "suspect"}
BAD_ADH   = {"deviated", "claimed_only"}
UNCERTAIN_PROV = {"unverified"}
UNCERTAIN_ADH  = {"self_declared_deviation"}
GOOD_PROV = {"corroborated", "outcome_consistent"}

def compute_step_score(adherence, provenance):
    if provenance in BAD_PROV:
        return 0.0
    if adherence in BAD_ADH:
        return 0.0
    if adherence in UNCERTAIN_ADH:
        return 0.5
    if provenance in UNCERTAIN_PROV:
        return 0.5
    if provenance in GOOD_PROV:
        return 1.0
    return 0.5

# ============================================================ plan_score 压缩表 (v6新增)
def compute_plan_score(success_rate, trunc_rate, trunc_veto_threshold=0.5):
    """3档压缩, 阈值选在N=4的天然刻度(1/4, 3/4)上; trunc_rate高时一票否决判BAD
    (幸存者偏差: 成功率只统计跑完的样本, 频繁烂尾的plan不该因为"跑完的全对"被洗白)。
    返回 None 表示无法判定(success_rate本身缺失, 即该plan没有已知的兄弟执行结果)。"""
    if success_rate is None:
        return None
    if trunc_rate is not None and trunc_rate >= trunc_veto_threshold:
        return 0.0
    if success_rate <= 0.25:
        return 0.0
    if success_rate >= 0.75:
        return 1.0
    return 0.5

# ============================================================ input 上下文 (与v5完全一致, 不变)
def build_input(problem, plan, steps_so_far, cur_step):
    plan_str = "\n".join(
        f"  {p['id']} ({p.get('action','?')}): {p['desc']}" for p in plan)
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

INSTRUCTION = (
    "You are given a PLAN (a multi-stage solution strategy) and an EXECUTION "
    "HISTORY ending in a CURRENT STEP. Output exactly two scores and nothing else:\n"
    "plan_score: how reliable is the PLAN itself (independent of how well any "
    "single execution carries it out)? Use exactly one of: 0.0 (this plan "
    "rarely leads to a correct answer, or frequently fails to even produce a "
    "complete result), 0.5 (mixed or inconsistent results across attempts), "
    "or 1.0 (this plan reliably leads to a correct answer).\n"
    "step_score: is the CURRENT STEP itself correct and trustworthy? Use "
    "exactly one of: 0.0 (no — wrong, deviates from the plan, or does no real "
    "work), 0.5 (cannot be independently determined), or 1.0 (yes — "
    "well-supported).\n"
    "Output ONLY the following two tags, in this exact order, with no other text:\n"
    "<plan_score>X.X</plan_score>\n"
    "<step_score>X.X</step_score>"
)

def build_output(plan_score, step_score):
    return f"<plan_score>{plan_score:.1f}</plan_score>\n<step_score>{step_score:.1f}</step_score>"

# ============================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", default="runs/v0/trajectories.jsonl")
    ap.add_argument("--labels", default="runs/v0/step_labels.jsonl")
    ap.add_argument("--c2", default="runs/v0/c2_labels.jsonl")
    ap.add_argument("--problems", default=None)
    ap.add_argument("--out", default="runs/v0/sft_dataset_v6.jsonl")
    ap.add_argument("--drop_suspect", action="store_true")
    ap.add_argument("--keep_uncertain", action="store_true")
    ap.add_argument("--keep_unknown_plan", action="store_true",
                    help="默认丢弃 success_rate=None(该plan没有已判定的兄弟执行)的样本; "
                         "加此开关则用0.5(UNCERTAIN)占位保留")
    ap.add_argument("--trunc_veto_threshold", type=float, default=0.5,
                    help="plan_trunc_rate达到此值时一票否决判BAD (该项目实测分布在"
                         "0.2和0.5之间有天然断层, 0.5是有数据支撑的切分点)")
    ap.add_argument("--oversample_bad_step", type=int, default=2,
                    help="step_score=BAD(0.0)的样本复制倍数")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    trajs = [json.loads(l) for l in open(args.traj)]
    labels = {(r["prob_idx"], r["plan_idx"], r["exec_idx"], r["step"]): r
              for r in (json.loads(l) for l in open(args.labels))}
    c2 = {(r["prob_idx"], r["plan_idx"], r["exec_idx"], r["step"]): r
          for r in (json.loads(l) for l in open(args.c2))}
    problems = [json.loads(l) for l in open(args.problems)] if args.problems else None

    base_records = []
    bad_step_records = []
    n_skipped = n_uncertain = n_unknown_plan = 0
    step_score_hist = defaultdict(int)
    plan_score_hist = defaultdict(int)

    for t in trajs:
        prob_text = (problems[t["prob_idx"]]["problem"] if problems
                     else t["plan"][0]["desc"])
        plan_score = compute_plan_score(t.get("plan_quality"), t.get("plan_trunc_rate"),
                                        args.trunc_veto_threshold)
        if plan_score is None:
            if not args.keep_unknown_plan:
                n_unknown_plan += len(t["steps"])
                continue
            plan_score = 0.5
        for i, s in enumerate(t["steps"]):
            key = (t["prob_idx"], t["plan_idx"], t["exec_idx"], i)
            lrow = labels.get(key, {})
            crow = c2.get(key, {})

            adh = lrow.get("adherence") or lrow.get("b2") or "unknown"
            if isinstance(adh, str) and adh.startswith("uncertain"):
                adh = "uncertain"
            prov = crow.get("provenance", "unverified")

            if adh in ("uncertain", "unknown") and not args.keep_uncertain:
                n_uncertain += 1
                continue
            if prov == "suspect" and args.drop_suspect:
                n_skipped += 1
                continue

            step_score = compute_step_score(adh, prov)
            inp = build_input(prob_text, t["plan"], t["steps"][:i], s)
            outp = build_output(plan_score, step_score)

            rec = {
                "instruction": INSTRUCTION, "input": inp, "output": outp,
                "meta_prob_idx": t["prob_idx"], "meta_plan_idx": t["plan_idx"],
                "meta_exec_idx": t["exec_idx"], "meta_step": i,
                "meta_claimed": s["claimed"],
                "meta_plan_score": plan_score,
                "meta_plan_quality_raw": t.get("plan_quality"),       # 追溯用
                "meta_plan_trunc_rate_raw": t.get("plan_trunc_rate"), # 追溯用
                "meta_adherence": adh, "meta_execution_provenance": prov,
                "meta_step_score": step_score,
                "meta_traj_correct": t.get("correct"),
            }
            base_records.append(rec)
            step_score_hist[step_score] += 1
            plan_score_hist[plan_score] += 1
            if step_score == 0.0:
                bad_step_records.append(rec)

    extra = []
    if args.oversample_bad_step > 1:
        for _ in range(args.oversample_bad_step - 1):
            extra.extend(bad_step_records)
    all_records = base_records + extra
    random.shuffle(all_records)

    out = open(args.out, "w")
    for rec in all_records:
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    out.close()

    n = len(all_records)
    print(f"written {n} samples (base={len(base_records)}, oversample_extra={len(extra)}) "
          f"(dropped: uncertain={n_uncertain}, suspect={n_skipped}, "
          f"unknown_plan={n_unknown_plan}) -> {args.out}")

    def _print_hist(name, hist):
        total = sum(hist.values()) or 1
        print(f"\n{name} 3档分布 (去重前):")
        for sc in [0.0, 0.5, 1.0]:
            label = {0.0: "BAD", 0.5: "UNCERTAIN", 1.0: "GOOD"}[sc]
            cnt = hist.get(sc, 0)
            print(f"  {sc:.1f} ({label}): {cnt:>5}  ({cnt/total*100:.1f}%)")
    _print_hist("step_score", step_score_hist)
    _print_hist("plan_score", plan_score_hist)
    print(f"\nstep_score BAD类过采样倍数: x{args.oversample_bad_step} "
          f"({len(bad_step_records)}条 -> 最终出现 {len(bad_step_records)*args.oversample_bad_step}次)")
    print(f"plan_score 未做过采样 (压缩后BAD类占比已经够大, 不需要)")

if __name__ == "__main__":
    main()