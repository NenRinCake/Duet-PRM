"""
compare_traj_batches.py — 对比两批trajectories.jsonl, 定位"这批数据到底差在哪"

不需要GPU, 纯读已有文件算统计量。专门用来回答: v6和v7这两批数据,
除了我们已经看过的plan_score 3档分布之外, 还有哪些维度真的不一样。
"""
import argparse, json
from collections import defaultdict

def analyze(path):
    trajs = [json.loads(l) for l in open(path)]
    n_traj = len(trajs)

    # ---- 轨迹级指标 ----
    n_fmt_bad = sum(1 for t in trajs if t.get("fmt_bad", 0) > 0)
    n_answered = sum(1 for t in trajs if t.get("answer") is not None)
    n_correct = sum(1 for t in trajs if t.get("correct") is True)
    n_wrong = sum(1 for t in trajs if t.get("correct") is False)
    n_censored = sum(1 for t in trajs if t.get("correct") is None)

    # ---- 步骤级指标 ----
    n_steps = sum(len(t.get("steps", [])) for t in trajs)
    n_deviation = sum(1 for t in trajs for s in t.get("steps", [])
                      if s.get("is_deviation"))

    # ---- plan级指标 (plan_quality/trunc_rate在同一plan下是重复值, 按plan去重算) ----
    by_plan = {}
    for t in trajs:
        key = (t["prob_idx"], t["plan_idx"])
        if key not in by_plan:
            by_plan[key] = {"quality": t.get("plan_quality"),
                            "trunc": t.get("plan_trunc_rate")}
    qualities = [v["quality"] for v in by_plan.values() if v["quality"] is not None]
    truncs = [v["trunc"] for v in by_plan.values() if v["trunc"] is not None]
    n_plans = len(by_plan)
    n_plan_allzero = sum(1 for q in qualities if q == 0.0)
    n_plan_allone = sum(1 for q in qualities if q == 1.0)

    return {
        "n_traj": n_traj, "n_plans": n_plans,
        "fmt_bad_rate": n_fmt_bad / n_traj if n_traj else None,
        "answer_rate": n_answered / n_traj if n_traj else None,
        "correct_rate_among_answered": n_correct / max(1, n_correct + n_wrong),
        "censored_rate": n_censored / n_traj if n_traj else None,
        "n_steps": n_steps,
        "deviation_rate": n_deviation / n_steps if n_steps else None,
        "plan_quality_mean": sum(qualities)/len(qualities) if qualities else None,
        "plan_quality_n": len(qualities),
        "plan_trunc_mean": sum(truncs)/len(truncs) if truncs else None,
        "plan_allzero_frac": n_plan_allzero / max(1, len(qualities)),
        "plan_allone_frac": n_plan_allone / max(1, len(qualities)),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj_a", required=True)
    ap.add_argument("--traj_b", required=True)
    ap.add_argument("--label_a", default="A")
    ap.add_argument("--label_b", default="B")
    args = ap.parse_args()

    sa = analyze(args.traj_a)
    sb = analyze(args.traj_b)

    rows = [
        ("轨迹总数", "n_traj", "{}"),
        ("plan总数", "n_plans", "{}"),
        ("格式违规率 (fmt_bad)", "fmt_bad_rate", "{:.1%}"),
        ("答案提取率", "answer_rate", "{:.1%}"),
        ("已判定里的正确率 (类pass@1)", "correct_rate_among_answered", "{:.1%}"),
        ("截断/无法判定率", "censored_rate", "{:.1%}"),
        ("总步骤数", "n_steps", "{}"),
        ("DEVIATION声明率", "deviation_rate", "{:.1%}"),
        ("plan_quality 均值", "plan_quality_mean", "{:.3f}"),
        ("plan_quality 有效样本数", "plan_quality_n", "{}"),
        ("plan_trunc_rate 均值", "plan_trunc_mean", "{:.3f}"),
        ("plan全错占比(quality=0)", "plan_allzero_frac", "{:.1%}"),
        ("plan全对占比(quality=1)", "plan_allone_frac", "{:.1%}"),
    ]
    name_w = max(len(r[0]) for r in rows)
    print(f"{'指标':<{name_w}}  {args.label_a:>12}  {args.label_b:>12}   差异")
    print("-" * (name_w + 40))
    for name, key, fmt in rows:
        va, vb = sa[key], sb[key]
        if va is None or vb is None:
            print(f"{name:<{name_w}}  {'n/a':>12}  {'n/a':>12}")
            continue
        sva, svb = fmt.format(va), fmt.format(vb)
        diff = ""
        if isinstance(va, float) and isinstance(vb, float):
            d = vb - va
            diff = f"  {d:+.3f}" if abs(d) < 1 else f"  {d:+.1f}"
        print(f"{name:<{name_w}}  {sva:>12}  {svb:>12}  {diff}")

if __name__ == "__main__":
    main()
