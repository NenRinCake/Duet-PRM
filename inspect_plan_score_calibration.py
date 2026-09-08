"""
inspect_plan_score_calibration.py — 测"plan_score有没有学到它被训练去预测的
目标(success_rate)", 不经过任何选择/投票策略, 纯粹是held-out相关性+校准。
跟BoN/majority/链级剪枝这些"下游任务"实验是两个不同性质的问题: 那些问的是
"拿这个分数去做选择有没有净收益", 这里问的是"模型预测的分数, 跟它被训练
去预测的目标, 吻合得怎么样" —— 后者不该被"下游任务表现一般"这件事连带
否定, 两者要分开看。

支持两种plan_score来源(用--source切换):
  probe   inspect_plan_only_probe.py的零执行探测结果(plan_only_probe.json)
  shard   eval_bon.py的eval shard文件(bon_*.json.shard*.json里的plan_score
          字段, 全程平均后的版本)

用法:
  python inspect_plan_score_calibration.py --traj runs/eval/trajectories.jsonl \
      --source probe --probe_json runs/eval/plan_probe_full01.json
  python inspect_plan_score_calibration.py --traj runs/eval/trajectories.jsonl \
      --source shard --shard_glob "runs/eval/bon_v6plus200-full01.json.shard*.json"
"""
import argparse, glob, json
from collections import defaultdict

def build_plan_groups(trajs):
    groups = defaultdict(list)
    for t in trajs:
        groups[(t["prob_idx"], t["plan_idx"])].append(t)
    plan_info = {}
    for (p, pl), ts in groups.items():
        decided = [t for t in ts if t.get("correct") is not None]
        real_acc = sum(1 for t in decided if t["correct"]) / len(decided) if decided else None
        plan_info[(p, pl)] = {"real_acc": real_acc, "n": len(ts)}
    return plan_info

def spearman(xs, ys):
    """手写Spearman相关系数(秩相关), 不依赖scipy。"""
    n = len(xs)
    if n < 2: return None
    def rank(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        ranks = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j+1]] == vals[order[i]]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1
            for k in range(i, j+1):
                ranks[order[k]] = avg_rank
            i = j + 1
        return ranks
    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx)/n, sum(ry)/n
    cov = sum((a-mx)*(b-my) for a, b in zip(rx, ry))
    vx = sum((a-mx)**2 for a in rx); vy = sum((b-my)**2 for b in ry)
    if vx < 1e-12 or vy < 1e-12: return None
    return cov / (vx*vy) ** 0.5

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    ap.add_argument("--source", required=True, choices=["probe", "shard"])
    ap.add_argument("--probe_json", default=None)
    ap.add_argument("--shard_glob", default=None)
    args = ap.parse_args()
    if args.source == "probe" and not args.probe_json:
        ap.error("--source probe 需要 --probe_json")
    if args.source == "shard" and not args.shard_glob:
        ap.error("--source shard 需要 --shard_glob")

    trajs = [json.loads(l) for l in open(args.traj)]
    plan_info = build_plan_groups(trajs)

    pred_scores = {}
    if args.source == "probe":
        raw = json.load(open(args.probe_json))["probe_scores"]
        for k, v in raw.items():
            p_str, pl_str = k.split(":")
            pred_scores[(int(p_str), int(pl_str))] = v[0]   # (plan_score, step_score)的第0个
    else:
        for f in glob.glob(args.shard_glob):
            data = json.load(open(f))
            for k, v in data.get("plan_score", {}).items():
                p_str, pl_str = k.split(":")
                pred_scores[(int(p_str), int(pl_str))] = v

    xs, ys = [], []
    for k, real_acc in ((k, v["real_acc"]) for k, v in plan_info.items()):
        if real_acc is None: continue
        psc = pred_scores.get(k)
        if psc is None: continue
        xs.append(psc); ys.append(real_acc)

    print(f"覆盖的plan数: {len(xs)}")
    rho = spearman(xs, ys)
    print(f"Spearman相关系数 (预测plan_score vs 真实success_rate): "
          f"{rho:.4f}" if rho is not None else "n/a (数据不足或全部相同)")
    print("(0=完全无关, 1=完美单调一致, -1=完全反向; 这是held-out相关性,")
    print(" 不涉及任何选择/投票策略, 跟BoN/majority的表现是两个独立的问题)")

    bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
    print(f"\n{'预测plan_score区间':<22}{'plan数':<10}{'真实success_rate均值':<20}")
    for lo, hi in bins:
        vals = [y for x, y in zip(xs, ys) if lo <= x < hi]
        if not vals:
            print(f"{lo:.1f}-{hi:.1f}{'':<14}{0:<10}{'n/a':<20}")
            continue
        print(f"{lo:.1f}-{hi:.1f}{'':<14}{len(vals):<10}{sum(vals)/len(vals):<20.3f}")
    print("\n解读: 如果区间从低到高, 真实success_rate均值也单调从低到高走,")
    print("说明plan_score确实学到了它被训练去预测的目标(校准方向是对的);")
    print("如果不单调或者各区间数值差不多, 说明哪怕不涉及任何下游选择策略,")
    print("模型本身预测的plan_score也没有准确反映真实的success_rate。")

if __name__ == "__main__":
    main()
