"""
inspect_gap_stratified.py — 验证"选对率高但实际提升微小"这件事是不是因为
大多数题目里两个plan的真实差距本来就很小。按真实差距(gap)分组, 看选对率
和实际提升幅度是不是集中在"差距大"的那一组。

需要inspect_plan_only_probe.py跑完保存的probe_scores(plan_only_probe.json
里的probe_scores字段), 复用同一份build_plan_groups逻辑, 不需要重新打分。

用法:
  python inspect_gap_stratified.py --traj runs/eval/trajectories.jsonl \
      --probe_json runs/eval/plan_only_probe.json
"""
import argparse, json
from collections import defaultdict
from inspect_plan_only_probe import build_plan_groups

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    ap.add_argument("--probe_json", required=True)
    args = ap.parse_args()

    trajs = [json.loads(l) for l in open(args.traj)]
    plan_info = build_plan_groups(trajs)
    raw = json.load(open(args.probe_json))["probe_scores"]
    probe_scores = {}
    for k, v in raw.items():
        p_str, pl_str = k.split(":")
        probe_scores[(int(p_str), int(pl_str))] = tuple(v)

    by_prob = defaultdict(list)
    for (p, pl) in plan_info:
        by_prob[p].append(pl)

    bins = [(0.0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 1.01)]
    bin_stats = {b: {"n": 0, "n_correct_pick": 0, "lift_sum": 0.0} for b in bins}

    for p, pls in by_prob.items():
        if len(pls) < 2: continue
        real_accs = {pl: plan_info[(p, pl)]["real_acc"] for pl in pls
                    if plan_info[(p, pl)]["real_acc"] is not None}
        scored = {pl: probe_scores.get((p, pl), (None, None))[0] for pl in pls}
        scored = {pl: s for pl, s in scored.items() if s is not None}
        if len(real_accs) < 2 or len(scored) < 2:
            continue
        gap = max(real_accs.values()) - min(real_accs.values())
        best_real_pl = max(real_accs, key=real_accs.get)
        winner_pl = max(scored, key=scored.get)
        avg_acc_no_select = sum(real_accs.values()) / len(real_accs)
        winner_acc = real_accs.get(winner_pl)
        if winner_acc is None: continue
        lift = winner_acc - avg_acc_no_select
        for lo, hi in bins:
            if lo <= gap < hi:
                bin_stats[(lo, hi)]["n"] += 1
                bin_stats[(lo, hi)]["n_correct_pick"] += int(winner_pl == best_real_pl)
                bin_stats[(lo, hi)]["lift_sum"] += lift
                break

    print(f"{'gap区间':<12}{'题目数':>8}{'选对率':>10}{'平均提升':>12}")
    for b in bins:
        st = bin_stats[b]
        if st["n"] == 0:
            print(f"{b[0]:.1f}-{b[1]:.1f}{'':<6}{0:>8}{'n/a':>10}{'n/a':>12}")
            continue
        acc = st["n_correct_pick"] / st["n"]
        avg_lift = st["lift_sum"] / st["n"]
        print(f"{b[0]:.1f}-{b[1]:.1f}{'':<6}{st['n']:>8}{acc*100:>9.1f}%{avg_lift*100:>+11.2f}pp")

    print("\n解读: 如果'平均提升'随着gap区间增大而明显增大, 说明probe的价值确实")
    print("集中在'两个plan真实差距大'的题目上, 只是这类题目占比不高, 被大量")
    print("'差距很小、选哪个都差不多'的题目平均稀释掉了 —— 这种情况下, 哪怕")
    print("整体平均提升很小, 针对性地只在probe给出'高置信度差距大'信号时才剪枝,")
    print("仍然可能是有价值的; 如果'平均提升'在各区间都同样小, 说明probe本身")
    print("即便面对真实差距大的情况也分不出来, 这条路的价值就更有限。")

if __name__ == "__main__":
    main()
