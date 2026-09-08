"""
inspect_A_win_B_lose.py — 追问inspect_plan_pruning_backtest.py那次"A对B错"
的具体案例: 输家plan被剪掉之后, 它原本贡献的票里, 有没有真的命中gold答案的
(不是噪声票, 是真正的"凑数"票)。

如果"A对B错"的案例里, 输家plan大多贡献了至少1条命中gold的票, 说明丢掉输家
丢的是真实有用的交叉印证, "给赢家补执行次数"这个思路大概率补不回来;
如果大多数案例里输家1条命中票都没贡献, 说明A赢B输更多是赢家自己4条内部的
并列偶然(可能跟clustering实现里的并列处理方式有关), 给赢家多采样大概率能解决。

用法: 跟inspect_plan_pruning_backtest.py一样的输入
  python inspect_A_win_B_lose.py --traj runs/eval/trajectories.jsonl \
      --shard_glob "runs/eval/bon_v6plus250.json.shard*.json" --problems ...
"""
import argparse, glob, json
from collections import defaultdict

try:
    from verifier import is_correct
    HAVE_VERIFIER = True
except ImportError:
    HAVE_VERIFIER = False
    def is_correct(a, g):
        try: return abs(float(a) - float(g)) < 1e-6
        except Exception: return str(a).strip() == str(g).strip()

def majority_answer_correct(cands, gold):
    valid = [c for c in cands if c is not None]
    if not valid: return None
    groups = []
    for a in valid:
        placed = False
        for g in groups:
            if is_correct(a, g[0]):
                g.append(a); placed = True; break
        if not placed:
            groups.append([a])
    groups.sort(key=len, reverse=True)
    return bool(is_correct(groups[0][0], gold)) if gold is not None else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    ap.add_argument("--shard_glob", required=True)
    ap.add_argument("--problems", required=True)
    args = ap.parse_args()

    problems = [json.loads(l) for l in open(args.problems)]
    trajs = [json.loads(l) for l in open(args.traj)]
    by_prob = defaultdict(list)
    for t in trajs:
        by_prob[t["prob_idx"]].append(t)

    plan_score_by_traj = {}
    for f in glob.glob(args.shard_glob):
        data = json.load(open(f))
        for k, v in data.get("plan_score_by_traj", {}).items():
            p, t = map(int, k.split(":"))
            plan_score_by_traj[(p, t)] = v

    n_AwinBlose = 0
    n_loser_contributed = 0   # 输家至少1条票命中gold
    n_loser_zero = 0          # 输家0条票命中gold(纯噪声, 没帮上忙)
    loser_hit_counts = []

    for prob_idx, cand in by_prob.items():
        if prob_idx >= len(problems): continue
        gold = problems[prob_idx].get("answer")
        if gold is None: continue

        by_plan = defaultdict(list)
        for tpos, t in enumerate(cand):
            by_plan[t["plan_idx"]].append(tpos)
        if len(by_plan) < 2: continue

        plan_avg_score = {}
        for plan_idx, tposs in by_plan.items():
            vals = [plan_score_by_traj.get((prob_idx, tp)) for tp in tposs]
            vals = [v for v in vals if v is not None]
            plan_avg_score[plan_idx] = sum(vals) / len(vals) if vals else 0.0
        winner_plan = max(plan_avg_score, key=plan_avg_score.get)
        loser_plans = [p for p in by_plan if p != winner_plan]
        winner_tposs = by_plan[winner_plan]
        loser_tposs = [tp for lp in loser_plans for tp in by_plan[lp]]

        all_answers = [cand[tp].get("answer") for tp in range(len(cand))]
        a_correct = majority_answer_correct(all_answers, gold)
        b_answers = [cand[tp].get("answer") for tp in winner_tposs]
        b_correct = majority_answer_correct(b_answers, gold)

        if a_correct and not b_correct:
            n_AwinBlose += 1
            loser_hits = sum(1 for tp in loser_tposs
                             if cand[tp].get("answer") is not None
                             and is_correct(cand[tp]["answer"], gold))
            loser_hit_counts.append(loser_hits)
            if loser_hits > 0:
                n_loser_contributed += 1
            else:
                n_loser_zero += 1

    print(f"'A对B错'的案例总数: {n_AwinBlose}")
    print(f"  输家plan贡献了>=1条命中gold的票: {n_loser_contributed} "
          f"({n_loser_contributed/n_AwinBlose*100:.1f}%)  <- 丢掉的是真实有用的交叉印证")
    print(f"  输家plan贡献了0条命中gold的票:   {n_loser_zero} "
          f"({n_loser_zero/n_AwinBlose*100:.1f}%)  <- 跟clustering并列处理方式有关, 不是真信号")
    if loser_hit_counts:
        print(f"  输家贡献票数分布: {sorted(loser_hit_counts)}")
    print("\n解读: 前者占比越高, 说明'给赢家补执行次数'这条思路越难解决问题"
          "(丢的是另一条独立路径的视角, 不是同一条路的噪声); "
          "后者占比越高, 说明问题更多出在小样本并列, 多采样大概率能解决。")

if __name__ == "__main__":
    main()
