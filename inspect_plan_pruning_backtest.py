"""
inspect_plan_pruning_backtest.py — 回放式验证: 如果用plan_score提前剪掉一个
plan(只留胜出的那个plan的N条候选, 不是全部M*N条都留着投票/排序), 能不能用
一半的计算量打平甚至超过"老老实实跑完全部候选再投票"的majority baseline。

完全复用已经存好的eval数据(trajectories.jsonl + bon分片里的plan_score预测),
不需要重新生成/重新打分。

对比三种方式, 在490题上各自的准确率:
  A. majority@全部(M*N条)         — 现有baseline, 用全部计算量
  B. majority@剪枝后(只留胜出plan的N条) — 只用一半计算量, 纯靠plan_score筛plan
  C. step_score重排@剪枝后(只留胜出plan的N条)  — 一半计算量 + 用PRM排序

如果B或C能追平甚至超过A, 说明"早期剪枝、把省下来的预算挪去做别的事"这条路
对你们是真实存在的结构性优势, 值得投入去建一套真正的实时剪枝流水线;
如果都明显不如A, 说明plan_score虽然能判断"这个plan好不好", 但还没好到能在
"少看一半候选"的情况下不丢信息, 这条路目前还不成熟。

用法:
  python inspect_plan_pruning_backtest.py --traj runs/eval/trajectories.jsonl \
      --shard_glob "runs/eval/bon_v6plus250.json.shard*.json"
"""
import argparse, glob, json
from collections import defaultdict, Counter

try:
    from verifier import is_correct
    HAVE_VERIFIER = True
except ImportError:
    HAVE_VERIFIER = False
    def is_correct(a, g):
        try: return abs(float(a) - float(g)) < 1e-6
        except Exception: return str(a).strip() == str(g).strip()

def majority_answer_correct(cands, gold):
    """cands: list of answer字符串(允许None)。聚类(数学等价归一类), 取最大簇,
    簇内任一答案跟gold比对一次即可代表整簇(同一簇内部对gold的判定理论上该一致,
    但仍以聚类自身的代表值实测为准, 不假设)。返回True/False/None(没有答案可投)。"""
    valid = [c for c in cands if c is not None]
    if not valid:
        return None
    groups = []
    for a in valid:
        placed = False
        for g in groups:
            if is_correct(a, g[0]):
                g.append(a); placed = True; break
        if not placed:
            groups.append([a])
    groups.sort(key=len, reverse=True)
    majority_ans = groups[0][0]
    return bool(is_correct(majority_ans, gold)) if gold is not None else None

def load_shards(shard_glob):
    step_score = defaultdict(dict)
    plan_score_by_traj = {}
    for f in glob.glob(shard_glob):
        data = json.load(open(f))
        for k, d in data.get("step_score", {}).items():
            p, t = map(int, k.split(":"))
            step_score[(p, t)].update({int(i): v for i, v in d.items()})
        for k, v in data.get("plan_score_by_traj", {}).items():
            p, t = map(int, k.split(":"))
            plan_score_by_traj[(p, t)] = v
    return step_score, plan_score_by_traj

def aggregate_step(seq, how="mean"):
    xs = [s if s is not None else 0.5 for s in seq]
    if not xs: return 0.5
    if how == "min": return min(xs)
    if how == "last": return xs[-1]
    return sum(xs) / len(xs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    ap.add_argument("--shard_glob", required=True)
    ap.add_argument("--gold_field", default="answer_gold",
                    help="如果trajectories.jsonl里没有直接存gold答案, 用--problems "
                         "另外传; 这里默认尝试从trajectory的correct字段反推无效, "
                         "实际需要gold文本来做majority判断, 见下方--problems")
    ap.add_argument("--problems", required=True,
                    help="跟当时生成candidate用的同一份--problems文件, 用来取gold answer")
    args = ap.parse_args()

    problems = [json.loads(l) for l in open(args.problems)]
    trajs = [json.loads(l) for l in open(args.traj)]
    by_prob = defaultdict(list)
    for t in trajs:
        by_prob[t["prob_idx"]].append(t)
    step_score, plan_score_by_traj = load_shards(args.shard_glob)

    hit_A, hit_B, hit_C, n_total = 0, 0, 0, 0
    plan_selection_quality = Counter()  # 验证: plan_score挑的plan是不是真的更好的那个

    for prob_idx, cand in by_prob.items():
        if prob_idx >= len(problems): continue
        gold = problems[prob_idx].get("answer")
        if gold is None: continue
        n_total += 1

        by_plan = defaultdict(list)
        for tpos, t in enumerate(cand):
            by_plan[t["plan_idx"]].append(tpos)
        if len(by_plan) < 2:
            continue   # 只有1个plan, 没有"剪枝选哪个"这个问题, 跳过(不计入对比)

        # A: 全部候选投票
        all_answers = [cand[tpos].get("answer") for tpos in range(len(cand))]
        a_correct = majority_answer_correct(all_answers, gold)
        if a_correct: hit_A += 1

        # 按各plan的平均plan_score排名, 选出"胜出"的plan
        plan_avg_score = {}
        for plan_idx, tposs in by_plan.items():
            vals = [plan_score_by_traj.get((prob_idx, tp)) for tp in tposs]
            vals = [v for v in vals if v is not None]
            plan_avg_score[plan_idx] = sum(vals) / len(vals) if vals else 0.0
        winner_plan = max(plan_avg_score, key=plan_avg_score.get)
        winner_tposs = by_plan[winner_plan]

        # 验证: plan_score选的这个plan, 真实成功率是不是确实更高(或至少不差)
        real_success = {pid: sum(1 for tp in tposs if cand[tp].get("correct") is True) / len(tposs)
                        for pid, tposs in by_plan.items()}
        best_real_plan = max(real_success, key=real_success.get)
        plan_selection_quality["选对(plan_score选的就是真实更好的那个)" if winner_plan == best_real_plan
                               else "选错(plan_score选的不是真实更好的那个)"] += 1

        # B: 只用胜出plan的候选投票
        b_answers = [cand[tp].get("answer") for tp in winner_tposs]
        b_correct = majority_answer_correct(b_answers, gold)
        if b_correct: hit_B += 1

        # C: 只用胜出plan的候选, 按step_score(mean聚合)排序选最高分那条
        scored = []
        for tp in winner_tposs:
            seq = [step_score.get((prob_idx, tp), {}).get(i)
                   for i in range(len(cand[tp]["steps"]))]
            scored.append((aggregate_step(seq, "mean"), tp))
        best_tp = max(scored, key=lambda x: x[0])[1]
        c_correct = bool(cand[best_tp].get("correct"))
        if c_correct: hit_C += 1

    print(f"共{n_total}题(且plan数>=2, 真正存在'剪枝选哪个'问题的题目)")
    print(f"\nA. majority@全部候选(基线, 用全部计算量):        {hit_A}/{n_total} = {hit_A/n_total*100:.1f}%")
    print(f"B. majority@剪枝后(只用一半计算量, 纯plan_score筛): {hit_B}/{n_total} = {hit_B/n_total*100:.1f}%")
    print(f"C. step_score重排@剪枝后(一半计算量+PRM排序):       {hit_C}/{n_total} = {hit_C/n_total*100:.1f}%")
    print(f"\nplan_score剪枝决策本身的质量(跟真实成功率比对): {dict(plan_selection_quality)}")
    print("\n解读: 如果B或C能追平甚至超过A, 说明用一半计算量、靠plan_score提前剪枝,")
    print("是个真实存在的结构性优势; 如果明显不如A, 说明这条路目前还不成熟,")
    print("plan_score的判断力还不足以支撑'少看一半候选也不丢信息'这件事。")

if __name__ == "__main__":
    main()
