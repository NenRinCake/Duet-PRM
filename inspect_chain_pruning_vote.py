"""
inspect_chain_pruning_vote.py — 按"链"这个细粒度(不是plan粒度)剪枝: 把8条
候选各自的PRM分算出来, 按不同的"保留比例"踢掉分数最低的那一批, 剩下的链
再做纯多数投票, 看能不能打平/超过"全部8条都投票"的majority baseline。

跟inspect_plan_pruning_backtest.py的关键区别: 那次是整个扔掉一个plan(粗粒度,
要求plan_score能精确判断哪个plan整体更好); 这次是逐条链各自评分, 只要求
"分得清明显差的链", 不要求精确排出谁是最好的那一条 —— 这是个结构上更容易
达成的任务, 用现有的AUC/分离度证据来看, 有理由认为这次更可能成功。

完全复用已存在的eval shard数据(step_score + plan_score_by_traj), 不需要
重新打分/重新生成。

用法:
  python inspect_chain_pruning_vote.py --traj runs/eval/trajectories.jsonl \
      --shard_glob "runs/eval/bon_v6plus250.json.shard*.json" --combine multiply --how mean
"""
import argparse, glob, json, random
from collections import defaultdict, Counter
import eval_bon as eb   # 直接复用aggregate()/combine_with_plan(), 不再自己另写一份

def majority_correct(kept_cands):
    """跟eval_bon.py的bon_majority逐字同一套算法: 按答案字符串完全相等分组
    计票(不做数学等价聚类), 用候选自带的correct字段判断对错(不重新调用
    verifier跟gold比) —— 这样keep_frac=1.0这一行才能精确复现75.9%那个
    基准, 不会因为换了一套不同的计票规则而产生额外的、跟剪枝本身无关的
    数字漂移。之前的版本用is_correct()做数学等价聚类计票, 跟这里不是
    同一套算法, 1.0这一行算出来是76.9%而不是75.9%, 就是这个不一致导致的。"""
    votes = Counter(t["answer"] for t in kept_cands if t.get("answer") is not None)
    if not votes: return False
    win_ans = votes.most_common(1)[0][0]
    pick = next(t for t in kept_cands if t.get("answer") == win_ans)
    return bool(pick.get("correct"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    ap.add_argument("--shard_glob", required=True)
    ap.add_argument("--combine", default="multiply",
                    choices=["none", "multiply", "weighted", "plan_trust"])
    ap.add_argument("--how", default="mean",
                    choices=["min", "mean", "last", "hybrid", "confident_mean", "adaptive_mean"])
    ap.add_argument("--plan_weight", type=float, default=0.3,
                    help="weighted模式下plan_score的权重, 跟eval_bon.py的--plan_weight同义")
    args = ap.parse_args()

    trajs = [json.loads(l) for l in open(args.traj)]
    by_prob = defaultdict(list)
    for t in trajs: by_prob[t["prob_idx"]].append(t)

    step_score = defaultdict(dict); plan_score_by_traj = {}
    files = sorted(glob.glob(args.shard_glob))
    if not files:
        raise SystemExit(f"未匹配到任何分片文件: {args.shard_glob} —— 路径写错了, "
                         f"不会继续往下跑(之前的版本这里不报错, 会悄悄算出一张"
                         f"看似正常但跟PRM打分完全无关的表格, 已修复)")
    covered = set()
    for f in files:
        data = json.load(open(f))
        for k, d in data.get("step_score", {}).items():
            p, t = map(int, k.split(":"))
            step_score[(p, t)].update({int(i): v for i, v in d.items()})
            covered.add(p)
        for k, v in data.get("plan_score", {}).items():
            p, t = map(int, k.split(":"))
            plan_score_by_traj[(p, t)] = v
    missing = set(by_prob.keys()) - covered
    if missing:
        print(f"  ⚠ 警告: {len(missing)}/{len(by_prob)} 题未被任何分片覆盖, "
              f"这些题目会退化成step_score全部按0.5处理(跟没有真实PRM分数一样)")

    keep_fracs = [1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25]
    results = {kf: {"hit": 0, "n": 0, "n_pruned_wrong": 0, "n_pruned_correct": 0} for kf in keep_fracs}
    N_RANDOM_TRIALS = 20
    rng = random.Random(42)
    random_results = {kf: {"hit": 0, "n": 0} for kf in keep_fracs}

    for prob_idx, cand in by_prob.items():
        scored = []
        for tpos, t in enumerate(cand):
            ss = step_score.get((prob_idx, tpos), {})
            seq = [ss.get(i) for i in range(len(t["steps"]))]
            agg = eb.aggregate(seq, args.how)
            psc = plan_score_by_traj.get((prob_idx, tpos))
            final = eb.combine_with_plan(agg, psc, args.combine, args.plan_weight)
            scored.append((final, tpos))
        scored.sort(key=lambda x: x[0], reverse=True)
        N = len(scored)

        for kf in keep_fracs:
            n_keep = max(1, round(N * kf))
            kept_tposs = sorted(tp for _, tp in scored[:n_keep])
            # 排序回原始tpos顺序(不是按分数排出来的顺序)再喂给投票函数 ——
            # 如果直接用按分数排出来的顺序, 遇到票数打平的题目时,
            # Counter.most_common()会按"谁先被数到"决定平局赢家, 而"谁先被
            # 数到"恰好就是分数排序的副产物, 跟分数判断准不准毫无关系。这样
            # 改完, "留下谁"仍然受分数影响(这正是剪枝要测的), 但"留下的这些
            # 人怎么投票/怎么打破平局"永远跟eval_bon.py的bon_majority一样,
            # 不会因为换一个打分策略而无端改变平局结果。
            pruned_tposs = [tp for _, tp in scored[n_keep:]]
            kept_cands = [cand[tp] for tp in kept_tposs]
            hit = majority_correct(kept_cands)
            results[kf]["hit"] += int(hit)
            results[kf]["n"] += 1
            for tp in pruned_tposs:
                if cand[tp].get("correct") is True:
                    results[kf]["n_pruned_correct"] += 1
                elif cand[tp].get("correct") is False:
                    results[kf]["n_pruned_wrong"] += 1

            # 随机剪枝对照组: 完全不看PRM分数, 随机丢掉同样比例的链, 重复
            # N_RANDOM_TRIALS次取平均 —— 这是用来判断"正确率没掉"到底是PRM
            # 真的挑对了, 还是单纯多数投票本身对丢票有冗余容忍度的关键对照。
            all_tposs = list(range(N))
            for _ in range(N_RANDOM_TRIALS):
                rng.shuffle(all_tposs)
                rand_kept = [cand[tp] for tp in all_tposs[:n_keep]]
                random_results[kf]["hit"] += int(majority_correct(rand_kept))
                random_results[kf]["n"] += 1

    print(f"{'保留比例':<10}{'剩余链数(约)':<14}{'PRM剪枝正确率':<14}{'随机剪枝正确率':<14}"
          f"{'剪掉的链里_错的':<16}{'剪掉的链里_对的':<16}")
    for kf in keep_fracs:
        r = results[kf]; rr = random_results[kf]
        acc = r["hit"] / r["n"] if r["n"] else 0
        racc = rr["hit"] / rr["n"] if rr["n"] else 0
        print(f"{kf:<10.3f}{round(8*kf):<14}{acc*100:<13.1f}%{racc*100:<13.1f}%"
              f"{r['n_pruned_wrong']:<16}{r['n_pruned_correct']:<16}")

    print("\n解读: 重点看'PRM剪枝正确率'和'随机剪枝正确率'这两列是否接近 ——")
    print("如果两列几乎一样, 说明'正确率没掉'纯粹是多数投票本身的冗余容忍度撑住的,")
    print("跟PRM打分准不准没有关系; 只有PRM剪枝那一列明显高于随机剪枝那一列,")
    print("才能说明PRM的打分真的在帮忙挑掉该剪的链。同时看后两列: 把'剪掉的链里")
    print("_对的'除以'剪掉的链里_错的', 跟整体数据里对:错的基础比例(约2.0)比较——")
    print("如果这个比例明显高于2.0且持续走高, 说明PRM排在末尾的链里, 正确的反而")
    print("比整体平均水平更多, 这是比'没有区分力'更糟的情况, 不是单纯无效。")

    # ===== 第二组: 绝对分数线丢弃(不是相对排名) =====
    # 跟上面那组的关键区别: 上面是"每题固定砍掉排名最后的x%, 不管这x%的实际
    # 分数高低"; 这里是"不管排第几, 只要分数低于这条线就丢, 高于线的全部留下"
    # —— 这意味着"整体都打高分"的题目会自然留下更多候选, "整体都打低分"的
    # 题目会自然留下更少甚至全部被丢弃(此时退回去保留全部候选, 不能让一道题
    # 真的0候选可投票)。这是更贴近"绝对质量门槛"直觉的丢弃方式, 跟相对排名
    # 是两种不同的机制, 不能假设两者结果一样, 需要单独测。
    print("\n" + "=" * 70)
    print("第二组: 绝对分数线丢弃(不看排名, 只看分数本身高低于阈值)")
    print("=" * 70)
    thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    thresh_results = {th: {"hit": 0, "n": 0, "n_pruned_wrong": 0, "n_pruned_correct": 0,
                           "n_survivors_sum": 0, "n_fallback": 0} for th in thresholds}
    thresh_random = {th: {"hit": 0, "n": 0} for th in thresholds}

    for prob_idx, cand in by_prob.items():
        per_chain_score = []
        for tpos, t in enumerate(cand):
            ss = step_score.get((prob_idx, tpos), {})
            seq = [ss.get(i) for i in range(len(t["steps"]))]
            agg = eb.aggregate(seq, args.how)
            psc = plan_score_by_traj.get((prob_idx, tpos))
            per_chain_score.append(eb.combine_with_plan(agg, psc, args.combine, args.plan_weight))
        N = len(cand)

        for th in thresholds:
            survivors = [tp for tp in range(N) if per_chain_score[tp] >= th]
            fallback = not survivors
            if fallback:
                survivors = list(range(N))   # 全军覆没时退回保留全部, 不能让这道题没法投票
                thresh_results[th]["n_fallback"] += 1
            pruned = [tp for tp in range(N) if tp not in survivors]
            kept_cands = [cand[tp] for tp in survivors]
            hit = majority_correct(kept_cands)
            thresh_results[th]["hit"] += int(hit)
            thresh_results[th]["n"] += 1
            thresh_results[th]["n_survivors_sum"] += len(survivors)
            for tp in pruned:
                if cand[tp].get("correct") is True:
                    thresh_results[th]["n_pruned_correct"] += 1
                elif cand[tp].get("correct") is False:
                    thresh_results[th]["n_pruned_wrong"] += 1

            # 随机对照: 同一批分数值, 但随机打乱"哪个分数对应哪条链"的配对关系,
            # 看丢弃机制本身(不看具体配对)能不能撑住正确率 —— 如果能, 说明
            # 哪条链拿到哪个分数其实不重要, 分数跟链的真实对应关系没有信息量。
            n_keep_equiv = len(survivors) if not fallback else N
            all_tposs = list(range(N))
            for _ in range(N_RANDOM_TRIALS):
                rng.shuffle(all_tposs)
                rand_kept = [cand[tp] for tp in all_tposs[:n_keep_equiv]]
                thresh_random[th]["hit"] += int(majority_correct(rand_kept))
                thresh_random[th]["n"] += 1

    print(f"{'阈值':<8}{'平均剩余链数':<14}{'全军覆没题数':<14}{'PRM剪枝正确率':<14}"
          f"{'随机对照正确率':<14}{'剪掉_错':<10}{'剪掉_对':<10}")
    n_prob_total = len(by_prob)
    for th in thresholds:
        r = thresh_results[th]; rr = thresh_random[th]
        acc = r["hit"] / r["n"] if r["n"] else 0
        racc = rr["hit"] / rr["n"] if rr["n"] else 0
        avg_surv = r["n_survivors_sum"] / r["n"] if r["n"] else 0
        print(f"{th:<8.1f}{avg_surv:<14.2f}{r['n_fallback']:<14}{acc*100:<13.1f}%"
              f"{racc*100:<13.1f}%{r['n_pruned_wrong']:<10}{r['n_pruned_correct']:<10}")
    print(f"\n('全军覆没题数'指该阈值下有多少题8条全部低于阈值、被迫退回保留全部, "
          f"共{n_prob_total}题)")
    print("解读跟上面那组一致: 看PRM剪枝正确率是否明显高于随机对照, 以及剪掉的")
    print("链里对错比例是否明显偏向'错的更多'。阈值太高时'全军覆没题数'会变大,")
    print("这部分题目其实没有被真正剪枝(退回了保留全部), 解读时要把这个考虑进去。")

if __name__ == "__main__":
    main()