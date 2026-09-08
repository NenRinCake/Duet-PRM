"""
analyze_step_score.py — 诊断: 模型预测的step_score到底准不准, 尤其是预测成
0.5(UNCERTAIN)的那些, 真实标签到底是BAD/UNCERTAIN/GOOD里哪一类占多数。

跟analyze_plan_score.py是同一个套路, 只是这次比对的是step级(不是plan级)的
预测值, 需要先给eval/test的trajectories跑一遍label_minimal.py+verify_rules.py
生成真值(这一步之前没做过, 测试集本来不需要标注, 这次为了验证才补上)。

用法:
  python analyze_step_score.py \
      --traj runs/eval/trajectories.jsonl \
      --step_labels runs/eval/step_labels_gt.jsonl \
      --c2 runs/eval/c2_labels_gt.jsonl \
      --shard_glob "runs/eval/bon_v6plus250.json.shard*.json"
"""
import argparse, json, glob
from collections import defaultdict, Counter

# ---- 跟assemble_dataset.py/diff_relabeling.py完全一致的压缩表 ----
BAD_PROV = {"localized_fail", "suspect"}
BAD_ADH = {"deviated", "claimed_only"}
UNCERTAIN_PROV = {"unverified"}
UNCERTAIN_ADH = {"self_declared_deviation"}
GOOD_PROV = {"corroborated", "outcome_consistent"}

def compute_step_score(adherence, provenance):
    if provenance in BAD_PROV: return 0.0
    if adherence in BAD_ADH: return 0.0
    if adherence in UNCERTAIN_ADH: return 0.5
    if provenance in UNCERTAIN_PROV: return 0.5
    if provenance in GOOD_PROV: return 1.0
    return 0.5

def bucketize(v):
    if v is None: return None
    if v <= 0.25: return 0.0
    if v >= 0.75: return 1.0
    return 0.5

BUCKET_NAMES = {0.0: "BAD", 0.5: "UNCERTAIN", 1.0: "GOOD"}

def load_ground_truth(traj_path, step_labels_path, c2_path):
    """返回 {(prob_idx, tpos, step_i): true_step_score}, tpos按文件读取顺序排,
    跟eval_bon.py的by_prob_full构建方式完全一致(简单顺序append)。"""
    trajs = [json.loads(l) for l in open(traj_path)]
    by_prob = defaultdict(list)
    for t in trajs:
        by_prob[t["prob_idx"]].append(t)

    adh = {}
    for line in open(step_labels_path):
        r = json.loads(line)
        key = (r["prob_idx"], r["plan_idx"], r["exec_idx"], r["step"])
        a = r.get("adherence") or r.get("b2") or "unknown"
        adh[key] = "uncertain" if str(a).startswith("uncertain") else a
    prov = {}
    for line in open(c2_path):
        r = json.loads(line)
        key = (r["prob_idx"], r["plan_idx"], r["exec_idx"], r["step"])
        prov[key] = r.get("provenance", "unverified")

    gt = {}
    for prob_idx, cand in by_prob.items():
        for tpos, t in enumerate(cand):
            for i, s in enumerate(t["steps"]):
                key4 = (t["prob_idx"], t["plan_idx"], t["exec_idx"], i)
                a = adh.get(key4, "unknown")
                p = prov.get(key4, "unverified")
                gt[(prob_idx, tpos, i)] = compute_step_score(a, p)
    return gt

def load_predicted(shard_glob):
    pred = {}
    for f in glob.glob(shard_glob):
        data = json.load(open(f))
        for k, d in data.get("step_score", {}).items():
            p_str, t_str = k.split(":")
            p, t = int(p_str), int(t_str)
            for i_str, v in d.items():
                pred[(p, t, int(i_str))] = v
    return pred

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", default="runs/eval/trajectories.jsonl")
    ap.add_argument("--step_labels", required=True)
    ap.add_argument("--c2", required=True)
    ap.add_argument("--shard_glob", required=True)
    args = ap.parse_args()

    gt = load_ground_truth(args.traj, args.step_labels, args.c2)
    pred = load_predicted(args.shard_glob)

    pairs = [(gt[k], pred[k]) for k in pred if k in gt and pred[k] is not None]
    n_missing = sum(1 for k in pred if k not in gt)
    n_pred_none = sum(1 for k in pred if pred.get(k) is None)
    print(f"匹配上的步骤数: {len(pairs)}  (真值缺失被跳过: {n_missing}, "
          f"预测为None/parse_fail被跳过: {n_pred_none})")
    if not pairs:
        print("没有匹配上任何数据, 检查路径/真值文件是否存在")
        return

    labels = [0.0, 0.5, 1.0]
    cm = {t: {p: 0 for p in labels} for t in labels}
    for true_v, pred_raw in pairs:
        pb = bucketize(pred_raw)
        if pb is None: continue
        cm[true_v][pb] += 1

    print("\n" + "=" * 64)
    print("  混淆矩阵 (真实step_score档 -> 模型预测档, 行=真实/列=预测)")
    print("=" * 64)
    row_label = "真实\\预测"
    header = f"  {row_label:<12}" + "".join(f"{BUCKET_NAMES[p]:>12}" for p in labels) + f"{'合计':>8}"
    print(header)
    for t in labels:
        row_total = sum(cm[t].values())
        row = f"  {BUCKET_NAMES[t]:<12}"
        for p in labels:
            cnt = cm[t][p]
            pct = f"{cnt/row_total*100:.0f}%" if row_total else "-"
            cell = f"{cnt}({pct})"
            row += f"{cell:>12}"
        row += f"{row_total:>8}"
        print(row)

    total = sum(sum(cm[t].values()) for t in labels)
    correct = sum(cm[t][t] for t in labels)
    print(f"\n总体准确率: {correct}/{total} = {correct/total*100:.1f}%  (瞎猜基线33.3%)")
    print("各档召回率 (真实是这一档, 预测对的占多少):")
    for t in labels:
        row_total = sum(cm[t].values())
        recall = cm[t][t] / row_total if row_total else None
        print(f"  {BUCKET_NAMES[t]:<10}: {recall*100:.1f}%" if recall is not None
              else f"  {BUCKET_NAMES[t]:<10}: n/a")

    # ---- 用户真正想知道的问题: 预测成UNCERTAIN(0.5)的, 真实标签分布是什么 ----
    print("\n" + "=" * 64)
    print("  你问的问题: 预测=UNCERTAIN(0.5) 这一列, 真实标签的分布")
    print("=" * 64)
    pred_unc_total = sum(cm[t][0.5] for t in labels)
    if pred_unc_total == 0:
        print("  模型从来没有预测过0.5, 这一列是空的")
    else:
        for t in labels:
            cnt = cm[t][0.5]
            print(f"  真实是{BUCKET_NAMES[t]:<10}: {cnt:>5} / {pred_unc_total} "
                  f"= {cnt/pred_unc_total*100:.1f}%")
        right = cm[0.5][0.5]
        wrong = pred_unc_total - right
        print(f"\n  -> 预测成UNCERTAIN里, 真实confirmed对的(GOOD)+错的(BAD)"
              f"共占 {(cm[0.0][0.5]+cm[1.0][0.5])/pred_unc_total*100:.1f}%, "
              f"真的是UNCERTAIN的占 {right/pred_unc_total*100:.1f}%")
        if cm[1.0][0.5] > cm[0.0][0.5]:
            print("  -> 真实GOOD被错判成UNCERTAIN的情况更多 "
                  "(模型对本该confirm的步骤过度谨慎)")
        elif cm[0.0][0.5] > cm[1.0][0.5]:
            print("  -> 真实BAD被错判成UNCERTAIN的情况更多 "
                  "(模型把本该挑出来的错误步骤藏进了模糊地带)")

if __name__ == "__main__":
    main()
