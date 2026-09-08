"""
inspect_excluded_step_outcomes.py — 查"被剔除的步骤"(原本落进0.5/兜底, 在
rebuild_step_binary.py里被排除出训练集的那批)所在的轨迹, 最终答案是对的多
还是错的多。

动机: 如果这批"无法判定"的步骤所在轨迹的正确率明显偏离pass@1基线(66.7%),
说明现有的跨执行投票验证方法漏掉了某种真实存在的信号; 如果跟基线很接近,
说明这批步骤是真的没有可提取的信号, 强行按轨迹结果给它们打标签纯粹是注入噪声。

同时给BAD/GOOD两组已确定步骤的同款统计当对照, 以及"按步骤数"和"按去重后
轨迹数"两个视角(同一条轨迹可能贡献多个被剔除的步骤, 只看步骤数会被这种
重复贡献带偏)。

用法:
  python inspect_excluded_step_outcomes.py --in_file runs/v0/sft_dataset_v6plus250.jsonl
"""
import argparse, json
from collections import Counter

BAD_PROV = {"localized_fail", "suspect"}
BAD_ADH = {"deviated", "claimed_only"}
UNCERTAIN_PROV = {"unverified"}
UNCERTAIN_ADH = {"self_declared_deviation"}
GOOD_PROV = {"corroborated", "outcome_consistent"}

def compute_step_score_binary(adherence, provenance):
    if provenance in BAD_PROV: return 0.0
    if adherence in BAD_ADH: return 0.0
    if adherence in UNCERTAIN_ADH: return None
    if provenance in UNCERTAIN_PROV: return None
    if provenance in GOOD_PROV: return 1.0
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_file", required=True)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.in_file)]
    seen = set()
    base_rows = []
    for r in rows:
        key = (r["meta_prob_idx"], r["meta_plan_idx"], r["meta_exec_idx"], r["meta_step"])
        if key not in seen:
            seen.add(key)
            base_rows.append(r)

    groups = {"excluded": [], "BAD(0.0)": [], "GOOD(1.0)": []}
    for r in base_rows:
        score = compute_step_score_binary(r["meta_adherence"], r["meta_execution_provenance"])
        bucket = "excluded" if score is None else ("BAD(0.0)" if score == 0.0 else "GOOD(1.0)")
        groups[bucket].append(r)

    def report(name, rs):
        n = len(rs)
        c = Counter(r.get("meta_traj_correct") for r in rs)
        true_n, false_n, none_n = c.get(True, 0), c.get(False, 0), c.get(None, 0)
        decided = true_n + false_n
        rate = true_n / decided * 100 if decided else float("nan")
        print(f"  [{name}] 共{n}个步骤  正确轨迹={true_n}({true_n/n*100:.1f}%)  "
              f"错误轨迹={false_n}({false_n/n*100:.1f}%)  无答案(censored)={none_n}({none_n/n*100:.1f}%)"
              f"  | 在已判定里的正确率(类pass@1)={rate:.1f}%")

    print("=" * 70)
    print("  按步骤数统计 (同一轨迹的多个步骤会被重复计入)")
    print("=" * 70)
    for name in ("excluded", "BAD(0.0)", "GOOD(1.0)"):
        report(name, groups[name])

    print("\n" + "=" * 70)
    print("  按去重后的轨迹数统计 (同一条轨迹只数一次, 避免被'一条轨迹贡献"
          "好几个被剔除步骤'这种情况带偏)")
    print("=" * 70)
    for name in ("excluded", "BAD(0.0)", "GOOD(1.0)"):
        traj_map = {}
        for r in groups[name]:
            tk = (r["meta_prob_idx"], r["meta_plan_idx"], r["meta_exec_idx"])
            traj_map[tk] = r.get("meta_traj_correct")   # 同一条轨迹反复写入同一个值, 天然去重
        n = len(traj_map)
        c = Counter(traj_map.values())
        true_n, false_n, none_n = c.get(True, 0), c.get(False, 0), c.get(None, 0)
        decided = true_n + false_n
        rate = true_n / decided * 100 if decided else float("nan")
        print(f"  [{name}] 共{n}条独立轨迹  正确={true_n}({true_n/n*100:.1f}%)  "
              f"错误={false_n}({false_n/n*100:.1f}%)  无答案={none_n}({none_n/n*100:.1f}%)"
              f"  | 已判定里的正确率={rate:.1f}%")

    print("\n  对照: 该数据集整体pass@1(随机选一条轨迹的正确率基线)约66.7%")
    print("  解读: excluded这一组的正确率如果接近66.7%, 说明这批步骤真的没有"
          "可提取的信号; 如果明显偏离, 说明现有验证方法漏掉了某种真实关联"
          "(但即便如此, 按轨迹结果强行回填标签仍有循环论证风险, 见对话讨论)")

if __name__ == "__main__":
    main()
