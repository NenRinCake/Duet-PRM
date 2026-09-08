"""
rebuild_plan_binary.py — 在已经把step_score处理成二分类的sft文件上, 再把
plan_score也从三分类(0.0/0.5/1.0)改成二分类(0.0/1.0)。

跟step_score那次(rebuild_step_binary.py)的关键区别: plan_score的UNCERTAIN
背后一直有一个真实的success_rate数字, 不是认知空白, 所以这次不剔除任何样本,
只是把原来"≤0.25→BAD, ≥0.75→GOOD, 中间→UNCERTAIN"的两道门槛, 合并成一道
"≥0.5→GOOD, 否则→BAD"。这个改法跟原规则在BAD/GOOD的边界上完全一致(原来的
confident BAD/GOOD案例换了门槛判断结果不变), 只是把原来的中间地带按0.5这道
单一门槛重新归边, 不存在剔除/编造标签的问题。trunc_rate>=0.5一票否决判BAD
的规则原样保留。

需要meta_plan_quality_raw(success_rate, decided-only)和meta_plan_trunc_rate_raw
(censored占比)这两个字段, sft_dataset里本来就保留着, 不需要回头重新跑
sample_minimal.py。

用法:
  python rebuild_plan_binary.py --in_file runs/v0/sft_dataset_v6plus250_stepbinary.jsonl \
      --out_file runs/v0/sft_dataset_v6plus250_fullbinary.jsonl
"""
import argparse, json, re
from collections import Counter

def compute_plan_score_binary(success_rate, trunc_rate):
    if trunc_rate is not None and trunc_rate >= 0.5:
        return 0.0
    if success_rate is None:
        return None   # 理论上不会触发: success_rate为None时trunc_rate必为1.0,
                       # 已被上面的一票否决先拦住; 留这一行只是防御性写法
    return 1.0 if success_rate >= 0.5 else 0.0

# plan_score那句话的原文(三分类版), 必须跟sft文件里实际出现的文字逐字匹配
# 才能安全替换。注意: 这次输入文件是已经把step_score改过的"step_01"文件,
# step_score那句话已经是二分类版了, 这里只动plan_score那句话, 不会影响它。
OLD_PLAN_INSTR = ("plan_score: how reliable is the PLAN itself (independent "
                  "of how well any single execution carries it out)? Use "
                  "exactly one of: 0.0 (this plan rarely leads to a correct "
                  "answer, or frequently fails to even produce a complete "
                  "result), 0.5 (mixed or inconsistent results across "
                  "attempts), or 1.0 (this plan reliably leads to a correct "
                  "answer).")
NEW_PLAN_INSTR = ("plan_score: how reliable is the PLAN itself (independent "
                  "of how well any single execution carries it out)? Use "
                  "exactly one of: 0.0 (this plan rarely leads to a correct "
                  "answer, or frequently fails to even produce a complete "
                  "result) or 1.0 (this plan reliably leads to a correct "
                  "answer).")

PLAN_TAG_RE = re.compile(r"<plan_score>[\d.]+</plan_score>")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_file", required=True)
    ap.add_argument("--out_file", required=True)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.in_file)]
    old_dist, new_dist = Counter(), Counter()
    n_instr_mismatch = 0
    n_unresolvable = 0
    out_rows = []

    for r in rows:
        old_dist[r["meta_plan_score"]] += 1
        new_score = compute_plan_score_binary(r["meta_plan_quality_raw"],
                                              r["meta_plan_trunc_rate_raw"])
        if new_score is None:
            n_unresolvable += 1
            continue   # 理论上不该发生, 真发生了就跳过并计数, 不要静默编造
        instr = r["instruction"]
        if OLD_PLAN_INSTR not in instr:
            n_instr_mismatch += 1
            continue
        new_r = dict(r)
        new_r["instruction"] = instr.replace(OLD_PLAN_INSTR, NEW_PLAN_INSTR)
        new_r["output"] = PLAN_TAG_RE.sub(f"<plan_score>{new_score}</plan_score>", r["output"])
        new_r["meta_plan_score"] = new_score
        new_dist[new_score] += 1
        out_rows.append(new_r)

    with open(args.out_file, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"原始行数(含step_score的过采样副本, 这次不需要先去重): {len(rows)}")
    print(f"旧三分类plan_score分布: {dict(old_dist)}")
    print(f"新二分类plan_score分布: {dict(new_dist)}")
    print(f"instruction格式不匹配被跳过: {n_instr_mismatch} 条")
    print(f"理论上不该出现的unresolvable: {n_unresolvable} 条")
    print(f"最终输出: {len(out_rows)} 条 -> {args.out_file}")
    bad_frac = new_dist.get(0.0, 0) / max(1, sum(new_dist.values()))
    print(f"\nBAD类占比: {bad_frac*100:.1f}%  "
          f"(原规则下'压缩后BAD类占比已经够大, 不需要过采样'的判断标准, "
          f"自行核对这次是否依然适用; 如果占比明显变小, 可能需要补一次过采样)")

if __name__ == "__main__":
    main()
