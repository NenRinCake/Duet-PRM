"""
rebuild_step_binary.py — 直接在已经assemble好的sft_dataset上, 把step_score从
三分类(0.0/0.5/1.0)改成二分类(0.0/1.0), 不需要回头重新跑label_minimal.py/
verify_rules.py —— sft_dataset里保留的meta_adherence + meta_execution_provenance
这两个字段, 就是计算step_score所需的全部原始证据。

处理流程:
  1. 按(meta_prob_idx, meta_plan_idx, meta_exec_idx, meta_step)去重, 拿到去掉
     过采样副本之后的"原始基础集合"(旧的过采样是按旧三分类规则做的, 不能直接
     沿用, 必须先回到基础集合, 再按新规则重新过采样一次)
  2. 对每条基础样本, 用跟原三分类函数完全相同的优先级逻辑重新判定:
     BAD/GOOD的判定逻辑逐字不变; 原来落进0.5的两种情况(UNCERTAIN_ADH,
     UNCERTAIN_PROV)以及原来的兜底分支, 现在直接丢弃这条样本(不强行二分类,
     理由: 这部分样本是"真的无法判定", 不是"模糊地带", 见对话里的论证)
  3. 对存活下来的样本, 同步重写两处文本, 保证instruction/output内部自洽:
     - output里的<step_score>X.X</step_score>改成新的二分类值
     - instruction里描述step_score选项的那句话, 同步去掉"0.5"这个选项
       (plan_score那句话原样保留, 这次只动step_score)
  4. 按新规则下的BAD类, 重新做一次过采样(默认x2, 跟项目里一贯的倍数一致)

用法:
  python rebuild_step_binary.py --in_file runs/v0/sft_dataset_v6plus250.jsonl \
      --out_file runs/v0/sft_dataset_v6plus250_stepbinary.jsonl --oversample 2
"""
import argparse, json, re
from collections import Counter

BAD_PROV = {"localized_fail", "suspect"}
BAD_ADH = {"deviated", "claimed_only"}
UNCERTAIN_PROV = {"unverified"}
UNCERTAIN_ADH = {"self_declared_deviation"}
GOOD_PROV = {"corroborated", "outcome_consistent"}

def compute_step_score_binary(adherence, provenance):
    """跟原三分类函数逐字相同的优先级判定; 原来输出0.5的分支(包括最后的兜底)
    现在统一返回None, 代表"该被剔除", 不强行塞进0或1。"""
    if provenance in BAD_PROV: return 0.0
    if adherence in BAD_ADH: return 0.0
    if adherence in UNCERTAIN_ADH: return None
    if provenance in UNCERTAIN_PROV: return None
    if provenance in GOOD_PROV: return 1.0
    return None

# instruction里描述step_score选项的那句话, 必须跟sft_dataset里实际出现的文字
# 逐字匹配才能安全替换 —— 如果哪天instruction模板改过没同步到这里, 下面的
# assert会先报错, 不会悄悄替换失败/替换错位置
OLD_STEP_INSTR = ("step_score: is the CURRENT STEP itself correct and "
                  "trustworthy? Use exactly one of: 0.0 (no — wrong, "
                  "deviates from the plan, or does no real work), 0.5 "
                  "(cannot be independently determined), or 1.0 (yes — "
                  "well-supported).")
NEW_STEP_INSTR = ("step_score: is the CURRENT STEP itself correct and "
                  "trustworthy? Use exactly one of: 0.0 (no — wrong, "
                  "deviates from the plan, or does no real work) or 1.0 "
                  "(yes — well-supported).")

STEP_TAG_RE = re.compile(r"<step_score>[\d.]+</step_score>")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_file", required=True)
    ap.add_argument("--out_file", required=True)
    ap.add_argument("--oversample", type=int, default=2)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.in_file)]

    # ---- 1. 去重, 回到过采样之前的基础集合 ----
    seen = set()
    base_rows = []
    for r in rows:
        key = (r["meta_prob_idx"], r["meta_plan_idx"], r["meta_exec_idx"], r["meta_step"])
        if key not in seen:
            seen.add(key)
            base_rows.append(r)

    # ---- 2. 重新判定 + 剔除 ----
    kept, dropped = [], 0
    old_dist, new_dist = Counter(), Counter()
    for r in base_rows:
        old_dist[r["meta_step_score"]] += 1
        new_score = compute_step_score_binary(r["meta_adherence"], r["meta_execution_provenance"])
        if new_score is None:
            dropped += 1
            continue
        new_dist[new_score] += 1
        kept.append((r, new_score))

    # ---- 3. 重写instruction/output, 保证内部自洽 ----
    rebuilt = []
    n_instr_mismatch = 0
    for r, new_score in kept:
        instr = r["instruction"]
        if OLD_STEP_INSTR not in instr:
            n_instr_mismatch += 1
            continue   # 格式跟预期不符, 跳过并计数, 不要静默替换出一份内部不一致的样本
        new_instr = instr.replace(OLD_STEP_INSTR, NEW_STEP_INSTR)
        new_output = STEP_TAG_RE.sub(f"<step_score>{new_score}</step_score>", r["output"])
        new_r = dict(r)
        new_r["instruction"] = new_instr
        new_r["output"] = new_output
        new_r["meta_step_score"] = new_score
        rebuilt.append(new_r)

    if n_instr_mismatch:
        print(f"[警告] {n_instr_mismatch} 条样本的instruction文本跟预期模板不匹配, "
              f"已跳过未处理 —— 需要人工确认是不是模板本身变了")

    # ---- 4. 按新规则重新过采样BAD类 ----
    final = []
    n_bad_oversampled = 0
    for r in rebuilt:
        final.append(r)
        if r["meta_step_score"] == 0.0:
            for _ in range(args.oversample - 1):
                final.append(dict(r))
            n_bad_oversampled += 1

    with open(args.out_file, "w") as f:
        for r in final:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"原始文件(含过采样副本): {len(rows)} 条")
    print(f"去重后基础集合: {len(base_rows)} 条")
    print(f"旧三分类分布(去重后): {dict(old_dist)}")
    print(f"剔除(原来落进0.5/兜底, 现在判定为真正无法确定): {dropped} 条")
    print(f"新二分类分布(去重后, 剔除之后): {dict(new_dist)}")
    print(f"BAD类(0.0)过采样: {n_bad_oversampled}条 -> 最终出现{n_bad_oversampled*args.oversample}次")
    print(f"最终输出(含新过采样副本): {len(final)} 条 -> {args.out_file}")

if __name__ == "__main__":
    main()
