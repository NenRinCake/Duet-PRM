"""
diff_relabeling.py — 检查"扩大N"是否悄悄改写了旧执行(exec_idx 0-3)的标签

假设: execution provenance(corroborated/suspect/outcome_consistent/localized_fail等)
是靠"跟兄弟执行互相比对"算出来的, 不是单条执行的固有属性。N从4扩到8后, 旧的
4次执行哪怕内容一个字没变, 它们的provenance标签也可能因为"对比池"变大而改变。
这个脚本直接对比v6和v6n8两批c2_labels.jsonl里, 同一批(prob,plan,exec0-3,step)
的标签有没有变, 以及这个变化是否足以跨过BAD/UNCERTAIN/GOOD的判定边界。

用法:
  python diff_relabeling.py \
      --old_c2 runs/v0/c2_labels.jsonl --new_c2 runs/v0/c2_labels-v6-n8.jsonl \
      --old_steps runs/v0/step_labels.jsonl --new_steps runs/v0/step_labels-v6-n8.jsonl
"""
import argparse, json
from collections import defaultdict, Counter

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

def load_keyed(path, field):
    d = {}
    for line in open(path):
        r = json.loads(line)
        key = (r["prob_idx"], r["plan_idx"], r["exec_idx"], r["step"])
        d[key] = r.get(field)
    return d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old_c2", required=True)
    ap.add_argument("--new_c2", required=True)
    ap.add_argument("--old_steps", required=True)
    ap.add_argument("--new_steps", required=True)
    ap.add_argument("--max_old_exec_idx", type=int, default=3,
                    help="旧执行的exec_idx上限(包含), 默认0-3共4次")
    args = ap.parse_args()

    old_prov = load_keyed(args.old_c2, "provenance")
    new_prov = load_keyed(args.new_c2, "provenance")
    old_adh_raw = load_keyed(args.old_steps, "adherence")
    new_adh_raw = load_keyed(args.new_steps, "adherence")
    def norm_adh(a):
        if a is None: return "unknown"
        return "uncertain" if str(a).startswith("uncertain") else a
    old_adh = {k: norm_adh(v) for k, v in old_adh_raw.items()}
    new_adh = {k: norm_adh(v) for k, v in new_adh_raw.items()}

    # 只看"旧执行"范围内的key (exec_idx <= max_old_exec_idx), 这些步骤内容
    # 在v6和v6n8之间应该是逐字节相同的, 只检查标签有没有被悄悄改写
    old_keys = {k for k in old_prov if k[2] <= args.max_old_exec_idx}
    common_keys = old_keys & set(new_prov.keys())
    print(f"旧执行范围内的步骤总数: {len(old_keys)}  "
          f"(在新文件里也找到对应key的: {len(common_keys)})")
    if len(common_keys) < len(old_keys):
        print(f"  ⚠ 有 {len(old_keys)-len(common_keys)} 条在新文件里找不到对应key, "
              f"可能是key格式不一致或exec_idx范围设错, 请检查 --max_old_exec_idx")

    prov_trans = Counter()
    adh_trans = Counter()
    score_trans = Counter()
    n_prov_changed = n_adh_changed = n_score_changed = 0
    for k in common_keys:
        op, npv = old_prov[k], new_prov[k]
        oa, na = old_adh.get(k, "unknown"), new_adh.get(k, "unknown")
        if op != npv:
            n_prov_changed += 1
            prov_trans[(op, npv)] += 1
        if oa != na:
            n_adh_changed += 1
            adh_trans[(oa, na)] += 1
        os_ = compute_step_score(oa, op)
        ns_ = compute_step_score(na, npv)
        if os_ != ns_:
            n_score_changed += 1
            score_trans[(os_, ns_)] += 1

    n = len(common_keys)
    print(f"\nprovenance 标签改变: {n_prov_changed}/{n} = {n_prov_changed/n*100:.1f}%")
    print(f"adherence  标签改变: {n_adh_changed}/{n} = {n_adh_changed/n*100:.1f}%"
          f"  (理论上该接近0%, 这个判定不依赖跨执行比对, 内容没变就不该变)")
    print(f"最终step_score改变: {n_score_changed}/{n} = {n_score_changed/n*100:.1f}%"
          f"  <- 这是真正影响训练的数字")

    if prov_trans:
        print("\nprovenance 变化方向 (旧 -> 新, 按次数排序):")
        for (op, npv), c in prov_trans.most_common(10):
            print(f"  {op:<18} -> {npv:<18}: {c}")
    if score_trans:
        names = {0.0: "BAD", 0.5: "UNCERTAIN", 1.0: "GOOD"}
        print("\nstep_score 跨档变化方向 (旧 -> 新):")
        for (os_, ns_), c in score_trans.most_common(10):
            print(f"  {names[os_]:<10} -> {names[ns_]:<10}: {c}")

if __name__ == "__main__":
    main()
