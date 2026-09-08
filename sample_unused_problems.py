"""
sample_unused_problems.py — 从官方MATH训练集里抽"从未被用过"的新题目, 并维护一份
持久化的"已使用题目"台账, 保证以后每一轮扩数据都不会跟之前任何一轮重叠。

为什么不能靠"换个随机种子"来保证不重叠:
  从6950题的池子里独立抽两次200题, 哪怕种子不同, 期望仍会重叠 200*200/6950≈5.8题
  —— 这是概率上"大概率不重叠", 不是"保证不重叠"。真正的保证来自显式排除,
  不是运气。

用法 (第一次, 这次要抽200题, train50用的是第0-49行, val500用的是第7000-7499行,
两者要先标记为"永久保留"不可被抽到):
  python sample_unused_problems.py \
      --source train.jsonl \
      --used_manifest runs/v0/used_problem_indices.json \
      --reserved_ranges "0:50,7000:7500" \
      --n_new 200 \
      --out runs/v0/train_round2_200.jsonl \
      --seed 44

以后再扩(比如再加250题), 同一个manifest文件会自动记得这次用过的200题, 直接:
  python sample_unused_problems.py \
      --source train.jsonl \
      --used_manifest runs/v0/used_problem_indices.json \
      --reserved_ranges "0:50,7000:7500" \
      --n_new 250 \
      --out runs/v0/train_round3_250.jsonl \
      --seed 45
不需要再手动想"这次该避开哪些", manifest会自动累积排除所有历史轮次用过的题目。
"""
import argparse, json, random, os

def parse_ranges(s):
    """'0:50,7000:7500' -> {0,1,...,49} | {7000,...,7499}"""
    out = set()
    if not s:
        return out
    for part in s.split(","):
        a, b = part.split(":")
        out.update(range(int(a), int(b)))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="官方训练集完整jsonl, 比如7500题那份")
    ap.add_argument("--used_manifest", required=True,
                    help="持久化的已使用索引台账(json数组), 不存在则视为空, 跑完会自动写入更新")
    ap.add_argument("--reserved_ranges", default="",
                    help="格式 'a:b,c:d', 表示[a,b)和[c,d)这些行号永久排除"
                         "(比如train50/val500当时占用的范围), 每次都会被排除, 不需要等"
                         "manifest里有记录")
    ap.add_argument("--n_new", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, required=True,
                    help="每轮换一个不同的种子(只影响这次抽到池子里的哪些题, 不影响"
                         "不重叠这个保证 —— 不重叠是靠排除集合保证的, 不是靠种子)")
    args = ap.parse_args()

    lines = open(args.source).readlines()
    total = len(lines)
    print(f"源文件总题数: {total}")

    reserved = parse_ranges(args.reserved_ranges)

    if os.path.exists(args.used_manifest):
        used = set(json.load(open(args.used_manifest)))
        print(f"已加载历史台账, 已记录使用过 {len(used)} 道题")
    else:
        used = set()
        print("未发现历史台账, 视为首次扩数据 (台账会在这次跑完后自动创建)")

    excluded = reserved | used
    candidates = [i for i in range(total) if i not in excluded]
    print(f"排除reserved({len(reserved)}题)和已用过({len(used)}题)后, "
          f"剩余候选池: {len(candidates)} 题")

    if len(candidates) < args.n_new:
        raise SystemExit(f"候选池只剩{len(candidates)}题, 不够抽{args.n_new}题, "
                         f"请检查reserved_ranges/manifest是否设对, 或减小n_new")

    rng = random.Random(args.seed)
    picked = sorted(rng.sample(candidates, args.n_new))

    with open(args.out, "w") as f:
        for i in picked:
            f.write(lines[i])
    print(f"抽出 {len(picked)} 题 -> {args.out}")

    new_used = used | set(picked)
    with open(args.used_manifest, "w") as f:
        json.dump(sorted(new_used), f)
    print(f"台账已更新, 累计记录已使用题目: {len(new_used)} 道 -> {args.used_manifest}")
    print(f"(下次再扩数据时, 这{len(picked)}道题会被自动排除, 不需要手动记)")

if __name__ == "__main__":
    main()
