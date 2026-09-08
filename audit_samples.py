"""
audit_samples.py — SFT 数据集人工核对抽样器 (纯 CPU)

目的: 分布健康不等于每条标签对。本脚本按"出错风险/信号价值"分层抽样, 把每条
渲染成便于肉眼判断的卡片 (题面 + 该步真实代码与输出 + 渲染出的判断 + score),
供人工核 verdict 对不对。优先抽方法最锋利、最易错的组合:
  - faithful × localized_fail : 忠实却算错 (B2 是否把该 deviated 误判成 faithful?)
  - deviated × *              : 偏离判定成不成立
  - claimed_only × *          : "嘴上做了"判得对不对 (严格规则的主要风险)
  - self_declared_deviation   : 自首是否真偏离
再随机补若干 faithful×corroborated / unverified 普通样本看文本与分数是否协调。

输出:
  runs/v0/audit_sheet.md   每条一张卡 + [ ]对 [ ]错 勾选位
  控制台同时打印精简版

用法:
  python audit_samples.py --sft runs/v0/sft_dataset.jsonl \
                          --traj runs/v0/trajectories.jsonl \
                          --per 8 --seed 1
"""
import argparse, json, random
from collections import defaultdict

# 分层: (adherence, provenance) 模式 -> 该层抽多少条。'*' 通配 provenance。
STRATA = [
    ("faithful", "localized_fail", None),   # None=该层全抽 (最金贵, 量小)
    ("deviated", "*", 8),
    ("claimed_only", "*", 8),
    ("self_declared_deviation", "*", 5),
    ("uncertain", "*", 5),                   # 未收口, 看 0.5 占位是否合理
    ("faithful", "corroborated", 5),         # 正例: 文本/分数协调性
    ("faithful", "unverified", 5),           # 中性档抽查
]

def match(r, adh, prov):
    if r["meta_adherence"] != adh:
        return False
    return prov == "*" or r["meta_execution_provenance"] == prov

def card(r, traj):
    """渲染一张人工核对卡。traj 用来取该步真实 code/output。"""
    t = traj.get((r["meta_prob_idx"], r["meta_plan_idx"], r["meta_exec_idx"]))
    st = t["steps"][r["meta_step"]] if t else None
    L = []
    L.append(f"## [{r['meta_adherence']} × {r['meta_execution_provenance']}]  "
             f"score={r['meta_score']}  "
             f"prob{r['meta_prob_idx']}/plan{r['meta_plan_idx']}"
             f"/exec{r['meta_exec_idx']}/step{r['meta_step']} "
             f"(claims {r['meta_claimed']})")
    if t:
        # 题面 + 该环节计划描述
        stage = next((p for p in t["plan"] if p["id"] == r["meta_claimed"]), None)
        if stage:
            L.append(f"**Claimed stage**: {stage['id']} ({stage.get('action','?')}) "
                     f"— {stage['desc']}")
    if st:
        L.append("**Step code**:\n```python\n" + st["code"] + "\n```")
        L.append("**Step output**:\n```\n" + (st["exec"] or "") + "\n```")
    # 渲染好的三段判断 (output 去掉 tag 行, 只看自然语言部分)
    judg = "\n".join(ln for ln in r["output"].splitlines()
                     if not ln.startswith("<"))
    L.append("**Rendered judgment**:\n> " + judg.replace("\n", "\n> "))
    L.append(f"**Tags**: adherence={r['meta_adherence']} "
             f"| execution={r['meta_execution_provenance']} "
             f"| verdict={r['meta_execution_verdict']} "
             f"| score={r['meta_score']}")
    L.append("**Human check**:  [ ] adherence对  [ ] execution对  "
             "[ ] score合理   备注: ____________")
    return "\n\n".join(L)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", default="runs/v0/sft_dataset.jsonl")
    ap.add_argument("--traj", default="runs/v0/trajectories.jsonl")
    ap.add_argument("--out", default="runs/v0/audit_sheet.md")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.sft)]
    traj = {(t["prob_idx"], t["plan_idx"], t["exec_idx"]): t
            for t in (json.loads(l) for l in open(args.traj))}
    rng = random.Random(args.seed)

    picked, seen = [], set()
    summary = []
    for adh, prov, k in STRATA:
        pool = [r for r in rows if match(r, adh, prov)
                and id(r) not in seen]
        rng.shuffle(pool)
        take = pool if k is None else pool[:k]
        for r in take:
            seen.add(id(r)); picked.append(r)
        summary.append((f"{adh} × {prov}", len(take), len(pool)))

    with open(args.out, "w") as f:
        f.write(f"# SFT 数据集人工核对清单 ({len(picked)} 条)\n\n")
        f.write("核对方法: 对每条, 看 [Step code/output] 与 [Rendered judgment] "
                "是否一致, 勾选三个维度对不对。重点关注 faithful×localized_fail "
                "(忠实却算错是否成立) 与 deviated/claimed_only (是否误判)。\n\n")
        for r in picked:
            f.write("---\n\n" + card(r, traj) + "\n\n")

    print(f"audit sheet -> {args.out}  ({len(picked)} cards)")
    print("\n抽样分层 (取/池):")
    for name, took, pool in summary:
        print(f"  {name:<34} {took:>3} / {pool}")
    print("\n--- 控制台速览前 3 条 ---\n")
    for r in picked[:3]:
        print(card(r, traj))
        print("\n" + "=" * 70 + "\n")

if __name__ == "__main__":
    main()
