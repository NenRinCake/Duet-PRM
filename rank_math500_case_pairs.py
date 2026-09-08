import json
from collections import defaultdict
from pathlib import Path

TRAJ_PATH = Path(
    "/public/home/ljt/lzm/code/PRM/runs/BoN/math500/trajectories.jsonl"
)
SCORE_PATH = Path(
    "/public/home/ljt/lzm/code/PRM/runs/eval/math500_bon_scores_dump.json"
)
PROBLEM_PATH = Path(
    "/public/home/ljt/lzm/code/Qwen2.5-Math/evaluation/data/math500/test.jsonl"
)

OUT_DIR = Path(
    "/public/home/ljt/lzm/code/PRM/runs/case_study/math500"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALPHA = 0.3

# 第一轮用相对宽松条件，避免漏掉好 case
MIN_STEP_ADV_WRONG = 0.02
MIN_PLAN_ADV_CORRECT = 0.10
MIN_FULL_ADV_CORRECT = 0.005


def problem_text(x):
    for k in ("problem", "question", "query", "prompt", "input"):
        if x.get(k) is not None:
            return str(x[k])
    return ""


def step_mean(score_dict, n_steps):
    vals = []
    for i in range(n_steps):
        v = score_dict.get(str(i))
        vals.append(0.5 if v is None else float(v))
    if not vals:
        return None
    return sum(vals) / len(vals)


def norm_answer(x):
    if x is None:
        return None
    return str(x).strip()


def plan_signature(plan):
    return tuple(
        (
            str(x.get("action", "")).strip(),
            str(x.get("desc", "")).strip().lower()
        )
        for x in plan
    )


# ============================================================
# load
# ============================================================

trajs = [json.loads(x) for x in open(TRAJ_PATH)]
problems = [json.loads(x) for x in open(PROBLEM_PATH)]
dump = json.load(open(SCORE_PATH))

step_scores = dump["step_score"]
plan_scores = dump["plan_score"]

by_prob = defaultdict(list)
for t in trajs:
    by_prob[int(t["prob_idx"])].append(t)


# ============================================================
# construct scored candidates
# ============================================================

scored = defaultdict(list)

for prob_idx, candidates in by_prob.items():

    for local_idx, t in enumerate(candidates):

        key = f"{prob_idx}:{local_idx}"

        if key not in step_scores or key not in plan_scores:
            continue

        steps = t.get("steps", [])
        plan = t.get("plan", [])

        sm = step_mean(step_scores[key], len(steps))
        if sm is None:
            continue

        ps = plan_scores.get(key)
        if ps is None:
            continue

        ps = float(ps)
        full = (1 - ALPHA) * sm + ALPHA * ps

        scored[prob_idx].append({
            "local_idx": local_idx,
            "plan_idx": t.get("plan_idx"),
            "exec_idx": t.get("exec_idx"),
            "answer": t.get("answer"),
            "correct": bool(t.get("correct", False)),
            "step_score": sm,
            "plan_score": ps,
            "combined_score": full,
            "n_steps": len(steps),
            "n_plan_stages": len(plan),
            "plan": plan,
            "steps": steps,
        })


# ============================================================
# all correct-vs-wrong pairs
# ============================================================

pairs = []

for prob_idx, candidates in scored.items():

    corrects = [x for x in candidates if x["correct"]]
    wrongs = [x for x in candidates if not x["correct"]]

    if not corrects or not wrongs:
        continue

    for c in corrects:
        for w in wrongs:

            # 不同 final answer 才有意义
            if norm_answer(c["answer"]) == norm_answer(w["answer"]):
                continue

            # 要有实质轨迹
            if not c["plan"] or not w["plan"]:
                continue
            if not c["steps"] or not w["steps"]:
                continue

            # Qualitative case: require genuinely different plans.
            if c["plan_idx"] == w["plan_idx"]:
                continue

            if plan_signature(c["plan"]) == plan_signature(w["plan"]):
                continue

            # 定义三个最关键 margin
            # 正数 = wrong 在 step 上更占优势
            step_wrong_adv = w["step_score"] - c["step_score"]

            # 正数 = correct 在 plan 上更占优势
            plan_correct_adv = c["plan_score"] - w["plan_score"]

            # 正数 = combined 后 correct 反超
            full_correct_adv = (
                c["combined_score"] - w["combined_score"]
            )

            if step_wrong_adv < MIN_STEP_ADV_WRONG:
                continue
            if plan_correct_adv < MIN_PLAN_ADV_CORRECT:
                continue
            if full_correct_adv < MIN_FULL_ADV_CORRECT:
                continue

            # ------------------------------------------------
            # qualitative suitability score
            #
            # 希望：
            # 1. plan difference 强
            # 2. step 确实误导
            # 3. combined 确实翻转
            # 4. 不要特别短的 trivial chain
            # ------------------------------------------------

            structure_bonus = min(
                (c["n_steps"] + w["n_steps"]) / 10.0,
                1.0
            )

            suitability = (
                3.0 * plan_correct_adv
                + 2.0 * step_wrong_adv
                + 2.0 * full_correct_adv
                + 0.10 * structure_bonus
            )

            pairs.append({
                "prob_idx": prob_idx,
                "problem": problem_text(problems[prob_idx]),

                "step_wrong_adv": step_wrong_adv,
                "plan_correct_adv": plan_correct_adv,
                "full_correct_adv": full_correct_adv,
                "suitability": suitability,

                "wrong": w,
                "correct": c,
            })


pairs.sort(
    key=lambda x: (
        x["suitability"],
        x["plan_correct_adv"],
        x["step_wrong_adv"],
    ),
    reverse=True,
)


# ============================================================
# save full JSONL
# ============================================================

jsonl_path = OUT_DIR / "pairwise_ranking_reversals.jsonl"

with open(jsonl_path, "w", encoding="utf-8") as f:
    for rank, p in enumerate(pairs, 1):
        q = dict(p)
        q["rank"] = rank
        f.write(json.dumps(q, ensure_ascii=False) + "\n")


# ============================================================
# save concise TSV for fast inspection
# ============================================================

tsv_path = OUT_DIR / "pairwise_ranking_reversals.tsv"

with open(tsv_path, "w", encoding="utf-8") as f:

    f.write(
        "rank\tprob_idx\tsuitability\t"
        "wrong_step\tcorrect_step\tstep_wrong_adv\t"
        "wrong_plan\tcorrect_plan\tplan_correct_adv\t"
        "wrong_full\tcorrect_full\tfull_correct_adv\t"
        "wrong_answer\tcorrect_answer\t"
        "wrong_idx\tcorrect_idx\n"
    )

    for rank, p in enumerate(pairs, 1):

        w = p["wrong"]
        c = p["correct"]

        f.write(
            f"{rank}\t"
            f"{p['prob_idx']}\t"
            f"{p['suitability']:.4f}\t"

            f"{w['step_score']:.4f}\t"
            f"{c['step_score']:.4f}\t"
            f"{p['step_wrong_adv']:.4f}\t"

            f"{w['plan_score']:.4f}\t"
            f"{c['plan_score']:.4f}\t"
            f"{p['plan_correct_adv']:.4f}\t"

            f"{w['combined_score']:.4f}\t"
            f"{c['combined_score']:.4f}\t"
            f"{p['full_correct_adv']:.4f}\t"

            f"{norm_answer(w['answer'])}\t"
            f"{norm_answer(c['answer'])}\t"

            f"{w['local_idx']}\t"
            f"{c['local_idx']}\n"
        )


# ============================================================
# Detailed markdown: top 30
# ============================================================

md_path = OUT_DIR / "top_pairwise_case_candidates.md"

with open(md_path, "w", encoding="utf-8") as f:

    f.write("# MATH500 Pairwise Ranking-Reversal Candidates\n\n")

    f.write(
        "Selection criterion: the incorrect trajectory receives a higher "
        "mean step score, while the correct trajectory receives a higher "
        "plan score and overtakes the incorrect trajectory after using "
        "`0.7 * step + 0.3 * plan`.\n\n"
    )

    f.write(f"Number of qualifying pairs: **{len(pairs)}**\n\n")

    for rank, p in enumerate(pairs[:30], 1):

        w = p["wrong"]
        c = p["correct"]

        f.write("---\n\n")
        f.write(
            f"# Rank {rank} | prob_idx={p['prob_idx']} | "
            f"suitability={p['suitability']:.4f}\n\n"
        )

        f.write("## Problem\n\n")
        f.write(p["problem"] + "\n\n")

        f.write("## Score comparison\n\n")

        f.write("| | Incorrect trajectory | Correct trajectory |\n")
        f.write("|---|---:|---:|\n")
        f.write(
            f"| Step score | {w['step_score']:.4f} | "
            f"{c['step_score']:.4f} |\n"
        )
        f.write(
            f"| Plan score | {w['plan_score']:.4f} | "
            f"{c['plan_score']:.4f} |\n"
        )
        f.write(
            f"| Combined score | {w['combined_score']:.4f} | "
            f"{c['combined_score']:.4f} |\n"
        )
        f.write(
            f"| Final answer | `{w['answer']}` | "
            f"`{c['answer']}` |\n\n"
        )

        f.write(
            f"- Wrong step advantage: "
            f"**{p['step_wrong_adv']:+.4f}**\n"
            f"- Correct plan advantage: "
            f"**{p['plan_correct_adv']:+.4f}**\n"
            f"- Correct combined advantage: "
            f"**{p['full_correct_adv']:+.4f}**\n\n"
        )

        for title, x in [
            ("Incorrect trajectory", w),
            ("Correct trajectory", c),
        ]:

            f.write(f"## {title}\n\n")

            f.write(
                f"local_idx={x['local_idx']}, "
                f"plan_idx={x['plan_idx']}, "
                f"exec_idx={x['exec_idx']}\n\n"
            )

            f.write("### Plan\n\n")

            for pp in x["plan"]:
                f.write(
                    f"- **{pp.get('id','?')} "
                    f"[{pp.get('action','?')}]**: "
                    f"{pp.get('desc','')}\n"
                )

            f.write("\n### Execution\n\n")

            for i, s in enumerate(x["steps"], 1):

                f.write(
                    f"**Step {i}** "
                    f"(claims `{s.get('claimed','?')}`)\n\n"
                )

                if s.get("code"):
                    f.write("```python\n")
                    f.write(str(s["code"]))
                    f.write("\n```\n\n")

                if s.get("exec"):
                    f.write("```text\n")
                    f.write(str(s["exec"]))
                    f.write("\n```\n\n")


print(f"Found {len(pairs)} qualifying correct-vs-wrong reversal pairs.")

print("\nTop 20:")
for rank, p in enumerate(pairs[:20], 1):
    w = p["wrong"]
    c = p["correct"]

    print(
        f"{rank:2d}. "
        f"prob={p['prob_idx']:3d} "
        f"step_adv_wrong={p['step_wrong_adv']:+.4f} "
        f"plan_adv_correct={p['plan_correct_adv']:+.4f} "
        f"full_adv_correct={p['full_correct_adv']:+.4f} "
        f"| step {w['step_score']:.3f}>{c['step_score']:.3f} "
        f"| plan {c['plan_score']:.3f}>{w['plan_score']:.3f} "
        f"| {w['answer']} -> {c['answer']}"
    )

print("\nSaved:")
print(jsonl_path)
print(tsv_path)
print(md_path)
