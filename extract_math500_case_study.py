import json
import math
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


def get_problem_text(x):
    for k in ("problem", "question", "query", "prompt", "input"):
        if x.get(k) is not None:
            return x[k]
    return ""


def mean_step_score(d, n_steps):
    vals = []
    for i in range(n_steps):
        v = d.get(str(i))
        # 与现有 eval aggregation 保持一致：
        # missing score -> uncertain 0.5
        vals.append(0.5 if v is None else float(v))

    if not vals:
        return 0.0
    return sum(vals) / len(vals)


# ------------------------------------------------------------
# Load
# ------------------------------------------------------------

trajs = [json.loads(x) for x in open(TRAJ_PATH)]
problems = [json.loads(x) for x in open(PROBLEM_PATH)]
score_dump = json.load(open(SCORE_PATH))

step_scores = score_dump["step_score"]
plan_scores = score_dump["plan_score"]

by_prob = defaultdict(list)
for t in trajs:
    by_prob[int(t["prob_idx"])].append(t)


# ------------------------------------------------------------
# Build per-trajectory records
# ------------------------------------------------------------

all_problem_rows = []
rescues = []
hurts = []
both_correct = []
both_wrong = []

missing_chains = 0

for prob_idx in sorted(by_prob):
    candidates = []

    for local_idx, traj in enumerate(by_prob[prob_idx]):
        key = f"{prob_idx}:{local_idx}"

        if key not in step_scores:
            missing_chains += 1
            continue

        step = mean_step_score(
            step_scores[key],
            len(traj.get("steps", []))
        )

        plan = plan_scores.get(key)
        if plan is None:
            # 没有 plan score 的 chain 不用于 full-selection case
            missing_chains += 1
            continue

        plan = float(plan)
        full = (1 - ALPHA) * step + ALPHA * plan

        candidates.append({
            "local_idx": local_idx,
            "plan_idx": traj.get("plan_idx"),
            "exec_idx": traj.get("exec_idx"),
            "answer": traj.get("answer"),
            "correct": bool(traj.get("correct", False)),
            "step_score": step,
            "plan_score": plan,
            "combined_score": full,
            "plan": traj.get("plan", []),
            "steps": traj.get("steps", []),
        })

    if not candidates:
        continue

    step_pick = max(
        candidates,
        key=lambda x: (x["step_score"], -x["local_idx"])
    )
    full_pick = max(
        candidates,
        key=lambda x: (x["combined_score"], -x["local_idx"])
    )

    if step_pick["correct"] and full_pick["correct"]:
        case = "both_correct"
    elif (not step_pick["correct"]) and full_pick["correct"]:
        case = "rescue"
    elif step_pick["correct"] and (not full_pick["correct"]):
        case = "hurt"
    else:
        case = "both_wrong"

    # pairwise margins between the selected wrong/correct chains
    step_margin = None
    plan_margin = None
    combined_margin = None

    if case == "rescue":
        step_margin = (
            full_pick["step_score"] - step_pick["step_score"]
        )
        plan_margin = (
            full_pick["plan_score"] - step_pick["plan_score"]
        )
        combined_margin = (
            full_pick["combined_score"]
            - step_pick["combined_score"]
        )

    row = {
        "prob_idx": prob_idx,
        "problem": get_problem_text(problems[prob_idx]),
        "case": case,

        "step_pick_local_idx": step_pick["local_idx"],
        "step_pick_answer": step_pick["answer"],
        "step_pick_correct": step_pick["correct"],
        "step_pick_step_score": step_pick["step_score"],
        "step_pick_plan_score": step_pick["plan_score"],
        "step_pick_combined_score": step_pick["combined_score"],

        "full_pick_local_idx": full_pick["local_idx"],
        "full_pick_answer": full_pick["answer"],
        "full_pick_correct": full_pick["correct"],
        "full_pick_step_score": full_pick["step_score"],
        "full_pick_plan_score": full_pick["plan_score"],
        "full_pick_combined_score": full_pick["combined_score"],

        "step_margin": step_margin,
        "plan_margin": plan_margin,
        "combined_margin": combined_margin,

        "candidates": candidates,
    }

    all_problem_rows.append(row)

    if case == "rescue":
        rescues.append(row)
    elif case == "hurt":
        hurts.append(row)
    elif case == "both_correct":
        both_correct.append(row)
    else:
        both_wrong.append(row)


# ------------------------------------------------------------
# Sort rescue cases:
# prefer large positive plan margin and clear ranking reversal
# ------------------------------------------------------------

rescues.sort(
    key=lambda r: (
        r["plan_margin"]
        if r["plan_margin"] is not None else -999,
        r["combined_margin"]
        if r["combined_margin"] is not None else -999,
    ),
    reverse=True,
)


# ------------------------------------------------------------
# Save per-problem JSONL
# ------------------------------------------------------------

with open(
    OUT_DIR / "per_problem_chain_selection.jsonl",
    "w",
    encoding="utf-8",
) as f:
    for r in all_problem_rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")


with open(
    OUT_DIR / "rescue_cases.jsonl",
    "w",
    encoding="utf-8",
) as f:
    for rank, r in enumerate(rescues, 1):
        x = dict(r)
        x["rank"] = rank
        f.write(json.dumps(x, ensure_ascii=False) + "\n")


# ------------------------------------------------------------
# Human-readable Markdown
# ------------------------------------------------------------

with open(
    OUT_DIR / "rescue_cases.md",
    "w",
    encoding="utf-8",
) as f:

    f.write("# MATH500 Qualitative Case Candidates\n\n")
    f.write(
        "Definition: step-only selects the trajectory with the highest "
        "mean step score; full selects the trajectory with the highest "
        "0.7 * step + 0.3 * plan score.\n\n"
    )

    f.write(f"Rescue cases: {len(rescues)}\n\n")
    f.write(f"Hurt cases: {len(hurts)}\n\n")

    for rank, r in enumerate(rescues, 1):
        f.write("---\n\n")
        f.write(
            f"# Rank {rank} | prob_idx={r['prob_idx']}\n\n"
        )

        f.write("## Problem\n\n")
        f.write(r["problem"] + "\n\n")

        f.write("## Selection flip\n\n")

        f.write(
            f"**Step-only selected WRONG trajectory "
            f"(local_idx={r['step_pick_local_idx']})**\n\n"
        )
        f.write(
            f"- answer: `{r['step_pick_answer']}`\n"
            f"- step: {r['step_pick_step_score']:.4f}\n"
            f"- plan: {r['step_pick_plan_score']:.4f}\n"
            f"- combined: {r['step_pick_combined_score']:.4f}\n\n"
        )

        f.write(
            f"**Full selected CORRECT trajectory "
            f"(local_idx={r['full_pick_local_idx']})**\n\n"
        )
        f.write(
            f"- answer: `{r['full_pick_answer']}`\n"
            f"- step: {r['full_pick_step_score']:.4f}\n"
            f"- plan: {r['full_pick_plan_score']:.4f}\n"
            f"- combined: {r['full_pick_combined_score']:.4f}\n\n"
        )

        f.write(
            f"- step margin (correct - wrong): "
            f"{r['step_margin']:+.4f}\n"
            f"- plan margin (correct - wrong): "
            f"{r['plan_margin']:+.4f}\n"
            f"- combined margin (correct - wrong): "
            f"{r['combined_margin']:+.4f}\n\n"
        )

        cand_map = {
            c["local_idx"]: c for c in r["candidates"]
        }

        wrong = cand_map[r["step_pick_local_idx"]]
        correct = cand_map[r["full_pick_local_idx"]]

        for name, c in [
            ("Wrong trajectory selected by step-only", wrong),
            ("Correct trajectory selected by full score", correct),
        ]:
            f.write(f"## {name}\n\n")

            f.write("### Plan\n\n")
            for p in c["plan"]:
                f.write(
                    f"- **{p.get('id','?')} "
                    f"[{p.get('action','?')}]**: "
                    f"{p.get('desc','')}\n"
                )
            f.write("\n")

            f.write("### Execution\n\n")

            for i, s in enumerate(c["steps"], 1):
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


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

summary = {
    "n_problems_analyzed": len(all_problem_rows),
    "both_correct": len(both_correct),
    "rescue": len(rescues),
    "hurt": len(hurts),
    "both_wrong": len(both_wrong),
    "missing_chains": missing_chains,
    "alpha": ALPHA,
}

json.dump(
    summary,
    open(OUT_DIR / "summary.json", "w"),
    ensure_ascii=False,
    indent=2,
)

print(json.dumps(summary, indent=2))

print("\nTop rescue cases:")
for rank, r in enumerate(rescues[:20], 1):
    print(
        f"{rank:2d}. prob={r['prob_idx']:3d} "
        f"step_margin={r['step_margin']:+.4f} "
        f"plan_margin={r['plan_margin']:+.4f} "
        f"combined_margin={r['combined_margin']:+.4f} "
        f"| {r['step_pick_answer']} -> {r['full_pick_answer']}"
    )

print("\nSaved to:")
print(OUT_DIR / "summary.json")
print(OUT_DIR / "per_problem_chain_selection.jsonl")
print(OUT_DIR / "rescue_cases.jsonl")
print(OUT_DIR / "rescue_cases.md")
