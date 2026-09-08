import json
import csv
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

TRAJ_PATH = "runs/BoN/math500/trajectories.jsonl"
SCORE_PATH = "runs/eval/math500_bon_scores_dump.json"

OUT_DIR = Path("runs/analysis_when_plan_helps/math500")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PLAN_WEIGHT = 0.3
STEP_WEIGHT = 0.7


def mean_step_score(step_dict):
    vals = [
        float(v)
        for v in step_dict.values()
        if v is not None
    ]

    if not vals:
        return None

    return float(np.mean(vals))


def normalize_answer(ans):
    if ans is None:
        return None
    return str(ans).strip()


def select_answer(cands, mode):
    groups = defaultdict(float)

    for c in cands:
        ans = normalize_answer(c["answer"])
        if not ans:
            continue

        if mode == "step":
            score = c["step_score"]
        elif mode == "full":
            score = STEP_WEIGHT * c["step_score"] + PLAN_WEIGHT * c["plan_score"]
        else:
            raise ValueError(mode)

        groups[ans] += score

    if not groups:
        return None

    return max(sorted(groups), key=lambda a: groups[a])


def group_score(cands, answer, key):
    if answer is None:
        return np.nan

    vals = [
        c[key]
        for c in cands
        if normalize_answer(c["answer"]) == normalize_answer(answer)
    ]

    if not vals:
        return np.nan

    return float(sum(vals))


by_prob = defaultdict(list)

with open(TRAJ_PATH, encoding="utf-8") as f:
    for line in f:
        x = json.loads(line)
        by_prob[int(x["prob_idx"])].append(x)


with open(SCORE_PATH, encoding="utf-8") as f:
    score_dump = json.load(f)

step_scores = score_dump["step_score"]
plan_scores = score_dump["plan_score"]


scored_by_prob = defaultdict(list)
missing = 0

for prob_idx, trajs in by_prob.items():
    for local_idx, traj in enumerate(trajs):
        key = f"{prob_idx}:{local_idx}"

        if key not in step_scores or key not in plan_scores:
            missing += 1
            continue

        step_score = mean_step_score(step_scores[key])

        if step_score is None:
            missing += 1
            continue

        plan_score = float(plan_scores[key])

        scored_by_prob[prob_idx].append({
            "prob_idx": prob_idx,
            "local_idx": local_idx,
            "plan_idx": traj.get("plan_idx"),
            "exec_idx": traj.get("exec_idx"),
            "answer": traj.get("answer"),
            "correct": bool(traj.get("correct", False)),
            "step_score": step_score,
            "plan_score": plan_score,
        })


print("Number of problems:", len(scored_by_prob))
print("Missing/unscored trajectories:", missing)

candidate_counts = Counter(len(v) for v in scored_by_prob.values())

print("\nCandidates per problem:")
for k in sorted(candidate_counts):
    print(f"  {k}: {candidate_counts[k]}")


rows = []

for prob_idx, cands in sorted(scored_by_prob.items()):
    if not cands:
        continue

    n_candidates = len(cands)
    n_correct = sum(c["correct"] for c in cands)

    step_ans = select_answer(cands, "step")
    full_ans = select_answer(cands, "full")

    step_correct = any(
        normalize_answer(c["answer"]) == normalize_answer(step_ans)
        and c["correct"]
        for c in cands
    )

    full_correct = any(
        normalize_answer(c["answer"]) == normalize_answer(full_ans)
        and c["correct"]
        for c in cands
    )

    if step_correct and full_correct:
        case = "both_correct"
    elif (not step_correct) and full_correct:
        case = "rescue"
    elif step_correct and (not full_correct):
        case = "hurt"
    else:
        case = "both_wrong"

    step_margin = np.nan
    plan_margin = np.nan

    if case == "rescue":
        correct_ans = full_ans
        wrong_ans = step_ans

        step_margin = (
            group_score(cands, correct_ans, "step_score")
            - group_score(cands, wrong_ans, "step_score")
        )

        plan_margin = (
            group_score(cands, correct_ans, "plan_score")
            - group_score(cands, wrong_ans, "plan_score")
        )

    rows.append({
        "prob_idx": prob_idx,
        "n_candidates": n_candidates,
        "n_correct_candidates": n_correct,
        "correct_fraction": n_correct / n_candidates,
        "step_answer": step_ans,
        "full_answer": full_ans,
        "step_correct": step_correct,
        "full_correct": full_correct,
        "case": case,
        "step_margin_rescue": step_margin,
        "plan_margin_rescue": plan_margin,
    })


with open(OUT_DIR / "per_problem.jsonl", "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")


case_counts = Counter(r["case"] for r in rows)

with open(OUT_DIR / "case_summary.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["case", "count", "percentage"])

    for case in ["both_correct", "rescue", "hurt", "both_wrong"]:
        count = case_counts[case]
        pct = 100 * count / len(rows)
        writer.writerow([case, count, f"{pct:.2f}"])


print("\nCase summary:")
for case in ["both_correct", "rescue", "hurt", "both_wrong"]:
    print(
        f"{case}: {case_counts[case]} "
        f"({100 * case_counts[case] / len(rows):.2f}%)"
    )


bucket = defaultdict(list)

for r in rows:
    bucket[r["n_correct_candidates"]].append(r)

summary_rows = []

for k in sorted(bucket):
    vals = bucket[k]

    step_acc = np.mean([r["step_correct"] for r in vals])
    full_acc = np.mean([r["full_correct"] for r in vals])

    summary_rows.append({
        "n_correct_candidates": k,
        "n_problems": len(vals),
        "step_acc": step_acc,
        "full_acc": full_acc,
        "gain": full_acc - step_acc,
    })


with open(
    OUT_DIR / "summary_by_correct_count.csv",
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "n_correct_candidates",
            "n_problems",
            "step_acc",
            "full_acc",
            "gain",
        ],
    )

    writer.writeheader()

    for r in summary_rows:
        writer.writerow({
            "n_correct_candidates": r["n_correct_candidates"],
            "n_problems": r["n_problems"],
            "step_acc": f"{100*r['step_acc']:.2f}",
            "full_acc": f"{100*r['full_acc']:.2f}",
            "gain": f"{100*r['gain']:.2f}",
        })


print("\nBy number of correct candidates:")
for r in summary_rows:
    print(
        f"k={r['n_correct_candidates']} "
        f"n={r['n_problems']} "
        f"step={100*r['step_acc']:.2f}% "
        f"full={100*r['full_acc']:.2f}% "
        f"gain={100*r['gain']:+.2f}"
    )


rescues = [r for r in rows if r["case"] == "rescue"]

step_margins = np.array([
    r["step_margin_rescue"]
    for r in rescues
    if np.isfinite(r["step_margin_rescue"])
])

plan_margins = np.array([
    r["plan_margin_rescue"]
    for r in rescues
    if np.isfinite(r["plan_margin_rescue"])
])


with open(
    OUT_DIR / "rescue_margin_summary.csv",
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.writer(f)
    writer.writerow(["metric", "n", "mean", "median"])

    writer.writerow([
        "step_margin",
        len(step_margins),
        np.mean(step_margins) if len(step_margins) else np.nan,
        np.median(step_margins) if len(step_margins) else np.nan,
    ])

    writer.writerow([
        "plan_margin",
        len(plan_margins),
        np.mean(plan_margins) if len(plan_margins) else np.nan,
        np.median(plan_margins) if len(plan_margins) else np.nan,
    ])


print("\nRescue cases:", len(rescues))

if len(step_margins):
    print("Mean Step margin:", step_margins.mean())
    print("Median Step margin:", np.median(step_margins))

if len(plan_margins):
    print("Mean Plan margin:", plan_margins.mean())
    print("Median Plan margin:", np.median(plan_margins))


fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))

ax = axes[0]

plot_rows = [
    r for r in summary_rows
    if r["n_problems"] >= 5
]

x = [r["n_correct_candidates"] for r in plot_rows]
step_y = [100 * r["step_acc"] for r in plot_rows]
full_y = [100 * r["full_acc"] for r in plot_rows]

ax.plot(x, step_y, marker="o", label="Step-only")
ax.plot(x, full_y, marker="o", label="Full")

ax.set_xlabel("Number of Correct Candidates")
ax.set_ylabel("Selection Accuracy (%)")
ax.set_title("(a) Selection by Candidate Composition")
ax.legend(frameon=False)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)


ax = axes[1]

means = [
    np.mean(step_margins) if len(step_margins) else 0,
    np.mean(plan_margins) if len(plan_margins) else 0,
]

bars = ax.bar(
    ["Step-score\nmargin", "Plan-score\nmargin"],
    means,
    width=0.5,
)

ax.axhline(0, linewidth=1)

ax.set_ylabel("Correct - Incorrect Answer-Group Score")
ax.set_title("(b) Score Margins in Rescued Cases")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

for rect, val in zip(bars, means):
    offset = 0.02 if val >= 0 else -0.02

    ax.text(
        rect.get_x() + rect.get_width()/2,
        val + offset,
        f"{val:.2f}",
        ha="center",
        va="bottom" if val >= 0 else "top",
    )


fig.tight_layout()

fig.savefig(
    OUT_DIR / "when_plan_helps.png",
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    OUT_DIR / "when_plan_helps.pdf",
    bbox_inches="tight",
)

plt.close(fig)

print("\nSaved to:", OUT_DIR)
