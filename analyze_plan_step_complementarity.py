import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd


ROOT = Path("/public/home/ljt/lzm/code/PRM")

DATASETS = {
    "MATH500": {
        "traj": ROOT / "runs/BoN_weak_policy/math500/trajectories.jsonl",
        "score": ROOT / "runs/eval/math500_bon_scores_dump.json",
    },
    "GSM8K": {
        "traj": ROOT / "runs/BoN_weak_policy/gsm8k/trajectories.jsonl",
        "score": ROOT / "runs/eval/gsm8k_bon_scores_dump.json",
    },
    "OlympiadBench": {
        "traj": ROOT / "runs/BoN_weak_policy/olympiadbench/trajectories.jsonl",
        "score": ROOT / "runs/eval/olympiadbench_bon_scores_dump.json",
    },
    "GaokaoEN": {
        "traj": ROOT / "runs/BoN_weak_policy/gaokaoen/trajectories.jsonl",
        "score": ROOT / "runs/eval/gaokaoen_bon_scores_dump.json",
    },
}

OUT_DIR = ROOT / "runs/analysis_plan_step_complementarity"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def step_mean(score_dict, n_steps):
    if n_steps <= 0:
        return None

    vals = []
    for i in range(n_steps):
        v = score_dict.get(str(i))
        vals.append(0.5 if v is None else float(v))

    return float(np.mean(vals))


def load_dataset(name, cfg):
    traj_path = cfg["traj"]
    score_path = cfg["score"]

    if not traj_path.exists():
        print(f"[WARN] trajectory not found: {traj_path}")
        return None

    if not score_path.exists():
        print(f"[WARN] score dump not found: {score_path}")
        return None

    trajectories = [
        json.loads(line)
        for line in open(traj_path, encoding="utf-8")
        if line.strip()
    ]

    dump = json.load(open(score_path, encoding="utf-8"))

    step_scores = dump["step_score"]
    plan_scores = dump["plan_score"]

    by_prob = defaultdict(list)
    for t in trajectories:
        by_prob[int(t["prob_idx"])].append(t)

    rows = []
    missing = 0

    for prob_idx, candidates in by_prob.items():
        for local_idx, t in enumerate(candidates):

            key = f"{prob_idx}:{local_idx}"

            if key not in step_scores or key not in plan_scores:
                missing += 1
                continue

            sm = step_mean(
                step_scores[key],
                len(t.get("steps", []))
            )

            ps = plan_scores.get(key)

            if sm is None or ps is None:
                missing += 1
                continue

            rows.append({
                "dataset": name,
                "prob_idx": prob_idx,
                "local_idx": local_idx,
                "correct": int(bool(t.get("correct", False))),
                "step_score": float(sm),
                "plan_score": float(ps),
            })

    if not rows:
        print(f"[WARN] no usable chains for {name}")
        return None

    df = pd.DataFrame(rows)

    print(
        f"{name}: "
        f"{len(df)} usable chains | "
        f"{missing} missing | "
        f"accuracy={df.correct.mean():.4f}"
    )

    return df


def tertile_bins(series):
    """
    Rank-based tertiles to avoid qcut failures caused by tied scores.
    """
    ranks = series.rank(method="average", pct=True)

    return pd.cut(
        ranks,
        bins=[0.0, 1/3, 2/3, 1.0],
        labels=["Low", "Mid", "High"],
        include_lowest=True,
    )


# ============================================================
# Load all datasets
# ============================================================

dfs = []

for name, cfg in DATASETS.items():
    df = load_dataset(name, cfg)

    if df is not None and len(df):
        dfs.append(df)

if not dfs:
    raise RuntimeError("No dataset could be loaded.")

all_df = pd.concat(dfs, ignore_index=True)

all_df.to_csv(
    OUT_DIR / "all_chain_scores.csv",
    index=False
)


# ============================================================
# Experiment 1
# Conditional utility of plan score given step score
# ============================================================

conditional_rows = []

for dataset, df in all_df.groupby("dataset"):

    df = df.copy()

    df["step_bin"] = tertile_bins(df["step_score"])
    df["plan_bin"] = tertile_bins(df["plan_score"])

    for sb in ["Low", "Mid", "High"]:
        for pb in ["Low", "Mid", "High"]:

            x = df[
                (df["step_bin"] == sb)
                & (df["plan_bin"] == pb)
            ]

            conditional_rows.append({
                "dataset": dataset,
                "step_bin": sb,
                "plan_bin": pb,
                "n": len(x),
                "accuracy": (
                    float(x["correct"].mean())
                    if len(x)
                    else np.nan
                ),
                "mean_step": (
                    float(x["step_score"].mean())
                    if len(x)
                    else np.nan
                ),
                "mean_plan": (
                    float(x["plan_score"].mean())
                    if len(x)
                    else np.nan
                ),
            })


conditional = pd.DataFrame(conditional_rows)

conditional.to_csv(
    OUT_DIR / "conditional_plan_utility.csv",
    index=False
)


# ============================================================
# Experiment 2
# Plan x Step 2x2 taxonomy
# ============================================================

taxonomy_rows = []

for dataset, df in all_df.groupby("dataset"):

    df = df.copy()

    df["plan_high"] = df["plan_score"] > 0.5
    df["step_high"] = df["step_score"] > 0.5

    def get_quadrant(r):
        if r["plan_high"] and r["step_high"]:
            return "HH"
        if r["plan_high"] and not r["step_high"]:
            return "HL"
        if not r["plan_high"] and r["step_high"]:
            return "LH"
        return "LL"

    df["quadrant"] = df.apply(get_quadrant, axis=1)

    for q in ["HH", "HL", "LH", "LL"]:

        x = df[df["quadrant"] == q]

        taxonomy_rows.append({
            "dataset": dataset,
            "quadrant": q,
            "n": len(x),
            "fraction": len(x) / len(df),
            "accuracy": (
                float(x["correct"].mean())
                if len(x)
                else np.nan
            ),
            "mean_plan": (
                float(x["plan_score"].mean())
                if len(x)
                else np.nan
            ),
            "mean_step": (
                float(x["step_score"].mean())
                if len(x)
                else np.nan
            ),
        })


taxonomy = pd.DataFrame(taxonomy_rows)

taxonomy.to_csv(
    OUT_DIR / "plan_step_taxonomy.csv",
    index=False
)


# ============================================================
# Print Experiment 1
# ============================================================

print("\n")
print("=" * 90)
print("EXPERIMENT 1: CONDITIONAL PLAN UTILITY")
print("=" * 90)

for dataset in conditional["dataset"].unique():

    print(f"\n[{dataset}]")

    x = conditional[
        conditional["dataset"] == dataset
    ]

    acc = x.pivot(
        index="step_bin",
        columns="plan_bin",
        values="accuracy"
    )

    acc = acc.reindex(
        index=["Low", "Mid", "High"],
        columns=["Low", "Mid", "High"]
    )

    n = x.pivot(
        index="step_bin",
        columns="plan_bin",
        values="n"
    )

    n = n.reindex(
        index=["Low", "Mid", "High"],
        columns=["Low", "Mid", "High"]
    )

    print("\nAccuracy:")
    print(acc.round(4).to_string())

    print("\nN:")
    print(n.to_string())


# ============================================================
# Conditional Plan High - Low effect
# ============================================================

effects = []

for dataset in conditional["dataset"].unique():

    x = conditional[
        conditional["dataset"] == dataset
    ]

    for sb in ["Low", "Mid", "High"]:

        xx = x[x["step_bin"] == sb]

        low = xx[
            xx["plan_bin"] == "Low"
        ]["accuracy"]

        high = xx[
            xx["plan_bin"] == "High"
        ]["accuracy"]

        if (
            len(low)
            and len(high)
            and not np.isnan(low.iloc[0])
            and not np.isnan(high.iloc[0])
        ):
            effects.append({
                "dataset": dataset,
                "step_bin": sb,
                "plan_high_minus_low": (
                    float(high.iloc[0])
                    - float(low.iloc[0])
                )
            })


effects = pd.DataFrame(effects)

effects.to_csv(
    OUT_DIR / "conditional_plan_effect.csv",
    index=False
)

print("\n")
print("=" * 90)
print("PLAN HIGH - PLAN LOW ACCURACY WITHIN EACH STEP BIN")
print("=" * 90)

if len(effects):
    print(effects.round(4).to_string(index=False))

    print("\nMacro average:")
    print(
        effects.groupby("step_bin")[
            "plan_high_minus_low"
        ].mean().round(4)
    )


# ============================================================
# Print Experiment 2
# ============================================================

print("\n")
print("=" * 90)
print("EXPERIMENT 2: PLAN x STEP TAXONOMY")
print("=" * 90)

for dataset in taxonomy["dataset"].unique():

    print(f"\n[{dataset}]")

    x = taxonomy[
        taxonomy["dataset"] == dataset
    ].copy()

    print(
        x[
            [
                "quadrant",
                "n",
                "fraction",
                "accuracy",
                "mean_plan",
                "mean_step"
            ]
        ].round(4).to_string(index=False)
    )


# ============================================================
# Macro taxonomy
# ============================================================

macro_tax = (
    taxonomy
    .groupby("quadrant")
    .agg(
        total_n=("n", "sum"),
        mean_fraction=("fraction", "mean"),
        mean_accuracy=("accuracy", "mean")
    )
    .reset_index()
)

macro_tax = macro_tax.set_index("quadrant").reindex(
    ["HH", "HL", "LH", "LL"]
).reset_index()

macro_tax.to_csv(
    OUT_DIR / "plan_step_taxonomy_macro.csv",
    index=False
)

print("\n[Macro-average across datasets]")

print(
    macro_tax.round(4).to_string(index=False)
)


print("\n")
print("=" * 90)
print("SAVED FILES")
print("=" * 90)

for p in sorted(OUT_DIR.iterdir()):
    print(p)
