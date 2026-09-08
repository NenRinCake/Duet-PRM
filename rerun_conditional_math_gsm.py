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
}


def step_mean(score_dict, n_steps):
    if n_steps <= 0:
        return None

    vals = []
    for i in range(n_steps):
        v = score_dict.get(str(i))
        vals.append(0.5 if v is None else float(v))

    return float(np.mean(vals))


def load_dataset(name, cfg):
    trajectories = [
        json.loads(line)
        for line in open(cfg["traj"], encoding="utf-8")
        if line.strip()
    ]

    dump = json.load(open(cfg["score"], encoding="utf-8"))

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
                "prob_idx": prob_idx,
                "correct": int(bool(t.get("correct", False))),
                "step_score": float(sm),
                "plan_score": float(ps),
            })

    df = pd.DataFrame(rows)

    print(
        f"{name}: {len(df)} usable chains | "
        f"{missing} missing | "
        f"accuracy={df.correct.mean():.4f}"
    )

    return df


for name, cfg in DATASETS.items():

    df = load_dataset(name, cfg)

    # Step score -> Low / Mid / High
    ranks = df["step_score"].rank(
        method="average",
        pct=True
    )

    df["step_bin"] = pd.cut(
        ranks,
        bins=[0.0, 1/3, 2/3, 1.0],
        labels=["Low", "Mid", "High"],
        include_lowest=True
    )

    # Plan score -> Low / High
    df["plan_group"] = np.where(
        df["plan_score"] > 0.5,
        "High",
        "Low"
    )

    print("\n" + "=" * 90)
    print(f"{name}: CONDITIONAL PLAN UTILITY")
    print("=" * 90)

    rows = []

    for sb in ["Low", "Mid", "High"]:
        for pg in ["Low", "High"]:

            x = df[
                (df["step_bin"] == sb)
                & (df["plan_group"] == pg)
            ]

            rows.append({
                "step_bin": sb,
                "plan": pg,
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

    out = pd.DataFrame(rows)

    print(out.round(4).to_string(index=False))

    print("\nHigh Plan - Low Plan accuracy gain:")

    for sb in ["Low", "Mid", "High"]:

        x = out[out["step_bin"] == sb]

        low = x[x["plan"] == "Low"]["accuracy"].iloc[0]
        high = x[x["plan"] == "High"]["accuracy"].iloc[0]

        gain = high - low

        print(
            f"{sb:>4}: "
            f"{gain:+.4f} "
            f"({100*gain:+.2f} pp)"
        )
