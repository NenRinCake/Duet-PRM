import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd


ROOT = Path("/public/home/ljt/lzm/code/PRM")

DATASETS = {
    "OlympiadBench": {
        "traj": ROOT / "runs/BoN_weak_policy/olympiadbench/trajectories.jsonl",
        "score": ROOT / "runs/eval/olympiad_bon_scores_dump.json",
    },
    "GaokaoEN": {
        "traj": ROOT / "runs/BoN_weak_policy/gaokao2023en/trajectories.jsonl",
        "score": ROOT / "runs/eval/gaokao_bon_scores.json",
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

    print(f"\n{name} score keys:", list(dump.keys()))

    if "step_score" not in dump or "plan_score" not in dump:
        raise RuntimeError(
            f"{name}: score file does not contain "
            f"'step_score' and 'plan_score'"
        )

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

    df = pd.DataFrame(rows)

    print(
        f"{name}: "
        f"{len(df)} usable chains | "
        f"{missing} missing | "
        f"accuracy={df.correct.mean():.4f}"
    )

    return df


# ============================================================
# Experiment 2: Plan x Step taxonomy
# ============================================================

for name, cfg in DATASETS.items():

    df = load_dataset(name, cfg)

    df["plan_high"] = df["plan_score"] > 0.5
    df["step_high"] = df["step_score"] > 0.5

    def quadrant(r):
        if r["plan_high"] and r["step_high"]:
            return "HH"
        elif r["plan_high"] and not r["step_high"]:
            return "HL"
        elif not r["plan_high"] and r["step_high"]:
            return "LH"
        else:
            return "LL"

    df["quadrant"] = df.apply(quadrant, axis=1)

    print("\n" + "=" * 90)
    print(f"{name}: PLAN x STEP TAXONOMY")
    print("=" * 90)

    rows = []

    for q in ["HH", "HL", "LH", "LL"]:

        x = df[df["quadrant"] == q]

        row = {
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
        }

        rows.append(row)

    out = pd.DataFrame(rows)

    print(out.round(4).to_string(index=False))


    # ========================================================
    # Conditional utility:
    # split step score into Low / Mid / High,
    # compare Plan Low vs High inside each regime
    # ========================================================

    ranks = df["step_score"].rank(
        method="average",
        pct=True
    )

    df["step_bin"] = pd.cut(
        ranks,
        bins=[0.0, 1/3, 2/3, 1.0],
        labels=["Low", "Mid", "High"],
        include_lowest=True,
    )

    df["plan_group"] = np.where(
        df["plan_score"] > 0.5,
        "High",
        "Low"
    )

    print("\n" + "-" * 90)
    print(f"{name}: CONDITIONAL PLAN UTILITY")
    print("-" * 90)

    cond_rows = []

    for sb in ["Low", "Mid", "High"]:

        for pg in ["Low", "High"]:

            x = df[
                (df["step_bin"] == sb)
                & (df["plan_group"] == pg)
            ]

            cond_rows.append({
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

    cond = pd.DataFrame(cond_rows)

    print(
        cond.round(4).to_string(index=False)
    )

    print("\nHigh Plan - Low Plan accuracy gain:")

    for sb in ["Low", "Mid", "High"]:

        x = cond[cond["step_bin"] == sb]

        low = x[x["plan"] == "Low"]
        high = x[x["plan"] == "High"]

        if (
            len(low)
            and len(high)
            and not np.isnan(low.iloc[0]["accuracy"])
            and not np.isnan(high.iloc[0]["accuracy"])
        ):
            gain = (
                high.iloc[0]["accuracy"]
                - low.iloc[0]["accuracy"]
            )

            print(
                f"{sb:>4}: {gain:+.4f} "
                f"({100*gain:+.2f} pp)"
            )

