import json
from collections import defaultdict
from pathlib import Path

ROOT = Path("/public/home/ljt/lzm/code/PRM")
OUT = ROOT / "runs/case_study/plan_step_contrast"
OUT.mkdir(parents=True, exist_ok=True)

BENCHES = {
    "math500": {
        "traj": ROOT / "runs/BoN/math500/trajectories.jsonl",
        "score": ROOT / "runs/eval/math500_bon_scores_dump.json",
    },
    "gsm8k": {
        "traj": ROOT / "runs/BoN/gsm8k/trajectories.jsonl",
        "score": ROOT / "runs/eval/gsm8k_bon_scores_dump.json",
    },
    "olympiadbench": {
        "traj": ROOT / "runs/BoN/olympiadbench/trajectories.jsonl",
        "score": ROOT / "runs/eval/olympiad_bon_scores_dump.json",
    },
    "gaokao2023en": {
        "traj": ROOT / "runs/BoN/gaokao2023en/trajectories.jsonl",
        "score": ROOT / "runs/eval/gaokao_scores.json",
    },
}

STRICT = {
    "high_plan": 0.7,
    "low_step": 0.4,
    "low_plan": 0.3,
    "high_step": 0.6,
}

MARGIN = 0.30


def mean_step_score(score_dict, n_steps):
    vals = []
    for i in range(n_steps):
        v = score_dict.get(str(i))
        vals.append(0.5 if v is None else float(v))
    if not vals:
        return None
    return sum(vals) / len(vals)


def plan_signature(plan):
    return tuple(
        (
            str(x.get("action", "")).strip(),
            str(x.get("desc", "")).strip().lower(),
        )
        for x in plan
    )


all_cases = []

for bench, cfg in BENCHES.items():
    print("\n" + "=" * 100)
    print("BENCH:", bench)

    trajs = [json.loads(x) for x in open(cfg["traj"], encoding="utf-8")]
    dump = json.load(open(cfg["score"], encoding="utf-8"))

    step_scores = dump["step_score"]
    plan_scores = dump["plan_score"]

    by_prob = defaultdict(list)
    for t in trajs:
        by_prob[int(t["prob_idx"])].append(t)

    scored = defaultdict(list)

    for prob_idx, candidates in by_prob.items():
        for local_idx, t in enumerate(candidates):
            key = f"{prob_idx}:{local_idx}"

            if key not in step_scores or key not in plan_scores:
                continue

            steps = t.get("steps", [])
            plan = t.get("plan", [])

            if not steps or not plan:
                continue

            ss = mean_step_score(step_scores[key], len(steps))
            ps = plan_scores.get(key)

            if ss is None or ps is None:
                continue

            scored[prob_idx].append({
                "local_idx": local_idx,
                "plan_idx": t.get("plan_idx"),
                "exec_idx": t.get("exec_idx"),
                "answer": t.get("answer"),
                "correct": bool(t.get("correct", False)),
                "step_score": float(ss),
                "plan_score": float(ps),
                "plan": plan,
                "steps": steps,
            })

    strict_cases = []
    loose_cases = []

    for prob_idx, candidates in scored.items():
        wrongs = [x for x in candidates if not x["correct"]]

        if len(wrongs) < 2:
            continue

        # A = high plan / low step
        strict_A = [
            x for x in wrongs
            if x["plan_score"] >= STRICT["high_plan"]
            and x["step_score"] <= STRICT["low_step"]
        ]

        # B = low plan / high step
        strict_B = [
            x for x in wrongs
            if x["plan_score"] <= STRICT["low_plan"]
            and x["step_score"] >= STRICT["high_step"]
        ]

        for a in strict_A:
            for b in strict_B:
                if a["local_idx"] == b["local_idx"]:
                    continue

                plan_gap = a["plan_score"] - b["plan_score"]
                step_gap = b["step_score"] - a["step_score"]

                score = plan_gap + step_gap

                strict_cases.append({
                    "benchmark": bench,
                    "prob_idx": prob_idx,
                    "type": "strict",
                    "contrast_score": score,
                    "plan_gap": plan_gap,
                    "step_gap": step_gap,
                    "same_plan_idx": a["plan_idx"] == b["plan_idx"],
                    "same_plan_text": (
                        plan_signature(a["plan"]) ==
                        plan_signature(b["plan"])
                    ),
                    "high_plan_low_step": a,
                    "low_plan_high_step": b,
                })

        # 宽松版：只要求方向差足够大
        loose_A = [
            x for x in wrongs
            if x["plan_score"] - x["step_score"] >= MARGIN
        ]

        loose_B = [
            x for x in wrongs
            if x["step_score"] - x["plan_score"] >= MARGIN
        ]

        for a in loose_A:
            for b in loose_B:
                if a["local_idx"] == b["local_idx"]:
                    continue

                plan_gap = a["plan_score"] - b["plan_score"]
                step_gap = b["step_score"] - a["step_score"]

                if plan_gap <= 0 or step_gap <= 0:
                    continue

                score = plan_gap + step_gap

                loose_cases.append({
                    "benchmark": bench,
                    "prob_idx": prob_idx,
                    "type": "loose",
                    "contrast_score": score,
                    "plan_gap": plan_gap,
                    "step_gap": step_gap,
                    "same_plan_idx": a["plan_idx"] == b["plan_idx"],
                    "same_plan_text": (
                        plan_signature(a["plan"]) ==
                        plan_signature(b["plan"])
                    ),
                    "high_plan_low_step": a,
                    "low_plan_high_step": b,
                })

    strict_cases.sort(
        key=lambda x: (
            not x["same_plan_idx"],
            x["contrast_score"],
        ),
        reverse=True,
    )

    loose_cases.sort(
        key=lambda x: (
            not x["same_plan_idx"],
            x["contrast_score"],
        ),
        reverse=True,
    )

    print("strict cases:", len(strict_cases))
    print("loose cases :", len(loose_cases))

    print("\nTop STRICT:")
    for i, x in enumerate(strict_cases[:10], 1):
        a = x["high_plan_low_step"]
        b = x["low_plan_high_step"]
        print(
            f"{i:2d}. prob={x['prob_idx']:4d} "
            f"contrast={x['contrast_score']:.3f} "
            f"same_plan={x['same_plan_idx']} | "
            f"A(plan={a['plan_score']:.3f}, step={a['step_score']:.3f}, ans={a['answer']}) | "
            f"B(plan={b['plan_score']:.3f}, step={b['step_score']:.3f}, ans={b['answer']})"
        )

    print("\nTop LOOSE:")
    for i, x in enumerate(loose_cases[:10], 1):
        a = x["high_plan_low_step"]
        b = x["low_plan_high_step"]
        print(
            f"{i:2d}. prob={x['prob_idx']:4d} "
            f"contrast={x['contrast_score']:.3f} "
            f"same_plan={x['same_plan_idx']} | "
            f"A(plan={a['plan_score']:.3f}, step={a['step_score']:.3f}, ans={a['answer']}) | "
            f"B(plan={b['plan_score']:.3f}, step={b['step_score']:.3f}, ans={b['answer']})"
        )

    # 优先 strict；没有的话保留 loose
    selected = strict_cases if strict_cases else loose_cases

    out_path = OUT / f"{bench}_contrast_cases.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for rank, x in enumerate(selected, 1):
            y = dict(x)
            y["rank"] = rank
            f.write(json.dumps(y, ensure_ascii=False) + "\n")

    all_cases.extend(selected)

    print("saved:", out_path)

all_cases.sort(
    key=lambda x: (
        x["type"] == "strict",
        not x["same_plan_idx"],
        x["contrast_score"],
    ),
    reverse=True,
)

combined_path = OUT / "all_benchmarks_ranked.jsonl"
with open(combined_path, "w", encoding="utf-8") as f:
    for rank, x in enumerate(all_cases, 1):
        y = dict(x)
        y["rank"] = rank
        f.write(json.dumps(y, ensure_ascii=False) + "\n")

print("\n" + "#" * 100)
print("GLOBAL TOP 20")
print("#" * 100)

for i, x in enumerate(all_cases[:20], 1):
    a = x["high_plan_low_step"]
    b = x["low_plan_high_step"]
    print(
        f"{i:2d}. {x['benchmark']:<14} "
        f"prob={x['prob_idx']:4d} "
        f"{x['type']:<6} "
        f"contrast={x['contrast_score']:.3f} "
        f"same_plan={x['same_plan_idx']} | "
        f"A: P={a['plan_score']:.3f} S={a['step_score']:.3f} ans={a['answer']} | "
        f"B: P={b['plan_score']:.3f} S={b['step_score']:.3f} ans={b['answer']}"
    )

print("\nSaved combined ranking:")
print(combined_path)
