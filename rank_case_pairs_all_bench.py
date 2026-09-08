import json
from collections import defaultdict
from pathlib import Path

ROOT = Path("/public/home/ljt/lzm/code/PRM")
OUT_ROOT = ROOT / "runs/case_study"

ALPHA = 0.3

BENCHES = {
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

# 先宽松筛，后面人工选
MIN_STEP_ADV_WRONG = 0.02
MIN_PLAN_ADV_CORRECT = 0.10
MIN_FULL_ADV_CORRECT = 0.005


def norm_answer(x):
    if x is None:
        return None
    return str(x).strip()


def plan_signature(plan):
    return tuple(
        (
            str(x.get("action", "")).strip(),
            str(x.get("desc", "")).strip().lower(),
        )
        for x in plan
    )


def mean_step_score(score_dict, n_steps):
    vals = []
    for i in range(n_steps):
        v = score_dict.get(str(i))
        vals.append(0.5 if v is None else float(v))
    if not vals:
        return None
    return sum(vals) / len(vals)


def load_problem_text_from_traj(t):
    # trajectory 内若没有 problem 文本也没关系，
    # qualitative shortlist 先看 plan/answer 即可
    for k in ("problem", "question", "query", "prompt", "input"):
        if t.get(k) is not None:
            return str(t[k])
    return ""


for bench, cfg in BENCHES.items():
    print("\n" + "=" * 100)
    print("BENCH:", bench)

    if not cfg["traj"].exists():
        print("SKIP: missing trajectory file:", cfg["traj"])
        continue
    if not cfg["score"].exists():
        print("SKIP: missing score file:", cfg["score"])
        continue

    trajs = [json.loads(x) for x in open(cfg["traj"], encoding="utf-8")]
    dump = json.load(open(cfg["score"], encoding="utf-8"))

    if not isinstance(dump, dict):
        print("SKIP: score file is not a dict")
        continue

    print("score keys:", list(dump.keys())[:20])

    if "step_score" not in dump or "plan_score" not in dump:
        print("SKIP: score file lacks step_score / plan_score")
        continue

    step_scores = dump["step_score"]
    plan_scores = dump["plan_score"]

    by_prob = defaultdict(list)
    for t in trajs:
        by_prob[int(t["prob_idx"])].append(t)

    scored = defaultdict(list)
    missing = 0

    for prob_idx, candidates in by_prob.items():
        for local_idx, t in enumerate(candidates):
            key = f"{prob_idx}:{local_idx}"

            if key not in step_scores or key not in plan_scores:
                missing += 1
                continue

            steps = t.get("steps", [])
            plan = t.get("plan", [])

            sm = mean_step_score(step_scores[key], len(steps))
            if sm is None:
                missing += 1
                continue

            ps = plan_scores.get(key)
            if ps is None:
                missing += 1
                continue

            ps = float(ps)
            full = 0.7 * sm + 0.3 * ps

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
                "plan": plan,
                "steps": steps,
                "problem": load_problem_text_from_traj(t),
            })

    pairs = []

    for prob_idx, candidates in scored.items():
        corrects = [x for x in candidates if x["correct"]]
        wrongs = [x for x in candidates if not x["correct"]]

        if not corrects or not wrongs:
            continue

        for c in corrects:
            for w in wrongs:
                if norm_answer(c["answer"]) == norm_answer(w["answer"]):
                    continue

                if not c["plan"] or not w["plan"]:
                    continue
                if not c["steps"] or not w["steps"]:
                    continue

                step_wrong_adv = w["step_score"] - c["step_score"]
                plan_correct_adv = c["plan_score"] - w["plan_score"]
                full_correct_adv = c["combined_score"] - w["combined_score"]

                if step_wrong_adv < MIN_STEP_ADV_WRONG:
                    continue
                if plan_correct_adv < MIN_PLAN_ADV_CORRECT:
                    continue
                if full_correct_adv < MIN_FULL_ADV_CORRECT:
                    continue

                same_plan_idx = c["plan_idx"] == w["plan_idx"]
                same_plan_text = (
                    plan_signature(c["plan"]) == plan_signature(w["plan"])
                )

                plan_relation = (
                    "different_plan"
                    if (not same_plan_idx and not same_plan_text)
                    else "same_or_equiv_plan"
                )

                suitability = (
                    3.0 * plan_correct_adv
                    + 2.0 * step_wrong_adv
                    + 2.0 * full_correct_adv
                    + (0.5 if plan_relation == "different_plan" else 0.0)
                )

                pairs.append({
                    "prob_idx": prob_idx,
                    "plan_relation": plan_relation,
                    "same_plan_idx": same_plan_idx,
                    "same_plan_text": same_plan_text,
                    "step_wrong_adv": step_wrong_adv,
                    "plan_correct_adv": plan_correct_adv,
                    "full_correct_adv": full_correct_adv,
                    "suitability": suitability,
                    "wrong": w,
                    "correct": c,
                })

    pairs.sort(
        key=lambda x: (
            x["plan_relation"] == "different_plan",
            x["suitability"],
            x["plan_correct_adv"],
            x["step_wrong_adv"],
        ),
        reverse=True,
    )

    out_dir = OUT_ROOT / bench
    out_dir.mkdir(parents=True, exist_ok=True)

    out_jsonl = out_dir / "pairwise_ranking_reversals.jsonl"
    out_tsv = out_dir / "pairwise_ranking_reversals.tsv"

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for rank, p in enumerate(pairs, 1):
            x = dict(p)
            x["rank"] = rank
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    with open(out_tsv, "w", encoding="utf-8") as f:
        f.write(
            "rank\tprob_idx\tplan_relation\t"
            "step_wrong\tstep_correct\tstep_adv_wrong\t"
            "plan_wrong\tplan_correct\tplan_adv_correct\t"
            "full_wrong\tfull_correct\tfull_adv_correct\t"
            "wrong_answer\tcorrect_answer\t"
            "wrong_plan_idx\tcorrect_plan_idx\n"
        )

        for rank, p in enumerate(pairs, 1):
            w = p["wrong"]
            c = p["correct"]

            f.write(
                f"{rank}\t{p['prob_idx']}\t{p['plan_relation']}\t"
                f"{w['step_score']:.4f}\t{c['step_score']:.4f}\t"
                f"{p['step_wrong_adv']:.4f}\t"
                f"{w['plan_score']:.4f}\t{c['plan_score']:.4f}\t"
                f"{p['plan_correct_adv']:.4f}\t"
                f"{w['combined_score']:.4f}\t{c['combined_score']:.4f}\t"
                f"{p['full_correct_adv']:.4f}\t"
                f"{norm_answer(w['answer'])}\t{norm_answer(c['answer'])}\t"
                f"{w['plan_idx']}\t{c['plan_idx']}\n"
            )

    n_diff = sum(1 for p in pairs if p["plan_relation"] == "different_plan")
    n_same = len(pairs) - n_diff

    print("problems:", len(by_prob))
    print("scored problems:", len(scored))
    print("missing/unscored chains:", missing)
    print("qualifying pairs:", len(pairs))
    print("different-plan pairs:", n_diff)
    print("same/equiv-plan pairs:", n_same)

    print("\nTop 15:")
    for rank, p in enumerate(pairs[:15], 1):
        w = p["wrong"]
        c = p["correct"]
        print(
            f"{rank:2d}. "
            f"prob={p['prob_idx']:4d} "
            f"{p['plan_relation']:<18} "
            f"step {w['step_score']:.3f}>{c['step_score']:.3f} "
            f"(+{p['step_wrong_adv']:.3f}) | "
            f"plan {c['plan_score']:.3f}>{w['plan_score']:.3f} "
            f"(+{p['plan_correct_adv']:.3f}) | "
            f"full {c['combined_score']:.3f}>{w['combined_score']:.3f} "
            f"| {w['answer']} -> {c['answer']}"
        )

    print("saved:", out_jsonl)
    print("saved:", out_tsv)
