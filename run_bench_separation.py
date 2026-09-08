"""
run_bench_separation.py — 一条龙: 给定一个 bench 的轨迹 jsonl + 一个 PRM,
直接跑出该 PRM 在这个 bench 上的链级 separation / Cohen's d / AUC。

省掉"存分片 -> merge"的中间步骤: 内存里打完分直接算指标。

== 复用, 不重复造轮子 ==
  打分: 直接调 eval_bon.py 的 score_ours / score_scalar / score_skywork
        (分类格式 --score_scheme、双分数解析都在里面, 口径与 BoN 评测一致)
  指标: 直接调 eval_separation.py 的 separation / cohen_d / auc + cluster bootstrap
  => 本文件只是编排, 不碰打分和指标的实现细节。

== 输入轨迹 ==
  sample_minimal.py 产出的 jsonl, 每行一条链 (prob_idx/plan_idx/exec_idx/plan/
  steps/answer/correct/...)。每题一条也行 (N=1), 每题多条也行。
  必须带 correct 字段 (verifier 终判): correct=True->正确链, False->错误链,
  None(截断)->自动剔除, 不进任何一类。
  强烈建议传 --problems (原始题目文件): PRM 输入要用题目原文, 不传则退化成
  用 plan 第一句当题面, 打分会失真。

== 用法 ==
  # 你自己的生成式 PRM (双分数, 三分类格式)
  python run_bench_separation.py \
      --traj runs/eval/math500.jsonl --problems data/math500.jsonl \
      --mode ours --dual_score --score_scheme v6 \
      --how mean,min

  # 外部 CoT PRM 基线 (打分头 / Skywork), 喂同一份 bench
  python run_bench_separation.py \
      --traj runs/eval/math500.jsonl --problems data/math500.jsonl \
      --mode scalar --prm /path/Qwen2.5-Math-PRM-7B --how mean,min

  # 顺手把分数存成分片, 之后多个 PRM 用 eval_separation.py 做成对显著性检验
  python run_bench_separation.py ... --dump_scores runs/eval/ours_full.shard0.json
"""
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import argparse, json, types
from collections import defaultdict

import eval_bon
from eval_separation import (chain_scores, point, cluster_bootstrap_ci,
                             separation, cohens_d, auc, fmt, fmt_ci)

SCORERS = {"ours": eval_bon.score_ours,
           "scalar": eval_bon.score_scalar,
           "skywork": eval_bon.score_skywork}

def build_scorer_args(a):
    """组装 eval_bon.score_* 需要的 args 命名空间 (非分片模式: num_shards=1)。"""
    return types.SimpleNamespace(
        prm=a.prm, tp=a.tp, max_model_len=a.max_model_len, max_gen=a.max_gen,
        repetition_penalty=a.repetition_penalty, step_sep=a.step_sep,
        dual_score=a.dual_score, score_scheme=a.score_scheme,
        num_shards=1, shard_id=0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True, help="bench 轨迹 jsonl (带 correct)")
    ap.add_argument("--problems", default=None, help="原始题目 jsonl (强烈建议传)")
    ap.add_argument("--mode", default="ours", choices=["ours", "scalar", "skywork"])
    ap.add_argument("--prm", default=None, help="PRM 模型路径 (三种 mode 都需要)")
    # ---- ours 双分数 (分类格式) ----
    ap.add_argument("--dual_score", action="store_true",
                    help="mode=ours 且模型直出 <plan_score>/<step_score> 时加")
    ap.add_argument("--score_scheme", choices=list(eval_bon.INSTRUCTION_DUAL_VARIANTS.keys()),
                    default=None,
                    help="checkpoint 对应的分类方案 v6/step01/full01; --dual_score 时必填")
    # ---- 链分计算 ----
    ap.add_argument("--how", default="min,last,mean,confident_mean",
                    help="聚合方式逗号分隔 (min/mean/last/hybrid/confident_mean/adaptive_mean)")
    ap.add_argument("--plan_combine", default="multiply",
                    choices=["none", "multiply", "weighted", "plan_trust"],
                    help="是否把 plan_score 掺进链分; 跟外部 PRM 对比时保持 none")
    ap.add_argument("--plan_weight", type=float, default=0.3)
    ap.add_argument("--plan_alone", action="store_true",
                    help="额外报 plan_score 单独(不掺step)的判别力")
    # ---- 模型/采样 ----
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--max_model_len", type=int, default=8192)
    ap.add_argument("--max_gen", type=int, default=400)
    ap.add_argument("--repetition_penalty", type=float, default=1.15)
    ap.add_argument("--step_sep", default="<extra_0>", help="scalar PRM 的步分隔符")
    # ---- bootstrap / 输出 ----
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/eval/separation_report.json")
    ap.add_argument("--dump_scores", default=None,
                    help="把逐步分数存成 eval_bon.save_shard 分片格式, 供 eval_separation.py "
                         "做多 PRM 成对检验")
    a = ap.parse_args()

    if not a.prm:
        ap.error(f"--mode {a.mode} 需要 --prm")
    if a.mode == "ours" and a.dual_score and a.score_scheme is None:
        ap.error("--dual_score 时必须显式指定 --score_scheme "
                 f"({'/'.join(eval_bon.INSTRUCTION_DUAL_VARIANTS.keys())})")
    if a.mode != "ours" and a.dual_score:
        print("⚠ --dual_score 只对 mode=ours 生效, 已忽略")
    if a.plan_combine != "none" and a.mode != "ours":
        print("⚠ 外部 PRM 没有 plan_score, --plan_combine 已强制回退 none")
        a.plan_combine = "none"

    os.makedirs("runs/eval", exist_ok=True)
    if os.path.dirname(a.out):
        os.makedirs(os.path.dirname(a.out), exist_ok=True)

    hows = [h.strip() for h in a.how.split(",") if h.strip()]

    # ---- 载入轨迹 + 标签检查 ----
    trajs = [json.loads(l) for l in open(a.traj)]
    by_prob = defaultdict(list)
    for t in trajs:
        by_prob[t["prob_idx"]].append(t)
    n_chains = len(trajs)
    n_labeled = sum(1 for t in trajs if t.get("correct") is not None)
    n_censored = n_chains - n_labeled
    n_corr_total = sum(1 for t in trajs if t.get("correct") is True)
    n_wrong_total = sum(1 for t in trajs if t.get("correct") is False)
    if n_labeled == 0:
        raise SystemExit("没有任何链带 correct 标签 —— 生成 bench 时要带 gold answer 跑 verifier")
    if n_corr_total == 0 or n_wrong_total == 0:
        print(f"⚠ 严重警告: 正确链={n_corr_total}, 错误链={n_wrong_total} —— 有一类为空, "
              f"分离度/AUC 无定义。换更难/更易的 bench 或调采样温度让两类都有量。")

    problems = [json.loads(l) for l in open(a.problems)] if a.problems else None
    if problems is None:
        print("⚠ 未传 --problems: PRM 输入将退化用 plan 首句当题面, 打分会失真, 强烈建议补上")
    else:
        # 字段归一化: 不同 bench 题面/答案字段名不同, 统一成 problem/answer,
        # 让 eval_bon 的 score_ours/score_scalar/score_skywork 都能取到 ["problem"]。
        for p in problems:
            if "problem" not in p:
                for k in ("question", "query", "prompt", "input"):
                    if k in p:
                        p["problem"] = p[k]; break
            if "answer" not in p:
                for k in ("solution", "gt", "gt_answer", "target", "label"):
                    if k in p:
                        p["answer"] = p[k]; break
        miss = sum(1 for p in problems if p.get("problem") is None)
        if miss:
            raise SystemExit(
                f"[字段错误] {miss}/{len(problems)} 条题目没有可识别的题面字段; "
                f"首行: {json.dumps(problems[0], ensure_ascii=False)[:200]}")

    print(f"\nbench={a.traj}  mode={a.mode}"
          + (f"  scheme={a.score_scheme}" if a.dual_score else "")
          + f"\n链总数={n_chains} (正确={n_corr_total}, 错误={n_wrong_total}, "
          f"截断剔除={n_censored})  题数={len(by_prob)}")

    # ---- 打分 (复用 eval_bon) ----
    scorer = SCORERS[a.mode]
    sargs = build_scorer_args(a)
    step_score, parse_fail, plan_score, plan_score_step0 = scorer(by_prob, problems, sargs)
    n_steps = sum(len(t["steps"]) for v in by_prob.values() for t in v)
    print(f"打分完成: parse_fail={parse_fail}/{n_steps} "
          f"({parse_fail/max(1,n_steps)*100:.1f}%)"
          + ("  ⚠ 解析失败偏高, 检查 score_scheme 是否跟 checkpoint 匹配"
             if parse_fail / max(1, n_steps) > 0.1 else ""))

    if a.plan_combine != "none" and not plan_score:
        print("⚠ 该模型没产出 plan_score, plan_combine 自动回退 none")
        a.plan_combine = "none"

    # ---- 逐 how 算指标 ----
    report = {"bench": a.traj, "mode": a.mode, "prm": a.prm,
              "score_scheme": a.score_scheme if a.dual_score else None,
              "plan_combine": a.plan_combine, "n_boot": a.n_boot,
              "n_chains": n_chains, "n_correct": n_corr_total, "n_wrong": n_wrong_total,
              "n_censored": n_censored, "parse_fail": parse_fail, "by_how": {}}

    print("\n" + "=" * 96)
    print(f"  {'聚合':<14}{'n_corr':>7}{'n_wrong':>8}"
          f"{'separation':>13}{'  95% CI':>22}"
          f"{'cohen_d':>11}{'AUC':>9}{'  95% CI':>22}")
    print("=" * 96)
    for how in hows:
        cs = chain_scores(by_prob, step_score, plan_score, how, a.plan_combine, a.plan_weight)
        sep_pt, nc, nw = point(cs, separation)
        d_pt, _, _ = point(cs, cohens_d)
        auc_pt, _, _ = point(cs, auc)
        sep_ci = cluster_bootstrap_ci(cs, separation, a.n_boot, a.seed)
        auc_ci = cluster_bootstrap_ci(cs, auc, a.n_boot, a.seed + 2)
        print(f"  {how:<14}{nc:>7}{nw:>8}"
              f"{fmt(sep_pt):>13}{fmt_ci(sep_ci):>22}"
              f"{fmt(d_pt):>11}"
              f"{('%.4f'%auc_pt if auc_pt is not None else 'n/a'):>9}{fmt_ci(auc_ci):>22}")
        report["by_how"][how] = {
            "n_correct": nc, "n_wrong": nw,
            "separation": sep_pt, "separation_ci": sep_ci,
            "cohen_d": d_pt, "auc": auc_pt, "auc_ci": auc_ci}
    print("=" * 96)

    # ---- plan_score 单独 ----
    if a.plan_alone and plan_score:
        cs_plan = {}
        for prob_idx, cand in by_prob.items():
            lst = []
            for tpos, t in enumerate(cand):
                c = t.get("correct")
                if c is None: continue
                v = plan_score.get((prob_idx, tpos))
                if v is None: continue
                lst.append((v, bool(c)))
            if lst: cs_plan[prob_idx] = lst
        sep_pt, nc, nw = point(cs_plan, separation)
        auc_pt, _, _ = point(cs_plan, auc)
        auc_ci = cluster_bootstrap_ci(cs_plan, auc, a.n_boot, a.seed + 5)
        print(f"\n  plan_score 单独(不掺step): sep={fmt(sep_pt)}  "
              f"AUC={('%.4f'%auc_pt if auc_pt is not None else 'n/a')}  CI={fmt_ci(auc_ci)}")
        report["plan_alone"] = {"separation": sep_pt, "auc": auc_pt, "auc_ci": auc_ci,
                                "n_correct": nc, "n_wrong": nw}

    json.dump(report, open(a.out, "w"), ensure_ascii=False, indent=2, default=list)
    print(f"\n  saved -> {a.out}")

    # ---- 可选: 存分片供跨 PRM 成对检验 ----
    if a.dump_scores:
        eval_bon.save_shard(a.dump_scores, step_score, parse_fail, list(by_prob.keys()),
                            plan_score, plan_score_step0,
                            a.score_scheme if (a.mode == "ours" and a.dual_score) else None)
        print(f"  分数已存 -> {a.dump_scores}  "
              f"(多个 PRM 各存一份后用 eval_separation.py 跑成对 ΔAUC 显著性)")

if __name__ == "__main__":
    main()
