"""
eval_separation.py — 链级分离度 / Cohen's d / AUC 评测 (PRM 判别力对照)

只回答一个问题: 给定一批已判定对错的链, 各打分源能把"正确链"和"错误链"分得多开。
跟 eval_bon.py 的 BoN 准确率彻底解耦 —— 这里只看"分数本身有没有把正确链排在
错误链前面", 不掺投票/选择策略。

== 三个指标 ==
  separation : mean(score|correct) - mean(score|wrong)。有量纲, 跟分数尺度绑定,
               同一模型内部可比, 跨模型不可比 (这就是 PPT 上 +0.35 那个数)。
  cohen_d    : separation / 合并标准差。量纲归一, 跨模型可比 —— 不同 PRM 分数
               尺度不同 (你的在{0,.5,1}附近, scalar PRM 是连续 sigmoid),
               光比 separation 不公平, 必须看 d。
  auc        : P(随机正确链分 > 随机错误链分)。尺度无关, 跨模型主对照指标。
               实现逐字沿用 eval_bon.compute_auc (与 sklearn 二分类下等价)。

== 为什么要 problem 级 cluster bootstrap ==
  同一题下多条链强相关 (共享题目+计划), 按"链"重采样会严重低估方差、把 CI
  做窄。这里按 problem 整块有放回重采样, 三个指标的 CI 才诚实。

== 多打分源对照 (核心用法) ==
  --scores NAME=GLOB 可重复, 每个源是 eval_bon.save_shard 存的分片 (支持多片 glob)。
  推荐三源, 第一个作为 paired 检验的参照:
    ours_full   你的PRM看完整轨迹(含<output>)         <- 列第一个, 作参照
    ours_masked 你的PRM看output被mask掉的轨迹         (控住格式, 隔离环境信号)
    cot_prm     外部CoT PRM (Qwen2.5-Math-PRM/Skywork) (文字视图)
  对每个其它源, 报"参照 - 该源"的 ΔAUC / Δd 及其 paired bootstrap CI 和近似 p,
  判断差距是否显著。

== plan_score 的处理 ==
  跟 CoT PRM 对照时 plan_score 无对应物 -> 默认 --plan_combine none (只用 step_score
  聚合), 保证可比。--plan_combine 仅用于你自己模型的内部消融。
  另: 加 --plan_alone 会额外报"你的 plan_score 单独(不掺step)"的判别力。

用法:
  python eval_separation.py \
      --traj runs/eval/trajectories.jsonl \
      --scores ours_full=runs/eval/ours_full.shard*.json \
      --scores ours_masked=runs/eval/ours_masked.shard*.json \
      --scores cot_prm=runs/eval/skywork.shard*.json \
      --how mean,min --n_boot 2000
"""
import argparse, json, glob, random, math, bisect
from collections import defaultdict

# ============================================================================
# 以下 4 个函数逐字沿用 eval_bon.py, 保证链分聚合与 AUC 口径与 BoN 评测完全一致。
# (若 eval_bon.py 可 import, 也可以改成 from eval_bon import ... 单一真源。)
# ============================================================================
def load_shard(path):
    data = json.load(open(path))
    ss = defaultdict(dict)
    for k, d in data["step_score"].items():
        p_str, t_str = k.split(":")
        p, t = int(p_str), int(t_str)
        for i_str, v in d.items():
            ss[(p, t)][int(i_str)] = v
    plan_score = {}
    for k, v in data.get("plan_score", {}).items():
        p_str, t_str = k.split(":")
        plan_score[(int(p_str), int(t_str))] = v
    return ss, data.get("parse_fail", 0), data.get("covered_probs", []), plan_score, data.get("score_scheme")

def aggregate(step_scores, how):
    if not step_scores: return 0.0
    xs = [s if s is not None else 0.5 for s in step_scores]
    mn, mu, lst = min(xs), sum(xs) / len(xs), xs[-1]
    if how == "confident_mean":
        weights = [abs(x - 0.5) * 2 for x in xs]
        wsum = sum(weights)
        if wsum < 1e-9:
            return mu
        return sum(x * w for x, w in zip(xs, weights)) / wsum
    if how == "adaptive_mean":
        weights = [abs(x - 0.5) * 2 for x in xs]
        wsum = sum(weights)
        conf_frac = sum(1 for w in weights if w > 1e-9) / len(weights)
        cm = mu if wsum < 1e-9 else sum(x * w for x, w in zip(xs, weights)) / wsum
        return conf_frac * cm + (1 - conf_frac) * mu
    return {"min": mn, "mean": mu, "last": lst, "hybrid": (mn + mu) / 2}[how]

def combine_with_plan(step_agg, plan_score, mode, weight):
    if mode == "none" or plan_score is None:
        return step_agg
    if mode == "multiply":
        return step_agg * plan_score
    if mode == "weighted":
        return (1 - weight) * step_agg + weight * plan_score
    if mode == "plan_trust":
        trust = abs(plan_score - 0.5) * 2
        return trust * plan_score + (1 - trust) * step_agg
    raise ValueError(f"unknown plan_combine mode: {mode}")

def compute_auc(scores_correct, scores_wrong):
    """AUC = P(随机正确链分 > 随机错误链分); 并列按0.5计。与sklearn二分类等价。"""
    n_c, n_w = len(scores_correct), len(scores_wrong)
    if n_c == 0 or n_w == 0:
        return None
    sw_sorted = sorted(scores_wrong)
    total = 0.0
    for sc in scores_correct:
        lo = bisect.bisect_left(sw_sorted, sc)
        hi = bisect.bisect_right(sw_sorted, sc)
        total += lo + 0.5 * (hi - lo)
    return total / (n_c * n_w)

# ============================================================================
# 指标 (统一签名: f(correct_list, wrong_list) -> float | None)
# ============================================================================
def _mean(xs):
    return sum(xs) / len(xs) if xs else None

def _var(xs):
    if len(xs) < 2: return None
    m = _mean(xs)
    return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)   # ddof=1

def separation(correct, wrong):
    if not correct or not wrong: return None
    return _mean(correct) - _mean(wrong)

def cohens_d(correct, wrong):
    if len(correct) < 2 or len(wrong) < 2: return None
    vc, vw = _var(correct), _var(wrong)
    nc, nw = len(correct), len(wrong)
    sp2 = ((nc - 1) * vc + (nw - 1) * vw) / (nc + nw - 2)
    sp = math.sqrt(sp2)
    if sp < 1e-12: return None
    return (_mean(correct) - _mean(wrong)) / sp

def auc(correct, wrong):
    return compute_auc(correct, wrong)

METRICS = {"separation": separation, "cohen_d": cohens_d, "auc": auc}

# ============================================================================
# 链分构造 + 重采样
# ============================================================================
def chain_scores(by_prob, step_score, plan_score, how, plan_combine, plan_weight):
    """{prob_idx: [(chain_score, is_correct_bool), ...]}; 截断链(correct=None)剔除。"""
    out = {}
    for prob_idx, cand in by_prob.items():
        lst = []
        for tpos, t in enumerate(cand):
            c = t.get("correct")
            if c is None:                       # 截断: 既非正确也非错误, 不进任何一类
                continue
            ss = step_score.get((prob_idx, tpos), {})
            seq = [ss.get(i) for i in range(len(t["steps"]))]
            agg = aggregate(seq, how)
            psc = plan_score.get((prob_idx, tpos))
            final = combine_with_plan(agg, psc, plan_combine, plan_weight)
            lst.append((final, bool(c)))
        if lst:
            out[prob_idx] = lst
    return out

def split_scores(cs_by_prob, prob_list):
    """按给定 problem 列表(可含重复, 供 bootstrap)摊平成 correct/wrong 两组分数。"""
    correct, wrong = [], []
    for p in prob_list:
        for sc, isc in cs_by_prob[p]:
            (correct if isc else wrong).append(sc)
    return correct, wrong

def point(cs_by_prob, metric_fn):
    c, w = split_scores(cs_by_prob, list(cs_by_prob.keys()))
    return metric_fn(c, w), len(c), len(w)

def cluster_bootstrap_ci(cs_by_prob, metric_fn, n_boot, seed):
    rng = random.Random(seed)
    probs = list(cs_by_prob.keys()); n = len(probs)
    vals = []
    for _ in range(n_boot):
        sample = [probs[rng.randrange(n)] for _ in range(n)]
        v = metric_fn(*split_scores(cs_by_prob, sample))
        if v is not None:
            vals.append(v)
    if not vals: return (None, None)
    vals.sort()
    return (vals[int(0.025 * len(vals))], vals[min(len(vals) - 1, int(0.975 * len(vals)))])

def paired_delta(cs_ref, cs_other, metric_fn, n_boot, seed):
    """参照源 - 其它源 的指标差, 同一轮重采样同一批 problem (paired)。
    返回 (Δ点估计, CI_lo, CI_hi, 近似双侧p)。"""
    rng = random.Random(seed)
    probs = [p for p in cs_ref if p in cs_other]; n = len(probs)
    pa = metric_fn(*split_scores(cs_ref, probs))
    pb = metric_fn(*split_scores(cs_other, probs))
    if pa is None or pb is None:
        return (None, None, None, None)
    deltas = []
    for _ in range(n_boot):
        sample = [probs[rng.randrange(n)] for _ in range(n)]
        va = metric_fn(*split_scores(cs_ref, sample))
        vb = metric_fn(*split_scores(cs_other, sample))
        if va is not None and vb is not None:
            deltas.append(va - vb)
    if not deltas:
        return (pa - pb, None, None, None)
    deltas.sort()
    lo = deltas[int(0.025 * len(deltas))]
    hi = deltas[min(len(deltas) - 1, int(0.975 * len(deltas)))]
    frac_le0 = sum(1 for d in deltas if d <= 0) / len(deltas)
    p = 2 * min(frac_le0, 1 - frac_le0)
    return (pa - pb, lo, hi, p)

def merge_shards(glob_pattern):
    files = sorted(glob.glob(glob_pattern))
    if not files:
        raise SystemExit(f"未匹配到分片文件: {glob_pattern}")
    step_score = defaultdict(dict); plan_score = {}; schemes = set(); covered = set()
    for f in files:
        ss, pf, cov, psc, scheme = load_shard(f)
        for k, d in ss.items():
            step_score[k].update(d)
        plan_score.update(psc); schemes.add(scheme); covered.update(cov)
    if len(schemes - {None}) > 1:
        print(f"  ⚠ 警告: {glob_pattern} 混入多种 score_scheme={schemes}, 可能是不同"
              f"checkpoint的分片混在一起, 结果可能不可信")
    return step_score, plan_score, len(files), len(covered)

# ============================================================================
def fmt(v, nd=4):
    return f"{v:+.{nd}f}" if v is not None else "  n/a "

def fmt_ci(ci, nd=4):
    if ci is None or ci[0] is None: return "[   n/a   ]"
    return f"[{ci[0]:+.{nd}f}, {ci[1]:+.{nd}f}]"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", default="runs/eval/trajectories.jsonl",
                    help="评测轨迹(带 correct 标签, 三源共用同一份)")
    ap.add_argument("--scores", action="append", required=True,
                    help="NAME=GLOB, 可重复; 第一个作为 paired 检验的参照源")
    ap.add_argument("--how", default="mean,min,confident_mean",
                    help="链分聚合方式, 逗号分隔 (mean/min/last/hybrid/"
                         "confident_mean/adaptive_mean)")
    ap.add_argument("--plan_combine", default="none",
                    choices=["none", "multiply", "weighted", "plan_trust"],
                    help="跟外部PRM对照时务必保持 none (公平); 仅自家消融时才改")
    ap.add_argument("--plan_weight", type=float, default=0.3)
    ap.add_argument("--plan_alone", action="store_true",
                    help="额外报每个源 plan_score 单独(不掺step)的判别力")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/Separation/separation_report.json")
    args = ap.parse_args()

    hows = [h.strip() for h in args.how.split(",") if h.strip()]

    # ---- 标签: 轨迹 ----
    trajs = [json.loads(l) for l in open(args.traj)]
    by_prob = defaultdict(list)
    for t in trajs:
        by_prob[t["prob_idx"]].append(t)

    # ---- 各打分源 ----
    sources = []   # [(name, step_score, plan_score)]
    for spec in args.scores:
        if "=" not in spec:
            raise SystemExit(f"--scores 需 NAME=GLOB 格式: {spec}")
        name, pattern = spec.split("=", 1)
        ss, psc, nfiles, ncov = merge_shards(pattern)
        print(f"[源] {name:<12} <- {nfiles} 分片, 覆盖 {ncov} 题, "
              f"{'有' if psc else '无'} plan_score")
        sources.append((name, ss, psc))

    if args.plan_combine != "none":
        print(f"\n⚠ 注意: plan_combine={args.plan_combine} 不为 none。跟外部 PRM "
              f"对照时这会让链分掺入 plan_score, 外部 PRM 没有对应物 -> 不公平。\n")

    report = {"traj": args.traj, "plan_combine": args.plan_combine,
              "n_boot": args.n_boot, "by_how": {}}

    for how in hows:
        print("\n" + "=" * 92)
        print(f"  聚合方式: {how}    (plan_combine={args.plan_combine})")
        print("=" * 92)
        # 预算每个源的链分
        cs = {name: chain_scores(by_prob, ss, psc, how, args.plan_combine, args.plan_weight)
              for (name, ss, psc) in sources}

        # 点估计 + CI 表
        header = (f"  {'源':<12}{'n_corr':>7}{'n_wrong':>8}"
                  f"{'separation':>13}{'  95% CI':>22}"
                  f"{'cohen_d':>11}{'  95% CI':>22}"
                  f"{'AUC':>9}{'  95% CI':>22}")
        print(header)
        print("  " + "-" * 116)
        how_rec = {}
        for name in cs:
            row_rec = {}
            sep_pt, nc, nw = point(cs[name], separation)
            d_pt, _, _ = point(cs[name], cohens_d)
            auc_pt, _, _ = point(cs[name], auc)
            sep_ci = cluster_bootstrap_ci(cs[name], separation, args.n_boot, args.seed)
            d_ci = cluster_bootstrap_ci(cs[name], cohens_d, args.n_boot, args.seed + 1)
            auc_ci = cluster_bootstrap_ci(cs[name], auc, args.n_boot, args.seed + 2)
            print(f"  {name:<12}{nc:>7}{nw:>8}"
                  f"{fmt(sep_pt):>13}{fmt_ci(sep_ci):>22}"
                  f"{fmt(d_pt):>11}{fmt_ci(d_ci):>22}"
                  f"{('%.4f'%auc_pt if auc_pt is not None else 'n/a'):>9}{fmt_ci(auc_ci):>22}")
            row_rec = {"n_correct": nc, "n_wrong": nw,
                       "separation": sep_pt, "separation_ci": sep_ci,
                       "cohen_d": d_pt, "cohen_d_ci": d_ci,
                       "auc": auc_pt, "auc_ci": auc_ci}
            how_rec[name] = row_rec

        # 成对显著性: 参照源(第一个) vs 其它
        ref_name = sources[0][0]
        if len(sources) > 1:
            print(f"\n  ---- 成对差异 (参照 = {ref_name}; Δ = 参照 − 该源; "
                  f"CI不含0 即显著) ----")
            print(f"  {'对比':<26}{'ΔAUC':>9}{'  95% CI':>24}{'  p≈':>8}"
                  f"{'   Δd':>9}{'  95% CI':>24}")
            how_rec["_paired_vs_ref"] = {"ref": ref_name, "pairs": {}}
            for name in cs:
                if name == ref_name: continue
                da, la, ha, pa = paired_delta(cs[ref_name], cs[name], auc,
                                              args.n_boot, args.seed + 3)
                dd, ld, hd, pd = paired_delta(cs[ref_name], cs[name], cohens_d,
                                              args.n_boot, args.seed + 4)
                sig = "" if (la is None or (la <= 0 <= ha)) else "  *"
                print(f"  {ref_name+' − '+name:<26}{fmt(da):>9}"
                      f"{fmt_ci((la, ha)):>24}{(f'{pa:.3f}' if pa is not None else 'n/a'):>8}"
                      f"{fmt(dd):>9}{fmt_ci((ld, hd)):>24}{sig}")
                how_rec["_paired_vs_ref"]["pairs"][name] = {
                    "delta_auc": da, "delta_auc_ci": (la, ha), "delta_auc_p": pa,
                    "delta_cohen_d": dd, "delta_cohen_d_ci": (ld, hd), "delta_cohen_d_p": pd}

        report["by_how"][how] = how_rec

    # plan_score 单独的判别力 (跟聚合方式无关, 只跑一次)
    if args.plan_alone:
        print("\n" + "=" * 60)
        print("  plan_score 单独 (不掺 step_score) 的判别力")
        print("=" * 60)
        report["plan_alone"] = {}
        for (name, ss, psc) in sources:
            if not psc:
                continue
            cs_plan = {}
            for prob_idx, cand in by_prob.items():
                lst = []
                for tpos, t in enumerate(cand):
                    c = t.get("correct")
                    if c is None: continue
                    v = psc.get((prob_idx, tpos))
                    if v is None: continue
                    lst.append((v, bool(c)))
                if lst: cs_plan[prob_idx] = lst
            sep_pt, nc, nw = point(cs_plan, separation)
            auc_pt, _, _ = point(cs_plan, auc)
            auc_ci = cluster_bootstrap_ci(cs_plan, auc, args.n_boot, args.seed + 5)
            print(f"  {name:<12} sep={fmt(sep_pt)}  AUC={('%.4f'%auc_pt if auc_pt is not None else 'n/a')}"
                  f"  CI={fmt_ci(auc_ci)}  (n_corr={nc}, n_wrong={nw})")
            report["plan_alone"][name] = {"separation": sep_pt, "auc": auc_pt,
                                          "auc_ci": auc_ci, "n_correct": nc, "n_wrong": nw}

    json.dump(report, open(args.out, "w"), ensure_ascii=False, indent=2, default=list)
    print(f"\n  saved -> {args.out}")

if __name__ == "__main__":
    main()
