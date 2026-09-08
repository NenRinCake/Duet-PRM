# """
# eval_bon.py — Best-of-N 评测 (因式分解 PRM), 支持 v5 双分数格式

# == v5 新增: --dual_score 模式 ==
# 配合 assemble_dataset.py v5 训出的模型: 不再生成长推理文本, 直出
# <plan_score>/<step_score> 两个数。step_score 用于聚合(min/mean/last,
# 逐步骤), plan_score 是该候选(plan)整体的成功率估计, 在聚合后作为独立
# 因子参与链分组合 (--plan_combine), 训练和组合公式解耦, 可单独做消融:
#   --plan_combine none      : 忽略plan_score, 等价于只用step_score (baseline)
#   --plan_combine weighted  : final = (1-w)*step_agg + w*plan_score (默认, 稳健)
#   --plan_combine multiply  : final = step_agg * plan_score (更激进, 对plan_score
#                               噪声敏感 —— 一个运气差被打低分的plan会把整条
#                               候选的分数直接清零, 哪怰step全对)
# 不加 --dual_score 时, 完全是旧的单分数 <score> 行为, 不受影响。

# 四种 --mode (候选解 trajectories.jsonl 完全复用, 保证对比公平):
#   ours      你的生成式 PRM (单分数 <score> 或 --dual_score 双分数)  (需 --prm)
#   scalar    打分头 PRM (如 Qwen2.5-Math-PRM-7B)                      (需 --prm)
#   skywork   Skywork-o1-Open-PRM 专用接口                              (需 --prm)
#   majority  多数投票, 不需要任何 PRM                                   (无需 --prm)
#   merge     合并多个分片(--shard_id/--num_shards)的结果, 出最终报告

# == 多卡数据并行 ==
# 若模型的 num_key_value_heads 限制了 vLLM tensor_parallel 的最大 tp, 改用数据
# 并行: N 张卡各自独立加载一份模型 (--tp 1), 按 prob_idx % num_shards 切题,
# 互不通信。用法见 run_eval_parallel.sh。
# """
# import os
# os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# import argparse, json, re, glob
# from collections import defaultdict, Counter

# try:
#     from verifier import is_correct
# except ImportError:
#     def is_correct(a, g):
#         try: return abs(float(a) - float(g)) < 1e-6
#         except Exception: return str(a).strip() == str(g).strip()

# # ===================== 解析: 单分数 (v1-v4 旧格式) =====================
# SCORE_RE = re.compile(r"<score>\s*([0-9]*\.?[0-9]+)\s*</score>")
# SUMMARY_TAG_RE = re.compile(r"<summary>\s*([0-9]*\.?[0-9]+)\s*</summary>")
# SUMMARY_TXT_RE = re.compile(r"summary score is\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)
# SCORE_COLON_RE = re.compile(r"\bscore\s*[:=]\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)

# def _norm01(v):
#     if v is None: return None
#     if 0.0 <= v <= 1.0: return v
#     if 1.0 < v <= 5.0: return round((v - 1.0) / 4.0, 3)
#     return None

# def parse_score(text):
#     t = text or ""
#     m = SCORE_RE.search(t)
#     if m:
#         try: return _norm01(float(m.group(1)))
#         except ValueError: pass
#     m = SUMMARY_TAG_RE.search(t)
#     if m:
#         try: return _norm01(float(m.group(1)))
#         except ValueError: pass
#     m = SUMMARY_TXT_RE.search(t)
#     if m:
#         try: return _norm01(float(m.group(1)))
#         except ValueError: pass
#     ms = SCORE_COLON_RE.findall(t)
#     if ms:
#         try: return _norm01(float(ms[-1]))
#         except ValueError: pass
#     return None

# # ===================== 解析: 双分数 (v5新格式) =====================
# PLAN_SCORE_RE = re.compile(r"<plan_score>\s*([0-9]*\.?[0-9]+)\s*</plan_score>")
# STEP_SCORE_RE = re.compile(r"<step_score>\s*([0-9]*\.?[0-9]+)\s*</step_score>")

# def parse_dual_score(text):
#     """返回 (plan_score, step_score), 任一未匹配则为 None。"""
#     t = text or ""
#     plan_score = step_score = None
#     m = PLAN_SCORE_RE.search(t)
#     if m:
#         try: plan_score = _norm01(float(m.group(1)))
#         except ValueError: pass
#     m = STEP_SCORE_RE.search(t)
#     if m:
#         try: step_score = _norm01(float(m.group(1)))
#         except ValueError: pass
#     return plan_score, step_score

# # ===================== 输入渲染: 单分数旧格式 (与v1-v4 assemble_dataset.py一致) =====================
# INSTRUCTION = ("You are a process reward model. Assess the given reasoning step "
#                "along three dimensions in order: plan quality, adherence to the "
#                "plan, and execution correctness. Give the reasoning for each "
#                "dimension, then output the dimension tags and a summary score.")

# def build_input(problem, plan, steps_so_far, cur_step):
#     plan_str = "\n".join(f"  {p['id']} ({p.get('action','?')}): {p['desc']}" for p in plan)
#     hist = ""
#     for j, s in enumerate(steps_so_far):
#         hist += (f"\n[Step {j+1}] claims {s['claimed']}\ncode:\n{s['code']}\noutput:\n{s['exec']}\n")
#     cur = (f"\n[Current step] claims {cur_step['claimed']}\ncode:\n{cur_step['code']}\noutput:\n{cur_step['exec']}\n")
#     return (f"Problem:\n{problem}\n\nPlan:\n{plan_str}\n\n"
#             f"Execution history so far:{hist if hist else ' (none)'}\n"
#             f"{cur}\nProvide the process-reward judgment for the [Current step].")

# # ===================== 输入渲染: 双分数v5新格式 (与v5 assemble_dataset.py一致) =====================
# # INSTRUCTION_DUAL = (
# #     "You are given a PLAN (a multi-stage solution strategy) and an EXECUTION "
# #     "HISTORY ending in a CURRENT STEP. Output exactly two scores and nothing else:\n"
# #     "plan_score: how reliable is the PLAN itself (independent of how well any "
# #     "single execution carries it out)? Use exactly one of: 0.0 (this plan "
# #     "rarely leads to a correct answer, or frequently fails to even produce a "
# #     "complete result), 0.5 (mixed or inconsistent results across attempts), "
# #     "or 1.0 (this plan reliably leads to a correct answer).\n"
# #     "step_score: is the CURRENT STEP itself correct and trustworthy? Use "
# #     "exactly one of: 0.0 (no — wrong, deviates from the plan, or does no real "
# #     "work), 0.5 (cannot be independently determined), or 1.0 (yes — "
# #     "well-supported).\n"
# #     "Output ONLY the following two tags, in this exact order, with no other text:\n"
# #     "<plan_score>X.X</plan_score>\n"
# #     "<step_score>X.X</step_score>"
# # )


# INSTRUCTION_DUAL = (
#     "You are given a PLAN (a multi-stage solution strategy) and an EXECUTION "
#     "HISTORY ending in a CURRENT STEP. Output exactly two scores and nothing else:\n"
#     "plan_score: how reliable is the PLAN itself (independent of how well any "
#     "single execution carries it out)? Use exactly one of: 0.0 (this plan "
#     "rarely leads to a correct answer, or frequently fails to even produce a "
#     "complete result), 0.5 (mixed or inconsistent results across attempts), "
#     "or 1.0 (this plan reliably leads to a correct answer).\n"
#     "step_score: is the CURRENT STEP itself correct and trustworthy? Use "
#     "exactly one of: 0.0 (no — wrong, deviates from the plan, or does no real "
#     "work) or 1.0 (yes — well-supported).\n"
#     "Output ONLY the following two tags, in this exact order, with no other text:\n"
#     "<plan_score>X.X</plan_score>\n"
#     "<step_score>X.X</step_score>"
# )

# def build_input_dual(problem, plan, steps_so_far, cur_step):
#     plan_str = "\n".join(f"  {p['id']} ({p.get('action','?')}): {p['desc']}" for p in plan)
#     hist = ""
#     for j, s in enumerate(steps_so_far):
#         hist += (f"\n[Step {j+1}] claims {s['claimed']}\n"
#                  f"code:\n{s['code']}\noutput:\n{s['exec']}\n")
#     cur = (f"claims {cur_step['claimed']}\n"
#            f"code:\n{cur_step['code']}\noutput:\n{cur_step['exec']}\n")
#     return (f"Problem:\n{problem}\n\n"
#             f"=== PLAN ===\n{plan_str}\n\n"
#             f"=== EXECUTION HISTORY ===\n{hist if hist else '(none)'}\n"
#             f"=== CURRENT STEP ===\n{cur}\n"
#             f"Output plan_score and step_score for the above.")

# def aggregate(step_scores, how):
#     if not step_scores: return 0.0
#     xs = [s if s is not None else 0.5 for s in step_scores]
#     mn, mu, lst = min(xs), sum(xs)/len(xs), xs[-1]
#     if how == "confident_mean":
#         # 给每一步按"离0.5有多远"加权: 0.5(诚实弃权)权重=0, 0/1(confident判断)权重=1。
#         # 动机: 难题数据里诚实的0.5变多了, 普通mean会被这些"弃权票"拉向中间稀释掉
#         # confident判断的信号; 这个聚合方式让confident的判断主导链分, 不被弃权票拖累。
#         weights = [abs(x - 0.5) * 2 for x in xs]
#         wsum = sum(weights)
#         if wsum < 1e-9:          # 全部都是0.5(整条链没有一步给出confident判断)
#             return mu            # 退化成普通mean, 没有更好的依据
#         return sum(x * w for x, w in zip(xs, weights)) / wsum
#     return {"min": mn, "mean": mu, "last": lst, "hybrid": (mn + mu) / 2}[how]

# def combine_with_plan(step_agg, plan_score, mode, weight):
#     """把(逐步聚合后的)step_agg和该候选的plan_score后处理组合成最终链分。
#     mode='none' 时忽略 plan_score (向后兼容/baseline消融)。"""
#     if mode == "none" or plan_score is None:
#         return step_agg
#     if mode == "multiply":
#         return step_agg * plan_score
#     if mode == "weighted":
#         return (1 - weight) * step_agg + weight * plan_score
#     if mode == "plan_trust":
#         # plan_score预测=0.5时已验证是"犹豫偷懒"(60.4%其实是GOOD), 不是诚实信号
#         # (跟step_score的0.5性质不同) —— 这里按plan_score自己离0.5多远来决定信任度:
#         # plan_score越confident(离0/1越近)越听它的; 越犹豫越退回去只信step_agg自己的判断
#         trust = abs(plan_score - 0.5) * 2
#         return trust * plan_score + (1 - trust) * step_agg
#     raise ValueError(f"unknown plan_combine mode: {mode}")

# # ===================== 打分: ours (生成式, 单分数 或 双分数) =====================
# def score_ours(by_prob, problems, args):
#     from vllm import LLM, SamplingParams
#     from transformers import AutoTokenizer
#     tok = AutoTokenizer.from_pretrained(args.prm, trust_remote_code=True)
#     llm = LLM(model=args.prm, tensor_parallel_size=args.tp,
#               max_model_len=args.max_model_len, gpu_memory_utilization=0.90,
#               trust_remote_code=True)

#     instruction = INSTRUCTION_DUAL if args.dual_score else INSTRUCTION
#     builder = build_input_dual if args.dual_score else build_input

#     def chat(inp):
#         return tok.apply_chat_template(
#             [{"role": "user", "content": instruction + "\n\n" + inp}],
#             tokenize=False, add_generation_prompt=True)

#     prompts, index = [], []
#     skipped_long = []
#     budget = args.max_model_len - args.max_gen - 16
#     for prob_idx, cand in by_prob.items():
#         ptext = problems[prob_idx]["problem"] if problems else cand[0]["plan"][0]["desc"]
#         for tpos, t in enumerate(cand):
#             for i, s in enumerate(t["steps"]):
#                 p = chat(builder(ptext, t["plan"], t["steps"][:i], s))
#                 ntok = len(tok(p)["input_ids"])
#                 if ntok > budget:
#                     skipped_long.append((prob_idx, tpos, i))
#                     continue
#                 prompts.append(p)
#                 index.append((prob_idx, tpos, i))
#     sp = SamplingParams(temperature=0.0, max_tokens=args.max_gen,
#                         repetition_penalty=args.repetition_penalty)
#     tag = f"[shard {args.shard_id}] " if args.num_shards > 1 else ""
#     print(f"{tag}[ours] scoring {len(prompts)} steps "
#           f"(skipped {len(skipped_long)} over-long > {budget} tok, "
#           f"dual_score={args.dual_score})...")
#     outs = llm.generate(prompts, sp)

#     step_score = defaultdict(dict)
#     plan_score_raw = defaultdict(list)   # (prob_idx,tpos) -> [模型对该候选每步给出的plan_score估计]
#     parse_fail = 0; parse_fail_plan = 0
#     fail_samples = []
#     for (p, tpos, i), o in zip(index, outs):
#         txt = o.outputs[0].text
#         if args.dual_score:
#             plan_sc, step_sc = parse_dual_score(txt)
#             if plan_sc is not None:
#                 plan_score_raw[(p, tpos)].append(plan_sc)
#             else:
#                 parse_fail_plan += 1
#             sc = step_sc
#         else:
#             sc = parse_score(txt)
#         if sc is None:
#             parse_fail += 1
#             if len(fail_samples) < 20:
#                 fail_samples.append(txt)
#         step_score[(p, tpos)][i] = sc
#     if fail_samples:
#         suffix = f".shard{args.shard_id}" if args.num_shards > 1 else ""
#         with open(f"runs/eval/parse_fail_samples{suffix}.txt", "w") as _f:
#             for k, t in enumerate(fail_samples):
#                 _f.write(f"===== parse-fail #{k+1} =====\n{t}\n\n")
#     for (p, tpos, i) in skipped_long:
#         step_score[(p, tpos)][i] = None

#     # 每个候选的 plan_score: 取该候选所有步的估计均值 (单次估计噪声大, 平均更稳)
#     plan_score_by_traj = {}
#     if args.dual_score:
#         for k, vals in plan_score_raw.items():
#             plan_score_by_traj[k] = sum(vals) / len(vals)
#         if parse_fail_plan:
#             print(f"{tag}[ours] plan_score 解析失败: {parse_fail_plan} 步 "
#                   f"(不计入主parse_fail, 该候选plan_score回退用0.5中性值)")
#     return step_score, parse_fail, plan_score_by_traj

# # ===================== 打分: scalar (打分头 PRM) =====================
# def score_scalar(by_prob, problems, args):
#     import torch
#     from transformers import AutoModel, AutoTokenizer
#     tok = AutoTokenizer.from_pretrained(args.prm, trust_remote_code=True)
#     model = AutoModel.from_pretrained(args.prm, trust_remote_code=True,
#                                       torch_dtype="auto", device_map="auto").eval()
#     SEP = args.step_sep
#     sep_id = tok.encode(SEP)[-1] if SEP else None

#     def _scalar_step_scores(problem, plan, steps):
#         sys = "Please reason step by step, and put your final answer within \\boxed{}."
#         chunks = [f"[{s['claimed']}] {s['code']}\n{s['exec']}" for s in steps]
#         convo = [
#             {"role": "system", "content": sys},
#             {"role": "user", "content": problem},
#             {"role": "assistant", "content": (SEP).join(chunks) + SEP},
#         ]
#         ids = tok.apply_chat_template(convo, tokenize=True, return_tensors="pt").to(model.device)
#         with torch.no_grad():
#             logits = model(ids).logits
#         probs = torch.softmax(logits, dim=-1)[0]
#         pos = (ids[0] == sep_id).nonzero(as_tuple=True)[0]
#         return [probs[p, 1].item() if probs.shape[-1] >= 2 else probs[p].max().item()
#                 for p in pos]

#     step_score = defaultdict(dict); parse_fail = 0
#     total = sum(len(v) for v in by_prob.values()); done = 0
#     tag = f"[shard {args.shard_id}] " if args.num_shards > 1 else ""
#     for prob_idx, cand in by_prob.items():
#         ptext = problems[prob_idx]["problem"] if problems else cand[0]["plan"][0]["desc"]
#         for tpos, t in enumerate(cand):
#             try:
#                 scs = _scalar_step_scores(ptext, t["plan"], t["steps"])
#             except Exception:
#                 scs = []; parse_fail += 1
#             for i in range(len(t["steps"])):
#                 step_score[(prob_idx, tpos)][i] = scs[i] if i < len(scs) else None
#             done += 1
#         if done % 50 == 0: print(f"{tag}[scalar] {done}/{total} candidates scored")
#     return step_score, parse_fail, {}

# # ===================== 打分: skywork =====================
# def score_skywork(by_prob, problems, args):
#     import torch
#     from transformers import AutoTokenizer
#     from model_utils.prm_model import PRM_MODEL
#     from model_utils.io_utils import (prepare_input, prepare_batch_input_for_model,
#                                       derive_step_rewards)
#     tok = AutoTokenizer.from_pretrained(args.prm, trust_remote_code=True)
#     model = PRM_MODEL.from_pretrained(args.prm, device_map="auto",
#                                       torch_dtype=torch.bfloat16).eval()

#     def _sky_step_scores(problem, steps):
#         response = "\n\n".join(f"[{s['claimed']}] {s['code']}\n{s['exec']}"
#                                for s in steps)
#         ids, mask, reward_idxs = prepare_input(problem, response,
#                                                tokenizer=tok, step_token="\n\n")
#         ids = torch.tensor([ids]).to(model.device)
#         with torch.no_grad():
#             _, _, rewards = model(input_ids=ids, attention_mask=torch.ones_like(ids),
#                                   return_probs=True)
#         return derive_step_rewards(rewards, [reward_idxs])[0]

#     step_score = defaultdict(dict); parse_fail = 0
#     total = sum(len(v) for v in by_prob.values()); done = 0
#     tag = f"[shard {args.shard_id}] " if args.num_shards > 1 else ""
#     for prob_idx, cand in by_prob.items():
#         ptext = problems[prob_idx]["problem"] if problems else cand[0]["plan"][0]["desc"]
#         for tpos, t in enumerate(cand):
#             try:
#                 scs = _sky_step_scores(ptext, t["steps"])
#             except Exception:
#                 scs = []; parse_fail += 1
#             for i in range(len(t["steps"])):
#                 step_score[(prob_idx, tpos)][i] = scs[i] if i < len(scs) else None
#             done += 1
#         if done % 50 == 0: print(f"{tag}[skywork] {done}/{total} candidates scored")
#     return step_score, parse_fail, {}

# # ===================== BoN 指标 =====================
# ALL_HOWS = ["min", "mean", "last", "hybrid", "confident_mean"]
# ALL_COMBINES = ["none", "weighted", "multiply", "plan_trust"]   # none=baseline, weighted=GPT方案A, multiply=GPT方案B, plan_trust=新增
# WEIGHT_SWEEP = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]   # v6实测mean行在0.2->0.5单调递增没见顶, 往上扩

# def bon_from_scores(by_prob, step_score, plan_score_by_traj=None,
#                     plan_combine="none", plan_weight=0.3, hows=None):
#     plan_score_by_traj = plan_score_by_traj or {}
#     HOWS = hows or ALL_HOWS
#     res = {h: 0 for h in HOWS}; per_problem = []
#     pass1_sum = 0.0; oracle_hit = 0; n_prob = len(by_prob)
#     diag_correct, diag_wrong = [], []
#     for prob_idx, cand in by_prob.items():
#         N = len(cand); corrects = [bool(t.get("correct")) for t in cand]
#         nc = sum(corrects); pass1_sum += nc / N
#         if nc > 0: oracle_hit += 1
#         rec = {"prob_idx": prob_idx, "N": N, "n_correct": nc}
#         for how in HOWS:
#             scored = []
#             for tpos, t in enumerate(cand):
#                 ss = step_score.get((prob_idx, tpos), {})
#                 seq = [ss.get(i) for i in range(len(t["steps"]))]
#                 agg = aggregate(seq, how)
#                 psc = plan_score_by_traj.get((prob_idx, tpos))
#                 final = combine_with_plan(agg, psc, plan_combine, plan_weight)
#                 scored.append((final, tpos))
#             best = max(scored, key=lambda x: x[0])[1]
#             res[how] += int(corrects[best]); rec[f"bon_{how}_hit"] = int(corrects[best])
#         for tpos, t in enumerate(cand):
#             ss = step_score.get((prob_idx, tpos), {})
#             seq = [ss.get(i) for i in range(len(t["steps"]))]
#             agg = aggregate(seq, "mean")
#             psc = plan_score_by_traj.get((prob_idx, tpos))
#             cs = combine_with_plan(agg, psc, plan_combine, plan_weight)
#             (diag_correct if corrects[tpos] else diag_wrong).append(cs)
#         per_problem.append(rec)
#     def _stats(xs):
#         if not xs: return (0, 0.0, 0.0)
#         xs2 = sorted(xs); n = len(xs2)
#         return (n, sum(xs2)/n, xs2[n//2])
#     nc_, mc, medc = _stats(diag_correct)
#     nw_, mw, medw = _stats(diag_wrong)
#     print("\n  ---- 诊断: PRM 链分能否区分对错 (mean聚合"
#           f"{f' + plan_combine={plan_combine}' if plan_combine!='none' else ''}) ----")
#     print(f"    正确链: n={nc_}  平均={mc:.3f}  中位={medc:.3f}")
#     print(f"    错误链: n={nw_}  平均={mw:.3f}  中位={medw:.3f}")
#     print(f"    分离度 (正确均分 - 错误均分): {mc-mw:+.3f}  "
#           f"{'<-- 几乎不分, PRM没学会区分' if abs(mc-mw)<0.03 else '<-- 有区分' if mc-mw>0.03 else '<-- 反了!错链分更高'}")
#     diag = {"n_correct_chains": nc_, "mean_correct": round(mc, 4), "median_correct": round(medc, 4),
#             "n_wrong_chains": nw_, "mean_wrong": round(mw, 4), "median_wrong": round(medw, 4),
#             "separation": round(mc - mw, 4)}
#     return res, pass1_sum/n_prob, oracle_hit/n_prob, per_problem, HOWS, diag

# def compute_bon_matrix(by_prob, step_score, plan_score_by_traj=None, weight_sweep=None):
#     """4(聚合)×(1 none + len(weight_sweep)个weighted权重 + 1 multiply) 矩阵。
#     之前的版本把weighted的权重锁死在单个值上, 这里改成扫一组权重网格, 因为
#     weighted本身的效果对权重选多少很敏感 (我们已经验证过: 同一个例子里
#     weight=0.5排序是错的, weight=0.3排序是对的) —— 不该只测一个随手取的点。
#     复用同一份 aggregate() 结果, 纯算术, 不需要重新跑模型。"""
#     plan_score_by_traj = plan_score_by_traj or {}
#     weight_sweep = weight_sweep or WEIGHT_SWEEP
#     n_prob = len(by_prob)
#     pass1_sum = 0.0; oracle_hit = 0
#     step_agg_cache = {how: {} for how in ALL_HOWS}
#     correctness = {}
#     for prob_idx, cand in by_prob.items():
#         corrects = [bool(t.get("correct")) for t in cand]
#         correctness[prob_idx] = corrects
#         nc = sum(corrects); N = len(cand)
#         pass1_sum += nc / N
#         if nc > 0: oracle_hit += 1
#         for tpos, t in enumerate(cand):
#             ss = step_score.get((prob_idx, tpos), {})
#             seq = [ss.get(i) for i in range(len(t["steps"]))]
#             for how in ALL_HOWS:
#                 step_agg_cache[how][(prob_idx, tpos)] = aggregate(seq, how)

#     columns = ["none"] + [f"weighted(w={w})" for w in weight_sweep] + ["multiply", "plan_trust"]
#     matrix = {how: {} for how in ALL_HOWS}
#     for how in ALL_HOWS:
#         for col in columns:
#             if col == "none":
#                 mode, w = "none", 0.0
#             elif col == "multiply":
#                 mode, w = "multiply", 0.0
#             elif col == "plan_trust":
#                 mode, w = "plan_trust", 0.0   # plan_trust不需要weight, 由plan_score自身置信度决定
#             else:
#                 mode, w = "weighted", float(col.split("w=")[1].rstrip(")"))
#             hit = 0
#             for prob_idx, cand in by_prob.items():
#                 scored = []
#                 for tpos in range(len(cand)):
#                     key = (prob_idx, tpos)
#                     agg = step_agg_cache[how][key]
#                     psc = plan_score_by_traj.get(key)
#                     final = combine_with_plan(agg, psc, mode, w)
#                     scored.append((final, tpos))
#                 best = max(scored, key=lambda x: x[0])[1]
#                 hit += int(correctness[prob_idx][best])
#             matrix[how][col] = hit / n_prob
#     return matrix, pass1_sum/n_prob, oracle_hit/n_prob, columns

# def print_matrix_report(matrix, pass1, oracle, columns):
#     colw = max(13, max(len(c) for c in columns) + 2)
#     width = 12 + colw*len(columns)
#     print("\n" + "=" * width)
#     print(f"  BoN 矩阵: 4种step聚合 × {len(columns)}种plan组合 (weighted权重已扫一组网格, 不锁单点)")
#     print(f"  pass@1={pass1*100:.1f}%  oracle={oracle*100:.1f}%")
#     print("=" * width)
#     col_header = "聚合\\组合"
#     header = f"  {col_header:<10}" + "".join(f"{c:>{colw}}" for c in columns)
#     print(header)
#     print("  " + "-" * (10 + colw*len(columns)))
#     for how in ALL_HOWS:
#         row = f"  {how:<10}"
#         for col in columns:
#             row += f"{matrix[how][col]*100:>{colw-1}.1f}%"
#         print(row)
#     print("=" * width)
#     print("  none=纯step_score(baseline) | weighted(w=...)=GPT方案A(扫权重) | multiply=GPT方案B")
#     best_how, best_col, best_val = None, None, -1
#     for how in ALL_HOWS:
#         for col in columns:
#             if matrix[how][col] > best_val:
#                 best_val = matrix[how][col]; best_how, best_col = how, col
#     print(f"  最佳组合: {best_how} × {best_col} = {best_val*100:.1f}%")

# # ===================== majority voting =====================
# def bon_majority(by_prob):
#     hit = 0; per_problem = []; pass1_sum = 0.0; oracle_hit = 0; n_prob = len(by_prob)
#     for prob_idx, cand in by_prob.items():
#         N = len(cand); corrects = [bool(t.get("correct")) for t in cand]
#         nc = sum(corrects); pass1_sum += nc / N
#         if nc > 0: oracle_hit += 1
#         votes = Counter(t["answer"] for t in cand if t.get("answer") is not None)
#         if votes:
#             win_ans = votes.most_common(1)[0][0]
#             pick = next(t for t in cand if t.get("answer") == win_ans)
#             h = bool(pick.get("correct"))
#         else:
#             h = False
#         hit += int(h)
#         per_problem.append({"prob_idx": prob_idx, "N": N, "n_correct": nc,
#                             "vote_winner_hit": int(h)})
#     return hit, pass1_sum/n_prob, oracle_hit/n_prob, per_problem

# # ===================== 分片结果的保存/加载 (数据并行用) =====================
# def save_shard(path, step_score, parse_fail, covered_probs, plan_score_by_traj=None):
#     data = {
#         "parse_fail": parse_fail,
#         "covered_probs": sorted(covered_probs),
#         "step_score": {f"{p}:{t}": {str(i): v for i, v in d.items()}
#                        for (p, t), d in step_score.items()},
#         "plan_score": {f"{p}:{t}": v for (p, t), v in (plan_score_by_traj or {}).items()},
#     }
#     json.dump(data, open(path, "w"))
#     print(f"  分片已保存 -> {path} (覆盖 {len(covered_probs)} 题, parse_fail={parse_fail})")

# def load_shard(path):
#     data = json.load(open(path))
#     ss = defaultdict(dict)
#     for k, d in data["step_score"].items():
#         p_str, t_str = k.split(":")
#         p, t = int(p_str), int(t_str)
#         for i_str, v in d.items():
#             ss[(p, t)][int(i_str)] = v
#     plan_score = {}
#     for k, v in data.get("plan_score", {}).items():
#         p_str, t_str = k.split(":")
#         plan_score[(int(p_str), int(t_str))] = v
#     return ss, data["parse_fail"], data["covered_probs"], plan_score

# def build_summary(mode_label, by_prob, step_score, parse_fail,
#                   plan_score_by_traj=None, plan_combine="none", plan_weight=0.3):
#     res, pass1, oracle, per_problem, HOWS, diag = bon_from_scores(
#         by_prob, step_score, plan_score_by_traj, plan_combine, plan_weight)
#     n = len(by_prob)
#     n_steps = sum(len(t["steps"]) for v in by_prob.values() for t in v)
#     summary = {"mode": mode_label, "n_problems": n,
#                "N_per_problem": len(next(iter(by_prob.values())))}
#     summary["parse_fail"] = parse_fail
#     summary["parse_fail_rate"] = round(parse_fail / max(1, n_steps), 4)
#     summary["pass@1 (random lower bound)"] = round(pass1, 4)
#     summary["oracle / pass@N (upper bound)"] = round(oracle, 4)
#     summary["BoN"] = {h: round(res[h]/n, 4) for h in HOWS}
#     gap = oracle - pass1
#     summary["BoN_gain_over_random"] = {h: round(res[h]/n - pass1, 4) for h in HOWS}
#     summary["BoN_fraction_of_oracle_gap"] = {
#         h: (round((res[h]/n - pass1)/gap, 4) if gap > 1e-9 else None) for h in HOWS}
#     summary["diagnostic_correct_vs_wrong_chain_score"] = diag
#     summary["plan_combine"] = plan_combine
#     summary["plan_weight"] = plan_weight if plan_combine == "weighted" else None
#     return summary, per_problem

# def print_report(summary):
#     print("\n" + "=" * 60)
#     print(f"mode: {summary['mode']} | problems: {summary['n_problems']} | N: {summary['N_per_problem']}")
#     if summary.get("plan_combine", "none") != "none":
#         print(f"plan_combine: {summary['plan_combine']}" +
#               (f" (weight={summary['plan_weight']})" if summary.get("plan_weight") else ""))
#     if "parse_fail_rate" in summary:
#         warn = " ⚠ 解析失败偏高" if summary["parse_fail_rate"] > 0.1 else ""
#         print(f"parse fail rate: {summary['parse_fail_rate']*100:.1f}%{warn}")
#     print(f"\n  pass@1 (随机选下限): {summary['pass@1 (random lower bound)']*100:.1f}%")
#     print(f"  oracle (完美选上限): {summary['oracle / pass@N (upper bound)']*100:.1f}%")
#     print(f"  --- BoN 准确率 ---")
#     for k, v in summary["BoN"].items():
#         frac = summary["BoN_fraction_of_oracle_gap"][k]
#         fs = f"{frac*100:.0f}% of oracle gap" if frac is not None else "n/a"
#         print(f"    {k:<14}: {v*100:.1f}%  (+{summary['BoN_gain_over_random'][k]*100:.1f} vs random, {fs})")

# # ===================== main =====================
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--mode", required=True,
#                     choices=["ours", "scalar", "skywork", "majority", "merge"])
#     ap.add_argument("--prm", default=None)
#     ap.add_argument("--traj", default="runs/eval/trajectories.jsonl")
#     ap.add_argument("--problems", default=None)
#     ap.add_argument("--out", default="runs/eval/bon_report.json")
#     ap.add_argument("--max_model_len", type=int, default=8192)
#     ap.add_argument("--tp", type=int, default=1)
#     ap.add_argument("--max_gen", type=int, default=400)
#     ap.add_argument("--step_sep", default="<extra_0>")
#     ap.add_argument("--repetition_penalty", type=float, default=1.15)
#     # ---- v5 双分数 ----
#     ap.add_argument("--dual_score", action="store_true",
#                     help="配合v5模型: 解析<plan_score>/<step_score>双tag, 而非旧版<score>")
#     ap.add_argument("--plan_combine", choices=["none", "multiply", "weighted", "plan_trust"], default="weighted",
#                     help="如何把plan_score组合进最终链分 (仅--dual_score时生效)")
#     ap.add_argument("--plan_weight", type=float, default=0.3,
#                     help="weighted模式下plan_score的权重 (0-1)")
#     # ---- 数据并行分片 ----
#     ap.add_argument("--num_shards", type=int, default=1)
#     ap.add_argument("--shard_id", type=int, default=0)
#     ap.add_argument("--shard_glob", default=None)
#     args = ap.parse_args()
#     if args.mode in ("ours", "scalar", "skywork") and not args.prm:
#         ap.error(f"--mode {args.mode} 需要 --prm")
#     if args.mode == "merge" and not args.shard_glob:
#         ap.error("--mode merge 需要 --shard_glob")

#     trajs = [json.loads(l) for l in open(args.traj)]
#     problems = [json.loads(l) for l in open(args.problems)] if args.problems else None
#     by_prob_full = defaultdict(list)
#     for t in trajs: by_prob_full[t["prob_idx"]].append(t)

#     # ---------------- merge ----------------
#     if args.mode == "merge":
#         files = sorted(glob.glob(args.shard_glob))
#         if not files:
#             raise SystemExit(f"未匹配到任何分片文件: {args.shard_glob}")
#         step_score = defaultdict(dict); parse_fail_total = 0; covered = set()
#         plan_score_by_traj = {}
#         for f in files:
#             ss, pf, cov, psc = load_shard(f)
#             for k, d in ss.items():
#                 step_score[k].update(d)
#             plan_score_by_traj.update(psc)
#             parse_fail_total += pf
#             covered.update(cov)
#         missing = set(by_prob_full.keys()) - covered
#         if missing:
#             print(f"  ⚠ 警告: {len(missing)} 题未被任何分片覆盖: "
#                   f"{sorted(missing)[:10]}{'...' if len(missing)>10 else ''}")
#         print(f"  已合并 {len(files)} 个分片, 覆盖 {len(covered)} 题, 总 parse_fail={parse_fail_total}")
#         summary, per_problem = build_summary("merged", by_prob_full, step_score, parse_fail_total,
#                                              plan_score_by_traj, args.plan_combine, args.plan_weight)
#         out_payload = {"summary": summary, "per_problem": per_problem}
#         if plan_score_by_traj:
#             matrix, mpass1, moracle, mcols = compute_bon_matrix(by_prob_full, step_score, plan_score_by_traj)
#             out_payload["bon_matrix"] = {how: matrix[how] for how in ALL_HOWS}
#             out_payload["bon_matrix"]["_columns"] = mcols
#         json.dump(out_payload, open(args.out, "w"), ensure_ascii=False, indent=2)
#         print_report(summary)
#         if plan_score_by_traj:
#             print_matrix_report(matrix, mpass1, moracle, mcols)
#         print(f"\n  saved -> {args.out}")
#         return

#     # ---------------- majority ----------------
#     if args.mode == "majority":
#         hit, pass1, oracle, per_problem = bon_majority(by_prob_full)
#         n = len(by_prob_full)
#         summary = {"mode": "majority", "n_problems": n,
#                    "N_per_problem": len(next(iter(by_prob_full.values())))}
#         summary.update({
#             "pass@1 (random lower bound)": round(pass1, 4),
#             "oracle / pass@N (upper bound)": round(oracle, 4),
#             "BoN": {"majority_vote": round(hit/n, 4)},
#             "BoN_gain_over_random": {"majority_vote": round(hit/n - pass1, 4)},
#         })
#         gap = oracle - pass1
#         summary["BoN_fraction_of_oracle_gap"] = {
#             "majority_vote": round((hit/n - pass1)/gap, 4) if gap > 1e-9 else None}
#         json.dump({"summary": summary, "per_problem": per_problem},
#                   open(args.out, "w"), ensure_ascii=False, indent=2)
#         print("\n" + "=" * 60)
#         print(f"mode: majority | problems: {n} | N: {summary['N_per_problem']}")
#         print(f"\n  pass@1 (随机选下限): {pass1*100:.1f}%")
#         print(f"  oracle (完美选上限): {oracle*100:.1f}%")
#         v = summary["BoN"]["majority_vote"]
#         print(f"    majority_vote : {v*100:.1f}%")
#         print(f"\n  saved -> {args.out}")
#         return

#     # ---------------- ours/scalar/skywork: 按需分片 ----------------
#     if args.num_shards > 1:
#         by_prob = {k: v for k, v in by_prob_full.items() if k % args.num_shards == args.shard_id}
#         if not by_prob:
#             raise SystemExit(f"分片 {args.shard_id}/{args.num_shards} 没分到任何题")
#     else:
#         by_prob = by_prob_full

#     if args.mode == "ours":
#         step_score, parse_fail, plan_score_by_traj = score_ours(by_prob, problems, args)
#     elif args.mode == "skywork":
#         step_score, parse_fail, plan_score_by_traj = score_skywork(by_prob, problems, args)
#     else:
#         step_score, parse_fail, plan_score_by_traj = score_scalar(by_prob, problems, args)

#     if args.num_shards > 1:
#         shard_path = f"{args.out}.shard{args.shard_id}.json"
#         save_shard(shard_path, step_score, parse_fail, list(by_prob.keys()), plan_score_by_traj)
#         return

#     summary, per_problem = build_summary(args.mode, by_prob, step_score, parse_fail,
#                                          plan_score_by_traj, args.plan_combine, args.plan_weight)
#     out_payload = {"summary": summary, "per_problem": per_problem}
#     if plan_score_by_traj:
#         matrix, mpass1, moracle, mcols = compute_bon_matrix(by_prob, step_score, plan_score_by_traj)
#         out_payload["bon_matrix"] = {how: matrix[how] for how in ALL_HOWS}
#         out_payload["bon_matrix"]["_columns"] = mcols
#     json.dump(out_payload, open(args.out, "w"), ensure_ascii=False, indent=2)
#     print_report(summary)
#     if plan_score_by_traj:
#         print_matrix_report(matrix, mpass1, moracle, mcols)
#     print(f"\n  saved -> {args.out}")

# if __name__ == "__main__":
#     main()


"""
eval_bon.py — Best-of-N 评测 (因式分解 PRM), 支持 v5 双分数格式

== v5 新增: --dual_score 模式 ==
配合 assemble_dataset.py v5 训出的模型: 不再生成长推理文本, 直出
<plan_score>/<step_score> 两个数。step_score 用于聚合(min/mean/last,
逐步骤), plan_score 是该候选(plan)整体的成功率估计, 在聚合后作为独立
因子参与链分组合 (--plan_combine), 训练和组合公式解耦, 可单独做消融:
  --plan_combine none      : 忽略plan_score, 等价于只用step_score (baseline)
  --plan_combine weighted  : final = (1-w)*step_agg + w*plan_score (默认, 稳健)
  --plan_combine multiply  : final = step_agg * plan_score (更激进, 对plan_score
                              噪声敏感 —— 一个运气差被打低分的plan会把整条
                              候选的分数直接清零, 哪怰step全对)
不加 --dual_score 时, 完全是旧的单分数 <score> 行为, 不受影响。

四种 --mode (候选解 trajectories.jsonl 完全复用, 保证对比公平):
  ours      你的生成式 PRM (单分数 <score> 或 --dual_score 双分数)  (需 --prm)
  scalar    打分头 PRM (如 Qwen2.5-Math-PRM-7B)                      (需 --prm)
  skywork   Skywork-o1-Open-PRM 专用接口                              (需 --prm)
  majority  多数投票, 不需要任何 PRM                                   (无需 --prm)
  merge     合并多个分片(--shard_id/--num_shards)的结果, 出最终报告

== 多卡数据并行 ==
若模型的 num_key_value_heads 限制了 vLLM tensor_parallel 的最大 tp, 改用数据
并行: N 张卡各自独立加载一份模型 (--tp 1), 按 prob_idx % num_shards 切题,
互不通信。用法见 run_eval_parallel.sh。
"""
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import argparse, json, re, glob
from collections import defaultdict, Counter

try:
    from verifier import is_correct
except ImportError:
    def is_correct(a, g):
        try: return abs(float(a) - float(g)) < 1e-6
        except Exception: return str(a).strip() == str(g).strip()

# ===================== 解析: 单分数 (v1-v4 旧格式) =====================
SCORE_RE = re.compile(r"<score>\s*([0-9]*\.?[0-9]+)\s*</score>")
SUMMARY_TAG_RE = re.compile(r"<summary>\s*([0-9]*\.?[0-9]+)\s*</summary>")
SUMMARY_TXT_RE = re.compile(r"summary score is\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)
SCORE_COLON_RE = re.compile(r"\bscore\s*[:=]\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)

def _norm01(v):
    if v is None: return None
    if 0.0 <= v <= 1.0: return v
    if 1.0 < v <= 5.0: return round((v - 1.0) / 4.0, 3)
    return None

def parse_score(text):
    t = text or ""
    m = SCORE_RE.search(t)
    if m:
        try: return _norm01(float(m.group(1)))
        except ValueError: pass
    m = SUMMARY_TAG_RE.search(t)
    if m:
        try: return _norm01(float(m.group(1)))
        except ValueError: pass
    m = SUMMARY_TXT_RE.search(t)
    if m:
        try: return _norm01(float(m.group(1)))
        except ValueError: pass
    ms = SCORE_COLON_RE.findall(t)
    if ms:
        try: return _norm01(float(ms[-1]))
        except ValueError: pass
    return None

# ===================== 解析: 双分数 (v5新格式) =====================
PLAN_SCORE_RE = re.compile(r"<plan_score>\s*([0-9]*\.?[0-9]+)\s*</plan_score>")
STEP_SCORE_RE = re.compile(r"<step_score>\s*([0-9]*\.?[0-9]+)\s*</step_score>")

def parse_dual_score(text):
    """返回 (plan_score, step_score), 任一未匹配则为 None。"""
    t = text or ""
    plan_score = step_score = None
    m = PLAN_SCORE_RE.search(t)
    if m:
        try: plan_score = _norm01(float(m.group(1)))
        except ValueError: pass
    m = STEP_SCORE_RE.search(t)
    if m:
        try: step_score = _norm01(float(m.group(1)))
        except ValueError: pass
    return plan_score, step_score

# ===================== 输入渲染: 单分数旧格式 (与v1-v4 assemble_dataset.py一致) =====================
INSTRUCTION = ("You are a process reward model. Assess the given reasoning step "
               "along three dimensions in order: plan quality, adherence to the "
               "plan, and execution correctness. Give the reasoning for each "
               "dimension, then output the dimension tags and a summary score.")

def build_input(problem, plan, steps_so_far, cur_step):
    plan_str = "\n".join(f"  {p['id']} ({p.get('action','?')}): {p['desc']}" for p in plan)
    hist = ""
    for j, s in enumerate(steps_so_far):
        hist += (f"\n[Step {j+1}] claims {s['claimed']}\ncode:\n{s['code']}\noutput:\n{s['exec']}\n")
    cur = (f"\n[Current step] claims {cur_step['claimed']}\ncode:\n{cur_step['code']}\noutput:\n{cur_step['exec']}\n")
    return (f"Problem:\n{problem}\n\nPlan:\n{plan_str}\n\n"
            f"Execution history so far:{hist if hist else ' (none)'}\n"
            f"{cur}\nProvide the process-reward judgment for the [Current step].")

# ===================== 输入渲染: 双分数v5新格式 (与v5 assemble_dataset.py一致) =====================
PREAMBLE_DUAL = (
    "You are given a PLAN (a multi-stage solution strategy) and an EXECUTION "
    "HISTORY ending in a CURRENT STEP. Output exactly two scores and nothing else:\n"
)
PLAN_INSTR_3CLASS = (
    "plan_score: how reliable is the PLAN itself (independent of how well any "
    "single execution carries it out)? Use exactly one of: 0.0 (this plan "
    "rarely leads to a correct answer, or frequently fails to even produce a "
    "complete result), 0.5 (mixed or inconsistent results across attempts), "
    "or 1.0 (this plan reliably leads to a correct answer)."
)
PLAN_INSTR_BINARY = (
    "plan_score: how reliable is the PLAN itself (independent of how well any "
    "single execution carries it out)? Use exactly one of: 0.0 (this plan "
    "rarely leads to a correct answer, or frequently fails to even produce a "
    "complete result) or 1.0 (this plan reliably leads to a correct answer)."
)
STEP_INSTR_3CLASS = (
    "step_score: is the CURRENT STEP itself correct and trustworthy? Use "
    "exactly one of: 0.0 (no — wrong, deviates from the plan, or does no real "
    "work), 0.5 (cannot be independently determined), or 1.0 (yes — "
    "well-supported)."
)
STEP_INSTR_BINARY = (
    "step_score: is the CURRENT STEP itself correct and trustworthy? Use "
    "exactly one of: 0.0 (no — wrong, deviates from the plan, or does no real "
    "work) or 1.0 (yes — well-supported)."
)
SUFFIX_DUAL = (
    "\nOutput ONLY the following two tags, in this exact order, with no other text:\n"
    "<plan_score>X.X</plan_score>\n"
    "<step_score>X.X</step_score>"
)

def _build_dual_instr(plan_part, step_part):
    return PREAMBLE_DUAL + plan_part + "\n" + step_part + SUFFIX_DUAL

# 三个版本分别对应训练时实际用过的三种标签方案, 必须跟--prm指向的checkpoint
# 严格一一对应, 错配会重演之前"忘了同步INSTRUCTION_DUAL"那次的bug(训练/评测
# 指令不一致, 不报错但分数不准, 非常隐蔽) —— 用--score_scheme显式选择,
# 不再靠手动改这个文件里的文字来切换, 避免再次靠记忆对齐出错。
INSTRUCTION_DUAL_VARIANTS = {
    # v6: plan_score/step_score都还是三分类(最早的版本, 50题+三分类基线)
    "v6": _build_dual_instr(PLAN_INSTR_3CLASS, STEP_INSTR_3CLASS),
    # step01: 只把step_score改成二分类, plan_score保留三分类 (追平v6, 72.7%那一版)
    "step01": _build_dual_instr(PLAN_INSTR_3CLASS, STEP_INSTR_BINARY),
    # full01: plan_score和step_score都改成二分类 (最新版本)
    "full01": _build_dual_instr(PLAN_INSTR_BINARY, STEP_INSTR_BINARY),
}

def build_input_dual(problem, plan, steps_so_far, cur_step):
    plan_str = "\n".join(f"  {p['id']} ({p.get('action','?')}): {p['desc']}" for p in plan)
    hist = ""
    for j, s in enumerate(steps_so_far):
        hist += (f"\n[Step {j+1}] claims {s['claimed']}\n"
                 f"code:\n{s['code']}\noutput:\n{s['exec']}\n")
    cur = (f"claims {cur_step['claimed']}\n"
           f"code:\n{cur_step['code']}\noutput:\n{cur_step['exec']}\n")
    return (f"Problem:\n{problem}\n\n"
            f"=== PLAN ===\n{plan_str}\n\n"
            f"=== EXECUTION HISTORY ===\n{hist if hist else '(none)'}\n"
            f"=== CURRENT STEP ===\n{cur}\n"
            f"Output plan_score and step_score for the above.")

def aggregate(step_scores, how):
    if not step_scores: return 0.0
    xs = [s if s is not None else 0.5 for s in step_scores]
    mn, mu, lst = min(xs), sum(xs)/len(xs), xs[-1]
    if how == "confident_mean":
        # 给每一步按"离0.5有多远"加权: 0.5(诚实弃权)权重=0, 0/1(confident判断)权重=1。
        # 动机: 难题数据里诚实的0.5变多了, 普通mean会被这些"弃权票"拉向中间稀释掉
        # confident判断的信号; 这个聚合方式让confident的判断主导链分, 不被弃权票拖累。
        weights = [abs(x - 0.5) * 2 for x in xs]
        wsum = sum(weights)
        if wsum < 1e-9:          # 全部都是0.5(整条链没有一步给出confident判断)
            return mu            # 退化成普通mean, 没有更好的依据
        return sum(x * w for x, w in zip(xs, weights)) / wsum
    if how == "adaptive_mean":
        # confident_mean在confident判断很稀少时(比如AIME24这种比round2更难的数据)
        # 有个没被早期测试覆盖到的风险: 全部权重压在仅剩的1-2个confident点上, 这1-2个
        # 点一旦是噪声, 整条链的分就被这一两个点完全主宰, 没有其他信息去稀释/纠正它 ——
        # 这正是AIME24上confident_mean变成最差的可能机制。这里按"这条链里confident
        # 判断占的比例"在confident_mean和mean之间自动插值: 占比高时趋近confident_mean
        # (沿用它在MATH上验证过的优势), 占比低时趋近mean(避免被极少数噪声点完全主宰)。
        weights = [abs(x - 0.5) * 2 for x in xs]
        wsum = sum(weights)
        conf_frac = sum(1 for w in weights if w > 1e-9) / len(weights)
        cm = mu if wsum < 1e-9 else sum(x * w for x, w in zip(xs, weights)) / wsum
        return conf_frac * cm + (1 - conf_frac) * mu
    return {"min": mn, "mean": mu, "last": lst, "hybrid": (mn + mu) / 2}[how]

def combine_with_plan(step_agg, plan_score, mode, weight):
    """把(逐步聚合后的)step_agg和该候选的plan_score后处理组合成最终链分。
    mode='none' 时忽略 plan_score (向后兼容/baseline消融)。"""
    if mode == "none" or plan_score is None:
        return step_agg
    if mode == "multiply":
        return step_agg * plan_score
    if mode == "weighted":
        return (1 - weight) * step_agg + weight * plan_score
    if mode == "plan_trust":
        # plan_score预测=0.5时已验证是"犹豫偷懒"(60.4%其实是GOOD), 不是诚实信号
        # (跟step_score的0.5性质不同) —— 这里按plan_score自己离0.5多远来决定信任度:
        # plan_score越confident(离0/1越近)越听它的; 越犹豫越退回去只信step_agg自己的判断
        trust = abs(plan_score - 0.5) * 2
        return trust * plan_score + (1 - trust) * step_agg
    raise ValueError(f"unknown plan_combine mode: {mode}")

# ===================== 打分: ours (生成式, 单分数 或 双分数) =====================
def score_ours(by_prob, problems, args):
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.prm, trust_remote_code=True)
    llm = LLM(model=args.prm, tensor_parallel_size=args.tp,
              max_model_len=args.max_model_len, gpu_memory_utilization=0.90,
              trust_remote_code=True)

    instruction = INSTRUCTION_DUAL_VARIANTS[args.score_scheme] if args.dual_score else INSTRUCTION
    builder = build_input_dual if args.dual_score else build_input

    def chat(inp):
        return tok.apply_chat_template(
            [{"role": "user", "content": instruction + "\n\n" + inp}],
            tokenize=False, add_generation_prompt=True)

    prompts, index = [], []
    skipped_long = []
    budget = args.max_model_len - args.max_gen - 16
    for prob_idx, cand in by_prob.items():
        ptext = problems[prob_idx]["problem"] if problems else cand[0]["plan"][0]["desc"]
        for tpos, t in enumerate(cand):
            for i, s in enumerate(t["steps"]):
                p = chat(builder(ptext, t["plan"], t["steps"][:i], s))
                ntok = len(tok(p)["input_ids"])
                if ntok > budget:
                    skipped_long.append((prob_idx, tpos, i))
                    continue
                prompts.append(p)
                index.append((prob_idx, tpos, i))
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_gen,
                        repetition_penalty=args.repetition_penalty)
    tag = f"[shard {args.shard_id}] " if args.num_shards > 1 else ""
    print(f"{tag}[ours] scoring {len(prompts)} steps "
          f"(skipped {len(skipped_long)} over-long > {budget} tok, "
          f"dual_score={args.dual_score})...")
    outs = llm.generate(prompts, sp)

    step_score = defaultdict(dict)
    plan_score_raw = defaultdict(list)   # (prob_idx,tpos) -> [模型对该候选每步给出的plan_score估计]
    parse_fail = 0; parse_fail_plan = 0
    fail_samples = []
    for (p, tpos, i), o in zip(index, outs):
        txt = o.outputs[0].text
        if args.dual_score:
            plan_sc, step_sc = parse_dual_score(txt)
            if plan_sc is not None:
                plan_score_raw[(p, tpos)].append(plan_sc)
            else:
                parse_fail_plan += 1
            sc = step_sc
        else:
            sc = parse_score(txt)
        if sc is None:
            parse_fail += 1
            if len(fail_samples) < 20:
                fail_samples.append(txt)
        step_score[(p, tpos)][i] = sc
    if fail_samples:
        suffix = f".shard{args.shard_id}" if args.num_shards > 1 else ""
        with open(f"runs/eval/parse_fail_samples{suffix}.txt", "w") as _f:
            for k, t in enumerate(fail_samples):
                _f.write(f"===== parse-fail #{k+1} =====\n{t}\n\n")
    for (p, tpos, i) in skipped_long:
        step_score[(p, tpos)][i] = None

    # 每个候选的 plan_score: 取该候选所有步的估计均值 (单次估计噪声大, 平均更稳)
    # 同时额外存一份"只看第一步"的估计(plan_score_raw里第0个元素) —— index按
    # i从0开始升序构建, vLLM按输入顺序返回输出, 所以列表第一个元素必然对应第一步,
    # 不会跟其他步骤错位。这部分计算本来就在算均值的过程里发生过, 只是算完就被
    # 扔掉了, 现在顺手存下来, 几乎不增加额外开销, 用于诊断"早期(只看第一步)的
    # plan_score判断力, 跟全程平均后的判断力比, 掉了多少" —— 这是验证"能不能在
    # 执行完之前就提前用plan_score剪枝"这件事的前提, 不是给现有BoN流程用的。
    plan_score_by_traj = {}
    plan_score_step0_by_traj = {}
    if args.dual_score:
        for k, vals in plan_score_raw.items():
            plan_score_by_traj[k] = sum(vals) / len(vals)
            plan_score_step0_by_traj[k] = vals[0]
        if parse_fail_plan:
            print(f"{tag}[ours] plan_score 解析失败: {parse_fail_plan} 步 "
                  f"(不计入主parse_fail, 该候选plan_score回退用0.5中性值)")
    return step_score, parse_fail, plan_score_by_traj, plan_score_step0_by_traj

# ===================== 打分: scalar (打分头 PRM) =====================
def score_scalar(by_prob, problems, args):
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.prm, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.prm, trust_remote_code=True,
                                      torch_dtype="auto", device_map="auto").eval()
    SEP = args.step_sep
    sep_id = tok.encode(SEP)[-1] if SEP else None

    # def _scalar_step_scores(problem, plan, steps):
    #     sys = "Please reason step by step, and put your final answer within \\boxed{}."
    #     chunks = [f"[{s['claimed']}] {s['code']}\n{s['exec']}" for s in steps]
    #     convo = [
    #         {"role": "system", "content": sys},
    #         {"role": "user", "content": problem},
    #         {"role": "assistant", "content": (SEP).join(chunks) + SEP},
    #     ]
    #     ids = tok.apply_chat_template(convo, tokenize=True, return_tensors="pt").to(model.device)
    #     with torch.no_grad():
    #         logits = model(ids).logits
    #     probs = torch.softmax(logits, dim=-1)[0]
    #     pos = (ids[0] == sep_id).nonzero(as_tuple=True)[0]
    #     return [probs[p, 1].item() if probs.shape[-1] >= 2 else probs[p].max().item() for p in pos]


    def _scalar_step_scores(problem, plan, steps):
        sys = "Please reason step by step, and put your final answer within \\boxed{}."
        chunks = [f"[{s['claimed']}] {s['code']}\n{s['exec']}" for s in steps]
        # [实验] 把 plan 作为上下文拼到题目后面 (开头一次, 不在每步重复)
        plan_txt = ""
        if plan:
            plan_lines = "\n".join(
                f"{p.get('id','?')} ({p.get('action','?')}): {p.get('desc','')}" for p in plan)
            plan_txt = "\n\nPlan:\n" + plan_lines
        convo = [
            {"role": "system", "content": sys},
            {"role": "user", "content": problem + plan_txt},   # plan 拼在题目后
            {"role": "assistant", "content": (SEP).join(chunks) + SEP},
        ]
        ids = tok.apply_chat_template(convo, tokenize=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            logits = model(ids).logits
        probs = torch.softmax(logits, dim=-1)[0]
        pos = (ids[0] == sep_id).nonzero(as_tuple=True)[0]
        return [probs[p, 1].item() if probs.shape[-1] >= 2 else probs[p].max().item()
                for p in pos]


    step_score = defaultdict(dict); parse_fail = 0
    total = sum(len(v) for v in by_prob.values()); done = 0
    tag = f"[shard {args.shard_id}] " if args.num_shards > 1 else ""
    for prob_idx, cand in by_prob.items():
        ptext = problems[prob_idx]["problem"] if problems else cand[0]["plan"][0]["desc"]
        for tpos, t in enumerate(cand):
            try:
                scs = _scalar_step_scores(ptext, t["plan"], t["steps"])
            except Exception:
                scs = []; parse_fail += 1
            for i in range(len(t["steps"])):
                step_score[(prob_idx, tpos)][i] = scs[i] if i < len(scs) else None
            done += 1
        if done % 50 == 0: print(f"{tag}[scalar] {done}/{total} candidates scored")
    return step_score, parse_fail, {}, {}

# ===================== 打分: skywork =====================
def score_skywork(by_prob, problems, args):
    import torch
    from transformers import AutoTokenizer
    from model_utils.prm_model import PRM_MODEL
    from model_utils.io_utils import (prepare_input, prepare_batch_input_for_model,
                                      derive_step_rewards)
    tok = AutoTokenizer.from_pretrained(args.prm, trust_remote_code=True)
    model = PRM_MODEL.from_pretrained(args.prm, device_map="auto",
                                      torch_dtype=torch.bfloat16).eval()

    def _sky_step_scores(problem, steps):
        response = "\n\n".join(f"[{s['claimed']}] {s['code']}\n{s['exec']}"
                               for s in steps)
        ids, mask, reward_idxs = prepare_input(problem, response,
                                               tokenizer=tok, step_token="\n\n")
        # ids = torch.tensor([ids]).to(model.device)
        device = next(model.parameters()).device
        ids = torch.tensor([ids]).to(device)
        with torch.no_grad():
            _, _, rewards = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                                  return_probs=True)
        rf_tensor = torch.tensor(reward_idxs).unsqueeze(0).to(rewards.device)
        return derive_step_rewards(rewards, rf_tensor)[0]

    # def _sky_step_scores(problem, plan, steps):
    #     # 拼plan, 用于对照实验, 和 score_scalar 的处理方式保持一致
    #     plan_txt = ""
    #     if plan:
    #         plan_lines = "\n".join(
    #             f"{p.get('id','?')} ({p.get('action','?')}): {p.get('desc','')}" for p in plan)
    #         plan_txt = "\n\nPlan:\n" + plan_lines
    #     response = "\n\n".join(f"[{s['claimed']}] {s['code']}\n{s['exec']}"
    #                            for s in steps)
    #     ids, mask, reward_idxs = prepare_input(problem + plan_txt, response,
    #                                            tokenizer=tok, step_token="\n\n")
    #     device = next(model.parameters()).device
    #     ids = torch.tensor([ids]).to(device)
    #     with torch.no_grad():
    #         _, _, rewards = model(input_ids=ids, attention_mask=torch.ones_like(ids),
    #                               return_probs=True)
    #     rf_tensor = torch.tensor(reward_idxs).unsqueeze(0).to(rewards.device)
    #     return derive_step_rewards(rewards, rf_tensor)[0]

    step_score = defaultdict(dict); parse_fail = 0
    total = sum(len(v) for v in by_prob.values()); done = 0
    tag = f"[shard {args.shard_id}] " if args.num_shards > 1 else ""
    for prob_idx, cand in by_prob.items():
        ptext = problems[prob_idx]["problem"] if problems else cand[0]["plan"][0]["desc"]
        for tpos, t in enumerate(cand):
            try:
                scs = _sky_step_scores(ptext, t["steps"])
            except Exception:
                scs = []; parse_fail += 1
            for i in range(len(t["steps"])):
                step_score[(prob_idx, tpos)][i] = scs[i] if i < len(scs) else None
            done += 1
        if done % 50 == 0: print(f"{tag}[skywork] {done}/{total} candidates scored")
    return step_score, parse_fail, {}, {}

# ===================== BoN 指标 =====================
ALL_HOWS = ["min", "mean", "last", "hybrid", "confident_mean", "adaptive_mean"]
ALL_COMBINES = ["none", "weighted", "multiply", "plan_trust"]   # none=baseline, weighted=GPT方案A, multiply=GPT方案B, plan_trust=新增
WEIGHT_SWEEP = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]   # v6实测mean行在0.2->0.5单调递增没见顶, 往上扩

def bon_from_scores(by_prob, step_score, plan_score_by_traj=None,
                    plan_combine="none", plan_weight=0.3, hows=None):
    plan_score_by_traj = plan_score_by_traj or {}
    HOWS = hows or ALL_HOWS
    res = {h: 0 for h in HOWS}; per_problem = []
    pass1_sum = 0.0; oracle_hit = 0; n_prob = len(by_prob)
    diag_correct, diag_wrong = [], []
    for prob_idx, cand in by_prob.items():
        N = len(cand); corrects = [bool(t.get("correct")) for t in cand]
        nc = sum(corrects); pass1_sum += nc / N
        if nc > 0: oracle_hit += 1
        rec = {"prob_idx": prob_idx, "N": N, "n_correct": nc}
        for how in HOWS:
            scored = []
            for tpos, t in enumerate(cand):
                ss = step_score.get((prob_idx, tpos), {})
                seq = [ss.get(i) for i in range(len(t["steps"]))]
                agg = aggregate(seq, how)
                psc = plan_score_by_traj.get((prob_idx, tpos))
                final = combine_with_plan(agg, psc, plan_combine, plan_weight)
                scored.append((final, tpos))
            best = max(scored, key=lambda x: x[0])[1]
            res[how] += int(corrects[best]); rec[f"bon_{how}_hit"] = int(corrects[best])
        for tpos, t in enumerate(cand):
            ss = step_score.get((prob_idx, tpos), {})
            seq = [ss.get(i) for i in range(len(t["steps"]))]
            agg = aggregate(seq, "mean")
            psc = plan_score_by_traj.get((prob_idx, tpos))
            cs = combine_with_plan(agg, psc, plan_combine, plan_weight)
            (diag_correct if corrects[tpos] else diag_wrong).append(cs)
        per_problem.append(rec)
    def _stats(xs):
        if not xs: return (0, 0.0, 0.0)
        xs2 = sorted(xs); n = len(xs2)
        return (n, sum(xs2)/n, xs2[n//2])
    nc_, mc, medc = _stats(diag_correct)
    nw_, mw, medw = _stats(diag_wrong)
    print("\n  ---- 诊断: PRM 链分能否区分对错 (mean聚合"
          f"{f' + plan_combine={plan_combine}' if plan_combine!='none' else ''}) ----")
    print(f"    正确链: n={nc_}  平均={mc:.3f}  中位={medc:.3f}")
    print(f"    错误链: n={nw_}  平均={mw:.3f}  中位={medw:.3f}")
    print(f"    分离度 (正确均分 - 错误均分): {mc-mw:+.3f}  "
          f"{'<-- 几乎不分, PRM没学会区分' if abs(mc-mw)<0.03 else '<-- 有区分' if mc-mw>0.03 else '<-- 反了!错链分更高'}")
    diag = {"n_correct_chains": nc_, "mean_correct": round(mc, 4), "median_correct": round(medc, 4),
            "n_wrong_chains": nw_, "mean_wrong": round(mw, 4), "median_wrong": round(medw, 4),
            "separation": round(mc - mw, 4)}
    return res, pass1_sum/n_prob, oracle_hit/n_prob, per_problem, HOWS, diag

def compute_bon_matrix(by_prob, step_score, plan_score_by_traj=None, weight_sweep=None):
    """4(聚合)×(1 none + len(weight_sweep)个weighted权重 + 1 multiply) 矩阵。
    之前的版本把weighted的权重锁死在单个值上, 这里改成扫一组权重网格, 因为
    weighted本身的效果对权重选多少很敏感 (我们已经验证过: 同一个例子里
    weight=0.5排序是错的, weight=0.3排序是对的) —— 不该只测一个随手取的点。
    复用同一份 aggregate() 结果, 纯算术, 不需要重新跑模型。"""
    plan_score_by_traj = plan_score_by_traj or {}
    weight_sweep = weight_sweep or WEIGHT_SWEEP
    n_prob = len(by_prob)
    pass1_sum = 0.0; oracle_hit = 0
    step_agg_cache = {how: {} for how in ALL_HOWS}
    correctness = {}
    for prob_idx, cand in by_prob.items():
        corrects = [bool(t.get("correct")) for t in cand]
        correctness[prob_idx] = corrects
        nc = sum(corrects); N = len(cand)
        pass1_sum += nc / N
        if nc > 0: oracle_hit += 1
        for tpos, t in enumerate(cand):
            ss = step_score.get((prob_idx, tpos), {})
            seq = [ss.get(i) for i in range(len(t["steps"]))]
            for how in ALL_HOWS:
                step_agg_cache[how][(prob_idx, tpos)] = aggregate(seq, how)

    columns = ["none"] + [f"weighted(w={w})" for w in weight_sweep] + ["multiply", "plan_trust"]
    matrix = {how: {} for how in ALL_HOWS}
    for how in ALL_HOWS:
        for col in columns:
            if col == "none":
                mode, w = "none", 0.0
            elif col == "multiply":
                mode, w = "multiply", 0.0
            elif col == "plan_trust":
                mode, w = "plan_trust", 0.0   # plan_trust不需要weight, 由plan_score自身置信度决定
            else:
                mode, w = "weighted", float(col.split("w=")[1].rstrip(")"))
            hit = 0
            for prob_idx, cand in by_prob.items():
                scored = []
                for tpos in range(len(cand)):
                    key = (prob_idx, tpos)
                    agg = step_agg_cache[how][key]
                    psc = plan_score_by_traj.get(key)
                    final = combine_with_plan(agg, psc, mode, w)
                    scored.append((final, tpos))
                best = max(scored, key=lambda x: x[0])[1]
                hit += int(correctness[prob_idx][best])
            matrix[how][col] = hit / n_prob
    return matrix, pass1_sum/n_prob, oracle_hit/n_prob, columns

def compute_auc(scores_correct, scores_wrong):
    """AUC = P(随机一条正确链分数 > 随机一条错误链分数), 用排序+二分实现,
    跟sklearn.metrics.roc_auc_score在二分类下数学等价(并列按0.5计)。
    这是个跟BoN准确率完全不同的问题: BoN准确率混杂了'打分准不准'和
    '挑选/排序/投票这套策略本身合不合理'两件事; AUC只看'分数本身有没有把
    正确链排在错误链前面', 跟下游具体怎么用这个分数(投票/排序/剪枝)完全
    解耦 —— majority voting这个外部基线没有'分数'这个概念, 没法直接套用
    AUC, 所以AUC不是用来跟majority直接比大小的, 是用来单独回答'PRM自己
    打分这件事做得怎么样'这个问题, 跟majority赢不赢是两件不绑定的事。"""
    import bisect
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

def compute_auc_matrix(by_prob, step_score, plan_score_by_traj=None, weight_sweep=None):
    """跟compute_bon_matrix同一套(how x combine)网格, 每个格子算AUC而不是
    BoN准确率, 方便直接对照: 同一个格子, 准确率高的是不是AUC也高。"""
    plan_score_by_traj = plan_score_by_traj or {}
    weight_sweep = weight_sweep or WEIGHT_SWEEP
    step_agg_cache = {how: {} for how in ALL_HOWS}
    correctness = {}
    for prob_idx, cand in by_prob.items():
        correctness[prob_idx] = [bool(t.get("correct")) for t in cand]
        for tpos, t in enumerate(cand):
            ss = step_score.get((prob_idx, tpos), {})
            seq = [ss.get(i) for i in range(len(t["steps"]))]
            for how in ALL_HOWS:
                step_agg_cache[how][(prob_idx, tpos)] = aggregate(seq, how)

    columns = ["none"] + [f"weighted(w={w})" for w in weight_sweep] + ["multiply", "plan_trust"]
    matrix = {how: {} for how in ALL_HOWS}
    for how in ALL_HOWS:
        for col in columns:
            if col == "none": mode, w = "none", 0.0
            elif col == "multiply": mode, w = "multiply", 0.0
            elif col == "plan_trust": mode, w = "plan_trust", 0.0
            else: mode, w = "weighted", float(col.split("w=")[1].rstrip(")"))
            sc_correct, sc_wrong = [], []
            for prob_idx, cand in by_prob.items():
                for tpos in range(len(cand)):
                    key = (prob_idx, tpos)
                    agg = step_agg_cache[how][key]
                    psc = plan_score_by_traj.get(key)
                    final = combine_with_plan(agg, psc, mode, w)
                    (sc_correct if correctness[prob_idx][tpos] else sc_wrong).append(final)
            matrix[how][col] = compute_auc(sc_correct, sc_wrong)
    return matrix, columns

def compute_plan_score_alone_auc(by_prob, plan_score_by_traj):
    """单独看plan_score自己(完全不掺step_score)的判别力 —— 矩阵里的'none'列
    是step_score自己不掺plan_score, 这里反过来, 是plan_score自己不掺step_score,
    两者对照能看出两个维度各自的判别力分别有多少。"""
    sc_correct, sc_wrong = [], []
    for prob_idx, cand in by_prob.items():
        for tpos, t in enumerate(cand):
            psc = plan_score_by_traj.get((prob_idx, tpos))
            if psc is None:
                continue
            (sc_correct if bool(t.get("correct")) else sc_wrong).append(psc)
    return compute_auc(sc_correct, sc_wrong), len(sc_correct), len(sc_wrong)

def print_auc_matrix_report(matrix, columns):
    colw = max(13, max(len(c) for c in columns) + 2)
    width = 12 + colw*len(columns)
    print("\n" + "=" * width)
    print(f"  AUC 矩阵: 同一套(how x combine)网格, 衡量'打分本身的判别力',")
    print(f"  跟上面的BoN准确率矩阵回答的是不同问题 (0.5=完全没有判别力, 1.0=完美区分,")
    print(f"  不受'怎么用分数去选/投票'这层策略影响, 不能直接跟majority的%数比大小)")
    print("=" * width)
    col_header = "聚合\\组合"
    header = f"  {col_header:<10}" + "".join(f"{c:>{colw}}" for c in columns)
    print(header)
    print("  " + "-" * (10 + colw*len(columns)))
    for how in ALL_HOWS:
        row = f"  {how:<10}"
        for col in columns:
            v = matrix[how][col]
            row += f"{v:>{colw-1}.4f}" if v is not None else f"{'n/a':>{colw-1}}"
        print(row)
    print("=" * width)
    best_how, best_col, best_val = None, None, -1
    for how in ALL_HOWS:
        for col in columns:
            v = matrix[how][col]
            if v is not None and v > best_val:
                best_val = v; best_how, best_col = how, col
    print(f"  最佳AUC组合: {best_how} × {best_col} = {best_val:.4f}")

def print_matrix_report(matrix, pass1, oracle, columns):
    colw = max(13, max(len(c) for c in columns) + 2)
    width = 12 + colw*len(columns)
    print("\n" + "=" * width)
    print(f"  BoN 矩阵: 4种step聚合 × {len(columns)}种plan组合 (weighted权重已扫一组网格, 不锁单点)")
    print(f"  pass@1={pass1*100:.1f}%  oracle={oracle*100:.1f}%")
    print("=" * width)
    col_header = "聚合\\组合"
    header = f"  {col_header:<10}" + "".join(f"{c:>{colw}}" for c in columns)
    print(header)
    print("  " + "-" * (10 + colw*len(columns)))
    for how in ALL_HOWS:
        row = f"  {how:<10}"
        for col in columns:
            row += f"{matrix[how][col]*100:>{colw-1}.1f}%"
        print(row)
    print("=" * width)
    print("  none=纯step_score(baseline) | weighted(w=...)=GPT方案A(扫权重) | multiply=GPT方案B")
    best_how, best_col, best_val = None, None, -1
    for how in ALL_HOWS:
        for col in columns:
            if matrix[how][col] > best_val:
                best_val = matrix[how][col]; best_how, best_col = how, col
    print(f"  最佳组合: {best_how} × {best_col} = {best_val*100:.1f}%")

# ===================== majority voting =====================
def bon_weighted_vote(by_prob, step_score, plan_score_by_traj, weight_sweep=None):
    """PRM加权投票: 按最终答案分组(跟bon_majority同一套分组逻辑, 保证跟那个
    majority基线可比), 组内对所有候选的PRM最终分数求和, 选总分最高的答案组 ——
    不是选票数最多的组。扫一遍跟矩阵同样的how×combine网格, 报告其中最好的组合。"""
    plan_score_by_traj = plan_score_by_traj or {}
    weight_sweep = weight_sweep or WEIGHT_SWEEP
    n_prob = len(by_prob)
    step_agg_cache = {how: {} for how in ALL_HOWS}
    for prob_idx, cand in by_prob.items():
        for tpos, t in enumerate(cand):
            ss = step_score.get((prob_idx, tpos), {})
            seq = [ss.get(i) for i in range(len(t["steps"]))]
            for how in ALL_HOWS:
                step_agg_cache[how][(prob_idx, tpos)] = aggregate(seq, how)

    combos = [("none", 0.0)] + [("weighted", w) for w in weight_sweep] \
            + [("multiply", 0.0), ("plan_trust", 0.0)]
    best = (-1.0, None, None)
    for how in ALL_HOWS:
        for mode, w in combos:
            hit = 0
            for prob_idx, cand in by_prob.items():
                groups = defaultdict(float)
                rep = {}   # answer -> 任意一条该答案候选(取correct标记用)
                for tpos, t in enumerate(cand):
                    ans = t.get("answer")
                    if ans is None:
                        continue
                    agg = step_agg_cache[how][(prob_idx, tpos)]
                    psc = plan_score_by_traj.get((prob_idx, tpos))
                    final = combine_with_plan(agg, psc, mode, w)
                    groups[ans] += final
                    rep.setdefault(ans, t)
                if not groups:
                    continue
                win_ans = max(groups, key=groups.get)
                hit += int(bool(rep[win_ans].get("correct")))
            acc = hit / n_prob
            if acc > best[0]:
                best = (acc, how, f"{mode}(w={w})" if mode == "weighted" else mode)
    return best  # (accuracy, best_how, best_combo_desc)

def bon_majority(by_prob):
    hit = 0; per_problem = []; pass1_sum = 0.0; oracle_hit = 0; n_prob = len(by_prob)
    for prob_idx, cand in by_prob.items():
        N = len(cand); corrects = [bool(t.get("correct")) for t in cand]
        nc = sum(corrects); pass1_sum += nc / N
        if nc > 0: oracle_hit += 1
        votes = Counter(t["answer"] for t in cand if t.get("answer") is not None)
        if votes:
            win_ans = votes.most_common(1)[0][0]
            pick = next(t for t in cand if t.get("answer") == win_ans)
            h = bool(pick.get("correct"))
        else:
            h = False
        hit += int(h)
        per_problem.append({"prob_idx": prob_idx, "N": N, "n_correct": nc,
                            "vote_winner_hit": int(h)})
    return hit, pass1_sum/n_prob, oracle_hit/n_prob, per_problem

# ===================== 分片结果的保存/加载 (数据并行用) =====================
def save_shard(path, step_score, parse_fail, covered_probs, plan_score_by_traj=None,
              plan_score_step0_by_traj=None, score_scheme=None):
    data = {
        "parse_fail": parse_fail,
        "covered_probs": sorted(covered_probs),
        "step_score": {f"{p}:{t}": {str(i): v for i, v in d.items()}
                       for (p, t), d in step_score.items()},
        "plan_score": {f"{p}:{t}": v for (p, t), v in (plan_score_by_traj or {}).items()},
        "plan_score_step0": {f"{p}:{t}": v for (p, t), v in (plan_score_step0_by_traj or {}).items()},
        "score_scheme": score_scheme,
    }
    json.dump(data, open(path, "w"))
    print(f"  分片已保存 -> {path} (覆盖 {len(covered_probs)} 题, parse_fail={parse_fail})")

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
    plan_score_step0 = {}
    for k, v in data.get("plan_score_step0", {}).items():
        p_str, t_str = k.split(":")
        plan_score_step0[(int(p_str), int(t_str))] = v
    return ss, data["parse_fail"], data["covered_probs"], plan_score, plan_score_step0, data.get("score_scheme")

def build_summary(mode_label, by_prob, step_score, parse_fail,
                  plan_score_by_traj=None, plan_combine="none", plan_weight=0.3):
    res, pass1, oracle, per_problem, HOWS, diag = bon_from_scores(
        by_prob, step_score, plan_score_by_traj, plan_combine, plan_weight)
    n = len(by_prob)
    n_steps = sum(len(t["steps"]) for v in by_prob.values() for t in v)
    summary = {"mode": mode_label, "n_problems": n,
               "N_per_problem": len(next(iter(by_prob.values())))}
    summary["parse_fail"] = parse_fail
    summary["parse_fail_rate"] = round(parse_fail / max(1, n_steps), 4)
    summary["pass@1 (random lower bound)"] = round(pass1, 4)
    summary["oracle / pass@N (upper bound)"] = round(oracle, 4)
    summary["BoN"] = {h: round(res[h]/n, 4) for h in HOWS}
    gap = oracle - pass1
    summary["BoN_gain_over_random"] = {h: round(res[h]/n - pass1, 4) for h in HOWS}
    summary["BoN_fraction_of_oracle_gap"] = {
        h: (round((res[h]/n - pass1)/gap, 4) if gap > 1e-9 else None) for h in HOWS}
    summary["diagnostic_correct_vs_wrong_chain_score"] = diag
    summary["plan_combine"] = plan_combine
    summary["plan_weight"] = plan_weight if plan_combine == "weighted" else None
    return summary, per_problem

def print_report(summary):
    print("\n" + "=" * 60)
    print(f"mode: {summary['mode']} | problems: {summary['n_problems']} | N: {summary['N_per_problem']}")
    if summary.get("plan_combine", "none") != "none":
        print(f"plan_combine: {summary['plan_combine']}" +
              (f" (weight={summary['plan_weight']})" if summary.get("plan_weight") else ""))
    if "parse_fail_rate" in summary:
        warn = " ⚠ 解析失败偏高" if summary["parse_fail_rate"] > 0.1 else ""
        print(f"parse fail rate: {summary['parse_fail_rate']*100:.1f}%{warn}")
    print(f"\n  pass@1 (随机选下限): {summary['pass@1 (random lower bound)']*100:.1f}%")
    print(f"  oracle (完美选上限): {summary['oracle / pass@N (upper bound)']*100:.1f}%")
    print(f"  --- BoN 准确率 ---")
    for k, v in summary["BoN"].items():
        frac = summary["BoN_fraction_of_oracle_gap"][k]
        fs = f"{frac*100:.0f}% of oracle gap" if frac is not None else "n/a"
        print(f"    {k:<14}: {v*100:.1f}%  (+{summary['BoN_gain_over_random'][k]*100:.1f} vs random, {fs})")

# ===================== main =====================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["ours", "scalar", "skywork", "majority", "merge"])
    ap.add_argument("--prm", default=None)
    ap.add_argument("--traj", default="runs/eval/trajectories.jsonl")
    ap.add_argument("--problems", default=None)
    ap.add_argument("--out", default="runs/eval/bon_report.json")
    ap.add_argument("--max_model_len", type=int, default=8192)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--max_gen", type=int, default=400)
    ap.add_argument("--step_sep", default="<extra_0>")
    ap.add_argument("--repetition_penalty", type=float, default=1.15)
    # ---- v5 双分数 ----
    ap.add_argument("--dual_score", action="store_true",
                    help="配合v5模型: 解析<plan_score>/<step_score>双tag, 而非旧版<score>")
    ap.add_argument("--score_scheme", choices=list(INSTRUCTION_DUAL_VARIANTS.keys()), default=None,
                    help="评测的是哪个版本的checkpoint: v6(plan/step都三分类) / "
                         "step01(仅step二分类) / full01(plan/step都二分类)。"
                         "--dual_score时必须显式指定, 不设默认值, 强制每次手动确认, "
                         "避免再发生'忘了同步指令跟checkpoint不一致'这类隐蔽bug。")
    ap.add_argument("--plan_combine", choices=["none", "multiply", "weighted", "plan_trust"], default="weighted",
                    help="如何把plan_score组合进最终链分 (仅--dual_score时生效)")
    ap.add_argument("--plan_weight", type=float, default=0.3,
                    help="weighted模式下plan_score的权重 (0-1)")
    # ---- 数据并行分片 ----
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--shard_glob", default=None)
    args = ap.parse_args()
    if args.mode in ("ours", "scalar", "skywork") and not args.prm:
        ap.error(f"--mode {args.mode} 需要 --prm")
    if args.mode == "merge" and not args.shard_glob:
        ap.error("--mode merge 需要 --shard_glob")
    if args.mode == "ours" and args.dual_score and args.score_scheme is None:
        ap.error("--dual_score 时必须显式指定 --score_scheme "
                f"({'/'.join(INSTRUCTION_DUAL_VARIANTS.keys())}), "
                "不允许用默认值, 必须明确这次评测的是哪个版本的checkpoint")

    trajs = [json.loads(l) for l in open(args.traj)]
    problems = [json.loads(l) for l in open(args.problems)] if args.problems else None
    by_prob_full = defaultdict(list)
    for t in trajs: by_prob_full[t["prob_idx"]].append(t)

    # ---------------- merge ----------------
    if args.mode == "merge":
        files = sorted(glob.glob(args.shard_glob))
        if not files:
            raise SystemExit(f"未匹配到任何分片文件: {args.shard_glob}")
        step_score = defaultdict(dict); parse_fail_total = 0; covered = set()
        plan_score_by_traj = {}
        plan_score_step0_by_traj = {}
        schemes_seen = set()
        for f in files:
            ss, pf, cov, psc, psc0, scheme = load_shard(f)
            for k, d in ss.items():
                step_score[k].update(d)
            plan_score_by_traj.update(psc)
            plan_score_step0_by_traj.update(psc0)
            parse_fail_total += pf
            covered.update(cov)
            schemes_seen.add(scheme)
        if len(schemes_seen) > 1:
            print(f"  ⚠⚠ 严重警告: 这次合并的分片里混入了不止一种--score_scheme: "
                  f"{schemes_seen} —— 说明可能不小心混入了不同checkpoint跑出来的"
                  f"分片文件, 这次的结果很可能不可信, 建议先排查清楚再看下面的数字")
        elif schemes_seen and next(iter(schemes_seen)) is not None:
            print(f"  本次合并的分片均使用 score_scheme={next(iter(schemes_seen))}")
        missing = set(by_prob_full.keys()) - covered
        if missing:
            print(f"  ⚠ 警告: {len(missing)} 题未被任何分片覆盖: "
                  f"{sorted(missing)[:10]}{'...' if len(missing)>10 else ''}")
        print(f"  已合并 {len(files)} 个分片, 覆盖 {len(covered)} 题, 总 parse_fail={parse_fail_total}")
        summary, per_problem = build_summary("merged", by_prob_full, step_score, parse_fail_total,
                                             plan_score_by_traj, args.plan_combine, args.plan_weight)
        out_payload = {"summary": summary, "per_problem": per_problem}
        if plan_score_step0_by_traj:
            out_payload["plan_score_step0"] = {f"{p}:{t}": v for (p, t), v in plan_score_step0_by_traj.items()}
        if plan_score_by_traj:
            matrix, mpass1, moracle, mcols = compute_bon_matrix(by_prob_full, step_score, plan_score_by_traj)
            out_payload["bon_matrix"] = {how: matrix[how] for how in ALL_HOWS}
            out_payload["bon_matrix"]["_columns"] = mcols
            auc_matrix, auc_cols = compute_auc_matrix(by_prob_full, step_score, plan_score_by_traj)
            out_payload["auc_matrix"] = {how: auc_matrix[how] for how in ALL_HOWS}
            out_payload["auc_matrix"]["_columns"] = auc_cols
            plan_alone_auc, n_pc, n_pw = compute_plan_score_alone_auc(by_prob_full, plan_score_by_traj)
            out_payload["plan_score_alone_auc"] = {"auc": plan_alone_auc, "n_correct": n_pc, "n_wrong": n_pw}
            maj_hit, _, _, _ = bon_majority(by_prob_full)
            maj_acc = maj_hit / len(by_prob_full)
            wv_acc, wv_how, wv_combo = bon_weighted_vote(by_prob_full, step_score, plan_score_by_traj)
            out_payload["majority_vote_plain"] = round(maj_acc, 4)
            out_payload["prm_weighted_vote_best"] = {
                "accuracy": round(wv_acc, 4), "how": wv_how, "combine": wv_combo}
        json.dump(out_payload, open(args.out, "w"), ensure_ascii=False, indent=2)
        print_report(summary)
        if plan_score_by_traj:
            print_matrix_report(matrix, mpass1, moracle, mcols)
            print_auc_matrix_report(auc_matrix, auc_cols)
            pa_str = f"{plan_alone_auc:.4f}" if plan_alone_auc is not None else "n/a"
            print(f"\n  plan_score单独(不掺step_score)的AUC: {pa_str} "
                  f"(n_correct={n_pc}, n_wrong={n_pw})")
            print(f"\n  ---- 三种选法对比 (同一批已存数据, 纯算术算出来的) ----")
            print(f"    纯投票(不看PRM, 只数最终答案票数)         : {maj_acc*100:.1f}%")
            print(f"    PRM加权投票(按答案分组, 组内PRM分数求和)  : {wv_acc*100:.1f}%  "
                  f"(how={wv_how}, combine={wv_combo})")
            print(f"    PRM单独挑分最高的那条(矩阵里的最佳组合)   : 见上方矩阵'最佳组合'那一行")
        print(f"\n  saved -> {args.out}")
        return

    # ---------------- majority ----------------
    if args.mode == "majority":
        hit, pass1, oracle, per_problem = bon_majority(by_prob_full)
        n = len(by_prob_full)
        summary = {"mode": "majority", "n_problems": n,
                   "N_per_problem": len(next(iter(by_prob_full.values())))}
        summary.update({
            "pass@1 (random lower bound)": round(pass1, 4),
            "oracle / pass@N (upper bound)": round(oracle, 4),
            "BoN": {"majority_vote": round(hit/n, 4)},
            "BoN_gain_over_random": {"majority_vote": round(hit/n - pass1, 4)},
        })
        gap = oracle - pass1
        summary["BoN_fraction_of_oracle_gap"] = {
            "majority_vote": round((hit/n - pass1)/gap, 4) if gap > 1e-9 else None}
        json.dump({"summary": summary, "per_problem": per_problem},
                  open(args.out, "w"), ensure_ascii=False, indent=2)
        print("\n" + "=" * 60)
        print(f"mode: majority | problems: {n} | N: {summary['N_per_problem']}")
        print(f"\n  pass@1 (随机选下限): {pass1*100:.1f}%")
        print(f"  oracle (完美选上限): {oracle*100:.1f}%")
        v = summary["BoN"]["majority_vote"]
        print(f"    majority_vote : {v*100:.1f}%")
        print(f"\n  saved -> {args.out}")
        return

    # ---------------- ours/scalar/skywork: 按需分片 ----------------
    if args.num_shards > 1:
        by_prob = {k: v for k, v in by_prob_full.items() if k % args.num_shards == args.shard_id}
        if not by_prob:
            raise SystemExit(f"分片 {args.shard_id}/{args.num_shards} 没分到任何题")
    else:
        by_prob = by_prob_full

    if args.mode == "ours":
        step_score, parse_fail, plan_score_by_traj, plan_score_step0_by_traj = score_ours(by_prob, problems, args)
    elif args.mode == "skywork":
        step_score, parse_fail, plan_score_by_traj, plan_score_step0_by_traj = score_skywork(by_prob, problems, args)
    else:
        step_score, parse_fail, plan_score_by_traj, plan_score_step0_by_traj = score_scalar(by_prob, problems, args)

    if args.num_shards > 1:
        shard_path = f"{args.out}.shard{args.shard_id}.json"
        save_shard(shard_path, step_score, parse_fail, list(by_prob.keys()),
                  plan_score_by_traj, plan_score_step0_by_traj,
                  args.score_scheme if (args.mode == "ours" and args.dual_score) else None)
        return

    summary, per_problem = build_summary(args.mode, by_prob, step_score, parse_fail,
                                         plan_score_by_traj, args.plan_combine, args.plan_weight)
    out_payload = {"summary": summary, "per_problem": per_problem}
    if plan_score_step0_by_traj:
        out_payload["plan_score_step0"] = {f"{p}:{t}": v for (p, t), v in plan_score_step0_by_traj.items()}
    if plan_score_by_traj:
        matrix, mpass1, moracle, mcols = compute_bon_matrix(by_prob, step_score, plan_score_by_traj)
        out_payload["bon_matrix"] = {how: matrix[how] for how in ALL_HOWS}
        out_payload["bon_matrix"]["_columns"] = mcols
        auc_matrix, auc_cols = compute_auc_matrix(by_prob, step_score, plan_score_by_traj)
        out_payload["auc_matrix"] = {how: auc_matrix[how] for how in ALL_HOWS}
        out_payload["auc_matrix"]["_columns"] = auc_cols
        plan_alone_auc, n_pc, n_pw = compute_plan_score_alone_auc(by_prob, plan_score_by_traj)
        out_payload["plan_score_alone_auc"] = {"auc": plan_alone_auc, "n_correct": n_pc, "n_wrong": n_pw}
    json.dump(out_payload, open(args.out, "w"), ensure_ascii=False, indent=2)
    print_report(summary)
    if plan_score_by_traj:
        print_matrix_report(matrix, mpass1, moracle, mcols)
        print_auc_matrix_report(auc_matrix, auc_cols)
        pa_str = f"{plan_alone_auc:.4f}" if plan_alone_auc is not None else "n/a"
        print(f"\n  plan_score单独(不掺step_score)的AUC: {pa_str} "
              f"(n_correct={n_pc}, n_wrong={n_pw})")
    print(f"\n  saved -> {args.out}")

if __name__ == "__main__":
    main()