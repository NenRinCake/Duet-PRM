import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")   # silence fork warning

"""
Minimal validation sampler — factorized PRM data construction (Phase 0).

Pipeline:
  Phase A (P1): for each problem, sample M plans (plan-only JSON, spoiler-checked).
  Phase B (P2): for each plan, sample N TIR executions (step -> run code -> feed back).
  Stats: format compliance, spoiler rate, DEVIATION rate, answer extraction,
         verifier-judged accuracy, per-plan plan_quality, per-problem diagnosis.

Run (single GPU is enough for 8B):
  CUDA_VISIBLE_DEVICES=0 python sample_minimal.py \
      --model /path/to/Qwen3-8B --problems math10.jsonl --out runs/v0

problems file: jsonl, each line {"problem": "...", "answer": "..."(optional)}
Put verifier.py next to this script to enable real math_equal judging.

Trajectory coordinates (saved in output):
  prob_idx : which problem
  plan_idx : which plan of that problem (0..M-1)
  exec_idx : which execution of that plan (0..N-1)
"""
import argparse, json, re, subprocess, sys, tempfile, textwrap
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

try:
    from verifier import is_correct          # your Qwen-style math_equal
    HAVE_VERIFIER = True
except ImportError:
    HAVE_VERIFIER = False
    def is_correct(a, g):                     # fallback: loose compare
        try: return abs(float(a) - float(g)) < 1e-6
        except Exception: return str(a).strip() == str(g).strip()

# ----------------------------- prompts --------------------------------------
P1_PLAN = """You are a math problem-solving strategist. Read the problem and produce ONLY a solution plan. Do NOT solve the problem, do NOT compute any numerical values, and do NOT state any intermediate or final results.

Problem:
{problem}

Write a plan of 3-6 ordered stages. Each stage must be:
- a single concrete action that could later be carried out with Python/sympy code;
- verifiable: after execution, one could check whether this stage was done;
- free of any computed values or answers.

For each stage, also give its action type, chosen from EXACTLY this list:
[TRANSLATE, SETUP_EQUATION, SIMPLIFY, SOLVE, CASE_ANALYSIS, ENUMERATE, COMPUTE, VERIFY_CHECK, FORMAT_ANSWER]

Output ONLY a JSON array, no other text:
[
  {{"id": "P1", "desc": "<what to do, no values>", "action": "<one type from the list>"}},
  ...
]"""

P2_SYSTEM = """You are a math problem solver that works strictly by executing a given plan with Python code.

You will be given a problem and a PLAN with stages P1, P2, .... Solve the problem by executing the plan stage by stage.

Rules:
1. Work in steps. Each step must follow EXACTLY this format:
<step>
executing: P<k>
thought: <one or two sentences on what this step does for stage P<k>>
```python
<code for this step; use sympy where appropriate; print() all results you need>
```
</step>
2. After each step, the system will run your code and return the result in <output>...</output>. Read it before writing the next step.
3. One stage may take more than one step; declare the same P<k> again.
4. If you must deviate from the plan (skip a stage, reorder, or do something not in the plan), you MUST declare it:
   executing: DEVIATION (reason: <brief reason>)
5. When the problem is solved, end with:
<answer>...</answer>  (final answer only, simplified, containing no natural language)

Problem:
{problem}

PLAN:
{plan_json}

Begin with your first step."""

# ----------------------------- code sandbox ---------------------------------
SANDBOX = textwrap.dedent("""\
    import sys, io, contextlib, ast
    prev_steps = {prev!r}        # list of past step codes
    cur  = {cur!r}
    g = {{}}
    # replay history ONE STEP AT A TIME, each isolated: a failed step is simply
    # skipped (its state never existed), later successful steps still replay.
    for _code in prev_steps:
        try:
            with contextlib.redirect_stdout(io.StringIO()), \\
                 contextlib.redirect_stderr(io.StringIO()):
                exec(_code, g)
        except Exception:
            pass
    # Jupyter-style: if the last statement is a bare expression, print its value
    try:
        tree = ast.parse(cur)
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last = ast.Expression(tree.body[-1].value)
            body = ast.Module(body=tree.body[:-1], type_ignores=[])
            exec(compile(body, "<step>", "exec"), g)
            _val = eval(compile(last, "<step>", "eval"), g)
            if _val is not None:
                print(repr(_val))
        else:
            exec(cur, g)
    except Exception:
        import traceback; traceback.print_exc()
""")

def run_code(prev_codes, cur_code, timeout=15):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(SANDBOX.format(prev=list(prev_codes), cur=cur_code)); path = f.name
    try:
        env = dict(os.environ, PYTHONWARNINGS="ignore")
        r = subprocess.run([sys.executable, path], capture_output=True,
                           text=True, timeout=timeout, env=env)
        out = (r.stdout + r.stderr).strip()
    except subprocess.TimeoutExpired:
        out = "[timeout]"
    finally:
        os.unlink(path)
    return out[:3000] if out else "[no output]"

# ----------------------------- parsing --------------------------------------
STEP_RE   = re.compile(r"<step>(.*?)</step>", re.DOTALL)
EXEC_RE   = re.compile(r"executing:\s*(P\d+|DEVIATION[^\n]*)", re.IGNORECASE)
CODE_RE   = re.compile(r"```python\s*(.*?)```", re.DOTALL)
ANS_RE    = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
NUM_RE    = re.compile(r"\d+(?:\.\d+)?")

def parse_step(chunk):
    """Returns dict or None (format violation)."""
    m = STEP_RE.search(chunk)
    body = m.group(1) if m else chunk          # tolerate missing <step> wrapper
    e = EXEC_RE.search(body); c = CODE_RE.search(body)
    if not (e and c):
        return None
    claim = e.group(1).strip()
    return {"claimed": claim,
            "is_deviation": claim.upper().startswith("DEVIATION"),
            "code": c.group(1).strip(),
            "raw": chunk,
            "wrapped": bool(m)}

def spoiler_check(plan, problem):
    """Numbers in plan descs that don't appear in the problem text -> spoiler."""
    prob_nums = set(NUM_RE.findall(problem))
    for st in plan:
        for n in NUM_RE.findall(st.get("desc", "")):
            if n not in prob_nums and n not in {"1", "2"}:
                return True
    return False

# ----------------------------- main -----------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--problems", required=True)
    ap.add_argument("--out", default="runs/Separation/gaokao2023en")
    ap.add_argument("--n_problems", type=int, default=500)
    ap.add_argument("--M", type=int, default=2, help="plans per problem")
    ap.add_argument("--N", type=int, default=4, help="executions per plan")
    ap.add_argument("--max_steps", type=int, default=15)
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--max_model_len", type=int, default=16384)
    ap.add_argument("--plan_temp", type=float, default=0.8)
    ap.add_argument("--exec_temp", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=64,
                    help="代码执行验证的并发线程数 (CPU, 等子进程I/O)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(model=args.model, tensor_parallel_size=args.tp,
              max_model_len=args.max_model_len, gpu_memory_utilization=0.90,
              trust_remote_code=True, seed=args.seed)

    def chat_prefix(user_msg, system_msg=None):
        msgs = ([{"role": "system", "content": system_msg}] if system_msg else []) \
               + [{"role": "user", "content": user_msg}]
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True,
                                       enable_thinking=False)   # Qwen3: no <think>

    problems = [json.loads(l) for l in open(args.problems)][:args.n_problems]

    # ---- 字段归一化: 不同 bench 的题目/答案字段名不同, 统一成 problem / answer ----
    # MATH: {"problem","answer"} | GSM8K: {"question","answer"(带 #### )} |
    # 其他常见: query/prompt/input(题面), solution/gt/gt_answer/target(答案)
    def _norm_problem(p):
        if "problem" not in p:
            for k in ("question", "query", "prompt", "input"):
                if k in p and p[k] is not None:
                    p["problem"] = p[k]; break
        if "answer" not in p:
            # final_answer 放最前(真答案); solution 放最后(它是解题过程, 不得已才用)
            for k in ("final_answer", "gt_answer", "gt", "target",
                      "answer_text", "solution", "label"):
                if k in p and p[k] is not None:
                    p["answer"] = p[k]; break
        # OlympiadBench 等: answer 可能是 list (如 ["14"]) -> 取第一个元素
        if isinstance(p.get("answer"), list) and p["answer"]:
            p["answer"] = p["answer"][0]
        # GSM8K: answer 是整段解析, 末尾 "#### 数字" 才是最终答案 -> 抽出来
        a = p.get("answer")
        if isinstance(a, str) and "####" in a:
            p["answer"] = a.split("####")[-1].strip().replace(",", "")
        return p

    problems = [_norm_problem(p) for p in problems]
    missing = sum(1 for p in problems if "problem" not in p or p.get("problem") is None)
    if missing:
        raise SystemExit(
            f"[字段错误] {missing}/{len(problems)} 条没有可识别的题目字段。"
            f"检查 jsonl 字段名, 首行: {json.dumps(problems[0], ensure_ascii=False)[:200]}")
    print(f"loaded {len(problems)} problems "
          f"(题目字段已归一化; 含答案: {sum('answer' in p for p in problems)}/{len(problems)})")

    # ---------------- Phase A: P1 plans (batched, 1 retry on fail) ----------
    plan_sp = SamplingParams(temperature=args.plan_temp, top_p=args.top_p, max_tokens=600)
    plans, spoiler_or_fail = {}, 0
    for attempt in range(2):
        need = [(prob_idx, plan_idx)
                for prob_idx in range(len(problems))
                for plan_idx in range(args.M)
                if (prob_idx, plan_idx) not in plans]
        if not need: break
        prompts = [chat_prefix(P1_PLAN.format(problem=problems[p]["problem"]))
                   for p, _ in need]
        outs = llm.generate(prompts, plan_sp)
        # print(outs[0].outputs[0].text)
        for (prob_idx, plan_idx), o in zip(need, outs):
            txt = o.outputs[0].text
            try:
                arr = json.loads(re.search(r"\[.*\]", txt, re.DOTALL).group(0))
                assert all("id" in s and "desc" in s and "action" in s for s in arr)
                if spoiler_check(arr, problems[prob_idx]["problem"]):
                    if attempt == 1:               # final attempt: keep but count
                        spoiler_or_fail += 1
                        plans[(prob_idx, plan_idx)] = arr
                    continue                       # first attempt: retry
                plans[(prob_idx, plan_idx)] = arr
            except Exception:
                if attempt == 1:                   # final attempt: count + drop
                    spoiler_or_fail += 1
                    plans[(prob_idx, plan_idx)] = None
    # [评测改动] 不再丢弃两次尝试都失败的 plan: 保留为 None, 下面 Phase B 仍为它
    # 产出 N 条空链, 保证每题严格 M×N 条、不丢题 (空链 answer=None -> 写文件时算错误链)。
    n_empty_plan_units = sum(1 for v in plans.values() if v is None)
    total_plans = len(problems) * args.M
    print(f"[#2] plan spoiler/parse-fail rate: {spoiler_or_fail}/{total_plans} "
          f"= {spoiler_or_fail/total_plans*100:.0f}%")

    # ---------------- Phase B: P2 TIR executions (batched stepping) ---------
    step_sp = SamplingParams(temperature=args.exec_temp, top_p=args.top_p, max_tokens=900,
                             stop=["</step>", "</answer>"],
                             include_stop_str_in_output=True)
    trajs = []
    for (prob_idx, plan_idx), plan in plans.items():
        if plan is None:
            # [评测改动] plan 两次尝试都失败: 仍产出 N 条空链占位 (严格保持每题 M×N),
            # done=True 直接跳过执行轮次, answer 保持 None -> 写文件时记为错误链。
            for exec_idx in range(args.N):
                trajs.append({"prob_idx": prob_idx, "plan_idx": plan_idx,
                              "exec_idx": exec_idx, "plan": [],
                              "prefix": "", "assistant": "", "steps": [],
                              "answer": None, "fmt_bad": 0, "done": True})
            continue
        base = chat_prefix(
            P2_SYSTEM.format(problem=problems[prob_idx]["problem"],
                             plan_json=json.dumps(plan, ensure_ascii=False)))
        for exec_idx in range(args.N):
            trajs.append({"prob_idx": prob_idx, "plan_idx": plan_idx,
                          "exec_idx": exec_idx, "plan": plan,
                          "prefix": base, "assistant": "", "steps": [],
                          "answer": None, "fmt_bad": 0, "done": False})

    # 长度安全阈值: 留出本轮即将生成的max_tokens, 再留一点余量给tokenize误差/特殊token。
    # 防止某条轨迹累积文本顶到模型硬上限时, vLLM把整批正常轨迹一起带崩。
    MAX_PROMPT_TOKENS = args.max_model_len - step_sp.max_tokens - 64

    for rnd in range(args.max_steps):
        active = [t for t in trajs if not t["done"]]
        if not active: break

        # 提前用tokenizer量长度, 超限的直接标记成done(不设answer, 自然落进现有的
        # "answer is None -> correct=None -> censored"统计口径, 跟max_steps用完
        # 没收敛是同一类情况, 不需要新字段) —— 关键是从这一批要发给vLLM的prompt里
        # 摘除掉, 不让vLLM看到它, 这样它不会拖累同批次其他正常轨迹。
        safe, too_long = [], 0
        for t in active:
            text = t["prefix"] + t["assistant"]
            if len(tok.encode(text, add_special_tokens=False)) > MAX_PROMPT_TOKENS:
                t["done"] = True
                too_long += 1
            else:
                safe.append(t)
        if too_long:
            print(f"  [警告] 本轮 {too_long} 条轨迹因累积长度超过安全阈值"
                  f"({MAX_PROMPT_TOKENS} tokens)被提前截断, 归入censored")
        active = safe
        if not active: break

        outs = llm.generate([t["prefix"] + t["assistant"] for t in active], step_sp)

        # print(outs[0].outputs[0].text)
        # 先做纯CPU解析(快): 分流出"已答完/格式坏/需跑代码"三类
        to_run = []   # (traj, step_dict, prev_codes)
        for t, o in zip(active, outs):
            chunk = o.outputs[0].text
            a = ANS_RE.search(chunk)
            if a:
                t["answer"] = a.group(1).strip(); t["done"] = True
                t["assistant"] += chunk; continue
            st = parse_step(chunk)
            if st is None:
                t["fmt_bad"] += 1; t["done"] = True       # unusable, stop traj
                t["assistant"] += chunk; continue
            prev = [s["code"] for s in t["steps"]]
            to_run.append((t, st, prev))

        # 并行执行代码(慢的部分: 每次起子进程跑sympy), 带进度条。
        # 不同轨迹互相独立, 故按轨迹级并行安全; 单条轨迹内仍按序(prev已固定)。
        def _exec(item):
            t, st, prev = item
            st["exec"] = run_code(prev, st["code"])
            return t, st

        if to_run:
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = [ex.submit(_exec, item) for item in to_run]
                for fut in tqdm(as_completed(futs), total=len(futs),
                                desc=f"round {rnd+1} 执行验证", ncols=80):
                    t, st = fut.result()
                    t["steps"].append(st)
                    t["assistant"] += st["raw"] + f"\n<output>\n{st['exec']}\n</output>\n"
        print(f"round {rnd+1}: {sum(t['done'] for t in trajs)}/{len(trajs)} done")

        # 每轮增量保存checkpoint, 覆盖写(不是追加), 只留最新一轮的快照。
        # 动机: 这次崩溃证明了一旦中途失败(不管是这次的长度问题, 还是以后任何
        # 别的原因), 之前跑完的轮次会随着进程一起全部丢失 —— 加上这一步, 下次
        # 哪怕中途崩了, 至少能从最近一轮的快照里把已经做完的部分捞回来,
        # 不需要从0开始重新跑。
        ckpt_path = os.path.join(args.out, "trajectories.jsonl")
        with open(ckpt_path, "w") as f:
            for t in trajs:
                keys = ("prob_idx", "plan_idx", "exec_idx",
                        "plan", "steps", "answer", "fmt_bad", "done")
                f.write(json.dumps({k: t[k] for k in keys}, ensure_ascii=False) + "\n")

    # ---------------- stats --------------------------------------------------
    n_steps = sum(len(t["steps"]) for t in trajs)
    n_bad   = sum(t["fmt_bad"] for t in trajs)
    n_dev   = sum(s["is_deviation"] for t in trajs for s in t["steps"])
    n_ans   = sum(t["answer"] is not None for t in trajs)
    print("\n========== VALIDATION NUMBERS ==========")
    print(f"trajectories: {len(trajs)} | parsed steps: {n_steps}")
    print(f"[#1] format compliance: {n_steps}/{n_steps+n_bad} "
          f"= {n_steps/max(1,n_steps+n_bad)*100:.0f}%  (target >90%)")
    print(f"[#5] DEVIATION-claimed step rate: {n_dev}/{n_steps} "
          f"= {n_dev/max(1,n_steps)*100:.0f}%  (ideal 10-30%)")
    print(f"answer extraction: {n_ans}/{len(trajs)} = {n_ans/len(trajs)*100:.0f}%")

    pq = {}    # (prob_idx, plan_idx) -> {"quality", "trunc_rate"}; filled if gold exists
    if all("answer" in p for p in problems):
        if not HAVE_VERIFIER:
            print("[warn] verifier.py not found -> falling back to loose compare")
        n_corr = 0
        for t in trajs:
            if t["answer"] is None:
                t["correct"] = None
            else:
                t["correct"] = bool(is_correct(t["answer"],
                                               problems[t["prob_idx"]]["answer"]))
                n_corr += t["correct"]
        print(f"verified-correct answers: {n_corr}/{n_ans} "
              f"({'real verifier' if HAVE_VERIFIER else 'loose'})")

        # ---- plan_quality (dual-quantity Phase-1 label) + diagnosis ----
        by_plan = defaultdict(list)
        for t in trajs:
            by_plan[(t["prob_idx"], t["plan_idx"])].append(t)
        print("\n---- plan labels: success rate (decided only) + truncation rate ----")
        print(f"{'prob':>4} {'plan':>4} {'N':>3} {'ok':>3} {'wrong':>5} {'cens':>4}"
              f"  {'quality':>8} {'trunc':>6}")
        pq = {}
        for (prob_idx, plan_idx), ts in sorted(by_plan.items()):
            ok = sum(1 for x in ts if x["correct"] is True)
            wrong = sum(1 for x in ts if x["correct"] is False)
            cens = sum(1 for x in ts if x["correct"] is None)
            q = ok / (ok + wrong) if (ok + wrong) else None   # decided-only
            trunc = cens / len(ts)                            # over ALL execs
            pq[(prob_idx, plan_idx)] = {"quality": q, "trunc_rate": trunc}
            qs = f"{q:.2f}" if q is not None else "n/a"
            print(f"{prob_idx:>4} {plan_idx:>4} {len(ts):>3} {ok:>3} {wrong:>5}"
                  f" {cens:>4}  {qs:>8} {trunc:>6.2f}")
        by_prob = defaultdict(list)
        for (prob_idx, plan_idx), lab in pq.items():
            by_prob[prob_idx].append(lab)
        print("\n---- per-problem (hard vs plan-split vs censored) ----")
        for prob_idx in sorted(by_prob):
            labs = by_prob[prob_idx]
            qs = [l["quality"] for l in labs if l["quality"] is not None]
            tag = ("all-censored" if not qs else
                   "hard/all-failed" if max(qs) == 0 else
                   "PLAN-SPLIT" if len(qs) > 1 and max(qs) - min(qs) >= 0.5 else
                   "ok")
            qstr = ", ".join(
                (f"q={l['quality']:.2f}" if l["quality"] is not None else "q=n/a")
                + f"/t={l['trunc_rate']:.2f}" for l in labs)
            print(f"  prob {prob_idx}: [{qstr}] -> {tag}")

    out = os.path.join(args.out, "trajectories.jsonl")
    with open(out, "w") as f:
        for t in trajs:
            keys = ("prob_idx", "plan_idx", "exec_idx",
                    "plan", "steps", "answer", "fmt_bad")
            rec = {k: t[k] for k in keys}
            rec["correct"] = t.get("correct")     # verifier verdict (or None)
            # [评测改动] 评测口径: 没答出(截断/格式坏/空plan -> correct=None)一律算错误链。
            # 只在写文件时转换, 不动上面的诊断统计(plan_quality 仍按 None 计 censored)。
            if rec["correct"] is None:
                rec["correct"] = False
            lab = pq.get((t["prob_idx"], t["plan_idx"]))
            rec["plan_quality"] = lab["quality"] if lab else None
            rec["plan_trunc_rate"] = lab["trunc_rate"] if lab else None
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\nsaved -> {out}")
    print("next: labeling script for [#3] B2 rule coverage and "
          "[#4] C2 verification coverage.")

if __name__ == "__main__":
    main()