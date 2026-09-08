import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

"""
gen_plan_prune_realloc.py — 真正的"剪枝+预算回收"实验: 用plan_score挑出每题
更好的那个plan, 给它额外生成3条新执行(配合原有4条共7条), 最后用PRM在这7条
里选分数最高的一条当最终答案。跟之前inspect_plan_pruning_backtest.py的回放
版本不同, 这次是真刀真枪重新生成, 不是用已有的4条凑数据。

三个阶段, 分开跑(避免在同一个vLLM进程里来回切换两个不同的模型):

  --phase score_plan    用PRM, 给每个(prob,plan)算plan_score(零执行占位符
                        探测法, 跟inspect_plan_only_probe.py同一套), 选出
                        每题的赢家plan, 存到--out
  --phase generate_exec 用执行模型(跟sample_minimal.py的--model同一个), 给
                        每题的赢家plan额外生成--n_new条新执行(exec_idx接着
                        原有的往后编号, 默认从4开始), 复用sample_minimal.py
                        里的P2_SYSTEM/SANDBOX/run_code/parse_step等组件,
                        保证生成机制跟原始数据完全一致, 不是另起一套
  --phase score_final   用PRM, 把赢家plan的"原有N条+新生成3条"合并成7条,
                        逐步打step_score, 选出每题分数最高的那条当最终答案,
                        算整体正确率, 跟majority/原始BoN baseline对照

用法:
  python gen_plan_prune_realloc.py --phase score_plan --prm /public/home/ljt/lzm/code/PRM/saves/prm-merged-v6plus200-full01 --score_scheme full01 --traj runs/eval/trajectories.jsonl --problems /public/home/ljt/lzm/code/Qwen2.5-Math/evaluation/data/math/test.jsonl --out runs/eval/realloc_plan_scores.json

  python gen_plan_prune_realloc.py --phase generate_exec --model /public/home/ljt/lzm/model/Qwen3-8B --traj runs/v0/trajectories.jsonl --problems /public/home/ljt/lzm/code/Qwen2.5-Math/evaluation/data/math/test.jsonl --plan_scores runs/eval/realloc_plan_scores.json --out runs/eval/realloc_new_execs.jsonl --n_new 3

  python gen_plan_prune_realloc.py --phase score_final --prm /public/home/ljt/lzm/code/PRM/saves/prm-merged-v6plus200-full01 --score_scheme full01 --traj runs/eval/trajectories.jsonl --new_traj runs/eval/realloc_new_execs.jsonl --plan_scores runs/eval/realloc_plan_scores.json --problems /public/home/ljt/lzm/code/Qwen2.5-Math/evaluation/data/math/test.jsonl --out runs/eval/realloc_final_report.json
"""

import argparse, json, os, re, sys, tempfile, textwrap, subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from verifier import is_correct
    HAVE_VERIFIER = True
except ImportError:
    HAVE_VERIFIER = False
    def is_correct(a, g):
        try: return abs(float(a) - float(g)) < 1e-6
        except Exception: return str(a).strip() == str(g).strip()

# ===================== phase: score_plan =====================
def phase_score_plan(args):
    from inspect_plan_only_probe import build_plan_groups, PLACEHOLDER_STEP
    import eval_bon as eb
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    trajs = [json.loads(l) for l in open(args.traj)]
    plan_info = build_plan_groups(trajs)
    problems = [json.loads(l) for l in open(args.problems)]

    tok = AutoTokenizer.from_pretrained(args.prm, trust_remote_code=True)
    llm = LLM(model=args.prm, tensor_parallel_size=args.tp,
              max_model_len=args.max_model_len, gpu_memory_utilization=0.90,
              trust_remote_code=True)
    instruction = eb.INSTRUCTION_DUAL_VARIANTS[args.score_scheme]

    def chat(inp):
        return tok.apply_chat_template(
            [{"role": "user", "content": instruction + "\n\n" + inp}],
            tokenize=False, add_generation_prompt=True)

    keys, prompts = [], []
    for (p, pl), info in plan_info.items():
        ptext = problems[p]["problem"]
        inp = eb.build_input_dual(ptext, info["plan"], [], PLACEHOLDER_STEP)
        prompts.append(chat(inp)); keys.append((p, pl))

    sp = SamplingParams(temperature=0.0, max_tokens=args.max_gen,
                        repetition_penalty=args.repetition_penalty)
    print(f"[score_plan] 探测 {len(prompts)} 个plan...")
    outs = llm.generate(prompts, sp)
    plan_scores = {}
    for k, o in zip(keys, outs):
        plan_sc, _ = eb.parse_dual_score(o.outputs[0].text)
        plan_scores[k] = plan_sc

    by_prob = defaultdict(list)
    for (p, pl) in plan_info: by_prob[p].append(pl)
    winners = {}
    for p, pls in by_prob.items():
        if len(pls) < 2: continue
        scored = [(plan_scores.get((p, pl)), pl) for pl in pls]
        scored = [(s, pl) for s, pl in scored if s is not None]
        if not scored: continue
        winners[p] = max(scored, key=lambda x: x[0])[1]

    json.dump({"plan_scores": {f"{p}:{pl}": s for (p, pl), s in plan_scores.items()},
              "winners": {str(p): pl for p, pl in winners.items()}},
             open(args.out, "w"), ensure_ascii=False, indent=2)
    print(f"[score_plan] 共{len(winners)}题选出赢家plan -> {args.out}")

# ===================== phase: generate_exec =====================
# 复用sample_minimal.py的核心组件, 保证生成机制跟原始数据完全一致
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

SANDBOX = textwrap.dedent("""\
    import sys, io, contextlib, ast
    prev_steps = {prev!r}
    cur  = {cur!r}
    g = {{}}
    for _code in prev_steps:
        try:
            with contextlib.redirect_stdout(io.StringIO()), \\
                 contextlib.redirect_stderr(io.StringIO()):
                exec(_code, g)
        except Exception:
            pass
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

STEP_RE = re.compile(r"<step>(.*?)</step>", re.DOTALL)
EXEC_RE = re.compile(r"executing:\s*(P\d+|DEVIATION[^\n]*)", re.IGNORECASE)
CODE_RE = re.compile(r"```python\s*(.*?)```", re.DOTALL)
ANS_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

def parse_step(chunk):
    m = STEP_RE.search(chunk)
    body = m.group(1) if m else chunk
    e = EXEC_RE.search(body); c = CODE_RE.search(body)
    if not (e and c): return None
    claim = e.group(1).strip()
    return {"claimed": claim, "is_deviation": claim.upper().startswith("DEVIATION"),
           "code": c.group(1).strip(), "raw": chunk, "wrapped": bool(m)}

def phase_generate_exec(args):
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    trajs = [json.loads(l) for l in open(args.traj)]
    problems = [json.loads(l) for l in open(args.problems)]
    info = json.load(open(args.plan_scores))
    winners = {int(p): pl for p, pl in info["winners"].items()}

    by_plan = {}
    for t in trajs:
        if (t["prob_idx"], t["plan_idx"]) not in by_plan:
            by_plan[(t["prob_idx"], t["plan_idx"])] = t["plan"]
    existing_exec_idx = defaultdict(set)
    for t in trajs:
        existing_exec_idx[(t["prob_idx"], t["plan_idx"])].add(t["exec_idx"])

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(model=args.model, tensor_parallel_size=args.tp,
              max_model_len=args.max_model_len, gpu_memory_utilization=0.90,
              trust_remote_code=True, seed=args.seed)

    def chat_prefix(user_msg):
        return tok.apply_chat_template(
            [{"role": "user", "content": user_msg}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)

    new_trajs = []
    for p, winner_pl in winners.items():
        plan = by_plan.get((p, winner_pl))
        if plan is None: continue
        base = chat_prefix(P2_SYSTEM.format(
            problem=problems[p]["problem"], plan_json=json.dumps(plan, ensure_ascii=False)))
        used = existing_exec_idx[(p, winner_pl)]
        next_idx = max(used) + 1 if used else 0
        for j in range(args.n_new):
            new_trajs.append({"prob_idx": p, "plan_idx": winner_pl,
                              "exec_idx": next_idx + j, "plan": plan,
                              "prefix": base, "assistant": "", "steps": [],
                              "answer": None, "fmt_bad": 0, "done": False})

    step_sp = SamplingParams(temperature=args.exec_temp, top_p=args.top_p, max_tokens=900,
                             stop=["</step>", "</answer>"], include_stop_str_in_output=True)
    MAX_PROMPT_TOKENS = args.max_model_len - step_sp.max_tokens - 64
    ckpt_path = args.out + ".ckpt.jsonl"

    for rnd in range(args.max_steps):
        active = [t for t in new_trajs if not t["done"]]
        if not active: break
        safe, too_long = [], 0
        for t in active:
            text = t["prefix"] + t["assistant"]
            if len(tok.encode(text, add_special_tokens=False)) > MAX_PROMPT_TOKENS:
                t["done"] = True; too_long += 1
            else:
                safe.append(t)
        if too_long:
            print(f"  [警告] 本轮 {too_long} 条轨迹因长度超限被提前截断, 归入censored")
        active = safe
        if not active: break

        outs = llm.generate([t["prefix"] + t["assistant"] for t in active], step_sp)
        to_run = []
        for t, o in zip(active, outs):
            chunk = o.outputs[0].text
            a = ANS_RE.search(chunk)
            if a:
                t["answer"] = a.group(1).strip(); t["done"] = True
                t["assistant"] += chunk; continue
            st = parse_step(chunk)
            if st is None:
                t["fmt_bad"] += 1; t["done"] = True
                t["assistant"] += chunk; continue
            prev = [s["code"] for s in t["steps"]]
            to_run.append((t, st, prev))

        def _exec(item):
            t, st, prev = item
            st["exec"] = run_code(prev, st["code"])
            return t, st

        if to_run:
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = [ex.submit(_exec, item) for item in to_run]
                for fut in as_completed(futs):
                    t, st = fut.result()
                    t["steps"].append(st)
                    t["assistant"] += st["raw"] + f"\n<output>\n{st['exec']}\n</output>\n"
        print(f"round {rnd+1}: {sum(t['done'] for t in new_trajs)}/{len(new_trajs)} done")

        with open(ckpt_path, "w") as f:
            for t in new_trajs:
                keys = ("prob_idx", "plan_idx", "exec_idx", "plan", "steps", "answer", "fmt_bad", "done")
                f.write(json.dumps({k: t[k] for k in keys}, ensure_ascii=False) + "\n")

    n_corr = 0; n_ans = 0
    for t in new_trajs:
        if t["answer"] is None:
            t["correct"] = None
        else:
            n_ans += 1
            t["correct"] = bool(is_correct(t["answer"], problems[t["prob_idx"]]["answer"]))
            n_corr += t["correct"]
    print(f"[generate_exec] 新生成{len(new_trajs)}条, 答完{n_ans}条, 答对{n_corr}条")

    with open(args.out, "w") as f:
        for t in new_trajs:
            keys = ("prob_idx", "plan_idx", "exec_idx", "plan", "steps", "answer", "fmt_bad", "correct")
            f.write(json.dumps({k: t[k] for k in keys}, ensure_ascii=False) + "\n")
    print(f"[generate_exec] saved -> {args.out}")

# ===================== phase: score_final =====================
def phase_score_final(args):
    import eval_bon as eb
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    orig_trajs = [json.loads(l) for l in open(args.traj)]
    new_trajs = [json.loads(l) for l in open(args.new_traj)]
    problems = [json.loads(l) for l in open(args.problems)]
    info = json.load(open(args.plan_scores))
    winners = {int(p): pl for p, pl in info["winners"].items()}

    by_prob_7 = defaultdict(list)
    for t in orig_trajs:
        if winners.get(t["prob_idx"]) == t["plan_idx"]:
            by_prob_7[t["prob_idx"]].append(t)
    for t in new_trajs:
        by_prob_7[t["prob_idx"]].append(t)

    tok = AutoTokenizer.from_pretrained(args.prm, trust_remote_code=True)
    llm = LLM(model=args.prm, tensor_parallel_size=args.tp,
              max_model_len=args.max_model_len, gpu_memory_utilization=0.90,
              trust_remote_code=True)
    instruction = eb.INSTRUCTION_DUAL_VARIANTS[args.score_scheme]

    def chat(inp):
        return tok.apply_chat_template(
            [{"role": "user", "content": instruction + "\n\n" + inp}],
            tokenize=False, add_generation_prompt=True)

    prompts, index = [], []
    budget = args.max_model_len - args.max_gen - 16
    for p, cand in by_prob_7.items():
        ptext = problems[p]["problem"]
        for tpos, t in enumerate(cand):
            for i, s in enumerate(t["steps"]):
                inp = chat(eb.build_input_dual(ptext, t["plan"], t["steps"][:i], s))
                if len(tok(inp)["input_ids"]) > budget:
                    continue
                prompts.append(inp); index.append((p, tpos, i))
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_gen,
                        repetition_penalty=args.repetition_penalty)
    print(f"[score_final] 打分 {len(prompts)} 步...")
    outs = llm.generate(prompts, sp)
    step_score = defaultdict(dict)
    for (p, tpos, i), o in zip(index, outs):
        _, step_sc = eb.parse_dual_score(o.outputs[0].text)
        step_score[(p, tpos)][i] = step_sc

    n_hit = 0; n_total = 0
    for p, cand in by_prob_7.items():
        scored = []
        for tpos, t in enumerate(cand):
            ss = step_score.get((p, tpos), {})
            seq = [ss.get(i) for i in range(len(t["steps"]))]
            scored.append((eb.aggregate(seq, "mean"), tpos))
        if not scored: continue
        best_tp = max(scored, key=lambda x: x[0])[1]
        n_total += 1
        n_hit += int(bool(cand[best_tp].get("correct")))

    acc = n_hit / n_total if n_total else 0.0
    print(f"\n[score_final] 剪枝+回收预算(赢家plan共7条)挑1条最终正确率: "
          f"{n_hit}/{n_total} = {acc*100:.1f}%")
    json.dump({"n_hit": n_hit, "n_total": n_total, "accuracy": acc},
             open(args.out, "w"), ensure_ascii=False, indent=2)
    print(f"saved -> {args.out}")

# ===================== main =====================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["score_plan", "generate_exec", "score_final"])
    ap.add_argument("--prm", default=None)
    ap.add_argument("--score_scheme", default=None)
    ap.add_argument("--model", default=None, help="generate_exec阶段用的执行模型路径")
    ap.add_argument("--traj", default="runs/v0/trajectories.jsonl")
    ap.add_argument("--new_traj", default=None)
    ap.add_argument("--problems", required=True)
    ap.add_argument("--plan_scores", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_new", type=int, default=3)
    ap.add_argument("--max_steps", type=int, default=15)
    ap.add_argument("--max_model_len", type=int, default=16384)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--max_gen", type=int, default=400)
    ap.add_argument("--exec_temp", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--repetition_penalty", type=float, default=1.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    if args.phase in ("score_plan", "score_final") and (not args.prm or not args.score_scheme):
        ap.error(f"--phase {args.phase} 需要 --prm 和 --score_scheme")
    if args.phase == "generate_exec" and not args.model:
        ap.error("--phase generate_exec 需要 --model (执行模型路径)")
    if args.phase == "generate_exec" and not args.plan_scores:
        ap.error("--phase generate_exec 需要 --plan_scores (上一阶段的输出)")
    if args.phase == "score_final" and (not args.new_traj or not args.plan_scores):
        ap.error("--phase score_final 需要 --new_traj 和 --plan_scores")

    if args.phase == "score_plan":
        phase_score_plan(args)
    elif args.phase == "generate_exec":
        phase_generate_exec(args)
    else:
        phase_score_final(args)

if __name__ == "__main__":
    main()
