import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

"""
extend_executions.py — 给已有的plan补跑更多执行 (N=4 -> N=8), 不重新生成plan

为什么需要这个脚本(而不是直接重跑sample_minimal.py加大--N):
  直接重跑sample_minimal.py会连plan一起重新生成(Phase A的LLM采样不是确定性的),
  这样N和plan两个变量同时变了, 没法干净地归因"扩大N有没有用"。这个脚本跳过
  Phase A, 直接从已有的trajectories-v6.jsonl里加载plan(逐字节复用, 不调用LLM),
  只针对每个已有plan补跑n_new次新执行(Phase B), 保证唯一变化的变量是N。

用法:
  python extend_executions.py \
      --model /public/home/ljt/lzm/model/Qwen3-8B \
      --base_traj /public/home/ljt/lzm/code/PRM/runs/v0/trajectories-v6.jsonl \
      --problems /path/train50.jsonl \
      --out runs/v0/trajectories-v6-n8.jsonl \
      --n_new 4 --tp 8

产出: 每个(prob_idx, plan_idx)下有 N_old(已有的, 原样不变) + n_new(新跑的) 条
trajectory, plan_quality/plan_trunc_rate 按合并后的全部N重新计算。
"""
import argparse, json, re, subprocess, sys, tempfile, textwrap
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

try:
    from verifier import is_correct
    HAVE_VERIFIER = True
except ImportError:
    HAVE_VERIFIER = False
    def is_correct(a, g):
        try: return abs(float(a) - float(g)) < 1e-6
        except Exception: return str(a).strip() == str(g).strip()

# ----------------------------- 与 sample_minimal.py 保持完全一致的部分 -------
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
ANS_RE  = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

def parse_step(chunk):
    m = STEP_RE.search(chunk)
    body = m.group(1) if m else chunk
    e = EXEC_RE.search(body); c = CODE_RE.search(body)
    if not (e and c):
        return None
    claim = e.group(1).strip()
    return {"claimed": claim,
            "is_deviation": claim.upper().startswith("DEVIATION"),
            "code": c.group(1).strip(),
            "raw": chunk,
            "wrapped": bool(m)}

# ----------------------------- 本脚本新增: 从已有文件加载plan和旧执行 --------
def load_plans_and_old_execs(base_traj_path):
    """返回:
      plans: {(prob_idx, plan_idx): plan_json_list}  (逐字节复用, 不重新生成)
      old_execs_by_plan: {(prob_idx, plan_idx): [old_traj_dict, ...]}
      next_exec_idx: {(prob_idx, plan_idx): int}  (= max已有exec_idx + 1)
    """
    plans = {}
    old_execs_by_plan = defaultdict(list)
    for line in open(base_traj_path):
        t = json.loads(line)
        key = (t["prob_idx"], t["plan_idx"])
        if key not in plans:
            plans[key] = t["plan"]          # 同一个plan下每条记录的plan字段都一样, 取一次即可
        old_execs_by_plan[key].append(t)
    next_exec_idx = {}
    for key, execs in old_execs_by_plan.items():
        next_exec_idx[key] = max(e["exec_idx"] for e in execs) + 1
    return plans, dict(old_execs_by_plan), next_exec_idx

def recompute_plan_labels(all_execs):
    """对合并后(旧+新)的全部执行, 重新算 plan_quality(decided-only成功率) 和
    plan_trunc_rate(censored占比) —— 逻辑与sample_minimal.py完全一致。"""
    ok = sum(1 for x in all_execs if x.get("correct") is True)
    wrong = sum(1 for x in all_execs if x.get("correct") is False)
    cens = sum(1 for x in all_execs if x.get("correct") is None)
    q = ok / (ok + wrong) if (ok + wrong) else None
    trunc = cens / len(all_execs) if all_execs else None
    return {"quality": q, "trunc_rate": trunc}

# ----------------------------- main ------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base_traj", required=True,
                    help="已有的trajectories.jsonl路径 (plan和旧执行从这里逐字节复用)")
    ap.add_argument("--problems", required=True,
                    help="必须是生成base_traj时用的同一份problems文件, 否则prob_idx对不上")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_new", type=int, default=4, help="每个plan补跑几条新执行")
    ap.add_argument("--max_steps", type=int, default=15)
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--max_model_len", type=int, default=16384)
    ap.add_argument("--exec_temp", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=43,
                    help="故意跟原始sample_minimal.py的默认seed(42)不同, 避免万一引擎"
                         "seed真的决定性地复现同一段随机数流, 新执行变成旧执行的复制品")
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    plans, old_execs_by_plan, next_exec_idx = load_plans_and_old_execs(args.base_traj)
    print(f"从 {args.base_traj} 加载到 {len(plans)} 个plan (逐字节复用, 未重新生成)")
    n_old_total = sum(len(v) for v in old_execs_by_plan.values())
    print(f"已有执行总数: {n_old_total}, 将为每个plan补跑 {args.n_new} 条新执行")

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(model=args.model, tensor_parallel_size=args.tp,
              max_model_len=args.max_model_len, gpu_memory_utilization=0.90,
              trust_remote_code=True, seed=args.seed)

    def chat_prefix(user_msg):
        return tok.apply_chat_template(
            [{"role": "user", "content": user_msg}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)

    problems = [json.loads(l) for l in open(args.problems)]

    # ---- 构造新执行的trajectory占位 (exec_idx 从 next_exec_idx 开始) ----
    new_trajs = []
    for (prob_idx, plan_idx), plan in plans.items():
        base = chat_prefix(P2_SYSTEM.format(
            problem=problems[prob_idx]["problem"],
            plan_json=json.dumps(plan, ensure_ascii=False)))
        start = next_exec_idx[(prob_idx, plan_idx)]
        for i in range(args.n_new):
            new_trajs.append({"prob_idx": prob_idx, "plan_idx": plan_idx,
                              "exec_idx": start + i, "plan": plan,
                              "prefix": base, "assistant": "", "steps": [],
                              "answer": None, "fmt_bad": 0, "done": False})

    # ---- Phase B 执行循环 (与sample_minimal.py逐行一致) ----
    step_sp = SamplingParams(temperature=args.exec_temp, top_p=args.top_p, max_tokens=900,
                             stop=["</step>", "</answer>"],
                             include_stop_str_in_output=True)
    for rnd in range(args.max_steps):
        active = [t for t in new_trajs if not t["done"]]
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
                for fut in tqdm(as_completed(futs), total=len(futs),
                                desc=f"round {rnd+1} 执行验证", ncols=80):
                    t, st = fut.result()
                    t["steps"].append(st)
                    t["assistant"] += st["raw"] + f"\n<output>\n{st['exec']}\n</output>\n"
        print(f"round {rnd+1}: {sum(t['done'] for t in new_trajs)}/{len(new_trajs)} done")

    # ---- 判正确性 (新执行) ----
    for t in new_trajs:
        if t["answer"] is None:
            t["correct"] = None
        else:
            t["correct"] = bool(is_correct(t["answer"],
                                           problems[t["prob_idx"]]["answer"]))

    # ---- 合并旧+新, 按合并后的全部N重新算plan_quality/trunc_rate ----
    new_by_plan = defaultdict(list)
    for t in new_trajs:
        new_by_plan[(t["prob_idx"], t["plan_idx"])].append(t)

    print("\n---- 对比: 仅旧N vs 合并后新N 的 plan_quality/trunc_rate 变化 ----")
    print(f"{'prob':>4} {'plan':>4} {'旧N':>4} {'旧quality':>9} {'旧trunc':>7}"
          f"  {'新N':>4} {'新quality':>9} {'新trunc':>7}  {'quality变化':>10}")
    out_records = []
    for key, plan in plans.items():
        old_list = old_execs_by_plan.get(key, [])
        new_list = new_by_plan.get(key, [])
        old_lab = recompute_plan_labels(old_list)
        merged = old_list + new_list
        new_lab = recompute_plan_labels(merged)
        oq = old_lab["quality"]; nq = new_lab["quality"]
        delta = (nq - oq) if (oq is not None and nq is not None) else None
        oqs = f"{oq:.2f}" if oq is not None else "n/a"
        nqs = f"{nq:.2f}" if nq is not None else "n/a"
        ds = f"{delta:+.2f}" if delta is not None else "n/a"
        print(f"{key[0]:>4} {key[1]:>4} {len(old_list):>4} {oqs:>9} "
              f"{old_lab['trunc_rate']:>7.2f}  {len(merged):>4} {nqs:>9} "
              f"{new_lab['trunc_rate']:>7.2f}  {ds:>10}")

        for t in merged:
            keys_ = ("prob_idx", "plan_idx", "exec_idx", "plan", "steps",
                    "answer", "fmt_bad")
            rec = {k: t[k] for k in keys_}
            rec["correct"] = t.get("correct")
            rec["plan_quality"] = new_lab["quality"]
            rec["plan_trunc_rate"] = new_lab["trunc_rate"]
            out_records.append(rec)

    with open(args.out, "w") as f:
        for rec in out_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\nsaved -> {args.out}  (共 {len(out_records)} 条执行记录, "
          f"覆盖 {len(plans)} 个plan)")

if __name__ == "__main__":
    main()
