"""
verify_rules.py (v3) — C2 规则引擎 (执行正确性维度, 零 LLM)

设计前提: TIR 代码真实执行过, 步内计算自洽已被解释器锚定 (C1); C2 验的是
意图层正确性, 只有两类锚: 相对锚 (兄弟执行) 与终局锚 (verifier 终判)。

  L1 完整性 (轨迹级, 不产生 step verdict):
     答案 vs 末端打印值 —— 终判对而不符 = 合法脑内改写 (final_reformulation,
     P4 残余职责样本池); 终判错而不符 = ungrounded_final_answer 旗标。
  L2 跨执行佐证 (B2=faithful 门卫 + 可比性门 + 对照共识门):
     - 可比性门: 只有同"类"的值才互比 (叙述/不等式不比; 容器须同括号同元数;
       Eq 对 Eq 走两边之差专比) —— 表示异质强行比 = 制造冤案;
     - 对照共识门: 定罪需 >=2 条答对执行在该环节互相一致, 孤证不定罪;
     - 一致 ∧ 有终判对 -> corroborated; 答错执行与共识首次分歧 -> localized_fail。
  L3 终局回灌: def-use 承重路径; 终判对 -> outcome_consistent (弱正标);
     终判错 -> suspect (仅排除/降权)。

provenance: localized_fail > corroborated > outcome_consistent > suspect > unverified
输出: c2_labels.jsonl (每步) + c2_labels_traj_flags.jsonl (轨迹级旗标)
用法: python verify_rules.py --traj runs/v0/trajectories.jsonl \
                             --labels runs/v0/step_labels.jsonl

v3.3 改动 (复杂度熔断, 三重防线):
  1. _too_complex_for_verifier(): 长度 / 括号嵌套深度 / 数学函数调用次数(含
     sin/cos/exp等) / im|re|conjugate复数专用函数出现 —— 任一命中即判"过于
     复杂, 不送验证器"。这是判断最快的一道, 覆盖已知的病态模式。
  2. stage_equal() 整体套 signal 超时(STAGE_EQUAL_TIMEOUT) —— 不管什么新
     模式导致慢, 只要这个函数总耗时超过阈值就直接放弃判定为不可比。这是
     兜底未知模式的一道, 不需要每次遇到新变体就手动加特例。
  3. values_equal() 内部对 verifier.is_correct 调用单独再加一层短超时
     (VALUES_EQUAL_TIMEOUT) —— 因为它也被 L1 阶段直接调用(不经过
     stage_equal), 需要独立的保护。
  已知局限: signal.SIGALRM 仅在主线程有效, 且对某些卡在C扩展层深处、不
  返回Python字节码的计算可能无法中断(实测对本项目遇到的sympy符号化简
  多数场景有效); 长度/关键词熔断作为更早的第一道防线可以避免大部分这类
  输入进入到需要靠超时兜底的阶段。
"""
import argparse, ast, json, re, signal, time
from collections import Counter, defaultdict
from functools import lru_cache
from contextlib import contextmanager
from tqdm import tqdm

# ----------------------------- 超时保护 (仅主线程有效) -----------------------
class _TO(Exception):
    pass

@contextmanager
def time_limit(seconds):
    def handler(signum, frame):
        raise _TO()
    old = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)

VALUES_EQUAL_TIMEOUT = 2    # values_equal 内部单次调用超时
STAGE_EQUAL_TIMEOUT = 3     # stage_equal 整体判断超时(含value_kind/_eq_sides/values_equal全流程)
STAGE_EQUAL_MAX_LEN = 150   # stage_equal 入口长度熔断阈值

# ----------------------------- 复杂度熔断 (供 values_equal 与 stage_equal 共用) ----
def _too_complex_for_verifier(a, b, max_len=150, max_paren_depth=6, max_func_calls=15, max_coefficient=100):
    for s in (str(a), str(b)):
        if len(s) > max_len:
            return True
        depth = max_depth = 0
        for ch in s:
            if ch == '(':
                depth += 1; max_depth = max(max_depth, depth)
            elif ch == ')':
                depth -= 1
        if max_depth > max_paren_depth:
            return True
        func_count = sum(s.count(f + "(") for f in
                         ("sin", "cos", "tan", "sqrt", "exp", "atan", "log", "Abs"))
        if func_count > max_func_calls:
            return True
        if re.search(r'\b(im|re|conjugate)\(', s):
            return True
        numbers = re.findall(r'\d+', s)
        if any(int(n) > max_coefficient for n in numbers if len(n) < 15):
            return True
        if re.search(r'\b(harmonic|Sum|Product|Integral|zeta|polygamma|hyper|Subs)\(', s):
            return True
        var_count = len(set(re.findall(r'\b[a-d]\b', s)))
        if var_count >= 3 and 'I' in s:
            return True
    return False

try:
    from verifier import is_correct as _vc

    @lru_cache(maxsize=200000)
    def values_equal(a, b):
        print(f"[DEBUG-L1] comparing len(a)={len(str(a))} a[:100]={str(a)[:100]!r}  len(b)={len(str(b))} b[:100]={str(b)[:100]!r}", flush=True)

        if _too_complex_for_verifier(a, b):
            return False
        try:
            with time_limit(VALUES_EQUAL_TIMEOUT):
                return bool(_vc(str(a), str(b)))
        except _TO:
            return False
        except Exception:
            return False
except ImportError:
    @lru_cache(maxsize=200000)
    def values_equal(a, b):
        try: return abs(float(a) - float(b)) < 1e-6
        except Exception: return str(a).strip() == str(b).strip()

PID_RE = re.compile(r"^P\d+$")
FLOAT_RE = re.compile(r"^-?\d+\.\d+$")

# ----------------------------- 打印值抽取 ------------------------------------
def _tail(line):
    """从一行打印里抽值: 冒号优先; 恰好一个等号才取尾; 其余取整行。"""
    if ": " in line:
        return line.rsplit(": ", 1)[1].strip()
    if line.count("=") == 1:
        return line.split("=", 1)[1].strip()
    return line.strip()

def last_printed_value(exec_out):
    if not exec_out or exec_out in ("[no output]", "[timeout]", "[output too long, discarded]") \
       or "Traceback" in exec_out:
        return None
    lines = [l for l in exec_out.strip().splitlines() if l.strip()]
    return _tail(lines[-1]) if lines else None

def terminal_candidates(steps, max_cands=8):
    """轨迹尾部往回收集打印值候选 (逐步、逐行倒序), 供 L1 比对。"""
    cands = []
    for s in reversed(steps):
        out = s.get("exec") or ""
        if out in ("[no output]", "[timeout]", "[output too long, discarded]") or "Traceback" in out:
            continue
        for line in reversed([l for l in out.strip().splitlines() if l.strip()]):
            cands.append(_tail(line))
            if len(cands) >= max_cands:
                return cands
    return cands

def norm_value(v):
    """句子尾数抽取: 'Angle at B is 56 degrees.' -> '56'。短值原样返回。"""
    if v is None:
        return None
    v = v.strip()
    if len(v.split()) > 4:
        m = re.search(r"(-?\d+(?:\.\d+)?(?:/\d+)?)\s*(?:degrees?|°)?\s*\.?$", v)
        return m.group(1) if m else v
    return v

def _mathy(v):
    return (len(v) <= 60 and len(v.split()) <= 6
            and bool(re.search(r"\d|sqrt|pi|frac", v)))

# ----------------------------- 可比性门 --------------------------------------
def value_kind(v):
    """值的"类": 只有同类才可互比/定罪。None = 不可比。"""
    s = str(v).strip()
    if not s:
        return None
    if re.search(r"<=|>=|<|>|\.\.\.", s):
        return None                          # 不等式/截断展示: 不比
    if s.startswith("Eq("):
        return ("eq",)
    if (s[0], s[-1]) in (("(", ")"), ("[", "]")):
        return ("seq", s[0], s.count(","))   # 容器: 同括号同元数才可比
    if len(s.split()) > 4 or re.search(r"[A-Za-z]{12,}", s):
        return None                          # 叙述文字
    if len(s) > 80:
        return None
    return ("expr",)

def _eq_sides(v):
    m = re.fullmatch(r"Eq\((.*)\)", (v or "").strip())
    if not m:
        return None
    s, depth = m.group(1), 0
    for i, ch in enumerate(s):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "," and depth == 0:
            return s[:i].strip(), s[i + 1:].strip()
    return None

def stage_equal(a, b):
    """同环节两值等价性: True / False / None(不可比, 不得据此定罪)。
    三道防线: 1) 长度熔断(最快)  2) 复杂度熔断(已知病态模式, 含复数专用函数)
    3) 整体超时(兜底未知模式) —— 不管什么原因导致慢, 超过 STAGE_EQUAL_TIMEOUT
    秒就直接放弃, 判不可比, 不据此定罪。"""
    print(f"[DEBUG] comparing len(a)={len(str(a))} a[:100]={str(a)[:100]!r}  len(b)={len(str(b))} b[:100]={str(b)[:100]!r}", flush=True)
    if len(str(a)) > STAGE_EQUAL_MAX_LEN or len(str(b)) > STAGE_EQUAL_MAX_LEN:
        return None
    if _too_complex_for_verifier(a, b):
        return None
    try:
        with time_limit(STAGE_EQUAL_TIMEOUT):
            ka, kb = value_kind(a), value_kind(b)
            if ka is None or ka != kb:
                return None
            if ka == ("eq",):
                ea, eb = _eq_sides(a), _eq_sides(b)
                if not (ea and eb):
                    return None
                d1, d2 = f"({ea[0]})-({ea[1]})", f"({eb[0]})-({eb[1]})"
                return bool(values_equal(d1, d2) or values_equal(d1, f"-({d2})"))
            return bool(values_equal(a, b))
    except _TO:
        return None

# ----------------------------- AST: def/use + 有界搜索 -----------------------
@lru_cache(maxsize=100000)
def _parse_tree(code):
    try:
        return ast.parse(code)
    except SyntaxError:
        return None

def defs_uses(code):
    tree = _parse_tree(code)
    if tree is None:
        return set(), set()
    d, u = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            (d if isinstance(node.ctx, ast.Store) else u).add(node.id)
        elif isinstance(node, ast.FunctionDef):
            d.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                d.add((a.asname or a.name).split(".")[0])
    return d, u - d

def bounded_search_flag(code):
    tree = _parse_tree(code)
    if tree is None:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
           and node.func.id == "range" \
           and any(isinstance(a, ast.Constant) and isinstance(a.value, int)
                   and a.value >= 10 for a in node.args):
            return True
    return False

# ----------------------------- main ------------------------------------------
def main():
    global VALUES_EQUAL_TIMEOUT, STAGE_EQUAL_TIMEOUT, STAGE_EQUAL_MAX_LEN
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", default="runs/v0/trajectories.jsonl")
    ap.add_argument("--labels", default="runs/v0/step_labels.jsonl")
    ap.add_argument("--out", default="runs/v0/c2_labels.jsonl")
    ap.add_argument("--timeout", type=float, default=VALUES_EQUAL_TIMEOUT,
                    help="values_equal 单次比较超时秒数 (默认2秒)")
    ap.add_argument("--stage_timeout", type=float, default=STAGE_EQUAL_TIMEOUT,
                    help="stage_equal 整体判断超时秒数 (默认3秒)")
    ap.add_argument("--stage_max_len", type=int, default=STAGE_EQUAL_MAX_LEN,
                    help="stage_equal 入口长度熔断阈值 (默认150字符)")
    args = ap.parse_args()

    VALUES_EQUAL_TIMEOUT = args.timeout
    STAGE_EQUAL_TIMEOUT = args.stage_timeout
    STAGE_EQUAL_MAX_LEN = args.stage_max_len

    trajs = [json.loads(l) for l in open(args.traj)]
    tmap = {(t["prob_idx"], t["plan_idx"], t["exec_idx"]): t for t in trajs}
    labels = [json.loads(l) for l in open(args.labels)]
    lab = {(r["prob_idx"], r["plan_idx"], r["exec_idx"], r["step"]): r
           for r in labels}

    def adherence_of(key):
        r = lab.get(key, {})
        return r.get("adherence") or r.get("b2", "")

    res = {}
    for t in tqdm(trajs, desc="Init results"):
        for i, _ in enumerate(t["steps"]):
            res[(t["prob_idx"], t["plan_idx"], t["exec_idx"], i)] = {
                "verdict": None, "provenance": "unverified", "evidence": ""}

    PREC = {"localized_fail": 5, "corroborated": 4,
            "outcome_consistent": 3, "suspect": 2, "unverified": 0}
    def assign(key, verdict, prov, ev):
        if PREC[prov] > PREC[res[key]["provenance"]]:
            res[key] = {"verdict": verdict, "provenance": prov, "evidence": ev}

    # ================= L1: 完整性 (轨迹级) =================
    l1 = Counter(); traj_flags = []
    t0 = time.time()
    for t in tqdm(trajs, desc="L1 Integrity"):
        if t.get("answer") is None or not t["steps"]:
            continue
        cands = terminal_candidates(t["steps"])
        if not cands:
            continue
        if any(values_equal(t["answer"], v) for v in cands):
            l1["transcription_ok"] += 1
            if any(FLOAT_RE.match(v) for v in cands) \
               and not FLOAT_RE.match(str(t["answer"])):
                l1["float_exact_ok"] += 1
            continue
        mathy_c = [v for v in cands if _mathy(v)]
        if not mathy_c:
            l1["transcription_skip_no_mathy"] += 1
            continue
        flag = ("final_reformulation" if t.get("correct") is True
                else "ungrounded_final_answer")
        l1[flag] += 1
        traj_flags.append({"prob_idx": t["prob_idx"], "plan_idx": t["plan_idx"],
                           "exec_idx": t["exec_idx"], "flag": flag,
                           "answer": t["answer"], "terminal": mathy_c[0]})

    # ================= L2: 跨执行佐证 =================
    stage_val = defaultdict(dict)   # (prob,plan,pid) -> {exec: (step_i, value)}
    print(f"L1 finished in {time.time()-t0:.1f}s")
    t0 = time.time()
    for t in tqdm(trajs, desc="Build stage values"):
        tk3 = (t["prob_idx"], t["plan_idx"])
        for i, s in enumerate(t["steps"]):
            pid = s["claimed"]
            if not PID_RE.match(pid):
                continue
            if adherence_of(tk3 + (t["exec_idx"], i)) != "faithful":
                continue
            v = norm_value(last_printed_value(s["exec"]))
            if v is not None and value_kind(v) is not None:
                stage_val[tk3 + (pid,)][t["exec_idx"]] = (i, v)

    l2 = Counter(); onset = {}
    print(f"Stage values built in {time.time()-t0:.1f}s")
    t0 = time.time()
    for (p, pl, pid), per_exec in tqdm(stage_val.items(), total=len(stage_val), desc="L2 Corroboration"):
        if len(per_exec) < 2:
            continue
        execs = sorted(per_exec)
        groups = []
        for e in execs:
            v = per_exec[e][1]
            placed = False
            for g in groups:
                if stage_equal(v, per_exec[g[0]][1]) is True:
                    g.append(e); placed = True; break
            if not placed:
                groups.append([e])
        corrects = {e for e in execs
                    if tmap[(p, pl, e)].get("correct") is True}
        if len(groups) == 1:
            if corrects:
                for e in execs:
                    i, v = per_exec[e]
                    assign((p, pl, e, i), "PASS", "corroborated",
                           f"{len(execs)} execs agree at {pid}; "
                           f"{len(corrects)} verified-correct")
                l2["corroborated_stage"] += 1
            continue
        l2["divergent_stage"] += 1
        # 对照共识门: >=2 条答对执行该环节互相一致, 才有定罪资格
        ref = None
        for g in groups:
            cg = [e for e in g if e in corrects]
            if len(cg) >= 2:
                ref = per_exec[cg[0]][1]; break
        if ref is None:
            l2["no_consensus_ref"] += 1
            continue
        stage_order = int(pid[1:])
        for e in execs:
            if tmap[(p, pl, e)].get("correct") is not False:
                continue
            if stage_equal(per_exec[e][1], ref) is False:   # None 不定罪
                k = (p, pl, e)
                if stage_order < onset.get(k, (10**9,))[0]:
                    onset[k] = (stage_order, per_exec[e][0],
                                per_exec[e][1], ref, pid)
    for (p, pl, e), (so, step_i, v, ref_v, pid) in onset.items():
        assign((p, pl, e, step_i), "FAIL", "localized_fail",
               f"first divergence at {pid}: {v!r} vs consensus of "
               f"correct execs {ref_v!r}")
        l2["localized_fail"] += 1

    # ================= L3: 终局回灌 =================
    l3 = Counter()
    print(f"L2 finished in {time.time()-t0:.1f}s")
    t0 = time.time()
    for t in tqdm(trajs, desc="L3 Outcome"):
        if t.get("correct") is None or not t["steps"]:
            continue
        tk = (t["prob_idx"], t["plan_idx"], t["exec_idx"])
        du = [defs_uses(s["code"]) for s in t["steps"]]
        need = set(du[-1][1]) | set(du[-1][0])
        path = {len(t["steps"]) - 1}
        for i in range(len(t["steps"]) - 2, -1, -1):
            if du[i][0] & need:
                path.add(i)
                need |= du[i][1]
        for i in path:
            if last_printed_value(t["steps"][i]["exec"]) is None:
                continue
            if t["correct"]:
                assign(tk + (i,), "PASS", "outcome_consistent",
                       "on load-bearing path of verified-correct trajectory")
                l3["outcome_consistent"] += 1
            else:
                assign(tk + (i,), None, "suspect",
                       "on load-bearing path of incorrect trajectory")
                l3["suspect"] += 1
    print(f"L3 finished in {time.time()-t0:.1f}s")

    # ================= 有界搜索 flag =================
    bounded = set()
    for t in trajs:
        for i, s in enumerate(t["steps"]):
            if bounded_search_flag(s["code"]):
                bounded.add((t["prob_idx"], t["plan_idx"], t["exec_idx"], i))
    l1["bounded_search_flag"] = len(bounded)

    # ================= 落盘 + 报告 =================
    with open(args.out, "w") as f:
        for key in sorted(res):
            p, pl, e, i = key
            rec = {"prob_idx": p, "plan_idx": pl, "exec_idx": e, "step": i,
                   **res[key], "bounded_search": key in bounded}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    fpath = args.out.replace(".jsonl", "_traj_flags.jsonl")
    with open(fpath, "w") as f:
        for r in traj_flags:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(res)
    prov = Counter(r["provenance"] for r in res.values())
    valued = sum(1 for key in res
                 if last_printed_value(
                     tmap[key[:3]]["steps"][key[3]]["exec"]) is not None)
    supervised = sum(v for k, v in prov.items()
                     if k in ("localized_fail", "corroborated",
                              "outcome_consistent"))
    print("=" * 64)
    print(f"steps: {n}  (value-producing: {valued})   saved -> {args.out}")
    print(f"\n[L1 integrity, trajectory-level] {dict(l1)}   -> {fpath}")
    print(f"[L2 corroboration] {dict(l2)}")
    print(f"[L3 outcome] {dict(l3)}")
    print(f"\n[provenance] {dict(prov)}")
    print(f"\n[#4-v3 C2 supervised coverage] {supervised}/{n} "
          f"= {supervised/n*100:.0f}% of all steps"
          f" | {supervised}/{valued} = {supervised/max(1,valued)*100:.0f}%"
          f" of value-producing steps  (suspect 不计入, 仅供排除/降权)")

if __name__ == "__main__":
    main()