"""
label_minimal.py — 规则标注器 (最小验证收口: 测 #3 与 #4)

吃 sample_minimal.py 产出的 trajectories.jsonl, 对每个 step 打:
  B1 序列级 adherence  (纯规则): 跳步 / 未覆盖环节 / 停滞stall / 弃计划
  B2 名实相符          (规则优先): code动作签名 vs plan环节的action
  C1 执行跑通          (规则): traceback / timeout
  C2 中间值程序验证    (模板, 本轮只跑规则模板, LLM兜底只统计不调用):
       T1 factorint乘积复核   T2 solve+代回   T3 末步answer(已有verifier判)

输出:
  runs/v0/step_labels.jsonl   每步一行的标签草稿
  终端报告: #3 = B2规则可判率,  #4 = C2可验率(PASS+FAIL占比), B1各项统计

用法:  python label_minimal.py --traj runs/v0/trajectories.jsonl
纯CPU, 不调LLM, 不需要GPU。需要 sympy。
"""
import argparse, json, re
from collections import Counter, defaultdict

import sympy as sp

# ---------------------------------------------------------------- B2 (v2: AST)
import ast as _ast

# plan 的 action 偶尔越出枚举表, 做别名归一
ACTION_ALIASES = {"COMPARE": "VERIFY_CHECK", "SUBSTITUTE": "COMPUTE",
                  "SUMMARIZE": "FORMAT_ANSWER", "COUNT": "COMPUTE",
                  "FILTER": "ENUMERATE", "SELECT_MIN": "COMPUTE"}
KNOWN_ACTIONS = {"TRANSLATE", "SETUP_EQUATION", "SIMPLIFY", "SOLVE", "CASE_ANALYSIS",
                 "ENUMERATE", "COMPUTE", "VERIFY_CHECK", "FORMAT_ANSWER"}

# 调用名 -> 动作类别 (AST 抽到的精确函数名查这张表)
CALL2ACT = {
    # SOLVE
    "solve": "SOLVE", "roots": "SOLVE", "linsolve": "SOLVE", "nsolve": "SOLVE",
    "solveset": "SOLVE", "rsolve": "SOLVE", "dsolve": "SOLVE",
    # SIMPLIFY
    "simplify": "SIMPLIFY", "factor": "SIMPLIFY", "expand": "SIMPLIFY",
    "cancel": "SIMPLIFY", "apart": "SIMPLIFY", "together": "SIMPLIFY",
    "radsimp": "SIMPLIFY", "nsimplify": "SIMPLIFY", "trigsimp": "SIMPLIFY",
    "limit_denominator": "SIMPLIFY",
    # COMPUTE (数值/代数运算)
    "subs": "COMPUTE", "evalf": "COMPUTE", "N": "COMPUTE", "factorint": "COMPUTE",
    "sqrt": "COMPUTE", "Rational": "COMPUTE", "gcd": "COMPUTE", "lcm": "COMPUTE",
    "summation": "COMPUTE", "Sum": "COMPUTE", "integrate": "COMPUTE",
    "diff": "COMPUTE", "limit": "COMPUTE", "binomial": "COMPUTE",
    "atan": "COMPUTE", "atan2": "COMPUTE", "sin": "COMPUTE", "cos": "COMPUTE",
    "tan": "COMPUTE", "log": "COMPUTE", "exp": "COMPUTE", "Abs": "COMPUTE",
    "norm": "COMPUTE", "dot": "COMPUTE", "cross": "COMPUTE", "det": "COMPUTE",
    "sum": "COMPUTE", "min": "COMPUTE", "max": "COMPUTE", "len": "COMPUTE",
    "round": "COMPUTE", "abs": "COMPUTE", "prod": "COMPUTE", "sorted": "COMPUTE",
    # SETUP_EQUATION (建符号/建方程/建结构)
    "symbols": "SETUP_EQUATION", "Symbol": "SETUP_EQUATION", "Eq": "SETUP_EQUATION",
    "Function": "SETUP_EQUATION", "Matrix": "SETUP_EQUATION",
    "parse_expr": "SETUP_EQUATION", "sympify": "SETUP_EQUATION",
    # ENUMERATE
    "range": "ENUMERATE", "enumerate": "ENUMERATE", "product": "ENUMERATE",
    "combinations": "ENUMERATE", "permutations": "ENUMERATE",
    # VERIFY_CHECK
    "equals": "VERIFY_CHECK", "isclose": "VERIFY_CHECK", "is_integer": "VERIFY_CHECK",
}

# expected action -> 可接受的签名集合
# 设计原则: 把已知量直接落成常量 (LITERAL) 是合法的计算/翻译/求解手法,
# 不算偏离 —— 否则 num_sides=5、x=3/2 这类正确步会被误判 deviated (假负标)。
COMPAT = {
    "TRANSLATE":      {"LITERAL", "SETUP_EQUATION", "COMPUTE", "DISPLAY"},
    "SETUP_EQUATION": {"SETUP_EQUATION", "LITERAL", "COMPUTE"},
    "SIMPLIFY":       {"SIMPLIFY", "COMPUTE"},
    "SOLVE":          {"SOLVE", "ENUMERATE", "LITERAL"},  # 手算/枚举求解; 不含通用COMPUTE
    "CASE_ANALYSIS":  {"CASE_ANALYSIS", "ENUMERATE"},
    "ENUMERATE":      {"ENUMERATE", "COMPUTE", "LITERAL"},
    "COMPUTE":        {"COMPUTE", "SOLVE", "SIMPLIFY", "ENUMERATE",
                       "SETUP_EQUATION", "LITERAL"},
    "VERIFY_CHECK":   {"VERIFY_CHECK"},   # 只认比较/assert/equals; 纯重算见下方特判
    "FORMAT_ANSWER":  {"FORMAT_ANSWER", "DISPLAY", "LITERAL", "COMPUTE"},
}

def ast_facts(code):
    """用 AST 抽该步代码的精确事实; 解析失败返回 None (语法残破 -> uncertain)。
    返回 dict:
      calls          非print调用名集合 (sp.solve -> 'solve')
      new_bindings   是否产生新变量绑定 (赋值/for目标/函数定义)
      loop/branch/compare/asserts   控制流与比较事实
      arith          是否有算术运算 (BinOp/UnaryOp, 字符串拼接除外)
      print_literal  是否有纯字符串字面量 print (叙述)
      print_var      是否有带变量/表达式的 print (展示状态)
      bare_names     语句级裸表达式 (auto-print 展示)
    """
    try:
        tree = _ast.parse(code)
    except SyntaxError:
        return None
    f = dict(calls=set(), new_bindings=False, loop=False, branch=False,
             compare=False, asserts=False, arith=False,
             print_literal=False, print_var=False, bare_names=False)
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Call):
            name = (node.func.id if isinstance(node.func, _ast.Name) else
                    node.func.attr if isinstance(node.func, _ast.Attribute) else None)
            if name == "print":
                lit = all(isinstance(a, _ast.Constant) and isinstance(a.value, str)
                          for a in node.args) and node.args
                f["print_literal" if lit else "print_var"] = True
            elif name:
                f["calls"].add(name)
        elif isinstance(node, (_ast.Assign, _ast.AugAssign, _ast.AnnAssign,
                               _ast.FunctionDef)):
            f["new_bindings"] = True
        elif isinstance(node, (_ast.For, _ast.While, _ast.comprehension)):
            f["loop"] = True; f["new_bindings"] = True
        elif isinstance(node, _ast.If):
            f["branch"] = True
        elif isinstance(node, _ast.Compare):
            f["compare"] = True
        elif isinstance(node, _ast.Assert):
            f["asserts"] = True
        elif isinstance(node, (_ast.BinOp, _ast.UnaryOp)):
            f["arith"] = True
        elif isinstance(node, _ast.Expr) and isinstance(node.value,
                                                        (_ast.Name, _ast.Attribute,
                                                         _ast.Tuple)):
            f["bare_names"] = True
    return f

def action_signature(code):
    """返回 (签名集合, facts)。签名含: 各动作类别 + LITERAL / DISPLAY / NARRATION
       / UNKNOWN_CALLS / SYNTAX_BROKEN"""
    f = ast_facts(code)
    if f is None:
        return {"SYNTAX_BROKEN"}, None
    sigs = set()
    unknown = set()
    DATA_XFORM = {"list", "dict", "set", "tuple", "sorted", "zip",
                  "items", "values", "keys", "map", "filter", "len"}
    TOOL_NEUTRAL = {"format", "join", "append", "get", "str", "int", "float"}
    for c in f["calls"]:
        if c in CALL2ACT:
            sigs.add(CALL2ACT[c])
        elif c in DATA_XFORM:
            sigs.add("ENUMERATE")     # 对真实数据做结构转换 = 实质操作, 非空话
        elif c not in TOOL_NEUTRAL:
            unknown.add(c)            # 真不认识的调用
    if f["loop"]:
        sigs.add("ENUMERATE")
    if f["branch"]:
        sigs.add("CASE_ANALYSIS")
    if f["compare"] or f["asserts"]:
        sigs.add("VERIFY_CHECK")
    if f["arith"]:
        sigs.add("COMPUTE")
    # 纯字面赋值: 有新绑定但无任何调用/运算 -> 给定数据落变量
    if f["new_bindings"] and not f["calls"] and not f["arith"] and not f["loop"]:
        sigs.add("LITERAL")
    if f["print_var"] or f["bare_names"]:
        sigs.add("DISPLAY")
    if f["print_literal"] and not sigs and not f["new_bindings"]:
        sigs.add("NARRATION")         # 纯叙述: 只 print 文字, 啥也没做
    if not sigs and not f["new_bindings"]:
        sigs.add("NARRATION")         # 纯注释/空代码同上
    if unknown and not sigs - {"DISPLAY"}:
        sigs.add("UNKNOWN_CALLS")     # 只有不认识的调用撑场 -> 留给兜底
    return sigs, f

COMPUTE_FAMILY = {"COMPUTE", "SOLVE", "SIMPLIFY", "ENUMERATE", "SETUP_EQUATION"}

def judge_b2(expected_action, code):
    """faithful / deviated / claimed_only / uncertain:<reason>"""
    exp = ACTION_ALIASES.get(expected_action, expected_action)
    if exp not in COMPAT:
        return "uncertain:unknown_action"
    sigs, f = action_signature(code)
    if "SYNTAX_BROKEN" in sigs:
        return "uncertain:syntax_broken"
    # 严格规则一: 纯叙述 (只 print 字符串字面量 / 纯注释空代码) 声称任何环节
    #             -> 嘴上做了。叙述不执行任何东西, 规则可确信。
    if sigs == {"NARRATION"}:
        return "claimed_only"
    if sigs <= {"NARRATION"}:              # 空代码 / 纯注释
        return "claimed_only"
    # 声称 FORMAT_ANSWER 而仅展示 -> 展示即本职, faithful
    if sigs <= {"DISPLAY", "NARRATION"} and exp == "FORMAT_ANSWER":
        return "faithful"
    # 仅展示已有变量 (无新计算) 声称计算/验证类: 规则无法判断所展示值是否
    # 为本环节应得的有意义结果 (可能是 SIMPLIFY/COMPUTE 的合法收尾, 也可能是
    # 空洞复读) -> 留给 P3 语义判断, 不硬扣 claimed_only (避免假负标)。
    if sigs <= {"DISPLAY", "NARRATION"}:
        return "uncertain:display_only"
    if "UNKNOWN_CALLS" in sigs:
        return "uncertain:unknown_calls"
    if sigs & COMPAT[exp]:
        return "faithful"
    # 特判: 声称 VERIFY 但只有计算无比较 —— 可能是"独立重算验证"(忠实),
    # 也可能是"硬赋值冒充验证"(嘴上做了), 规则分不开 -> 留给 P3
    if exp == "VERIFY_CHECK" and sigs & COMPUTE_FAMILY:
        return "uncertain:verify_ambiguous"
    # 特判: 声称 SOLVE 但只有通用计算签名 (无 solve/枚举) —— 手算求解(忠实)
    # 与调错求解器(偏离)在 AST 层不可分 -> 留给 P3
    if exp == "SOLVE" and sigs & {"COMPUTE"} and not (sigs & {"SOLVE", "ENUMERATE"}):
        return "uncertain:solve_ambiguous"
    return "deviated"

# ---------------------------------------------------------------- C2 模板
def last_printed_value(exec_out):
    if not exec_out or exec_out in ("[no output]", "[timeout]"):
        return None
    lines = [l for l in exec_out.strip().splitlines() if l.strip()]
    if not lines or "Traceback" in exec_out:
        return None
    last = lines[-1]
    m = re.search(r"[:=]\s*(.+)$", last)    # "r = 3" / "Final answer: 14/3"
    return (m.group(1) if m else last).strip()

def c2_factorint(code, exec_out):
    """T1: factorint(N) 的输出 {p:e,...} 乘回去等不等于 N"""
    m = re.search(r"factorint\s*\(\s*(\d+)\s*\)", code)
    if not m:
        return None
    n = int(m.group(1))
    d = re.search(r"\{([^}]*)\}", exec_out or "")
    if not d:
        return None
    try:
        prod = 1
        for pair in d.group(1).split(","):
            p, e = pair.split(":")
            prod *= int(p) ** int(e)
        return "PASS" if prod == n else "FAIL"
    except Exception:
        return None

def c2_solve_subs(code, exec_out):
    """T2: solve(expr, var) + 输出根列表 -> 独立代回 expr 验根"""
    m = re.search(r"solve\s*\(\s*([^,]+?)\s*,\s*(\w+)\s*\)", code)
    if not m:
        return None
    expr_txt, var_txt = m.group(1), m.group(2)
    roots_m = re.search(r"\[([^\]]*)\]", exec_out or "")
    if not roots_m:
        return None
    try:
        var = sp.symbols(var_txt)
        expr = sp.sympify(expr_txt, locals={var_txt: var})
        roots = [sp.sympify(r) for r in roots_m.group(1).split(",") if r.strip()]
        if not roots:
            return None
        ok = all(sp.simplify(expr.subs(var, r)) == 0 for r in roots)
        return "PASS" if ok else "FAIL"
    except Exception:
        return None

# ---------------------------------------------------------------- B1 序列规则
def b1_trajectory(plan, steps):
    plan_ids = [st["id"] for st in plan]
    order = {pid: i for i, pid in enumerate(plan_ids)}
    claimed = [s["claimed"] for s in steps]
    pid_seq = [c for c in claimed if re.fullmatch(r"P\d+", c)]

    executed = set(pid_seq)
    uncovered = [p for p in plan_ids if p not in executed]

    skips = []                              # 跳到 P_j 时, 前面有从未执行过的 P_i
    seen = set()
    for pid in pid_seq:
        for earlier in plan_ids[:order.get(pid, 0)]:
            if earlier not in seen and earlier not in skips:
                skips.append(earlier)
        seen.add(pid)
    skips = [p for p in skips if p in uncovered or p not in seen]

    # stall: 同一 pid + 完全相同代码 连续出现 >= 3 次
    stall_idx = set()
    run_start, run_len = 0, 1
    for i in range(1, len(steps)):
        same = (steps[i]["claimed"] == steps[i-1]["claimed"]
                and steps[i]["code"].strip() == steps[i-1]["code"].strip())
        if same:
            run_len += 1
        else:
            if run_len >= 3:
                stall_idx.update(range(run_start, i))
            run_start, run_len = i, 1
    if run_len >= 3:
        stall_idx.update(range(run_start, len(steps)))

    # 弃计划: 轨迹尾部连续 >=3 个 DEVIATION 再没回到 plan
    tail_dev = 0
    for s in reversed(steps):
        if s["is_deviation"]:
            tail_dev += 1
        else:
            break
    abandoned = tail_dev >= 3

    return {"uncovered": uncovered, "skips": skips,
            "stall_idx": sorted(stall_idx), "abandoned": abandoned}

# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", default="runs/v0/trajectories.jsonl")
    ap.add_argument("--out", default="runs/v0/step_labels.jsonl")
    args = ap.parse_args()

    trajs = [json.loads(l) for l in open(args.traj)]
    b2_counts = Counter(); c2_counts = Counter(); b1_stats = Counter()
    enum_violations = Counter()
    rows = []

    for t in trajs:
        plan = t["plan"]
        act_of = {st["id"]: st.get("action", "?") for st in plan}
        for st in plan:
            a = st.get("action", "?")
            if a not in KNOWN_ACTIONS and a not in ACTION_ALIASES:
                enum_violations[a] += 1
        b1 = b1_trajectory(plan, t["steps"])
        b1_stats["traj"] += 1
        b1_stats["traj_with_skip"] += bool(b1["skips"])
        b1_stats["traj_with_uncovered"] += bool(b1["uncovered"])
        b1_stats["traj_with_stall"] += bool(b1["stall_idx"])
        b1_stats["traj_abandoned"] += bool(b1["abandoned"])

        for i, s in enumerate(t["steps"]):
            ran_ok = ("Traceback" not in (s["exec"] or "")
                      and s["exec"] != "[timeout]")
            if s["is_deviation"]:
                b2 = "self_declared_deviation"
            else:
                b2 = judge_b2(act_of.get(s["claimed"], "?"), s["code"])
            b2_counts[b2] += 1

            # C2: 模板逐个试, 命中即裁决
            c2 = None
            if ran_ok:
                c2 = c2_factorint(s["code"], s["exec"]) \
                     or c2_solve_subs(s["code"], s["exec"])
            if c2 is None:
                val = last_printed_value(s["exec"]) if ran_ok else None
                c2 = ("UNCHECKABLE_value" if val is not None else
                      "UNCHECKABLE_novalue")
            c2_counts[c2] += 1

            rows.append({
                "prob_idx": t["prob_idx"], "plan_idx": t["plan_idx"],
                "exec_idx": t["exec_idx"], "step": i,
                "claimed": s["claimed"], "b2": b2, "ran_ok": ran_ok,
                "c2": c2, "in_stall": i in b1["stall_idx"],
                "traj_skips": b1["skips"], "traj_uncovered": b1["uncovered"],
                "traj_abandoned": b1["abandoned"],
            })

    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(rows)
    decided = sum(b2_counts[k] for k in
                  ("faithful", "deviated", "claimed_only",
                   "self_declared_deviation"))
    uncertain_n = sum(v for k, v in b2_counts.items()
                      if k.startswith("uncertain"))
    checked = c2_counts["PASS"] + c2_counts["FAIL"]
    novalue = c2_counts["UNCHECKABLE_novalue"]

    print("=" * 64)
    print(f"steps labeled: {n}   (saved -> {args.out})")
    print(f"\n[B1 per-trajectory]  skip {b1_stats['traj_with_skip']}/{b1_stats['traj']}"
          f" | uncovered {b1_stats['traj_with_uncovered']}/{b1_stats['traj']}"
          f" | stall {b1_stats['traj_with_stall']}/{b1_stats['traj']}"
          f" | abandoned {b1_stats['traj_abandoned']}/{b1_stats['traj']}")
    print(f"\n[#3 B2 rule coverage] decided-by-rule {decided}/{n} "
          f"= {decided/n*100:.0f}%   (uncertain -> LLM fallback pool: "
          f"{uncertain_n})")
    print("    breakdown:", dict(b2_counts))
    if enum_violations:
        print(f"    plan action enum violations: {dict(enum_violations)}")
    print(f"\n[#4 C2 verification coverage] PASS+FAIL {checked}/{n} "
          f"= {checked/n*100:.0f}% of all steps")
    vsteps = n - novalue
    if vsteps:
        print(f"    over value-producing steps: {checked}/{vsteps} "
              f"= {checked/vsteps*100:.0f}%   (target >60% 的口径建议用这个)")
    print("    breakdown:", dict(c2_counts))
    print("\nnote: 本轮纯规则; UNCHECKABLE_value 是 P4(LLM写校验码) 的目标池,"
          " uncertain 是 P3 的目标池 — 两个池子的大小就是LLM兜底的真实需求量。")

if __name__ == "__main__":
    main()