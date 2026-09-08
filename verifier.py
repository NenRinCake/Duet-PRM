r"""
verifier.py — 答案验证器（Qwen2.5-Math 风格 math_equal 为主力判等）

判等逻辑取自 Qwen2.5-Math 官方评测脚本的 math_equal,对数学答案的边界情况
(百分比、数值近似、矩阵、区间/列表、方程、选择题、括号归一化)覆盖最全。

本文件自包含:
  - 若项目里已有 Qwen 的完整脚本,可改成 `from qwen_math_eval import math_equal`;
  - 否则直接用本文件内补全的 math_equal + 所有辅助函数,开箱即用。

对外接口:
  is_correct(pred, gold) -> bool        # 判答案等价(默认开 timeout 防 sympy 卡死)
  extract_final_answer(text) -> str     # 从一段文本里抽最后一个 \boxed{...}

== 本版改动 (v2) ==
  1) is_correct 入口新增 _normalize_answer: 剥 \left/\right(及 \big 系),
     统一 unicode 数学符号 (π→pi, √→sqrt, ×·→*, ÷→/, −→-)。
     这些是纯排版/字符层归一, 不改变数学含义, 修复:
       pred "(3, pi/2)"  vs gold "\\left( 3, \\frac{\\pi}{2} \\right)"  的假阴性
       (元组逐元素分支因 gold 以 \\left( 开头而匹配不上)。
  2) _parse_any 新增第三级 fallback: 常规解析失败后, 用 sympy 的
     implicit_multiplication_application 重试。只在原本必判 False 的
     不可解析串上生效, 修复:
       pred "3*sqrt(13)" vs gold "3\\sqrt{13}" (清洗后 "3sqrt(13)" 无乘号解析失败)。
  3) 自测新增上述两类用例 + 反例守卫 (确保没引入假阳性)。
  明确不做: 从整句话里抽答案表达式 (误抽风险, 维持原行为)。
"""

import re
import signal
from typing import Union, Optional

import sympy
from sympy import simplify, N
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
)
from sympy.parsing.latex import parse_latex


# ===========================================================================
# 超时保护:symbolic_equal 用 sympy,某些式子会卡死,必须能打断。
# ===========================================================================
class _TimeoutError(Exception):
    pass


def _handler(signum, frame):
    raise _TimeoutError()


def call_with_timeout(func, *args, timeout: int = 5, **kwargs):
    """在 timeout 秒内执行 func,超时返回 False。仅在主线程/Unix 下用 signal。"""
    try:
        signal.signal(signal.SIGALRM, _handler)
        signal.alarm(timeout)
        try:
            result = func(*args, **kwargs)
        finally:
            signal.alarm(0)
        return result
    except _TimeoutError:
        return False
    except Exception:
        return False


# ===========================================================================
# 数值相关辅助
# ===========================================================================
def parse_digits(num):
    """把字符串解析成 float,支持百分号和千分位逗号。失败返回 None。"""
    num = str(num).replace(",", "")
    try:
        return float(num)
    except Exception:
        if num.endswith("%"):
            num = num[:-1]
            if num.endswith("\\"):
                num = num[:-1]
            try:
                return float(num) / 100
            except Exception:
                pass
    return None


def is_digit(num):
    """能否解析成数字。"""
    return parse_digits(num) is not None


def numeric_equal(prediction: float, reference: float) -> bool:
    """数值近似相等(相对容差),用于浮点比较。"""
    return abs(prediction - reference) <= 1e-4 * max(1.0, abs(reference))


# ===========================================================================
# 字符串/格式归一化辅助
# ===========================================================================
def choice_answer_clean(pred: str):
    """从模型输出里清洗出选择题答案字母(A-E)。"""
    pred = str(pred).strip().strip(".").rstrip("/").strip()
    tmp = re.findall(r"\b(A|B|C|D|E)\b", pred.upper())
    if tmp:
        return tmp[-1]
    return pred


def str_to_pmatrix(s: str) -> str:
    """把形如 {a,b;c,d} 的串转成 pmatrix 形式(简化处理)。"""
    s = s.strip()
    if s.startswith("{") and s.endswith("}"):
        s = s[1:-1]
    rows = s.split(";")
    body = " \\\\ ".join(r.strip() for r in rows)
    return r"\begin{pmatrix}" + body + r"\end{pmatrix}"


def _normalize_answer(s: str) -> str:
    """(v2 新增) 纯排版/字符层归一, 不改变数学含义:
       - 剥 \\left \\right 及 \\big/\\Big/\\bigg/\\Bigg(l/r) 定界指令
       - 剥 \\! \\, \\; \\: 间距符
       - unicode 数学符号 -> ascii: π→pi, √→sqrt(...), ×·→*, ÷→/, −→-
    """
    if s is None:
        return s
    s = str(s)
    s = re.sub(r"\\left|\\right", "", s)
    s = re.sub(
        r"\\(?:text|mathrm|textrm|mbox)\{([^{}]*)\}",
        r"\1",
        s,
    )
    s = re.sub(r"\\[Bb]igg?[lr]?", "", s)
    s = re.sub(r"\\[!,;:]", "", s)
    s = (s.replace("×", "*").replace("·", "*").replace("÷", "/")
          .replace("−", "-"))
    s = re.sub(r"√\s*\(", "sqrt(", s)
    s = re.sub(r"√\s*([0-9a-zA-Z]+)", r"sqrt(\1)", s)
    s = s.replace("π", "pi")
    s = re.sub(r"\^\{?\\circ\}?", "", s)
    s = s.replace("°", "")
    return s.strip()


# ===========================================================================
# 符号判等
# ===========================================================================
def _clean_latex(s: str) -> str:
    """把常见 LaTeX 写法清成可 sympify 的形式(antlr 不可用时的兜底)。"""
    s = s.strip().strip("$").strip()
    s = s.replace(r"\left", "").replace(r"\right", "")
    s = s.replace(r"\!", "").replace(r"\,", "").replace("\\ ", "")
    s = s.replace(r"\times", "*").replace(r"\cdot", "*").replace(r"\div", "/")
    s = s.replace(r"\pi", "pi")
    for _ in range(3):  # \frac{a}{b} -> ((a)/(b)),处理嵌套
        s = re.sub(r"\\d?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"((\1)/(\2))", s)
    s = re.sub(r"\\sqrt\s*\{([^{}]*)\}", r"sqrt(\1)", s)
    s = re.sub(r"\\sqrt\s*(\w+)", r"sqrt(\1)", s)
    s = s.replace(r"\sqrt", "sqrt")
    s = s.replace("^", "**")
    s = s.replace("{", "(").replace("}", ")")
    s = re.sub(r"\\[a-zA-Z]+", "", s)  # 删掉残余的 \command
    return s


_IMPLICIT = standard_transformations + (implicit_multiplication_application,)


def _parse_any(s: str):
    """1) 仅当串含反斜杠(真 LaTeX)才走 parse_latex —— antlr 在场时会把
       ascii 串如 '3*sqrt(13)'、'pi/2' "成功"误解析成 3*s*q*r*t*(13)、p*i/2,
       且因解析成功而短路后面所有 fallback,必须挡在门外;
       2) 清洗后 parse_expr; 3) sympify; 4) 隐式乘法重试。"""
    s = str(s).strip()
    if "\\" in s:                     # gold 这类真 LaTeX 照旧走 antlr(最准)
        try:
            return parse_latex(s)
        except Exception:
            pass
    cleaned = _clean_latex(s)
    try:
        return parse_expr(cleaned, evaluate=True)
    except Exception:
        pass
    try:
        return sympy.sympify(cleaned)
    except Exception:
        pass
    try:                              # 隐式乘法 fallback: 救 '3sqrt(13)' 这类
        return parse_expr(cleaned, transformations=_IMPLICIT, evaluate=True)
    except Exception:
        return None


def symbolic_equal(a: str, b: str) -> bool:
    """两个字符串是否符号等价。"""
    ea, eb = _parse_any(a), _parse_any(b)
    if ea is None or eb is None:
        return False
    # antlr 把 \pi 解析成 Symbol('pi') 而非 π 常量, 在唯一入口统一替换
    try:
        ea = ea.subs(sympy.Symbol('pi'), sympy.pi)
        eb = eb.subs(sympy.Symbol('pi'), sympy.pi)
    except Exception:
        pass
    try:
        if simplify(ea - eb) == 0:
            return True
    except Exception:
        pass
    try:
        if abs(float(N(ea)) - float(N(eb))) <= 1e-6:
            return True
    except Exception:
        pass
    try:
        if ea.equals(eb):
            return True
    except Exception:
        pass
    return False


def symbolic_equal_process(a, b):
    return symbolic_equal(a, b)


# ===========================================================================
# 主判等函数(Qwen2.5-Math 风格)
# ===========================================================================
def math_equal(
    prediction: Union[bool, float, str],
    reference: Union[float, str],
    include_percentage: bool = True,
    is_close: bool = True,
    timeout: bool = True,
) -> bool:
    """
    数学答案精确匹配,当且仅当:
      1. 数值相等:都能转 float 且相等(支持百分比、近似)
      2. 符号相等:都能转 sympy 表达式且等价
    另处理:选择题、矩阵、区间/列表、方程式、括号归一化。
    """
    if prediction is None or reference is None:
        return False
    if str(prediction).strip().lower() == str(reference).strip().lower():
        return True
    if reference in ["A", "B", "C", "D", "E"] and choice_answer_clean(prediction) == reference:
        return True

    # 1) 数值相等
    try:
        if is_digit(prediction) and is_digit(reference):
            p = parse_digits(prediction)
            r = parse_digits(reference)
            if include_percentage:
                gt_result = [r / 100, r, r * 100]
            else:
                gt_result = [r]
            for item in gt_result:
                try:
                    if is_close:
                        if numeric_equal(p, item):
                            return True
                    else:
                        if item == p:
                            return True
                except Exception:
                    continue
            return False
    except Exception:
        pass

    if not prediction and prediction not in [0, False]:
        return False

    # 2) 符号相等
    reference = str(reference).strip()
    prediction = str(prediction).strip()

    if "pmatrix" in prediction and "pmatrix" not in reference:
        reference = str_to_pmatrix(reference)

    pred_str, ref_str = prediction, reference
    if (prediction.startswith("[") and prediction.endswith("]") and not reference.startswith("(")) or \
       (prediction.startswith("(") and prediction.endswith(")") and not reference.startswith("[")):
        pred_str = pred_str.strip("[]()")
        ref_str = ref_str.strip("[]()")
    for sym in ["{", "}", "(", ")"]:
        ref_str = ref_str.replace(sym, "")
        pred_str = pred_str.replace(sym, "")
    if pred_str.lower() == ref_str.lower():
        return True

    if re.match(r"(\(|\[).+(\)|\])", prediction) is not None and \
       re.match(r"(\(|\[).+(\)|\])", reference) is not None:
        pred_parts = prediction[1:-1].split(",")
        ref_parts = reference[1:-1].split(",")
        if len(pred_parts) == len(ref_parts):
            if all(math_equal(pred_parts[i], ref_parts[i], include_percentage, is_close)
                   for i in range(len(pred_parts))):
                return True

    if (prediction.startswith("\\begin{pmatrix}") or prediction.startswith("\\begin{bmatrix}")) and \
       (prediction.endswith("\\end{pmatrix}") or prediction.endswith("\\end{bmatrix}")) and \
       (reference.startswith("\\begin{pmatrix}") or reference.startswith("\\begin{bmatrix}")) and \
       (reference.endswith("\\end{pmatrix}") or reference.endswith("\\end{bmatrix}")):
        pred_lines = [l.strip() for l in prediction[len("\\begin{pmatrix}"):-len("\\end{pmatrix}")].split("\\\\") if l.strip()]
        ref_lines = [l.strip() for l in reference[len("\\begin{pmatrix}"):-len("\\end{pmatrix}")].split("\\\\") if l.strip()]
        matched = True
        if len(pred_lines) == len(ref_lines):
            for pl, rl in zip(pred_lines, ref_lines):
                pp, rp = pl.split("&"), rl.split("&")
                if len(pp) == len(rp):
                    if not all(math_equal(pp[i], rp[i], include_percentage, is_close) for i in range(len(pp))):
                        matched = False
                        break
                else:
                    matched = False
                if not matched:
                    break
        else:
            matched = False
        if matched:
            return True

    if prediction.count("=") == 1 and reference.count("=") == 1:
        pred = prediction.split("=")
        pred = f"{pred[0].strip()} - ({pred[1].strip()})"
        ref = reference.split("=")
        ref = f"{ref[0].strip()} - ({ref[1].strip()})"
        if symbolic_equal(pred, ref) or symbolic_equal(f"-({pred})", ref):
            return True
    elif prediction.count("=") == 1 and len(prediction.split("=")[0].strip()) <= 2 and "=" not in reference:
        if math_equal(prediction.split("=")[1], reference, include_percentage, is_close):
            return True
    elif reference.count("=") == 1 and len(reference.split("=")[0].strip()) <= 2 and "=" not in prediction:
        if math_equal(prediction, reference.split("=")[1], include_percentage, is_close):
            return True

    if timeout:
        if call_with_timeout(symbolic_equal_process, prediction, reference):
            return True
    else:
        if symbolic_equal(prediction, reference):
            return True

    return False


# ===========================================================================
# 答案抽取(从一段文本里取最后一个 \boxed{...},处理嵌套花括号)
# ===========================================================================
def extract_final_answer(text: str) -> Optional[str]:
    if not text:
        return None
    idx = text.rfind(r"\boxed")
    if idx != -1:
        i = text.find("{", idx)
        if i != -1:
            depth = 0
            for j in range(i, len(text)):
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                    if depth == 0:
                        return text[i + 1:j].strip()
    m = re.findall(r"[-+]?\d+\.?\d*(?:/\d+)?", text)
    return m[-1] if m else None


# ===========================================================================
# 对外主接口
# ===========================================================================
def _strip_units(s: str) -> str:
    """
    剥离答案里的单位、货币符号、\\text{...} 等"非数学"修饰,
    避免 '180' vs '180\\text{ miles}'、'30' vs '\\$30' 这种被误判为不等。
    """
    if s is None:
        return s
    s = str(s).strip()
    # 去掉 \text{...} / \mbox{...} (单位多写在这里面)
    s = re.sub(r"\\(?:text|mbox|mathrm|textrm)\s*\{[^{}]*\}", "", s)
    # 去掉货币符号 \$ $ ￥ 等
    s = s.replace(r"\$", "").replace("$", "").replace("￥", "").replace("%","")
    # 去掉 \! \, 等间距符
    s = re.sub(r"\\[!,;:]", "", s)
    # 去掉末尾的纯单位词(miles, dollars, cm, square centimeters 等英文单位)
    s = re.sub(r"\s*(square|cubic)?\s*[a-zA-Z]+\.?\s*$", lambda m: "" if not re.search(r"\d", m.group(0)) else m.group(0), s)
    return s.strip()


def is_correct(pred: str, gold: str) -> bool:
    """判 pred 与 gold 是否数学等价(默认开 timeout 防卡死)。
    v2: 入口先做排版/unicode归一 (_normalize_answer), 再按原流程比;
    不等时, 剥离单位/货币/\\text 后再比一次。
    """
    if pred is None or gold is None:
        return False
    p, g = _normalize_answer(pred), _normalize_answer(gold)
    # 1) 归一后按原样比
    if math_equal(p, g, timeout=True):
        return True
    # 2) 剥单位后再比(救 '180' vs '180 miles' 这类)
    ps, gs = _strip_units(p), _strip_units(g)
    if (ps != p or gs != g) and ps and gs:
        if math_equal(ps, gs, timeout=True):
            return True
    return False


# ===========================================================================
if __name__ == "__main__":
    print("verifier (Qwen-style math_equal) self-test  [v2]")
    print("=" * 60)

    extract_cases = [
        (r"所以面积是 \boxed{40} cm²", "40"),
        (r"得到 \boxed{\frac{1}{2}}", r"\frac{1}{2}"),
        (r"答案为 \boxed{\sqrt{2}}.", r"\sqrt{2}"),
        ("没有boxed,最后算出 42", "42"),
    ]
    print("extract_final_answer:")
    ep = 0
    for text, exp in extract_cases:
        got = extract_final_answer(text)
        ok = got == exp; ep += ok
        print(f"  [{'PASS' if ok else 'FAIL'}] -> {got} (expect {exp})")

    equiv = [
        ("40", "40"),
        ("1/2", "0.5"),
        (r"\frac{1}{2}", "0.5"),
        ("0.5", "0.50"),
        (r"\sqrt{2}", r"\sqrt{2}"),
        (r"\sqrt{2}", r"2^{1/2}"),
        ("2*3", "6"),
        (r"\frac{3}{6}", "0.5"),
        ("50\\%", "0.5"),
        ("x=3", "3"),
        # ---- v2 新增: 本轮修复的两类 ----
        ("3*sqrt(13)", r"3\sqrt{13}"),
        ("(3, pi/2)", r"\left( 3, \frac{\pi}{2} \right)"),
        ("(3, π/2)", r"\left( 3, \frac{\pi}{2} \right)"),
        ("2π", r"2\pi"),
    ]
    nonequiv = [
        ("40", "41"),
        ("1/2", "1/3"),
        (r"\sqrt{2}", r"\sqrt{3}"),
        ("6", "7"),
        # ---- v2 新增: 反例守卫 (确保修复没引入假阳性) ----
        ("2*sqrt(13)", r"3\sqrt{13}"),
        ("(3, pi/3)", r"\left( 3, \frac{\pi}{2} \right)"),
        ("(3, pi/2, 0)", r"\left( 3, \frac{\pi}{2} \right)"),
    ]
    print("\nequiv pairs (expect True):")
    cp = 0
    for a, b in equiv:
        got = is_correct(a, b); cp += got
        print(f"  [{'PASS' if got else 'FAIL'}] {a} == {b} -> {got}")
    print("non-equiv pairs (expect False):")
    for a, b in nonequiv:
        got = is_correct(a, b); cp += (not got)
        print(f"  [{'PASS' if not got else 'FAIL'}] {a} != {b} -> {got}")

    total = len(extract_cases) + len(equiv) + len(nonequiv)
    print("-" * 60)
    print(f"passed {ep + cp}/{total}")