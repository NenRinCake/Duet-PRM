"""
inspect_incomparable.py — 抽样看"value_kind可比性门"到底在拦截什么样的真实案例

动机: witnesses=1这个补丁效果远低于预期(8026条样本里只挪动了17条), 说明
no_consensus_ref偏高的真正原因更可能是stage_equal/value_kind把很多本该能比的
值判成了"不可比"(返回None), 不是票数不够。但具体是哪种值被错误拦住, 没有
真实样本看过, 不能瞎猜着去改这段逻辑 —— 这个脚本就是为了在动手改之前先看证据。

输出两类样本:
  [A] 两边value_kind都不是None(都长得像正常的可比较值), 但kind不一致(比如一个
      是tuple一个是list, 或者其中一个落进了"expr"类别) 被拒绝比较的案例
      —— 这类是最值得人工看一眼的: 可能是同一个数学量写法不同, 也可能真的是
      不同的东西, 需要肉眼判断
  [B] 至少一边value_kind本身就是None(叙述文字/不等式/超长文本等) 被拒绝比较的
      案例 —— 这类大概率是合理拒绝, 抽几条只是做个对照, 不是重点

用法:
  python inspect_incomparable.py --traj runs/v0_round2/trajectories.jsonl \
      --labels runs/v0_round2/step_labels.jsonl --n_samples 30
"""
import argparse, json, re
from collections import defaultdict

PID_RE = re.compile(r"^P\d+$")

def _tail(line):
    if ": " in line:
        return line.rsplit(": ", 1)[1].strip()
    if line.count("=") == 1:
        return line.split("=", 1)[1].strip()
    return line.strip()

def last_printed_value(exec_out):
    if not exec_out or exec_out in ("[no output]", "[timeout]") \
       or "Traceback" in exec_out:
        return None
    lines = [l for l in exec_out.strip().splitlines() if l.strip()]
    return _tail(lines[-1]) if lines else None

def norm_value(v):
    if v is None:
        return None
    v = v.strip()
    if len(v.split()) > 4:
        m = re.search(r"(-?\d+(?:\.\d+)?(?:/\d+)?)\s*(?:degrees?|°)?\s*\.?$", v)
        return m.group(1) if m else v
    return v

def value_kind(v):
    s = str(v).strip()
    if not s:
        return None
    if re.search(r"<=|>=|<|>|\.\.\.", s):
        return None
    if s.startswith("Eq("):
        return ("eq",)
    if (s[0], s[-1]) in (("(", ")"), ("[", "]")):
        return ("seq", s[0], s.count(","))
    if len(s.split()) > 4 or re.search(r"[A-Za-z]{12,}", s):
        return None
    if len(s) > 80:
        return None
    return ("expr",)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--n_samples", type=int, default=30)
    args = ap.parse_args()

    trajs = [json.loads(l) for l in open(args.traj)]
    labels = [json.loads(l) for l in open(args.labels)]
    lab = {(r["prob_idx"], r["plan_idx"], r["exec_idx"], r["step"]): r
           for r in labels}

    def adherence_of(key):
        r = lab.get(key, {})
        a = r.get("adherence") or r.get("b2", "")
        return "uncertain" if str(a).startswith("uncertain") else a

    stage_val = defaultdict(dict)
    for t in trajs:
        tk3 = (t["prob_idx"], t["plan_idx"])
        for i, s in enumerate(t["steps"]):
            pid = s["claimed"]
            if not PID_RE.match(pid):
                continue
            if adherence_of(tk3 + (t["exec_idx"], i)) != "faithful":
                continue
            v = norm_value(last_printed_value(s["exec"]))
            stage_val[tk3 + (pid,)][t["exec_idx"]] = v   # 这里先不做value_kind过滤,
                                                          # 保留None也记下来, 方便分类统计

    samples_A, samples_B = [], []
    n_pairs_checked = 0
    n_kind_mismatch = 0    # 两边都非None, 但kind不同 -> [A]类
    n_either_none = 0      # 至少一边是None -> [B]类
    n_same_kind = 0        # 两边kind相同(正常参与后续比较, 不算"被门拦住")

    for key, per_exec in stage_val.items():
        execs = sorted(v for v in per_exec if per_exec[v] is not None)
        # 两两比较该阶段下所有有值的兄弟执行
        for i in range(len(execs)):
            for j in range(i + 1, len(execs)):
                a, b = per_exec[execs[i]], per_exec[execs[j]]
                ka, kb = value_kind(a), value_kind(b)
                n_pairs_checked += 1
                if ka is None or kb is None:
                    n_either_none += 1
                    if len(samples_B) < args.n_samples:
                        samples_B.append((key, a, ka, b, kb))
                elif ka != kb:
                    n_kind_mismatch += 1
                    if len(samples_A) < args.n_samples:
                        samples_A.append((key, a, ka, b, kb))
                else:
                    n_same_kind += 1

    print(f"共检查 {n_pairs_checked} 对兄弟执行的(同阶段)取值比较")
    print(f"  两边kind相同(正常参与比较): {n_same_kind} "
          f"({n_same_kind/max(1,n_pairs_checked)*100:.1f}%)")
    print(f"  [A] 两边都非None但kind不同(被可比性门拒绝, 最值得看): {n_kind_mismatch} "
          f"({n_kind_mismatch/max(1,n_pairs_checked)*100:.1f}%)")
    print(f"  [B] 至少一边是None(叙述/不等式/超长文本等, 大概率合理拒绝): {n_either_none} "
          f"({n_either_none/max(1,n_pairs_checked)*100:.1f}%)")

    print(f"\n{'='*70}\n  [A] 抽样: kind不同被拒绝的真实案例 (最多{args.n_samples}条)\n{'='*70}")
    for key, a, ka, b, kb in samples_A:
        print(f"  prob={key[0]} plan={key[1]} stage={key[2]}")
        print(f"    值A: {a!r:<40} kind={ka}")
        print(f"    值B: {b!r:<40} kind={kb}")
        print()

    print(f"\n{'='*70}\n  [B] 抽样: 至少一边None被拒绝的真实案例 (最多{args.n_samples}条, 仅供对照)\n{'='*70}")
    for key, a, ka, b, kb in samples_B:
        print(f"  prob={key[0]} plan={key[1]} stage={key[2]}")
        print(f"    值A: {a!r:<40} kind={ka}")
        print(f"    值B: {b!r:<40} kind={kb}")
        print()

if __name__ == "__main__":
    main()
