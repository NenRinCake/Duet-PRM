"""
p3_adherence.py — adherence 维度收口: P3 兜底 + 规则结果合并

定位: B1/B2 规则已在 label_minimal.py 拍板了 ~69% 的步; 本脚本只处理剩下的
uncertain 池 (规则拿不准的), 用 P3 prompt 让 LLM 判忠实度 —— 只判 "做的是不是
声称的事", 不判对错 (对错是 C 层的事, 维度不许混)。

流程:
  1. 读 step_labels.jsonl, 取 b2 == "uncertain" 的步;
  2. 回连 trajectories.jsonl 拿上下文 (plan环节desc/action, 步的thought/code/输出);
  3. P3 prompt 批量判定, 解析 JSON {verdict, evidence, actually_doing};
  4. 合并: 规则拍板的步保留规则结论 (source=rule), uncertain 步用 LLM 结论
     (source=llm), 写出全量 adherence_final.jsonl;
  5. 报告: 最终 adherence 分布 / LLM 占比 / 解析失败数; 抽样 audit 清单
     (deviated 优先) 供人工核一致率。

用法 (单卡):
  CUDA_VISIBLE_DEVICES=0 python p3_adherence.py \
      --model /path/to/Qwen3-8B \
      --traj runs/v0/trajectories.jsonl \
      --labels runs/v0/step_labels.jsonl
"""
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse, json, re, textwrap
from collections import Counter

# --------------- P3 prompt (与 factorized_prm_prompts_v0 一致) ---------------
P3_ADHERENCE_JUDGE = """You are auditing whether an execution step actually carries out the plan stage it claims to execute. Judge ONLY whether the step's actual behavior matches the claimed stage. Do NOT judge whether the step is mathematically correct.

Plan stage claimed:
  id: {pid}
  description: {pid_desc}
  expected action type: {pid_action}

The step:
  thought: {thought}
  code:
{code}
  execution output:
{exec_output}

Question: does the code's actual behavior carry out the claimed stage?
- faithful: the code does what the stage describes.
- partial: the code does part of the stage, or mixes it with another stage's work.
- deviated: the code does something different from the claimed stage.

Output ONLY this JSON:
{{"verdict": "faithful|partial|deviated",
  "evidence": "<quote the specific code fragment or output line that supports your verdict>",
  "actually_doing": "<one short phrase: what the code is really doing>"}}"""

THOUGHT_RE = re.compile(r"thought:\s*(.+?)(?:\n```|\n<|\Z)", re.DOTALL)
JSON_RE    = re.compile(r"\{.*\}", re.DOTALL)

def extract_thought(raw):
    m = THOUGHT_RE.search(raw or "")
    return m.group(1).strip()[:500] if m else "(no thought line)"

def parse_p3(txt):
    """解析 P3 的 JSON 输出; 失败返回 None。verdict 必须是三值之一。"""
    m = JSON_RE.search(txt or "")
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
        v = str(d.get("verdict", "")).strip().lower()
        if v not in ("faithful", "partial", "deviated"):
            return None
        return {"verdict": v,
                "evidence": str(d.get("evidence", ""))[:300],
                "actually_doing": str(d.get("actually_doing", ""))[:200]}
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--traj", default="runs/v0/trajectories.jsonl")
    ap.add_argument("--labels", default="runs/v0/step_labels.jsonl")
    ap.add_argument("--out", default="runs/v0")
    ap.add_argument("--temp", type=float, default=0.2,
                    help="判定任务用低温求稳")
    ap.add_argument("--max_model_len", type=int, default=16384)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    trajs = {(t["prob_idx"], t["plan_idx"], t["exec_idx"]): t
             for t in (json.loads(l) for l in open(args.traj))}
    labels = [json.loads(l) for l in open(args.labels)]
    todo = [r for r in labels if r["b2"] == "uncertain"]
    print(f"steps total: {len(labels)} | rule-decided: {len(labels)-len(todo)}"
          f" | uncertain -> P3: {len(todo)}")

    # ---------------- P3 批量判定 ----------------
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(model=args.model, tensor_parallel_size=args.tp,
              max_model_len=args.max_model_len, gpu_memory_utilization=0.90,
              trust_remote_code=True, seed=args.seed)

    def chat(msg):
        return tok.apply_chat_template(
            [{"role": "user", "content": msg}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)

    prompts = []
    for r in todo:
        t = trajs[(r["prob_idx"], r["plan_idx"], r["exec_idx"])]
        st = t["steps"][r["step"]]
        stage = next((p for p in t["plan"] if p["id"] == st["claimed"]),
                     {"id": st["claimed"], "desc": "(stage not in plan)",
                      "action": "?"})
        prompts.append(P3_ADHERENCE_JUDGE.format(
            pid=stage["id"], pid_desc=stage["desc"],
            pid_action=stage.get("action", "?"),
            thought=extract_thought(st.get("raw", "")),
            code=textwrap.indent(st["code"], "    "),
            exec_output=textwrap.indent(st["exec"], "    ")))

    sp = SamplingParams(temperature=args.temp, top_p=0.95, max_tokens=400,
                        seed=args.seed)
    outs = llm.generate([chat(p) for p in prompts], sp)

    parse_fail = 0
    for r, o in zip(todo, outs):
        j = parse_p3(o.outputs[0].text)
        if j is None:
            parse_fail += 1
            r["p3"] = {"verdict": "uncertain", "evidence": "",
                       "actually_doing": "(P3 parse failed)"}
        else:
            r["p3"] = j

    # ---------------- 合并: 规则优先, LLM 只填空 ----------------
    final = Counter(); by_source = Counter()
    fpath = os.path.join(args.out, "adherence_final.jsonl")
    with open(fpath, "w") as f:
        for r in labels:
            if r["b2"] != "uncertain":
                r["adherence"], r["adherence_source"] = r["b2"], "rule"
            else:
                r["adherence"] = r["p3"]["verdict"]
                r["adherence_source"] = "llm"
            final[r["adherence"]] += 1
            by_source[r["adherence_source"]] += 1
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---------------- audit 抽样 (deviated/partial 优先) ----------------
    apath = os.path.join(args.out, "p3_audit.md")
    judged = [r for r in todo if r["p3"]["verdict"] != "uncertain"]
    judged.sort(key=lambda r: {"deviated": 0, "partial": 1,
                               "faithful": 2}[r["p3"]["verdict"]])
    with open(apath, "w") as f:
        f.write("# P3 人工核对清单 (核 verdict 与 evidence 是否一致, 建议核20条)\n\n")
        for i, r in enumerate(judged[:30]):
            t = trajs[(r["prob_idx"], r["plan_idx"], r["exec_idx"])]
            st = t["steps"][r["step"]]
            f.write(f"---\n## #{i+1} [{r['p3']['verdict']}] "
                    f"prob{r['prob_idx']}/plan{r['plan_idx']}"
                    f"/exec{r['exec_idx']}/step{r['step']} "
                    f"(claimed {st['claimed']})\n\n")
            f.write(f"**LLM认为实际在做**: {r['p3']['actually_doing']}\n\n")
            f.write(f"**证据**: `{r['p3']['evidence']}`\n\n")
            f.write("**代码**:\n```python\n" + st["code"] + "\n```\n")
            f.write("人工判定: [ ] 对  [ ] 错\n\n")

    n = len(labels)
    print("\n========== ADHERENCE FINAL ==========")
    print(f"  分布: {dict(final)}")
    print(f"  来源: rule {by_source['rule']}/{n} "
          f"= {by_source['rule']/n*100:.0f}% | llm {by_source['llm']}/{n} "
          f"= {by_source['llm']/n*100:.0f}%   (P3 解析失败: {parse_fail})")
    neg = final["deviated"] + final["claimed_only"] + final["partial"]
    print(f"  负信号存量 (deviated+claimed_only+partial): {neg}/{n}")
    print(f"  saved -> {fpath}")
    print(f"  audit -> {apath}")

if __name__ == "__main__":
    main()
