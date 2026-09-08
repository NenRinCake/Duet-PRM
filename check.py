# """
# within_problem_check.py — 验证"GOOD占比 vs 对错"这个相关性, 是不是被"题目难度"
# 这个混淆变量制造出来的假象。

# 用户的原始脚本把所有题目的所有候选混在一起算GOOD占比, 再按对/错分组比均值 ——
# 这个统计方式没有控制"题目难度"这个变量: 如果难题整体GOOD占比低、正确率也低,
# 简单题整体GOOD占比高、正确率也高, 混着算会得出"GOOD占比跟对错强相关"的假象,
# 但这个相关性可能完全来自题目难度本身, 跟"step_score能不能在同一题的候选里
# 挑出更好的那条"(BoN真正需要的能力)毫无关系。

# 这个脚本改成"题目内部比较": 对每道题, 分别算它自己的correct候选和wrong候选
# 的GOOD占比均值, 取差值(这一步控制掉了题目难度, 因为同一题自己跟自己比);
# 再把所有题目的这个差值取平均。

# 用法:
#   python within_problem_check.py --traj runs/eval/trajectories.jsonl \
#       --shard_glob "runs/eval/bon_v6plus250.json.shard*.json" --label v6plus250
# """
# import argparse, json, glob
# from collections import defaultdict

# def load_by_prob(traj_path):
#     trajs = [json.loads(l) for l in open(traj_path)]
#     by_prob = defaultdict(list)
#     for t in trajs:
#         by_prob[t["prob_idx"]].append(t)
#     return by_prob

# def load_step_score(shard_glob):
#     step_score = {}
#     for fp in glob.glob(shard_glob):
#         obj = json.load(open(fp))
#         step_score.update(obj.get("step_score", {}))
#     return step_score

# def analyze(traj_path, shard_glob, label):
#     by_prob = load_by_prob(traj_path)
#     step_score = load_step_score(shard_glob)

#     pooled_correct, pooled_wrong = [], []
#     per_problem_gaps = []
#     n_no_contrast = 0

#     for prob_idx, cand in by_prob.items():
#         good_correct, good_wrong = [], []
#         for tpos, t in enumerate(cand):
#             key = f"{prob_idx}:{tpos}"
#             steps = step_score.get(key)
#             if not steps:
#                 continue
#             vals = list(steps.values())
#             good_frac = sum(v == 1.0 for v in vals) / len(vals)
#             (good_correct if t.get("correct") else good_wrong).append(good_frac)
#             (pooled_correct if t.get("correct") else pooled_wrong).append(good_frac)
#         if good_correct and good_wrong:
#             gap = sum(good_correct)/len(good_correct) - sum(good_wrong)/len(good_wrong)
#             per_problem_gaps.append(gap)
#         else:
#             n_no_contrast += 1   # 这道题8条候选全对或全错, 题目内部没有对照, 跳过

#     pooled_gap = (sum(pooled_correct)/len(pooled_correct) if pooled_correct else 0) \
#                - (sum(pooled_wrong)/len(pooled_wrong) if pooled_wrong else 0)
#     within_gap = sum(per_problem_gaps) / len(per_problem_gaps) if per_problem_gaps else None

#     print(f"[{label}]")
#     print(f"  混着算(用户原脚本那种, 跨题目pool): GOOD占比差距(对-错) = {pooled_gap:.3f}")
#     print(f"  题目内部比(控制掉难度这个混淆变量): GOOD占比差距(对-错) = "
#           f"{within_gap:.3f}  (基于{len(per_problem_gaps)}道题有内部对照, "
#           f"{n_no_contrast}道题8条候选全对或全错被跳过)")
#     return pooled_gap, within_gap

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--traj", required=True)
#     ap.add_argument("--shard_glob", required=True)
#     ap.add_argument("--label", default="model")
#     args = ap.parse_args()
#     analyze(args.traj, args.shard_glob, args.label)

# if __name__ == "__main__":
#     main()

# 单独存成 diag.py 跑, 看一条真实链的 reward 数
import torch, json, sys
sys.path.insert(0, "/public/home/ljt/lzm/code/PRM/skywork-o1-prm-inference-main")
from transformers import AutoTokenizer
from model_utils.prm_model import PRM_MODEL
from model_utils.io_utils import prepare_input, derive_step_rewards

prm = "/public/home/ljt/lzm/model/Skywork-o1-Open-PRM-Qwen-2.5-7B"
tok = AutoTokenizer.from_pretrained(prm, trust_remote_code=True)
model = PRM_MODEL.from_pretrained(prm, device_map="auto", torch_dtype=torch.bfloat16).eval()

# 取一条真实轨迹
t = None
for l in open("/public/home/ljt/lzm/code/PRM/runs/Separation/math500/trajectories.jsonl"):
    tt = json.loads(l)
    if len(tt["steps"]) >= 3:   # 找一条多步的
        t = tt; break
print("这条链 step 数:", len(t["steps"]))

# 看每步内容里有没有 \n\n
for i, s in enumerate(t["steps"]):
    body = f"[{s['claimed']}] {s['code']}\n{s['exec']}"
    print(f"  step{i}: 内部含\\n\\n? {'是' if chr(10)+chr(10) in body else '否'}  exec前80字符: {repr(s['exec'][:80])}")

# 用原始拼法(不clean)看 reward 数
resp1 = "\n\n".join(f"[{s['claimed']}] {s['code']}\n{s['exec']}" for s in t["steps"])
ids, _, rf = prepare_input(t["plan"][0]["desc"], resp1, tokenizer=tok, step_token="\n\n")
device = next(model.parameters()).device
ids = torch.tensor([ids]).to(device)
with torch.no_grad():
    _,_,r = model(input_ids=ids, attention_mask=torch.ones_like(ids), return_probs=True)
rf_tensor = torch.tensor(rf).unsqueeze(0).to(r.device)
print("原始拼法 reward 数:", len(derive_step_rewards(r, rf_tensor)[0]))