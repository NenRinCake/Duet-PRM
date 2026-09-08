#!/usr/bin/env bash
# run_prm_sft.sh — 一键: 校验数据 -> 注册到 LLaMA-Factory -> 启动 8卡 SFT
# 用法: bash run_prm_sft.sh /public/home/ljt/lzm/code/PRM/LLaMA-Factory
set -e

LF_DIR="${1:?用法: bash run_prm_sft.sh <LLaMA-Factory根目录>}"
SFT_DATA="/public/home/ljt/lzm/code/PRM/runs/train_data_test/sft_dataset_5981q.jsonl"
MODEL="/public/home/ljt/lzm/model/Qwen2.5-Math-7B-Instruct"

echo "$SFT_DATA"
echo "==> [1/4] 校验数据集"
python - <<PY
import json
n=0; bad=0
for l in open("$SFT_DATA"):
    r=json.loads(l); n+=1
    if not all(k in r for k in ("instruction","input","output")): bad+=1
print(f"  样本数: {n}, 缺字段: {bad}")
assert bad==0, "存在缺字段样本"
PY

echo "==> [2/4] token 长度速查 (确认 cutoff_len 够用)"
# python - <<PY
# import json
# try:
#     from transformers import AutoTokenizer
#     tok=AutoTokenizer.from_pretrained("$MODEL", trust_remote_code=True)
#     L=[len(tok(r["instruction"]+r["input"]+r["output"])["input_ids"])
#        for r in map(json.loads, open("$SFT_DATA"))]
#     L.sort(); n=len(L)
#     print(f"  min {L[0]} | median {L[n//2]} | p95 {L[int(n*0.95)]} | max {L[-1]}")
#     over=sum(x>8192 for x in L)
#     print(f"  超过 8192 的样本: {over}/{n}" + ("  <-- 考虑增大 cutoff_len 或截断历史" if over else "  (OK)"))
# except Exception as e:
#     print("  跳过 token 统计:", e)
# PY

echo "==> [3/4] 注册数据集到 $LF_DIR/data/dataset_info.json"
python - <<PY
import json, os
info_path="$LF_DIR/data/dataset_info.json"
info=json.load(open(info_path))
info["factorized_prm_500k"]={
    "file_name":"$SFT_DATA",
    "formatting":"alpaca",
    "columns":{"prompt":"instruction","query":"input","response":"output"}
}
json.dump(info, open(info_path,"w"), ensure_ascii=False, indent=2)
print("  已注册 factorized_prm_500k -> $SFT_DATA")
PY

echo "==> [4/4] 启动 8 卡 SFT"
cd "$LF_DIR"
FORCE_TORCHRUN=1 NPROC_PER_NODE=8 \
  llamafactory-cli train /public/home/ljt/lzm/code/PRM/prm_sft.yaml

echo "==> 完成. 权重在 saves/qwen25math7b-factorized-prm-700q"