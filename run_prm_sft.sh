#!/usr/bin/env bash
# run_prm_sft.sh — 一键: 校验数据 -> 注册到 LLaMA-Factory -> 启动 8卡 SFT
# 用法: bash run_prm_sft.sh /path/to/Duet-PRM/LLaMA-Factory
# 必填环境变量: SFT_DATA、MODEL、OUTPUT_DIR
set -euo pipefail

LF_DIR="${1:?用法: bash run_prm_sft.sh <LLaMA-Factory根目录>}"
SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_DIR="${SCRIPT_PATH%/*}"
if [[ "$SCRIPT_DIR" == "$SCRIPT_PATH" ]]; then
  SCRIPT_DIR=.
fi
SCRIPT_DIR="$(cd "$SCRIPT_DIR" && pwd)"
SFT_DATA="${SFT_DATA:-/path/to/sft_dataset.jsonl}"
MODEL="${MODEL:-/path/to/Qwen2.5-Math-7B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-/path/to/output_dir}"

for path in "$LF_DIR" "$SFT_DATA" "$MODEL" "$OUTPUT_DIR"; do
  if [[ "$path" == /path/to/* ]]; then
    echo "错误: 请把占位路径替换为实际路径: $path" >&2
    exit 2
  fi
done

python - "$OUTPUT_DIR" <<'PY'
import os, sys
output_dir = sys.argv[1]
if os.path.isdir(output_dir) and any(os.scandir(output_dir)):
    raise SystemExit(
        f"错误: OUTPUT_DIR 已存在且不是空目录，拒绝混入旧 checkpoint: {output_dir}\n"
        "请指定一个新的空目录，从头开始训练。"
    )
PY

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
info["factorized_prm"]={
    "file_name":"$SFT_DATA",
    "formatting":"alpaca",
    "columns":{"prompt":"instruction","query":"input","response":"output"}
}
json.dump(info, open(info_path,"w"), ensure_ascii=False, indent=2)
print("  已注册 factorized_prm -> $SFT_DATA")
PY

echo "==> [4/4] 启动 8 卡 SFT"
cd "$LF_DIR"
TRAIN_OVERRIDES=(
  "model_name_or_path=$MODEL"
  "deepspeed=$LF_DIR/examples/deepspeed/ds_z3_config.json"
  "output_dir=$OUTPUT_DIR"
)
FORCE_TORCHRUN=1 NPROC_PER_NODE=8 \
  llamafactory-cli train "$SCRIPT_DIR/prm_sft.yaml" "${TRAIN_OVERRIDES[@]}"

echo "==> 完成. 权重在 $OUTPUT_DIR"
