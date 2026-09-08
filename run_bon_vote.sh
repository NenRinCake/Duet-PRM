#!/bin/bash
# 多卡数据并行跑 bon_vote.py: 每张卡一个 shard 独立加载模型各跑各的, 全跑完再 merge。
# 用法:
#   bash run_bon_vote.sh <mode> <PRM路径> <候选轨迹jsonl> <problems jsonl> <score_scheme> <输出前缀> [卡数]
#   mode = ours | scalar | skywork
# 例:
#   bash run_bon_vote.sh ours    /path/PRM   runs/bon/cand.jsonl data/olympiad.jsonl v6 runs/bon/ours    8
#   bash run_bon_vote.sh scalar  /path/Qwen  runs/bon/cand.jsonl data/olympiad.jsonl v6 runs/bon/qwen    8
#   bash run_bon_vote.sh skywork /path/Sky   runs/bon/cand.jsonl data/olympiad.jsonl v6 runs/bon/skywork 8
# (majority 不用 PRM、不用多卡, 直接:
#    python bon_vote.py --mode majority --traj <cand> --problems <prob> --out runs/bon/majority.json )

set -e
MODE=$1
PRM=$2
TRAJ=$3
PROBLEMS=$4
SCHEME=${5:-v6}
OUT=${6:-runs/bon/report}
NGPU=${7:-8}

mkdir -p "$(dirname "$OUT")"

echo ">>> mode=$MODE, 启动 $NGPU 个分片 (每卡一个, tp=1)..."
pids=()
for i in $(seq 0 $((NGPU-1))); do
    CUDA_VISIBLE_DEVICES=$i python -u bon_vote.py \
        --mode "$MODE" --traj "$TRAJ" --problems "$PROBLEMS" \
        --prm "$PRM" --score_scheme "$SCHEME" --combine multiply \
        --tp 1 --num_shards "$NGPU" --shard_id "$i" \
        --out "${OUT}.json" > "${OUT}.shard${i}.log" 2>&1 &
    pids+=($!)
    echo "  shard $i -> GPU $i (pid ${pids[-1]}, log ${OUT}.shard${i}.log)"
done

echo ">>> 等待所有分片完成..."
fail=0
for p in "${pids[@]}"; do
    if ! wait "$p"; then echo "  ⚠ pid $p 失败"; fail=1; fi
done
if [ $fail -ne 0 ]; then
    echo "有分片失败, 检查 ${OUT}.shard*.log"; exit 1
fi

echo ">>> 合并分片出报告..."
python -u bon_vote.py --mode merge --traj "$TRAJ" \
    --combine multiply \
    --shard_glob "${OUT}.json.shard*.json" --out "${OUT}.json"

echo ">>> 完成: ${OUT}.json"
