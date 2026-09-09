#!/usr/bin/env bash
# run_eval_parallel.sh — 数据并行评测: N张卡各自独立打一份模型, 各管1/N的题,
# 跑完自动合并出最终报告。用于 tp 被模型结构(如 num_key_value_heads)卡住、
# 或张量并行通信开销让多卡反而不划算的情况 —— 这种"海量短请求批量打分"负载,
# 数据并行通常比张量并行快得多。
#
# 用法: bash run_eval_parallel.sh <mode> <prm_or_-> <traj> <problems> <out> [num_shards]
#   mode: ours / scalar / skywork  (majority 不用这个脚本, 直接单进程跑, 不吃GPU)
#   prm_or_-: PRM 权重路径
#   num_shards: 默认 4 (对应你能用的卡数)
#
# v5双分数模型(--dual_score)等新参数, 通过 EXTRA_ARGS 环境变量透传给每个分片
# 进程和最后的合并步骤, 不需要改这个脚本本身:
#   EXTRA_ARGS="--dual_score" bash run_eval_parallel.sh ours /path/prm-merged-v5 \
#       runs/eval/trajectories.jsonl /path/test.jsonl runs/eval/bon_v5.json 8
#
# 示例(旧版单分数模型):
#   bash run_eval_parallel.sh ours /path/prm-merged-v2 runs/eval/trajectories.jsonl \
#       /path/test.jsonl runs/eval/bon_ours_v2.json 4
set -e

MODE="${1:?mode: ours/scalar/skywork}"
PRM="${2:?prm权重路径}"
TRAJ="${3:?trajectories.jsonl路径}"
PROBLEMS="${4:?problems文件路径}"
OUT="${5:?输出文件路径}"
NUM_SHARDS="${6:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-12288}"
MAX_GEN="${MAX_GEN:-512}"
EXTRA_ARGS="${EXTRA_ARGS:-}"   # 例如 "--dual_score" 或 "--dual_score --plan_combine multiply"

mkdir -p "$(dirname "$OUT")"

echo "============================================================"
echo "数据并行评测: mode=$MODE, $NUM_SHARDS 张卡各独立打分 1/$NUM_SHARDS 的题"
echo "============================================================"

echo "  清理同名旧分片文件 (避免和这次的新分片混合误合并)..."
rm -f "${OUT}".shard*.json "${OUT}".shard*.log
echo ""

PIDS=()
for ((i=0; i<NUM_SHARDS; i++)); do
    echo "  启动分片 $i -> GPU $i"
    CUDA_VISIBLE_DEVICES=$i python eval_bon.py \
        --mode "$MODE" --prm "$PRM" \
        --traj "$TRAJ" --problems "$PROBLEMS" \
        --out "$OUT" \
        --tp 1 --num_shards "$NUM_SHARDS" --shard_id "$i" \
        --max_model_len "$MAX_MODEL_LEN" --max_gen "$MAX_GEN" \
        $EXTRA_ARGS \
        > "${OUT}.shard${i}.log" 2>&1 &
    PIDS+=($!)
done

echo "  ${NUM_SHARDS} 个分片进程已启动 (PID: ${PIDS[*]}), 等待全部完成..."
echo "  实时看进度: tail -f ${OUT}.shard0.log  (换数字看其他分片)"
FAIL=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || FAIL=1
done
if [ "$FAIL" -ne 0 ]; then
    echo "  ⚠ 至少一个分片进程出错, 检查 ${OUT}.shard*.log"
    exit 1
fi
echo "  全部分片完成。"

echo ""
echo "============================================================"
echo "合并分片结果, 出最终报告"
echo "============================================================"
python eval_bon.py --mode merge \
    --shard_glob "${OUT}.shard*.json" \
    --traj "$TRAJ" --problems "$PROBLEMS" \
    --out "$OUT" \
    $EXTRA_ARGS

echo ""
echo "完成. 最终报告: $OUT"
