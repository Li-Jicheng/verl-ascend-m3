export PYTHONPATH=$PYTHONPATH:/home/.../sft
export PYTHONPATH=$PYTHONPATH:/home/.../sft/verl
export WANDB_MODE=offline

# NPU 特定配置
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_RT_MEMORY_POOL_ALLOCATE_POLICY=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export VERL_SFT_LOGGING_LEVEL=DEBUG

export TORCH_HCCL_ZERO_COPY=0
export HCCL_BUFFSIZE=1024
export HCCL_EXEC_TIMEOUT=7200
export MULTI_STREAM_MEMORY_REUSE=2
export ENABLE_TASK_QUEUE=2

# ==========================================
# 打印确认信息（带颜色）
# ==========================================
# 获取颜色代码（可选，让关键信息更醒目）
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 获取当前 Rank（如果是分布式训练）
RANK=${RANK:-0}

echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}[Rank $RANK] HCCL 环境变量配置确认${NC}"
echo -e "${YELLOW}========================================${NC}"

# 检查并打印每个变量
vars=("TORCH_HCCL_ZERO_COPY" "HCCL_BUFFSIZE" "MULTI_STREAM_MEMORY_REUSE")
for var in "${vars[@]}"; do
    value=$(printenv "$var")
    if [ -z "$value" ]; then
        echo -e "[Rank $RANK] ${RED}✗ $var: 未设置${NC}"
    else
        echo -e "[Rank $RANK] ${GREEN}✓ $var: $value${NC}"
    fi
done

# 额外信息：显示当前主机名和HCCL相关环境
echo -e "[Rank $RANK] 主机名: $(hostname)"
echo -e "[Rank $RANK] HCCL 可用: $(if [ -n "$HCCL_INCOMING_HANDLE" ]; then echo "是"; else echo "未知"; fi)"
echo -e "${YELLOW}========================================${NC}"

nohup bash kernel/scripts/coldstart_30b.sh \
  --train_batch_size 64 \
  --micro_batch_size_per_gpu 1 \
  --max_length 65536 \
  --learning_rate 2e-5 \
  --total_epochs 4 \
  --dataset_name triton_sft_trajectories_8k \
  --train_data_path /home/.../triton_sft_trajectories_8k.parquet \
  --model_name Qwen3-30B-A3B-Thinking-2507 > ./Triton_sft_distill_v3sandbox_$(date +%Y%m%d_%H%M%S).log 2>&1 &