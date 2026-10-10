#!/usr/bin/env bash
# SFT | GSM8K | FSDP engine | Ascend A2/A3 NPU

# Examples:
#   # SFT on Ascend A2/A3 device
#   bash run_sft_qwen3_0_6b_npu.sh

set -x

# ================== Hardware configuration setting ==================
## For Ascend A2
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NNODES=${NNODES:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
## For Ascend A3
#export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
#NNODES=${NNODES:-1}
#NPROC_PER_NODE=${NPROC_PER_NODE:-16}

MASTER_PORT=$(shuf -i 20000-65535 -n 1)
# ====================================================================


# ==================== User adjustable parameters ====================
# Data path
TRAIN_DATA=${TRAIN_DATA:-"${HOME}/data/gsm8k_sft/train.parquet"}
TEST_DATA=${TEST_DATA:-"${HOME}/data/gsm8k_sft/test.parquet"}
# Model and output path
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-0.6B"}
SAVE_PATH=${SAVE_PATH:-"sft_outputs"}
# Training hyperparameters
SP_SIZE=${SP_SIZE:-1}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
MICRO_BATCH_SIZE_PER_GPU=${MICRO_BATCH_SIZE_PER_GPU:-1}
LR=${LR:-3e-5}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-2}
MAX_LENGTH=${MAX_LENGTH:-4096}
# Other
PROJECT_NAME=${PROJECT_NAME:-"sft-gsm8k"}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"qwen3_0.6b_sft"}
# ====================================================================

# Run SFT training
torchrun --nnodes=${NNODES} --nproc_per_node=${NPROC_PER_NODE} --master_port=${MASTER_PORT} \
  -m verl.trainer.sft_trainer \
  data.train_files=${TRAIN_DATA} \
  data.val_files=${TEST_DATA} \
  data.train_batch_size=${TRAIN_BATCH_SIZE} \
  data.truncation=right \
  data.max_length=${MAX_LENGTH} \
  data.micro_batch_size_per_gpu=${MICRO_BATCH_SIZE_PER_GPU} \
  data.ignore_input_ids_mismatch=True \
  optim.lr=${LR} \
  model.path="${MODEL_PATH}" \
  model.trust_remote_code=True \
  model.use_remove_padding=True \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.total_epochs=${TOTAL_EPOCHS} \
  trainer.default_local_dir="${SAVE_PATH}" \
  trainer.logger='["console"]' \
  data.use_dynamic_bsz=False \
  engine.ulysses_sequence_parallel_size=${SP_SIZE}
