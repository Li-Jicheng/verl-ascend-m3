#!/usr/bin/env bash
# RL | GSM8K | FSDP engine | Ascend A2/A3 NPU

# Examples:
#   # RL on Ascend A2/A3 device
#   bash run_rl_qwen3_8b_npu.sh

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
TRAIN_DATA=${TRAIN_DATA:-"${HOME}/data/gsm8k_rl/train.parquet"}
TEST_DATA=${TEST_DATA:-"${HOME}/data/gsm8k_rl/test.parquet"}

# Model and output path
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-8B"}
SAVE_PATH=${SAVE_PATH:-"rl_outputs"}

# Training hyperparameters
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-16}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-512}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-512}
LR=${LR:-1e-6}
TP=${TP:-1}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-2}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.001}
ENTROPY_COEFF=${ENTROPY_COEFF:-0}
ROLLOUT_N=${ROLLOUT_N:-5}
SAVE_FREQ=${SAVE_FREQ:-100}

# Other
PROJECT_NAME=${PROJECT_NAME:-"rl-gsm8k"}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"qwen3_8b_rl"}
# ====================================================================

# Run RL training
python3 -m verl.trainer.main_ppo \
      algorithm.adv_estimator=grpo \
      data.train_files="${TRAIN_DATA}" \
      data.val_files="${TEST_DATA}" \
      data.train_batch_size=${TRAIN_BATCH_SIZE} \
      data.max_prompt_length=${MAX_PROMPT_LENGTH} \
      data.max_response_length=${MAX_RESPONSE_LENGTH} \
      data.filter_overlong_prompts=False \
      data.truncation='error' \
      data.trust_remote_code=True \
      actor_rollout_ref.model.path=${MODEL_PATH} \
      actor_rollout_ref.model.trust_remote_code=True \
      actor_rollout_ref.actor.optim.lr=${LR} \
      actor_rollout_ref.model.use_remove_padding=True \
      actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
      actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU} \
      actor_rollout_ref.actor.use_kl_loss=True \
      actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF} \
      actor_rollout_ref.actor.kl_loss_type=low_var_kl \
      actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF} \
      actor_rollout_ref.actor.use_torch_compile=False \
      actor_rollout_ref.ref.use_torch_compile=False \
      actor_rollout_ref.model.enable_gradient_checkpointing=True \
      actor_rollout_ref.actor.fsdp_config.param_offload=False \
      actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
      actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU} \
      actor_rollout_ref.rollout.tensor_model_parallel_size=${TP} \
      actor_rollout_ref.rollout.name=vllm \
      actor_rollout_ref.rollout.load_format=safetensors \
      actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
      actor_rollout_ref.rollout.n=${ROLLOUT_N} \
      actor_rollout_ref.rollout.enforce_eager=True \
      actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
      actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
      actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU} \
      actor_rollout_ref.ref.fsdp_config.param_offload=True \
      algorithm.use_kl_in_reward=False \
      trainer.critic_warmup=0 \
      trainer.logger='["console"]' \
      trainer.project_name=${PROJECT_NAME} \
      trainer.experiment_name=${EXPERIMENT_NAME} \
      trainer.n_gpus_per_node=${NPROC_PER_NODE} \
      trainer.nnodes=${NNODES} \
      trainer.default_local_dir=${SAVE_PATH} \
      trainer.resume_mode=auto \
      actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
      actor_rollout_ref.ref.fsdp_config.forward_prefetch=True \
      ++actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
      ++actor_rollout_ref.ref.entropy_from_logits_with_chunking=True \
      trainer.val_before_train=False \
      trainer.save_freq=${SAVE_FREQ} \
      trainer.test_freq=-1 \
      trainer.total_epochs=${TOTAL_EPOCHS}

