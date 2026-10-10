#!/usr/bin/env bash
# GRPO | Qwen3-1.7B | VeOmni quick test | Ascend NPU

set -x
set -e

FSDP_SIZE=${FSDP_SIZE:-4}
SP_SIZE=${SP_SIZE:-1}
EP_SIZE=${EP_SIZE:-1}
NUM_GPUS=${NUM_GPUS:-16}

TRAIN_FILE=/mnt/share/l00991686/m3_env_setup/dapo-math-17k.parquet
TEST_FILE=/mnt/share/l00991686/m3_env_setup/aime_2024_problems.parquet
LOG_DIR=/mnt/share/l00991686/.logs
MODEL_PATH=/mnt/share/l00991686/m3_env_setup/Qwen3-1.7B

max_prompt_length=512
max_response_length=1024


DATA=(
    algorithm.adv_estimator=grpo
    algorithm.kl_ctrl.kl_coef=0.001

    data.train_files="${TRAIN_FILE}"
    data.val_files="${TEST_FILE}"

    # quick test
    data.train_batch_size=8
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=False
    data.truncation='error'
)


MODEL=(
    actor_rollout_ref.model.path=$MODEL_PATH
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)


ACTOR=(
    actor_rollout_ref.actor.optim.lr=5e-7
    actor_rollout_ref.actor.veomni.param_offload=True
    actor_rollout_ref.actor.veomni.optimizer_offload=True
    actor_rollout_ref.actor.ppo_mini_batch_size=8
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.001
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.actor.veomni.fsdp_size="${FSDP_SIZE}"
    actor_rollout_ref.actor.veomni.ulysses_parallel_size="${SP_SIZE}"
    actor_rollout_ref.actor.veomni.expert_parallel_size="${EP_SIZE}"
)


ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.max_num_batched_tokens=512
    actor_rollout_ref.rollout.tensor_model_parallel_size=2
    actor_rollout_ref.rollout.enable_chunked_prefill=False
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.enforce_eager=False

    # GRPO samples
    actor_rollout_ref.rollout.n=2
)


REF=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.veomni.param_offload=True
    actor_rollout_ref.ref.use_torch_compile=False
)


TRAINER=(
    +trainer.use_legacy_worker_impl=disable

    trainer.critic_warmup=0

    trainer.logger=['console','tensorboard']

    trainer.default_local_dir="${LOG_DIR}"

    trainer.project_name='verl_veomni_quick_test'

    trainer.n_gpus_per_node="${NUM_GPUS}"
    trainer.nnodes=1

    trainer.save_freq=5

    trainer.test_freq=-1

    # quick run
    trainer.total_epochs=1
    trainer.total_training_steps=10
)


EXTRA=(
    model_engine=veomni
)


python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "$@"