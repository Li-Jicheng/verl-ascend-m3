# Copyright 2024 Bytedance Ltd. and/or its affiliates
# bert_paddingpad_input
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A lightweight one-file FSDP SFT Trainer
TODO(zhangchi.usc1992)
- Add calculation of mfu
- Add validation
"""

import os

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import logging
import re
import time  # 在原有 imports 后添加
from contextlib import nullcontext

import hydra
import torch
import torch.distributed
import torch_npu

# from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
# try:
#     from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
# except ImportError:
#     # NPU or other devices may not have flash_attn
#     index_first_axis = None
#     pad_input = None
#     rearrange = None
#     unpad_input = None
from peft import LoraConfig, TaskType, get_peft_model
from tensordict import TensorDict
from torch import nn, optim
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import CPUOffload, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel
from verl_patch.trainer.code.constant import QWEN3CHATTEMPLATE, QWEN3CODERCHATTEMPLATE
from verl_patch.utils.dataset import SFTDataset

import verl.utils.hdfs_io as hdfs_io
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.device import get_device_id, get_device_name
from verl.utils.distributed import initialize_global_process_group
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import (
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    init_fn,
)
from verl.utils.torch_functional import (
    get_cosine_schedule_with_warmup,
    get_wsd_schedule_with_warmup,
)
from verl.utils.tracking import Tracking
from verl.utils.ulysses import (
    gather_outputs_and_unpad,
    get_ulysses_sequence_parallel_world_size,
    ulysses_pad_and_slice_inputs,
)
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager

from .npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input

# logger = logging.getLogger(__file__)
logger = logging.getLogger("SFTTrainer")
# logger.setLevel(os.getenv("VERL_SFT_LOGGING_LEVEL", "WARN"))
# print(f"🔍 Logger ID: {id(logger)}")
# print(f"🔍 Logger Level: {logger.level} ({logging.getLevelName(logger.level)})")
# print(f"🔍 Handlers 数量: {len(logger.handlers)}")
# for i, h in enumerate(logger.handlers):
#     print(f"   Handler {i}: {type(h).__name__} | Level: {h.level} | {getattr(h, 'baseFilename', 'N/A')}")

# # 测试写入
# logger.info("🧪 测试写入文件")


def extract_step(path):
    match = re.search(r"global_step_(\d+)", path)
    if match:
        return int(match.group(1))
    return None


def convert_to_regular_types(obj):
    """Convert Hydra configs and other special types to regular Python types."""
    from omegaconf import DictConfig, ListConfig

    if isinstance(obj, (ListConfig, DictConfig)):
        return {k: convert_to_regular_types(v) for k, v in obj.items()} if isinstance(obj, DictConfig) else list(obj)
    elif isinstance(obj, (list, tuple)):
        return [convert_to_regular_types(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: convert_to_regular_types(v) for k, v in obj.items()}
    return obj


class FSDPSFTTrainer:
    def __init__(
        self,
        config,
        device_mesh: DeviceMesh,
        ulysses_device_mesh: DeviceMesh,
        tokenizer,
        train_dataset: Dataset,
        val_dataset: Dataset,
    ):
        self.config = config
        self.device_mesh = device_mesh
        self.ulysses_device_mesh = ulysses_device_mesh
        self.sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)
        self.tokenizer = tokenizer
        if self.config.data.chat_template is not None:
            raise ValueError("Apply Chat template from config is not supported yet.")

        # normalize dp size
        self._normalize_config_bsz()

        # Set sequence parallel size
        self.config.ulysses_sequence_parallel_size = getattr(self.config, "ulysses_sequence_parallel_size", 1)
        self.use_remove_padding = getattr(self.config, "use_remove_padding", False)
        if self.device_mesh.get_rank() == 0:
            logger.info(f"Using sequence parallel size: {self.config.ulysses_sequence_parallel_size}")
            logger.info(f"Using remove padding: {self.use_remove_padding}")

        self._build_dataloader(train_dataset, val_dataset)
        # build model
        self._build_model_optimizer()

        # TODO: add checkpoint manager
        if self.device_mesh.get_rank() == 0:
            print(self.config)

    def _normalize_config_bsz(self):
        dp_size = self.device_mesh.size(0) if not self.ulysses_device_mesh else self.ulysses_device_mesh.size(0)
        if self.device_mesh.get_rank() == 0:
            print(f"Normalize batch size by dp {dp_size}")

        assert self.config.data.train_batch_size % dp_size == 0, (
            f"Global batch size {self.config.data.train_batch_size} is not divisible by dp size {dp_size}"
        )

        self.config.data.train_batch_size //= dp_size

        assert self.config.data.train_batch_size % self.config.data.micro_batch_size_per_gpu == 0

    def _build_dataloader(self, train_dataset, val_dataset):
        # build dataset
        config = self.config
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        # build dataloader
        # Use data parallel rank and size instead of global rank and world size_build_model_optimizer

        # If doing SP, we need to use the local rank and size
        if self.config.ulysses_sequence_parallel_size > 1:
            rank = self.ulysses_device_mesh.get_local_rank("dp")
            world_size = self.ulysses_device_mesh.size(0)
            if self.ulysses_device_mesh.get_rank() == 0:
                print(f"Using SP rank {rank} and size {world_size} for data distribution")
                print("Each SP rank gets different data, but the same data WITHIN the same rank")
        else:
            rank = self.device_mesh.get_rank()
            world_size = self.device_mesh.size()
        if self.device_mesh.get_rank() == 0:
            print(f"Using FSDP rank {rank} and size {world_size} for data distribution")

        self.train_sampler = DistributedSampler(
            self.train_dataset, shuffle=True, num_replicas=world_size, rank=rank, drop_last=True
        )
        self.train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=config.data.train_batch_size,
            sampler=self.train_sampler,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
        )

        # self.val_sampler = DistributedSampler(
        #     self.val_dataset, shuffle=False, num_replicas=world_size, rank=rank, drop_last=True
        # )
        # self.val_dataloader = DataLoader(
        #     dataset=self.val_dataset,
        #     batch_size=config.data.micro_batch_size_per_gpu,
        #     sampler=self.val_sampler,
        #     num_workers=8,
        #     pin_memory=True,
        #     drop_last=True,
        # )

    def _build_model_optimizer(self):
        # TODO (zhangchi.usc1992):
        # 1. support pretrain from random weights
        # 2. support init directly from sharded weights
        local_model_path = copy_to_local(src=self.config.model.partial_pretrain, verbose=True)

        if self.config.model.get("external_lib", None) is not None:
            # This is used to import external_lib into the huggingface systems
            import importlib

            importlib.import_module(self.config.model.external_lib)

        # log_gpu_memory_usage("Before model allocation", logger=logger)

        trust_remote_code = self.config.model.trust_remote_code
        # load config first
        config = AutoConfig.from_pretrained(local_model_path, trust_remote_code=trust_remote_code)
        if self.config.ulysses_sequence_parallel_size > 1:
            assert self.use_remove_padding, "Sequence parallel is only supported when remove_padding is enabled"

        # This may be very large
        init_context = get_init_weight_context_manager(
            use_meta_tensor=not config.tie_word_embeddings, mesh=self.device_mesh
        )

        with init_context():
            # attn_impl = "eager" if is_npu_available else "flash_attention_2"
            self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
                local_model_path,
                config=config,
                torch_dtype=torch.float32,
                attn_implementation="flash_attention_2",
                # attn_implementation=attn_impl,
                trust_remote_code=trust_remote_code,
            )

            if self.use_remove_padding or self.config.ulysses_sequence_parallel_size > 1:
                from verl.models.transformers.monkey_patch import apply_monkey_patch

                apply_monkey_patch(model=self.model, ulysses_sp_size=self.config.ulysses_sequence_parallel_size)

            # Apply Liger kernel if use_liger is enabled
            if self.config.model.get("use_liger", False):
                from liger_kernel.transformers.monkey_patch import (
                    _apply_liger_kernel_to_instance,
                )

                _apply_liger_kernel_to_instance(model=self.model)

            if self.config.model.get("lora_rank", 0) > 0:
                self.model.enable_input_require_grads()
                # Convert config to regular Python types before creating PEFT model
                lora_config = {
                    "task_type": TaskType.CAUSAL_LM,
                    "r": self.config.model.lora_rank,
                    "lora_alpha": self.config.model.lora_alpha,
                    "target_modules": convert_to_regular_types(self.config.model.target_modules),
                    "bias": "none",
                }
                self.model = get_peft_model(self.model, LoraConfig(**lora_config))

        if self.config.model.enable_gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        # log_gpu_memory_usage("After model allocation", logger=logger)
        if get_device_name() == "cuda":
            log_gpu_memory_usage("After model allocation", logger=logger)

        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32
        )

        auto_wrap_policy = get_fsdp_wrap_policy(
            self.model,
            config=self.config.model.fsdp_config.wrap_policy,
            is_lora=self.config.model.get("lora_rank", 0) > 0,
        )
        if self.device_mesh.get_rank() == 0:
            print(auto_wrap_policy)

        if not self.config.model.fsdp_config.cpu_offload:
            cpu_offload = None
        else:
            cpu_offload = CPUOffload(offload_params=self.config.model.fsdp_config.offload_params)

        self.fsdp_model = FSDP(
            module=self.model,
            auto_wrap_policy=auto_wrap_policy,
            param_init_fn=init_fn,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mixed_precision,
            device_mesh=self.device_mesh,
            sync_module_states=True,
            # device_id=torch.cuda.current_device(),
            device_id=get_device_id(),
            cpu_offload=cpu_offload,
            use_orig_params=False,
        )

        if self.device_mesh.get_rank() == 0:
            logger.info("✅ FSDP 包装完成")

        # log_gpu_memory_usage("After FSDP wrapping", logger=logger)
        if get_device_name() == "cuda":
            log_gpu_memory_usage("After FSDP wrapping", logger=logger)

        self.optimizer = optim.AdamW(
            self.fsdp_model.parameters(),
            lr=self.config.optim.lr,
            betas=self.config.optim.betas,
            weight_decay=self.config.optim.weight_decay,
        )

        # log_gpu_memory_usage("After initialize optimizer", logger=logger)
        if get_device_name() == "cuda":
            log_gpu_memory_usage("After initialize optimizer", logger=logger)

        self.steps_per_epoch = len(self.train_dataloader)
        self.total_steps = self.steps_per_epoch * self.config.trainer.total_epochs

        if self.device_mesh.get_rank() == 0:
            warmup_steps = int(self.total_steps * self.config.optim.warmup_steps_ratio)
            logger.info(f"✅ 优化器就绪: lr={self.config.optim.lr}, warmup_steps={warmup_steps}")

        if self.device_mesh.get_rank() == 0:
            print(
                f"Number of steps/epoch {self.steps_per_epoch}, number of epochs {self.config.trainer.total_epochs}, "
                f"total number of steps {self.total_steps}"
            )

        num_warmup_steps = int(self.total_steps * self.config.optim.warmup_steps_ratio)

        if not hasattr(self.config.optim, "lr_scheduler") or self.config.optim.lr_scheduler == "cosine":
            self.lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=self.total_steps
            )
        elif self.config.optim.lr_scheduler == "wsd":
            self.lr_scheduler = get_wsd_schedule_with_warmup(
                optimizer=self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=self.total_steps
            )
        else:
            raise ValueError(f"Unknown lr scheduler: {self.config.optim.lr_scheduler}")

    def _compute_loss_and_backward(self, batch, do_backward=True):
        """Compute loss with optional sequence parallelism and remove padding features"""
        use_sp = self.use_remove_padding and self.config.ulysses_sequence_parallel_size > 1

        # Move inputs to GPU and prepare loss mask
        # input_ids = batch["input_ids"].cuda()
        # attention_mask = batch["attention_mask"].cuda()
        # position_ids = batch["position_ids"].cuda()
        # loss_mask = batch.pop("loss_mask")[:, :-1].reshape(-1).cuda()
        device = get_device_id()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        position_ids = batch["position_ids"].to(device)
        loss_mask = batch.pop("loss_mask")[:, :-1].reshape(-1).to(device)
        loss_fct = nn.CrossEntropyLoss(reduction="none")

        # Context manager for sequence parallel if needed
        context = self.sharding_manager if use_sp else nullcontext()
        # with context, torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        with context, torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
            if not use_sp:
                # Standard forward pass without sequence parallel
                labels = input_ids[:, 1:].contiguous()
                output = self.fsdp_model(
                    input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False
                )
                logits = output.logits

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels.contiguous()
                # Flatten the tokens
                shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)
                loss = loss * loss_mask.to(loss.device)
            else:
                # IMPORTANT: We have a big assumption here, so we can shard the SAME sequence across SP ranks
                # i.e., each GPU has <1 sequence, and each SP group has 1 sequence
                # 1. All SP ranks will receive the *SAME* batch
                # 2. Different SP groups will receive *DIFFERENT* batches
                # This is implemented by the DistributedSampler

                batch_size, seqlen = input_ids.shape
                # Remove padding
                input_ids_rmpad, indices, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # Unpad position_ids to align rotary
                position_ids_rmpad = index_first_axis(
                    rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                ).transpose(0, 1)

                # Pad and slice inputs for sequence parallelism
                # ulysses_pad_and_slice_inputs已适配
                # get_ulysses_sequence_parallel_world_size已适配
                input_ids_rmpad_sliced, position_ids_rmpad_padded, pad_size = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad, position_ids_rmpad, sp_size=get_ulysses_sequence_parallel_world_size()
                )
                # For computing loss
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)
                input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad_rolled, None, get_ulysses_sequence_parallel_world_size()
                )
                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # Forward pass
                output = self.fsdp_model(
                    input_ids=input_ids_rmpad_sliced,
                    attention_mask=None,  # Not needed with flash attention varlen
                    position_ids=position_ids_rmpad_padded,
                    use_cache=False,
                )

                # logger.info(f"🔥 caculate output through fsdp_model")
                # Compute loss locally then aggregate
                logits_rmpad = output.logits.squeeze(0)
                input_ids_rmpad_rolled = input_ids_rmpad_rolled.to(logits_rmpad.device)
                loss = loss_fct(logits_rmpad, input_ids_rmpad_rolled)
                # Gather and unpad for sequence parallelism
                loss = gather_outputs_and_unpad(loss, gather_dim=0, unpad_dim=0, padding_size=pad_size)

                # This is the loss collected from all ulysses ranks
                full_loss = pad_input(
                    hidden_states=loss.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
                )
                full_loss = full_loss.squeeze(-1)[:, :-1]  # Remove last token's loss
                full_loss = full_loss.reshape(-1)
                loss_mask = loss_mask.to(full_loss.device)
                loss = full_loss * loss_mask

            valid_token_this_rank = torch.sum(loss_mask)

            if self.config.data.balance_dp_token:
                torch.distributed.all_reduce(valid_token_this_rank)
                dp_size = self.ulysses_device_mesh.size("dp") if use_sp else torch.distributed.get_world_size()
            else:
                dp_size = 1

            loss = torch.sum(loss) / (valid_token_this_rank + 1e-8) * dp_size

            if do_backward:
                loss.backward()
            return loss

    def training_step(self, batch: TensorDict):
        self.fsdp_model.train()

        # log_gpu_memory_usage("Before optimizer zero_grad", logger=logger)
        if get_device_name() == "cuda":
            log_gpu_memory_usage("Before optimizer zero_grad", logger=logger)

        if get_device_name() == "npu":
            torch.npu.synchronize()
        step_start = time.time()

        self.optimizer.zero_grad()

        # log_gpu_memory_usage("After optimizer zero_grad", logger=logger)
        if get_device_name() == "cuda":
            log_gpu_memory_usage("After optimizer zero_grad", logger=logger)

        micro_batches = batch.split(self.config.data.micro_batch_size_per_gpu)
        n_micro_batches = len(micro_batches)
        step_loss = 0
        logger.info(f"✅ micro_batches 的大小为 {n_micro_batches}")
        # for micro_batch in micro_batches:
        #     loss = self._compute_loss_and_backward(batch=micro_batch) / n_micro_batches
        #     step_loss += loss.item()
        #     logger.info(f"✅ 一个 micro_batch 的 loss计算完成 step_loss为{step_loss}\n")

        total_step_start = time.perf_counter()
        step_loss = 0.0

        for i, micro_batch in enumerate(micro_batches):
            batch_start = time.perf_counter()

            loss = self._compute_loss_and_backward(batch=micro_batch) / n_micro_batches
            step_loss += loss.item()

            batch_end = time.perf_counter()
            duration = batch_end - batch_start

            # 记录单个 micro_batch 的耗时和累计 loss
            logger.info(
                f"✅ [Batch {i + 1}/{len(micro_batches)}] loss calculation completed. "
                f"Current step_loss: {step_loss:.4f}, Duration: {duration:.4f}s"
            )

        total_step_end = time.perf_counter()
        total_duration = total_step_end - total_step_start

        logger.info(
            f"✅ All micro_batches completed. Total step_loss: {step_loss:.4f}, "
            f"Total Duration: {total_duration:.4f}s, Avg per batch: {total_duration / len(micro_batches):.4f}s"
        )

        logger.info("✅ micro_batches 的 loss计算完成 ✅")
        grad_norm = self.fsdp_model.clip_grad_norm_(max_norm=self.config.optim.clip_grad)

        # log_gpu_memory_usage("Before optimizer step", logger=logger)
        if get_device_name() == "cuda":
            log_gpu_memory_usage("Before optimizer step", logger=logger)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: grad_norm is not finite: {grad_norm}")
            self.optimizer.zero_grad()
        else:
            self.optimizer.step()

        # log_gpu_memory_usage("After optimizer step", logger=logger)
        if get_device_name() == "cuda":
            log_gpu_memory_usage("After optimizer step", logger=logger)

        self.lr_scheduler.step()
        logger.info("✅ 学习率lr_scheduler更新完成 ✅")

        # reduce loss across dp ranks
        lr = self.lr_scheduler.get_last_lr()[0]

        if get_device_name() == "npu":
            torch.npu.synchronize()
        # log_gpu_memory_usage("After offload weights", logger=logger)
        if get_device_name() == "cuda":
            log_gpu_memory_usage("After offload weights", logger=logger)

        # step_loss = torch.tensor(step_loss).cuda()
        step_loss = torch.tensor(step_loss).to(get_device_id())
        torch.distributed.all_reduce(step_loss, op=torch.distributed.ReduceOp.AVG)

        logger.info("✅ step_loss 计算完成 ")

        step_time = time.time() - step_start
        if self.device_mesh.get_rank() == 0 and hasattr(self, "_global_step"):
            self._global_step += 1
            # 每10步记录一次性能
            if self._global_step % 10 == 0:
                logger.info(
                    f"⏱️  Step {self._global_step} | 耗时: {step_time:.3f}s | "
                    f"Loss: {step_loss.item():.4f} | LR: {lr * 1e3:.2f}e-3"
                )
        elif self.device_mesh.get_rank() == 0:
            self._global_step = 1

        return {"train/loss": step_loss.detach().item(), "train/lr(1e-3)": lr * 1e3}

    def validation_step(self, batch: TensorDict):
        self.fsdp_model.eval()
        with torch.no_grad():
            loss = self._compute_loss_and_backward(batch, do_backward=False)
            torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG)
        return loss

    def save_checkpoint(self, step):
        # save checkpoint
        save_start = time.time()
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType

        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.fsdp_model, StateDictType.FULL_STATE_DICT, cfg):
            state_dict = self.fsdp_model.state_dict()

        path = os.path.join(self.config.trainer.default_local_dir, f"global_step_{step}")
        # save huggingface model
        if self.device_mesh.get_rank() == 0:
            os.makedirs(path, exist_ok=True)
            self.model.save_pretrained(path, state_dict=state_dict)
            self.tokenizer.save_pretrained(path)
            if self.config.trainer.default_hdfs_dir:
                hdfs_io.makedirs(self.config.trainer.default_hdfs_dir, exist_ok=True)
                hdfs_io.copy(src=path, dst=self.config.trainer.default_hdfs_dir, dirs_exist_ok=True)
        torch.distributed.barrier()
        if self.device_mesh.get_rank() == 0:
            logger.info(f"✅ 检查点保存完成 | 耗时: {time.time() - save_start:.2f}s | 路径: {path}")

    def get_profiler(self):
        # if args.profile_level == 'level_none':
        #     profiler_level = torch_npu.profiler.ProfilerLevel.Level_none
        # elif args.profile_level == 'level0':
        #     profiler_level = torch_npu.profiler.ProfilerLevel.Level0
        # elif args.profile_level == 'level1':
        profiler_level = torch_npu.profiler.ProfilerLevel.Level1
        # elif args.profile_level == 'level2':
        #     profiler_level = torch_npu.profiler.ProfilerLevel.Level2
        # else:
        #     raise ValueError(f"profiler_level only supports level0,"
        #                      f" 1, 2, and level_none, but gets {args.profile_level}")

        # if args.profile_export_type == 'text':
        profile_export_type = torch_npu.profiler.ExportType.Text
        # elif args.profile_export_type == 'db':
        #     profile_export_type = torch_npu.profiler.ExportType.Db
        # else:
        #     raise ValueError(f"profile_export_type only supports text or db,"
        #                      f"but gets {args.export_type}")

        experimental_config = torch_npu.profiler._ExperimentalConfig(
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
            profiler_level=profiler_level,
            export_type=profile_export_type,
            data_simplification=False,
        )

        activites = [torch_npu.profiler.ProfilerActivity.NPU]
        activites.append(torch_npu.profiler.ProfilerActivity.CPU)

        prof = torch_npu.profiler.profile(
            with_stack=False,
            record_shapes=True,
            profile_memory=False,
            activities=activites,
            schedule=torch_npu.profiler.schedule(wait=0, warmup=1, active=5, repeat=1, skip_first=0),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./profiler_output"),
            experimental_config=experimental_config,
        )

        return prof

    def fit(self):
        logger.info("enter fit !!!!")
        rank = self.device_mesh.get_rank()

        # TODO: add a unified tracking
        if rank == 0:
            tracking = Tracking(
                project_name=self.config.trainer.project_name,
                experiment_name=self.config.trainer.experiment_name,
                default_backend=self.config.trainer.logger,
            )
            logger.info(
                f"🏃 训练开始 | 实验: {self.config.trainer.experiment_name} | Backend: {self.config.trainer.logger}"
            )

        global_step = 0
        # compute the total training steps.
        # the total training steps in SFT is mainly for early exit
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps

        if rank == 0:
            logger.info(f"🎯 训练目标: {self.total_training_steps} steps ({self.config.trainer.total_epochs} epochs)")

            # NPU预热（避免冷启动影响计时）
            if get_device_name() == "npu":
                logger.info("🔥 NPU预热中...")
                dummy_tensor = torch.randn(2, 2).to(get_device_id())
                for _ in range(3):
                    dummy_tensor = dummy_tensor * 2
                    torch.npu.synchronize()
                logger.info("✅ NPU预热完成")

        # TODO (zhangchi.usc1992) add back checkpoint manager.
        # Currently, it blocks when uploading to hdfs. So very slow.

        # prof = self.get_profiler()
        # prof.start()

        for epoch in range(self.config.trainer.total_epochs):
            self.train_sampler.set_epoch(epoch=epoch)
            if rank == 0:
                logger.info(f"🔄 Epoch {epoch + 1}/{self.config.trainer.total_epochs} 开始")

            for data in tqdm(
                self.train_dataloader,
                total=self.steps_per_epoch,
                desc=f"Epoch {epoch + 1}/{self.config.trainer.total_epochs}",
                disable=rank != 0,
            ):
                global_step += 1
                # data = TensorDict(data, batch_size=self.config.data.train_batch_size).cuda()
                data = TensorDict(data, batch_size=self.config.data.train_batch_size).to(get_device_id())

                # NPU同步以确保准确计时
                if get_device_name() == "npu":
                    torch.npu.synchronize()
                step_start = time.time()

                logger.info("enter self.training_step !!!")
                metric = self.training_step(data)

                # 计算step耗时
                if get_device_name() == "npu":
                    torch.npu.synchronize()
                step_time = time.time() - step_start
                # prof.step()
                if rank == 0:
                    tracking.log(data=metric, step=global_step)

                    # 每50步输出详细进度
                    if global_step % 50 == 0:
                        logger.info(
                            f"📈 Step {global_step}/{self.total_training_steps} | "
                            f"Loss: {metric['train/loss']:.4f} | "
                            f"LR: {metric['train/lr(1e-3)']:.3f}e-3 | "
                            f"耗时: {step_time:.3f}s"
                        )

                if self.config.trainer.save_freq > 0 and global_step % self.config.trainer.save_freq == 0:
                    if rank == 0:
                        logger.info(f"💾 触发检查点保存于 step {global_step}")
                    self.save_checkpoint(step=global_step)

                # for early exit validation
                if global_step >= self.total_training_steps:
                    if rank == 0:
                        logger.info(f"🎯 达到目标步数 {self.total_training_steps}，执行最终验证")

                    # # Perform final validation
                    # val_losses = []
                    # if rank == 0:
                    #     logger.info("🔍 最终验证开始...")
                    #
                    # for val_data in self.val_dataloader:
                    # val_data = (
                    #     TensorDict(val_data, batch_size=self.config.data.micro_batch_size_per_gpu)
                    #     .to(get_device_id())
                    # )
                    #     val_loss = self.validation_step(val_data)
                    #     val_losses.append(val_loss)
                    #
                    # if rank == 0:
                    #     avg_val_loss = torch.mean(torch.stack(val_losses))
                    #     metric = {"val/loss": avg_val_loss.detach().item()}
                    #     tracking.log(data=metric, step=global_step)
                    #     logger.info(f"📊 最终验证完成 | Loss: {avg_val_loss.item():.4f}")

                    torch.distributed.barrier()

                    # Save final checkpoint
                    if rank == 0:
                        logger.info("💾 保存最终检查点...")
                    self.save_checkpoint(step=global_step)

                    if rank == 0:
                        logger.info("🏁 训练完成！")
                    return

            # validation
            if rank == 0:
                logger.info(f"🔍 Epoch {epoch + 1} 验证开始...")

            # val_losses = []
            # for data in self.val_dataloader:
            #     data = TensorDict(data, batch_size=self.config.data.micro_batch_size_per_gpu).to(get_device_id())
            #     # data = TensorDict(data, batch_size=self.config.data.micro_batch_size_per_gpu).cuda()
            #     val_loss = self.validation_step(data)
            #     val_losses.append(val_loss)
            #
            # if rank == 0:
            #     val_loss = torch.mean(torch.stack(val_losses))
            #     metric = {"val/loss": val_loss.detach().item()}
            #     tracking.log(data=metric, step=global_step)
            #     logger.info(f"📊 Epoch {epoch + 1} 验证完成 | Loss: {val_loss.item():.4f}")

            torch.distributed.barrier()

            # save checkpoint
            if rank == 0:
                logger.info(f"💾 保存 Epoch {epoch + 1} 检查点...")
            self.save_checkpoint(step=global_step)

            if rank == 0:
                logger.info(f"✅ Epoch {epoch + 1}/{self.config.trainer.total_epochs} 完成")

            # prof.stop()

        if rank == 0:
            logger.info("🏁 所有 Epoch 训练结束")


@hydra.main(config_path="config", config_name="sft_trainer", version_base=None)
def main(config):
    local_rank, rank, world_size = initialize_global_process_group()

    # device_mesh = init_device_mesh(device_type="cuda", mesh_shape=(world_size,), mesh_dim_names=("fsdp",))
    device_type = get_device_name()
    device_mesh = init_device_mesh(device_type=device_type, mesh_shape=(world_size,), mesh_dim_names=("fsdp",))

    dp_size = world_size // config.ulysses_sequence_parallel_size
    ulysses_device_mesh = init_device_mesh(
        # device_type="cuda", mesh_shape=(dp_size, config.ulysses_sequence_parallel_size), mesh_dim_names=("dp", "sp")
        device_type=device_type,
        mesh_shape=(dp_size, config.ulysses_sequence_parallel_size),
        mesh_dim_names=("dp", "sp"),
    )
    # build tokenizer and datasets first
    from verl.utils import hf_tokenizer

    local_model_path = copy_to_local(src=config.model.partial_pretrain, verbose=True)
    tokenizer = hf_tokenizer(local_model_path, trust_remote_code=config.model.trust_remote_code)

    if "qwen3" in local_model_path.lower() or "qwen-3" in local_model_path.lower():
        # fix qwen3 chat template
        if "coder" not in local_model_path.lower():
            tokenizer.chat_template = QWEN3CHATTEMPLATE
        elif "coder" in local_model_path.lower():
            tokenizer.chat_template = QWEN3CODERCHATTEMPLATE

    train_dataset = create_sft_dataset(config.data.train_files, config.data, tokenizer)
    val_dataset = create_sft_dataset(config.data.val_files, config.data, tokenizer)

    trainer = FSDPSFTTrainer(
        config=config,
        device_mesh=device_mesh,
        ulysses_device_mesh=ulysses_device_mesh,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )

    trainer.fit()


def create_sft_dataset(data_paths, data_config, tokenizer):
    """Create a dataset."""
    # build dataset
    # First check if a custom dataset class is specified
    if data_config.custom_cls.get("path", None):
        from verl.utils.import_utils import load_extern_type

        dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
    # Then check if multi-turn dataset should be used
    elif data_config.get("multiturn", {}).get("enable", False):
        dataset_cls = MultiTurnSFTDataset
    # Default to single-turn dataset
    else:
        dataset_cls = SFTDataset

    # Create datasets based on the selected class
    dataset = dataset_cls(parquet_files=data_paths, tokenizer=tokenizer, config=data_config)
    return dataset


if __name__ == "__main__":
    main()
