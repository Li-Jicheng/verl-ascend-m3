import logging
import os
import time
from logging.handlers import RotatingFileHandler

import hydra
from torch.distributed.device_mesh import init_device_mesh
from verl_patch.trainer.code.fsdp_sft_trainer import FSDPSFTTrainer, create_sft_dataset

from verl.utils.device import get_device_name
from verl.utils.distributed import initialize_global_process_group
from verl.utils.fs import copy_to_local

from .constant import QWEN3CHATTEMPLATE


def setup_logger(rank: int = 0, log_dir: str = "./logs") -> logging.Logger:
    """
    配置双通道日志：控制台 + 文件
    - 控制台：INFO级别以上，带颜色/emoji，方便实时查看
    - 文件：DEBUG级别以上，包含完整上下文，用于事后分析
    """
    logger = logging.getLogger("SFTTrainer")
    logger.setLevel(logging.DEBUG)

    # 避免重复配置
    if logger.handlers:
        return logger

    # 创建日志目录（自动创建，支持多级路径）
    os.makedirs(log_dir, exist_ok=True)

    # 1. 控制台 Handler (INFO以上，结构化格式)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(f"[%(asctime)s] [Rank {rank}] [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # 2. 文件 Handler (DEBUG以上，详细格式，按Rank分文件)
    # 使用 RotatingFileHandler：单个文件最大100MB，保留5个备份，防止磁盘占满
    log_file = os.path.join(log_dir, f"sft_rank_{rank}.log")
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=100 * 1024 * 1024,  # 100MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    # 文件日志包含更多信息：文件名、行号、函数名（方便定位代码位置）
    file_format = logging.Formatter(
        f"[%(asctime)s] [Rank {rank}] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    # Rank 0 额外写入一个全局主日志（汇总所有关键节点）
    if rank == 0:
        master_log = os.path.join(log_dir, "sft_master.log")
        master_handler = RotatingFileHandler(master_log, maxBytes=100 * 1024 * 1024, backupCount=10, encoding="utf-8")
        master_handler.setLevel(logging.INFO)
        master_handler.setFormatter(file_format)
        logger.addHandler(master_handler)

    logger.info(f"📁 日志系统初始化完成 | 控制台: INFO+ | 文件: {log_file} (DEBUG+)")
    if rank == 0:
        logger.info(f"📁 Master日志: {os.path.join(log_dir, 'sft_master.log')}")

    return logger


@hydra.main(config_path="../verl_patch/trainer/code/config", config_name="sft_trainer", version_base=None)
def main(config):
    # 初始化分布式环境
    local_rank, rank, world_size = initialize_global_process_group()

    # 从config读取日志目录，默认为./logs，可在yaml配置中覆盖
    log_dir = getattr(config, "log_dir", os.path.abspath(os.path.join(os.path.dirname(__file__), "../logs")))
    logger = setup_logger(rank, log_dir)

    logger.info("=" * 70)
    logger.info("🚀 SFT Training Job Started")
    logger.info("=" * 70)
    logger.info(f"分布式环境 | local_rank={local_rank}, rank={rank}, world_size={world_size}")
    logger.info(f"日志目录: {os.path.abspath(log_dir)}")

    # 关键配置记录（仅Rank 0写入主日志，避免重复）
    if rank == 0:
        logger.info(f"📋 完整配置:\n{config}")
        logger.info(
            f"🔧 关键参数: model={config.model.partial_pretrain}, "
            f"seq_parallel={config.ulysses_sequence_parallel_size}, "
            f"train_files={config.data.train_files}"
        )

    # Device Mesh 初始化（带异常捕获和详细计时）
    device_type = get_device_name()
    logger.info(f"🖥️  设备类型: {device_type}")

    try:
        start_time = time.time()
        device_mesh = init_device_mesh(device_type=device_type, mesh_shape=(world_size,), mesh_dim_names=("fsdp",))
        logger.info(f"✅ FSDP DeviceMesh 完成 | shape=({world_size},) | 耗时: {time.time() - start_time:.2f}s")
    except Exception as e:
        logger.exception(f"❌ FSDP DeviceMesh 初始化失败: {e}")
        raise

    # Ulysses序列并行
    dp_size = world_size // config.ulysses_sequence_parallel_size
    try:
        start_time = time.time()
        ulysses_device_mesh = init_device_mesh(
            device_type=device_type,
            mesh_shape=(dp_size, config.ulysses_sequence_parallel_size),
            mesh_dim_names=("dp", "sp"),
        )
        logger.info(
            f"✅ Ulysses DeviceMesh 完成 | shape=({dp_size}, {config.ulysses_sequence_parallel_size}) | "
            f"耗时: {time.time() - start_time:.2f}s"
        )
    except Exception as e:
        logger.exception(f"❌ Ulysses DeviceMesh 初始化失败: {e}")
        raise

    # 模型路径处理
    from verl.utils import hf_tokenizer

    logger.info(f"📥 拷贝模型到本地: {config.model.partial_pretrain}")
    try:
        start_time = time.time()
        local_model_path = copy_to_local(src=config.model.partial_pretrain, verbose=True)
        copy_time = time.time() - start_time
        logger.info(f"✅ 模型拷贝完成 | 路径: {local_model_path} | 耗时: {copy_time:.2f}s")

        # 记录模型大小信息（如果有）
        if os.path.exists(local_model_path):
            import subprocess

            try:
                du_result = subprocess.run(["du", "-sh", local_model_path], capture_output=True, text=True, timeout=10)
                if du_result.returncode == 0:
                    logger.info(f"📦 模型大小: {du_result.stdout.strip()}")
            except Exception as e:
                logger.exception(f"❌ 获取模型大小失败: {e}")
    except Exception as e:
        logger.exception(f"❌ 模型拷贝失败: {e}")
        raise

    # Tokenizer加载
    logger.info(f"🔤 加载Tokenizer | trust_remote_code={config.model.trust_remote_code}")
    try:
        start_time = time.time()
        tokenizer = hf_tokenizer(local_model_path, trust_remote_code=config.model.trust_remote_code)
        vocab_size = len(tokenizer)
        logger.info(f"✅ Tokenizer加载完成 | vocab_size={vocab_size} | 耗时: {time.time() - start_time:.2f}s")
    except Exception as e:
        logger.exception(f"❌ Tokenizer加载失败: {e}")
        raise

    # ChatTemplate处理（详细记录变更）
    model_path_lower = local_model_path.lower()
    is_qwen3 = "qwen3" in model_path_lower or "qwen-3" in model_path_lower

    if is_qwen3:
        if "coder" not in model_path_lower:
            logger.warning("⚠️  Qwen3非Coder模型检测到，应用ChatTemplate修复")
            logger.debug(f"原始template: {tokenizer.chat_template[:200] if tokenizer.chat_template else 'None'}...")
            tokenizer.chat_template = QWEN3CHATTEMPLATE
            logger.info("✅ ChatTemplate已替换为自定义模板")
            logger.debug(f"新template: {tokenizer.chat_template[:200]}...")
        else:
            logger.info("ℹ️  Qwen3 Coder模型，保留原始ChatTemplate")
    else:
        logger.info(f"ℹ️  使用模型默认ChatTemplate | 模型类型: {model_path_lower.split('/')[-1]}")

    # 数据集构建（详细计时和样本统计）
    logger.info("📚 开始构建数据集...")
    data_start = time.time()

    try:
        # 训练集
        logger.info(f"📝 训练集路径: {config.data.train_files}")
        train_start = time.time()
        train_dataset = create_sft_dataset(config.data.train_files, config.data, tokenizer)
        train_time = time.time() - train_start
        train_samples = len(train_dataset)
        logger.info(
            f"✅ 训练集构建完成 | samples={train_samples} | 耗时: {train_time:.2f}s "
            f"({train_samples / train_time:.1f} samples/s)"
        )

        # 验证集
        if config.data.val_files:
            # logger.info(f"📝 验证集路径: {config.data.val_files}")
            # val_start = time.time()
            # val_dataset = create_sft_dataset(config.data.val_files, config.data, tokenizer)
            # val_time = time.time() - val_start
            # val_samples = len(val_dataset)
            # logger.info(f"✅ 验证集构建完成 | samples={val_samples} | 耗时: {val_time:.2f}s")
            logger.info("set val_dataset None.")
            val_dataset = None
        else:
            logger.warning("⚠️  未配置验证集(val_files为空)")
            val_dataset = None

        total_data_time = time.time() - data_start
        logger.info(f"📊 数据集构建总耗时: {total_data_time:.2f}s")

    except Exception as e:
        logger.exception(f"❌ 数据集构建失败: {e}")
        raise

    # Trainer初始化（记录显存使用情况）
    logger.info("🎯 初始化FSDP SFT Trainer...")
    try:
        import torch

        start_time = time.time()

        # 记录初始化前显存状态
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            mem_before = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"💾 初始化前显存占用: {mem_before:.2f} GB")

        trainer = FSDPSFTTrainer(
            config=config,
            device_mesh=device_mesh,
            ulysses_device_mesh=ulysses_device_mesh,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
        )

        init_time = time.time() - start_time

        if torch.cuda.is_available():
            mem_after = torch.cuda.memory_allocated() / 1024**3
            mem_peak = torch.cuda.max_memory_allocated() / 1024**3
            logger.info(
                f"✅ Trainer初始化完成 | 耗时: {init_time:.2f}s | "
                f"显存占用: {mem_after:.2f} GB (峰值: {mem_peak:.2f} GB)"
            )
        else:
            logger.info(f"✅ Trainer初始化完成 | 耗时: {init_time:.2f}s")

    except Exception as e:
        logger.exception(f"❌ Trainer初始化失败: {e}")
        raise

    val_samples = len(val_dataset) if val_dataset else 0

    # 训练前关键信息汇总（仅Rank 0）
    if rank == 0:
        logger.info("🚀🚀🚀 训练即将开始 | 环境汇总:")
        logger.info(f"   • 数据并行(DP)大小: {dp_size}")
        logger.info(f"   • 序列并行(SP)大小: {config.ulysses_sequence_parallel_size}")
        logger.info(f"   • 总训练样本数: {train_samples}")
        logger.info(f"   • 总验证样本数: {val_samples if val_dataset else 0}")
        logger.info(f"   • 词汇表大小: {vocab_size}")
        logger.info(f"   • 日志保存路径: {os.path.abspath(log_dir)}")

    # 启动训练
    logger.info("🏃 启动训练流程...")
    train_start_time = time.time()

    try:
        trainer.fit()
        total_train_time = time.time() - train_start_time
        logger.info(f"🎉 训练成功完成 | 总耗时: {total_train_time:.2f}s ({total_train_time / 3600:.2f}h)")

        # 保存训练结束标记（方便外部监控脚本检测）
        if rank == 0:
            completion_file = os.path.join(log_dir, "training_completed.signal")
            with open(completion_file, "w") as f:
                f.write(f"completed_at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"duration_seconds: {total_train_time}\n")
            logger.info(f"📄 完成标记已写入: {completion_file}")

    except Exception as e:
        total_time = time.time() - train_start_time
        logger.exception(f"💥 训练失败（已运行 {total_time:.2f}s): {e}")

        # 保存错误标记
        if rank == 0:
            error_file = os.path.join(log_dir, "training_failed.signal")
            with open(error_file, "w") as f:
                f.write(f"failed_at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"error: {str(e)}\n")
        raise

    logger.info("=" * 70)
    logger.info("🏁 SFT Training Job Finished Successfully")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
