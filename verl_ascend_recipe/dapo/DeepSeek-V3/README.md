# Deepseek-V3-Base on Ascend NPU
本recipe是基于Deepseek-V3-Base模型在NPU上进行RLHF后训练的样例，基于DAPO与规则奖励，使用GSM8K数据集进行训练。


# 环境配套

| **组件**  | **配套版本** | **备注**                     |
| ----------------- | -------------------- | ------------------------------------ |
| CANN            | 8.5.0/9.0.0 | |
| python          | 3.11               |                                    |
| pytorch         | 2.9.0              |                                    |
| vllm            | v0.18.0	 | |
| vllm-ascend     | releases/v0.18.0| |
| verl            | main |   |
| transformers    | 4.57.1             | |
| Megatron        | core0.16.0 | |
| MindSpeed       | core_0.16.0 |   |
| mbridge | v0.15.1 | |

# 激活CANN

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
```

# 权重准备

权重下载地址：[deepseek-ai/DeepSeek-V3-Base at main](https://huggingface.co/deepseek-ai/DeepSeek-V3-Base/tree/main)

因为官网上deepseekv3的权重是fp8格式的，需要将其转至fp16格式

参考脚本：[gitee.com/ascend/ModelZoo-PyTorch/blob/master/MindIE/LLM/DeepSeek/DeepSeek-V2/NPU\_inference/fp8\_cast\_bf16.py](https://gitee.com/ascend/ModelZoo-PyTorch/blob/master/MindIE/LLM/DeepSeek/DeepSeek-V2/NPU_inference/fp8_cast_bf16.py)

# 环境安装

除了 **mbridge 的安装方式和 transformers 的版本要求**，其余均与 verl 主线分支的[安装脚本](https://github.com/verl-project/verl/blob/main/scripts/install_vllm_mcore_npu.sh)和依赖一致。
## 安装 verl-ascend-recipe
```bash
git clone https://github.com/verl-project/verl-ascend-recipe.git
```

## 安装 vllm && vllm_ascend

```bash
git clone --depth 1 --branch v0.18.0 https://github.com/vllm-project/vllm.git
cd vllm
VLLM_TARGET_DEVICE=empty pip install -v -e .
cd ..

git clone -b releases/v0.18.0 https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
git submodule update --init --recursive
pip install -v -e . --no-build-isolation --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple/ --trusted-host triton-ascend.osinfra.cn
cd ..
```

## 安装 MindSpeed && Megatron-LM

```bash
git clone https://gitcode.com/Ascend/MindSpeed.git
cd MindSpeed && git checkout core_r0.16.0 && cd ..
git clone --depth 1 --branch core_r0.16.0 https://github.com/NVIDIA/Megatron-LM.git

# 安装 Megatron & MindSpeed
pip install -e Megatron-LM
pip install -e MindSpeed
```

## 安装 mbridge

```bash
git clone https://github.com/ISEEKYAN/mbridge.git
cd mbridge && git checkout v0.15.1
git apply ../verl-ascend-recipe/dapo/DeepSeek-V3/patch/mbridge.patch
pip install -e .
cd ..
```

## 安装 verl

```bash
git clone --recursive https://github.com/volcengine/verl.git
cd verl && pip install -r requirements-npu.txt
pip install -v -e .
cd recipe && git checkout main
cd ..
```
## 安装其他 Python 依赖
```bash
pip uninstall -y triton triton-ascend
# 安装与 vLLM-Ascend 0.18.0 对应的软件包
pip install torchvision==0.24.0
pip install torchaudio==2.9.0
pip install triton-ascend==3.2.1 --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple/ --trusted-host triton-ascend.osinfra.cn
pip install "transformers==4.57.1"
pip install "setuptools==80.10.2"
```

# 训练启动

```bash
cd verl
# 修改ray_start.sh和run_dapo_deepseekv3_671b_megatron_8k_npu.sh中对应的网卡、主节点IP、权重、数据集地址等
bash ../verl-ascend-recipe/dapo/DeepSeek-V3/scripts/ray_start.sh
```