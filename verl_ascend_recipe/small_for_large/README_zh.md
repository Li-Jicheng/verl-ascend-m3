# DeepSeek-V4-Flash 以小验大实践（Ascend NPU）

## 背景
业界DeepSeekV4等大模型的RL单步训练耗时极长，且需要海量的算力资源。

因此，本recipe尝试在维持模型基础与架构前提下，裁剪出小模型，用于快速RL训练精度、问题定位以及自定义特性调试。

本recipe是基于DeepSeek-V4-Flash模型在NPU上进行后训练的样例，基于GRPO，使用gsm8k数据集。

## 一、准备

### 准备环境

#### 推理场景

方式一，直接使用镜像：

参考vllm-ascend官方[DeepSeek-V4](https://docs.vllm.ai/projects/ascend/en/v0.13.0/tutorials/DeepSeek-V4.html)部署文档，使用支持DeepSeek-V4的镜像

方式二，源码编译：
```
# vLLM (v0.13.0)
git clone --branch v0.13.0 https://github.com/vllm-project/vllm.git
cd vllm
VLLM_TARGET_DEVICE=empty pip install -v -e . -i https://repo.huaweicloud.com/repository/pypi/simple

# vLLM-Ascend (tag: v0.13.0rc3)
git clone --branch v0.13.0rc3 https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
git submodule update --init --recursive
pip install -v -e . -i https://repo.huaweicloud.com/repository/pypi/simple
```

#### 训推场景（TODO）


### 准备模型权重
DeepSeek-V4-Flash[开源权重](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)
- 需要将权重转为BF16权重，此步骤需要目录所在磁盘约有 692GB（开源权重149GB + 类型转换后543GB）以上空间。
- `--output_hf_path`输出路径下，需
    - 修改config.json中，将quantization_config字典删除
    - 新建chat_template.jinja并拷贝内容：https://modelscope.cn/models/Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp/file/view/master/chat_template.jinja?status=0

```
# 模型权重类型转换脚本
python convert_model_deepseekV4_flash.py --input_fp8_hf_path=/path/DeepSeek-V4-Flash --output_hf_path=/path/DeepSeek-V4-Flash-bf16 --quant_type=bfloat16 
```

### 准备模型评测工具
可按照[vllm-ascend官方文档](https://docs.vllm.ai/projects/ascend/en/v0.13.0/developer_guide/evaluation/index.html)，选择评测工具使用。


## 二、裁剪小模型

### 主要思想
DeepSeek-V4-Flash模型有43个moe层，均256专家，前三层hash路由选择6个激活专家，后续层根据输入token以及scoring_func计算出6个激活专家。

256专家使模型在不同数据集下均有很高得分，针对单个数据集，可通过仅保留高频专家方式，裁剪掉冗余专家，维持不错得分。

- vllm-ascend仓增加保存高频专家代码
- 原模型拉起推理服务、推理指定数据集、获取到该数据集的token级别的激活专家数据
- 将token级别激活专家数据，转为数据集级别各moe层高频专家
- 按高频专家，裁剪后40层的冗余专家
- 前3层因为使用hash路由选择激活专家，因此不能裁剪
- 后40层裁剪，仅保留高频64专家

### 主要操作
1、应用以小验大的patch补丁

```
# 查看vllm-ascend路径
pip show vllm-ascend

# 将vllm-ascend.patch补丁拷贝到vllm-ascend路径，应用补丁
cd /path/vllm-ascend
git apply vllm-ascend.patch
```

2、单机拉起推理服务

```shell
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export ACL_OP_INIT_MODE=1
export ASCEND_A3_ENABLE=1
export USE_MULTI_BLOCK_POOL=1

export HCCL_OP_EXPANSION_MODE=AIV
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

# 注：想保存各moe层每个token的激活专家时，需设置该环境变量，不需要时需注释
export TOPK_SAVE_PATH="/path/topk_ids_by_layer"    

python -m vllm.entrypoints.openai.api_server \
  --model /path/DeepSeek-V4-Flash-bf16 \
  --host 0.0.0.0 \
  --max_model_len 22528 \
  --max-num-batched-tokens 22528 \
  --served-model-name auto \
  --gpu-memory-utilization 0.9 \
  --data-parallel-size 4 \
  --tensor-parallel-size 4 \
  --max-num-seqs 48 \
  --async-scheduling \
  --enable-expert-parallel \
  --chat-template /path/DeepSeek-V4-Flash-bf16/chat_template.jinja \
  --port 8006 \
  --enforce-eager

```

3、清空dummy_run时的激活专家缓存

在推理服务就绪时，删除掉`TOPK_SAVE_PATH`文件夹下所有.txt文件

```shell
rm -rf topk_ids_by_layer/*
```

4、收集指定数据集的激活专家缓存

以`ais_bench`为例，指定数据集评测

新增`benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_general_V4.py`,内容如下
```
from ais_bench.benchmark.models import VLLMCustomAPIChat
from ais_bench.benchmark.utils.postprocess.model_postprocessors import extract_non_reasoning_content

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr="vllm-api-general-chat",
        path="/path/DeepSeek-V4-Flash-bf16",
        model="auto",   # 与拉起推理服务时，served-model-name一致
        stream=False,
        request_rate=0,
        use_timestamp=False,
        retry=2,
        host_ip="localhost",
        host_port=8006,
        max_out_len=1024,
        batch_size=48,
        trust_remote_code=False,
        generation_kwargs=dict(
            temperature=0.01,
            ignore_eos=False,
        ),
        pred_postprocessor=dict(type=extract_non_reasoning_content),
    )
]

```

```shell
# 校验是否存在
ais_bench --models vllm_api_general_V4 --dataset gsm8k_gen --search
# 评测500条，也可以按需评测全部数据
ais_bench --models vllm_api_general_V4 --dataset gsm8k_gen --num-prompt 500 --num-warmups=0
```

原模型得分效果如下

| dataset | 专家数 | 激活数 | num-prompt | vllm-api-general-chat |
| ------- | ------ | ------ | ---------- | --------------------- |
| gsm8k   | 256    | 6      | 500        | 92.60                 |

最终`TOPK_SAVE_PATH`下生成所有moe层在`gsm8k-500`条数据集的所有激活专家

5、识别高频专家

```shell
# --log-dir是之前`TOPK_SAVE_PATH`指定的路径，将生成top64_experts.json文件
python stat_moe.py --top-k=64 --log-dir=""
```

6、按高频专家裁剪模型

```shell
# --json-path是上一步生成的top64_experts.json
python expert_pruned.py --top-k=64 --json-path=top64_experts.json --model-dir=/path/DeepSeek-V4-Flash-bf16
```

7、裁剪模型效果校验

小模型拉起推理服务

```shell
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export ACL_OP_INIT_MODE=1
export ASCEND_A3_ENABLE=1
export USE_MULTI_BLOCK_POOL=1

export HCCL_OP_EXPANSION_MODE=AIV
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

# 注释该环境变量
# export TOPK_SAVE_PATH="/path/topk_ids_by_layer"    

# 使用裁剪模型路径
vllm serve /path/DeepSeek-V4-Flash-bf16-top64-pruned \
  --host 0.0.0.0 \
  --max_model_len 22528 \
  --max-num-batched-tokens 22528 \
  --served-model-name auto \
  --gpu-memory-utilization 0.9 \
  --data-parallel-size 4 \
  --tensor-parallel-size 4 \
  --max-num-seqs 48 \
  --async-scheduling \
  --enable-expert-parallel \
  --chat-template /path/DeepSeek-V4-Flash-bf16-top64-pruned/chat_template.jinja \
  --port 8006 \
  --enforce-eager
```

小模型使用`ais_bench`，进行指定数据集评测
新增`benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_general_V4_pruned.py`,内容如下
```
from ais_bench.benchmark.models import VLLMCustomAPIChat
from ais_bench.benchmark.utils.postprocess.model_postprocessors import extract_non_reasoning_content

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr="vllm-api-general-chat",
        path="/path/DeepSeek-V4-Flash-bf16-top64-pruned",
        model="auto",   # 与拉起推理服务时，served-model-name一致
        stream=False,
        request_rate=0,
        use_timestamp=False,
        retry=2,
        host_ip="localhost",
        host_port=8006,
        max_out_len=1024,
        batch_size=48,
        trust_remote_code=False,
        generation_kwargs=dict(
            temperature=0.01,
            ignore_eos=False,
        ),
        pred_postprocessor=dict(type=extract_non_reasoning_content),
    )
]
```
执行评测
```shell
# 校验是否存在
ais_bench --models vllm_api_general_V4_pruned --dataset gsm8k_gen --search
# 评测500条，也可以按需评测全部数据
ais_bench --models vllm_api_general_V4_pruned --dataset gsm8k_gen --num-prompt 500 --num-warmups=0
```

小模型得分效果如下

| dataset | 专家数 | 激活数 | num-prompt | vllm-api-general-chat |
| ------- | ------ | ------ | ---------- | --------------------- |
| gsm8k   | 64     | 6      | 500        | 39.58                 |


## 三、训练侧patch适配（TODO）
