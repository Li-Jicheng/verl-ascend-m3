import argparse
import json
import os
import re
import shutil
from collections import defaultdict

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SKIP_LAYERS = {0, 1, 2}


def load_keep_dict_from_json(json_path: str) -> dict[int, list[int]]:
    """
    从 JSON 加载 keep_dict。
    兼容键名格式：
        "layers.0" / "model.layers.0" / "model.layers.0.mlp.experts" 等
    """
    with open(json_path, encoding="utf-8") as f:
        raw = json.load(f)

    keep_dict = {}
    for key, val in raw.items():
        # 关键修改：去掉 $，支持 "model.layers.0.mlp.experts" 这种格式
        m = re.search(r"layers\.(\d+)", key)
        if not m:
            print(f"[WARN] 无法解析层索引: {key}，跳过")
            continue
        layer_idx = int(m.group(1))

        # 兼容新旧两种格式
        if isinstance(val, dict):
            expert_ids = [int(x) for x in val["topk_ids"]]
        elif isinstance(val, list):
            expert_ids = [int(x) for x in val]
        else:
            print(f"[WARN] {key} 的数据格式异常，跳过")
            continue

        keep_dict[layer_idx] = expert_ids

    return keep_dict


def prune_moe_experts(
    model_dir: str,
    output_dir: str,
    keep_dict: dict[int, list[int]],
    target_num_experts: int = 16,
):
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as f:
        config = json.load(f)

    with open(os.path.join(model_dir, "model.safetensors.index.json"), encoding="utf-8") as f:
        index = json.load(f)

    num_layers = config["num_hidden_layers"]
    old_n_experts = config["n_routed_experts"]

    full_keep = {}
    for i in range(num_layers):
        if i in SKIP_LAYERS:
            full_keep[i] = list(range(old_n_experts))
        elif i in keep_dict:
            experts = sorted(set(keep_dict[i]))
            assert len(experts) == target_num_experts, (
                f"Layer {i}: 期望 {target_num_experts} 个专家，实际给了 {len(experts)} 个"
            )
            assert all(0 <= e < old_n_experts for e in experts), f"Layer {i}: 专家索引越界 [0, {old_n_experts})"
            full_keep[i] = experts
        else:
            full_keep[i] = list(range(target_num_experts))

    old_to_new_maps = {}
    for layer_idx, keep in full_keep.items():
        if layer_idx in SKIP_LAYERS:
            continue
        mapping = torch.full((old_n_experts,), -1, dtype=torch.long)
        for new_idx, old_idx in enumerate(keep):
            mapping[old_idx] = new_idx
        for old_idx in range(old_n_experts):
            if mapping[old_idx] == -1:
                mapping[old_idx] = old_idx % target_num_experts
        old_to_new_maps[layer_idx] = mapping

    weight_map = index["weight_map"]
    file_to_keys = defaultdict(list)
    for key, filename in weight_map.items():
        file_to_keys[filename].append(key)

    new_weight_map = {}

    for filename, keys in file_to_keys.items():
        new_tensors = {}
        filepath = os.path.join(model_dir, filename)

        with safe_open(filepath, framework="pt", device="cpu") as f:
            for key in keys:
                tensor = f.get_tensor(key)
                new_key, new_tensor = _process_single_key(key, tensor, full_keep, old_to_new_maps, old_n_experts)
                if new_key is not None:
                    new_tensors[new_key] = new_tensor
                    new_weight_map[new_key] = filename

        if new_tensors:
            save_file(new_tensors, os.path.join(output_dir, filename))

    processed_files = set(file_to_keys.keys())

    for item in os.listdir(model_dir):
        src_path = os.path.join(model_dir, item)
        dst_path = os.path.join(output_dir, item)

        # 跳过已经处理过的 safetensors
        if item in processed_files:
            continue

        # 跳过索引文件（后续会重写）
        if item == "model.safetensors.index.json" or item == "config.json":
            continue

        # 文件：直接复制
        if os.path.isfile(src_path):
            shutil.copy2(src_path, dst_path)

        # 文件夹：递归复制
        elif os.path.isdir(src_path):
            if os.path.exists(dst_path):
                shutil.rmtree(dst_path)
            shutil.copytree(src_path, dst_path)

    config["n_routed_experts"] = target_num_experts
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    new_index = {
        "metadata": index.get("metadata", {}),
        "weight_map": new_weight_map,
    }
    with open(os.path.join(output_dir, "model.safetensors.index.json"), "w", encoding="utf-8") as f:
        json.dump(new_index, f, indent=2)

    print(f"\n✅ 裁剪完成，输出目录: {output_dir}")
    print(f"   跳过层 {SKIP_LAYERS}: 保留全部 {old_n_experts} 个专家")
    print(f"   其余层路由专家: {old_n_experts} -> {target_num_experts}")
    print("   共享专家: 保留不变")
    print(f"   总层数: {num_layers}")


def _process_single_key(key, tensor, full_keep, old_to_new_maps, old_n):
    """
    处理单个权重 key。同时兼容 ffn 和 mlp 两套命名。
    """
    m = re.search(r"(?:model\.)?layers\.(\d+)", key)
    if not m:
        return key, tensor
    layer_idx = int(m.group(1))

    # 前三层原样保留
    if layer_idx in SKIP_LAYERS:
        return key, tensor

    # ---------- A. 路由专家权重 ----------
    # 同时兼容：
    #   ffn.experts.E.w1/w2/w3.weight
    #   mlp.experts.E.gate_proj/up_proj/down_proj.weight
    m_exp = re.search(
        r"(?:model\.)?layers\.\d+\.(?:ffn|mlp)\.experts\.(\d+)\.(?:w[123]|gate_proj|up_proj|down_proj)\.weight$", key
    )
    if m_exp:
        expert_idx = int(m_exp.group(1))
        keep = full_keep[layer_idx]
        if expert_idx not in keep:
            return None, None

        new_expert_idx = keep.index(expert_idx)
        new_key = re.sub(r"experts\.\d+", f"experts.{new_expert_idx}", key, count=1)
        return new_key, tensor

    # ---------- B. Gate 权重 + Bias ----------
    # 同时兼容 ffn.gate 和 mlp.gate
    m_gate = re.search(r"(?:model\.)?layers\.\d+\.(?:ffn|mlp)\.gate\.(weight|bias)$", key)
    if m_gate and "experts" not in key:
        keep = full_keep[layer_idx]

        target_dim = None
        for dim, size in enumerate(tensor.shape):
            if size == old_n:
                target_dim = dim
                break

        if target_dim is None:
            if tensor.numel() == old_n:
                return key, tensor[keep]
            raise ValueError(f"{key} 形状 {list(tensor.shape)} 中找不到专家维度 {old_n}，无法裁剪")

        idx = [slice(None)] * tensor.dim()
        idx[target_dim] = keep
        return key, tensor[tuple(idx)]

    # ---------- C. 其他权重 ----------
    return key, tensor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MoE 模型专家裁剪工具（根据 TopK experts JSON）")
    parser.add_argument("--top-k", type=int, default=32, help="保留的专家数量")
    parser.add_argument("--json-path", type=str, required=True, help="topk experts json 文件路径")
    parser.add_argument("--model-dir", type=str, required=True, help="模型目录")
    args = parser.parse_args()

    keep_dict = load_keep_dict_from_json(args.json_path)

    model_dir = args.model_dir.rstrip("/")
    output_dir = f"{model_dir}-top{args.top_k}-pruned"

    prune_moe_experts(
        model_dir=args.model_dir,
        output_dir=output_dir,
        keep_dict=keep_dict,
        target_num_experts=args.top_k,
    )
