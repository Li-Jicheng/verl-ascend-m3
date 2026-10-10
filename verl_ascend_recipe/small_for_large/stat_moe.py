import argparse
import json
import os
from collections import Counter


# ===================== 命令行参数解析 =====================
def parse_args():
    parser = argparse.ArgumentParser(description="统计MoE层TopK专家ID并保存JSON")
    parser.add_argument("--top-k", type=int, default=64, help="要统计的TopK数量")
    parser.add_argument("--log-dir", type=str, required=True, help="topk_ids日志文件所在目录")
    return parser.parse_args()


def stat_and_save_topk():
    args = parse_args()

    TOP_K = args.top_k
    LOG_DIR = args.log_dir
    parent_dir = os.path.dirname(LOG_DIR)
    SAVE_JSON_PATH = os.path.join(parent_dir, f"top{TOP_K}_experts.json")

    layer_expert_counter: dict[str, Counter] = {}

    # 1. 遍历目录下所有 layer 文件
    if not os.path.isdir(LOG_DIR):
        raise FileNotFoundError(f"日志目录不存在: {LOG_DIR}")

    for filename in os.listdir(LOG_DIR):
        if not filename.endswith(".txt"):
            continue

        layer_name = filename[:-4]
        filepath = os.path.join(LOG_DIR, filename)

        counter = Counter()
        with open(filepath, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                expert_ids = line.split(",")
                expert_ids = [eid for eid in expert_ids if eid != ""]

                if not expert_ids:
                    print(f"警告：{filename} 第{line_num}行无有效数据，已跳过")
                    continue

                counter.update(expert_ids)

        if counter:
            layer_expert_counter[layer_name] = counter
        else:
            print(f"警告：{filename} 无有效专家数据")

    if not layer_expert_counter:
        raise ValueError(f"未在 {LOG_DIR} 中找到任何有效的 topk_ids 数据")

    # 2. 生成 topk 专家ID + 出现次数
    result = {}
    for layer_name, counter in layer_expert_counter.items():
        # most_common 返回 [(expert_id, count), ...]
        topk = counter.most_common(TOP_K)

        topk_ids = [expert_id for expert_id, _ in topk]
        topk_counts = [count for _, count in topk]

        # 同时保留完整频次映射（可选，如需全量统计可取消注释）
        # all_counts = dict(counter.most_common())

        result[layer_name] = {
            "topk_ids": topk_ids,
            "topk_counts": topk_counts,
            # "all_counts": all_counts,  # 如需保存全部专家频次，取消注释此行
        }

    # 按 layer 序号排序
    def sort_key(item):
        name = item[0]
        try:
            return int(name.split(".")[-1])
        except (ValueError, IndexError):
            return name

    result = dict(sorted(result.items(), key=sort_key))

    # 3. 保存为 JSON 文件
    with open(SAVE_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # 4. 打印摘要
    print(f"✅ 统计完成！共处理 {len(layer_expert_counter)} 个 MoE 层")
    print(f"   topk 专家已保存至：{SAVE_JSON_PATH}")

    # 打印每层 Top5 作为预览
    print("\n📊 各层 Top5 专家预览：")
    for layer_name, data in result.items():
        ids = data["topk_ids"][:5]
        counts = data["topk_counts"][:5]
        pairs = [f"{eid}({cnt})" for eid, cnt in zip(ids, counts, strict=False)]
        print(f"   {layer_name}: {', '.join(pairs)}")


if __name__ == "__main__":
    stat_and_save_topk()
