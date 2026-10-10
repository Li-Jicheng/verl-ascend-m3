import glob
import json
import os

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask


class SFTDataset(Dataset):
    def __init__(self, parquet_files: str | list[str], tokenizer, config):
        prompt_key = config.get("prompt_key", "prompt")
        prompt_dict_keys = config.get("prompt_dict_keys", None)
        response_key = config.get("response_key", "response")
        response_dict_keys = config.get("response_dict_keys", None)
        max_length = config.get("max_length", 1024)
        truncation = config.get("truncation", "error")

        assert truncation in ["error", "left", "right"]
        self.truncation = truncation

        # 格式检测逻辑
        self.use_hf_load = False
        self.use_json_load = False

        if isinstance(parquet_files, str) and os.path.isdir(parquet_files):
            files = os.listdir(parquet_files)
            has_json = any(f.endswith(".json") for f in files)

            if has_json:
                self.use_json_load = True
                self.parquet_files = parquet_files
            else:
                self.use_hf_load = True
        elif not isinstance(parquet_files, list):
            parquet_files = [parquet_files]
            self.parquet_files = parquet_files

        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer

        self.prompt_key = prompt_key if isinstance(prompt_key, (tuple, list)) else [prompt_key]
        self.response_key = response_key if isinstance(response_key, (tuple, list)) else [response_key]
        self.prompt_dict_keys = prompt_dict_keys if prompt_dict_keys else []
        self.response_dict_keys = response_dict_keys if response_dict_keys else []

        self.max_length = max_length

        if not self.use_hf_load and not self.use_json_load:
            self._download()

        self._read_files_and_tokenize()
        self._log_response_stats(num_samples=min(1000, len(self)))

    def _load_json_files(self):
        """加载 JSON 文件夹格式的数据 - 简化版：只取 input 和 output"""

        if isinstance(self.parquet_files, str) and os.path.isdir(self.parquet_files):
            json_pattern = os.path.join(self.parquet_files, "*.json")
            json_files = sorted(glob.glob(json_pattern))
        else:
            json_files = self.parquet_files if isinstance(self.parquet_files, list) else [self.parquet_files]
            json_files = [f for f in json_files if f.endswith(".json")]

        all_prompts = []
        all_responses = []
        skipped = 0

        for file_path in json_files:
            if not os.path.exists(file_path):
                continue

            with open(file_path, encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    if not isinstance(data, list):
                        continue
                except json.JSONDecodeError:
                    continue

            for item in data:
                if not isinstance(item, dict):
                    skipped += 1
                    continue

                # 只取 input 作为 prompt，output 作为 response
                input_text = str(item.get("input", "")).strip()
                output_text = str(item.get("output", "")).strip()

                if not input_text or not output_text:
                    skipped += 1
                    continue

                all_prompts.append(input_text)
                all_responses.append(output_text)

        self.prompts = all_prompts
        self.responses = all_responses

        if os.environ.get("RANK", "0") == "0":
            print(
                f"[SFTDataset] Loaded {len(self.prompts)} samples from {len(json_files)} JSON files "
                f"(skipped {skipped} invalid items)"
            )

    def _download(self):
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_to_local(parquet_file, verbose=True)

    def _read_files_and_tokenize(self):
        if self.use_hf_load:
            from datasets import load_dataset

            ds = load_dataset(self.parquet_files)

            def get_pair(example):
                assert len(example["conversations"]) == 2
                assert example["conversations"][0]["from"] == self.prompt_key[0]
                assert example["conversations"][1]["from"] == self.response_key[0]
                prompt = example["conversations"][0]["value"]
                response = example["conversations"][1]["value"]
                return {"prompt": prompt, "response": response}

            train_pairs = ds["train"].map(get_pair, remove_columns=ds["train"].column_names, num_proc=4)
            self.prompts = train_pairs["prompt"]
            self.responses = train_pairs["response"]

        elif self.use_json_load:
            self._load_json_files()

        else:
            # 原有 Parquet 处理逻辑保持不变
            def series_to_item(ls):
                import numpy
                import pandas

                while isinstance(ls, (pandas.core.series.Series, numpy.ndarray)) and len(ls) == 1:
                    if isinstance(ls, pandas.core.series.Series):
                        ls = ls.iloc[0]
                    else:
                        ls = ls[0]
                if not isinstance(ls, (pandas.core.series.Series, numpy.ndarray, list)):
                    return ls
                if isinstance(ls, list) and len(ls) == 1:
                    return series_to_item(ls[0])
                return ls

            dataframes = []
            for parquet_file in self.parquet_files:
                dataframe = pd.read_parquet(parquet_file)
                dataframes.append(dataframe)
            self.dataframe = pd.concat(dataframes)

            self.prompts = self.dataframe[self.prompt_key]
            if len(self.prompt_dict_keys) > 0:
                for key in self.prompt_dict_keys:
                    try:
                        if isinstance(self.prompts, pd.Series):
                            self.prompts = self.prompts.apply(lambda x, k=key: series_to_item(x)[k])
                        else:
                            self.prompts = self.prompts.apply(lambda row, k=key: series_to_item(row)[k], axis=1)
                    except Exception:
                        print(f"self.prompts={self.prompts}")
                        raise
            else:
                try:
                    self.prompts = self.prompts.apply(lambda x, k=key: series_to_item(x)[k], axis=1)
                except Exception:
                    print(f"self.prompts={self.prompts}")
                    raise
            self.prompts = self.prompts.tolist()

            self.responses = self.dataframe[self.response_key]
            if len(self.response_dict_keys) > 0:
                for key in self.response_dict_keys:
                    try:
                        if isinstance(self.responses, pd.Series):
                            self.responses = self.responses.apply(lambda x, k=key: series_to_item(x)[k])
                        else:
                            self.responses = self.responses.apply(lambda row, k=key: series_to_item(row)[k], axis=1)
                    except Exception:
                        print(f"self.responses={self.responses}")
                        raise
            else:
                try:
                    self.responses = self.responses.apply(lambda x, k=key: series_to_item(x)[k], axis=1)
                except Exception:
                    print(f"self.responses={self.responses}")
                    raise
            self.responses = self.responses.tolist()

        if os.environ.get("RANK", "0") == "0":
            response_lens = [len(str(r)) for r in self.responses[:1000]]
            if response_lens:
                avg_chars = sum(response_lens) / len(response_lens)
                print(
                    f"[SFTDataset] Sampled response length (chars): avg={avg_chars:.0f}, "
                    f"max={max(response_lens)}, min={min(response_lens)}"
                )

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, item):
        tokenizer = self.tokenizer

        prompt = self.prompts[item]
        response = self.responses[item]

        # 构造单轮对话（input -> output）
        prompt_chat = [{"role": "user", "content": prompt}]
        prompt_chat_str = tokenizer.apply_chat_template(prompt_chat, add_generation_prompt=True, tokenize=False)
        response_chat_str = response + tokenizer.eos_token

        prompt_ids_output = tokenizer(prompt_chat_str, return_tensors="pt", add_special_tokens=False)
        prompt_ids = prompt_ids_output["input_ids"][0]
        prompt_attention_mask = prompt_ids_output["attention_mask"][0]

        response_ids_output = tokenizer(response_chat_str, return_tensors="pt", add_special_tokens=False)
        response_ids = response_ids_output["input_ids"][0]
        response_attention_mask = response_ids_output["attention_mask"][0]

        prompt_length = prompt_ids.shape[0]
        response_length = response_ids.shape[0]

        input_ids = torch.cat((prompt_ids, response_ids), dim=-1)
        attention_mask = torch.cat((prompt_attention_mask, response_attention_mask), dim=-1)

        sequence_length = input_ids.shape[0]
        if sequence_length < self.max_length:
            padded_input_ids = (
                torch.ones(size=(self.max_length - sequence_length,), dtype=input_ids.dtype)
                * self.tokenizer.pad_token_id
            )
            padded_attention_mask = torch.zeros(size=(self.max_length - sequence_length,), dtype=attention_mask.dtype)

            input_ids = torch.cat((input_ids, padded_input_ids))
            attention_mask = torch.cat((attention_mask, padded_attention_mask))
        elif sequence_length > self.max_length:
            if self.truncation == "left":
                input_ids = input_ids[-self.max_length :]
                attention_mask = attention_mask[-self.max_length :]
            elif self.truncation == "right":
                input_ids = input_ids[: self.max_length]
                attention_mask = attention_mask[: self.max_length]
            elif self.truncation == "error":
                raise NotImplementedError(f"{sequence_length=} is larger than {self.max_length=}")
            else:
                raise NotImplementedError(f"Unknown truncation method {self.truncation}")

        position_ids = compute_position_id_with_mask(attention_mask)

        loss_mask = attention_mask.clone()
        if prompt_length > 1:
            loss_mask[: min(prompt_length, loss_mask.size(0)) - 1] = 0
        loss_mask[min(prompt_length + response_length, loss_mask.size(0)) - 1] = 0

        if item < 5 and os.environ.get("RANK", "0") == "0":
            print(
                f"[SFTDataset] Sample {item}: prompt_tokens={prompt_length}, "
                f"response_tokens={response_length}, total={input_ids.shape[0]}"
            )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }

    def _log_response_stats(self, num_samples=100):
        """统计 response 的 token 数量分布"""
        import os

        import numpy as np

        if os.environ.get("RANK", "0") != "0":
            return

        indices = np.random.choice(len(self), min(num_samples, len(self)), replace=False)
        lengths = []

        for idx in indices:
            response = self.responses[idx]
            response_chat_str = response + self.tokenizer.eos_token
            response_ids = self.tokenizer(response_chat_str, return_tensors="pt", add_special_tokens=False)[
                "input_ids"
            ][0]
            lengths.append(response_ids.shape[0])

        if lengths:
            lengths = np.array(lengths)
            print(f"\n{'=' * 60}")
            print(f"[SFTDataset] Response Token Statistics (sampled {len(lengths)}/{len(self)} items):")
            print(f"  Mean:   {lengths.mean():.1f}")
            print(f"  Median: {np.median(lengths):.1f}")
            print(f"  Min:    {lengths.min()}")
            print(f"  Max:    {lengths.max()}")
            print(f"  P90:    {np.percentile(lengths, 90):.1f}")
            print(f"  P95:    {np.percentile(lengths, 95):.1f}")
            print(f"  P99:    {np.percentile(lengths, 99):.1f}")
            print(f"{'=' * 60}\n")
