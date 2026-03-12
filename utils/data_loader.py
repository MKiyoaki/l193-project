# utils/data_loader.py

import json
import random
import torch as t
from transformers import AutoTokenizer
from typing import Literal
from utils.configs_utils import SEED, DEVICE, DATA_PATH, MODEL_NAME


class SycophancyDataLoader:
    """
    Data loader class for the Sycophancy dataset.
    """

    def __init__(self, file_path: str = DATA_PATH, model_name: str = MODEL_NAME):
        self.file_path = file_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "left"

    def get_text_batches(self, split: Literal["train", "test"] = "train", batch_size: int = 32, seed: int = SEED):
        """
        Reads JSONL dataset, splits it, and yields batches of texts and token IDs.
        """
        with open(self.file_path, 'r', encoding='utf-8') as f:
            all_data = [json.loads(line) for line in f]

        random.Random(seed).shuffle(all_data)

        split_idx = int(len(all_data) * 0.8)
        data = all_data[:split_idx] if split == "train" else all_data[split_idx:]

        batches = []
        for i in range(0, len(data), batch_size):
            batch_data = data[i:i+batch_size]
            batch_texts = []
            batch_fac = []
            batch_syc = []

            for item in batch_data:
                batch_texts.append(item["clean_text"])

                syc_char = item["sycophantic_token"].strip().replace(
                    "(", "").replace(")", "")
                fac_char = item["factual_token"].strip().replace(
                    "(", "").replace(")", "")

                syc_id = self.tokenizer.encode(
                    syc_char, add_special_tokens=False)[0]
                fac_id = self.tokenizer.encode(
                    fac_char, add_special_tokens=False)[0]

                batch_syc.append(syc_id)
                batch_fac.append(fac_id)

            batches.append((
                batch_texts,
                t.tensor(batch_fac, device=DEVICE),
                t.tensor(batch_syc, device=DEVICE)
            ))

        return batches
