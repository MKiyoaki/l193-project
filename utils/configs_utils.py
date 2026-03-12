# utils/configs_utils.py

import torch

DTYPE = torch.bfloat16
DEVICE = 'cuda:0'
SEED = 42
BATCH_SIZE = 1
N_BATCHES = 100
DATA_PATH = "data/sycophancy_nlp.jsonl"
MODEL_NAME = "google/gemma-2-2b"
EFFECTS_DIR = "effects"
TOP_K_TO_ABLATE = 20
