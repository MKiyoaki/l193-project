# utils/configs_utils.py

import torch

DTYPE = torch.bfloat16
DEVICE = 'cuda:0'
SEED = 42
BATCH_SIZE = 1
N_BATCHES = 100

DATA_OUTPUT_DIR = "/root/autodl-tmp/projects/feature-circuits/data/"

DATA_PATH = "data/sycophancy_nlp.jsonl"
DATA_CONTROLLED_PATH = None

MODEL_NAME = "google/gemma-2-2b"
EFFECTS_DIR = "effects"
TOP_K_TO_ABLATE = 20
