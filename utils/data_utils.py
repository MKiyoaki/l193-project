from pathlib import Path


def get_dataset_path(subset_name: str) -> str:
    dataset_collection = {
        "nlp": Path("data") / "sycophancy_nlp.jsonl",
        "phi": Path("data") / "sycophancy_phi.jsonl",
        "political": Path("data") / "sycophancy_political.jsonl",
    }

    return dataset_collection[subset_name]
