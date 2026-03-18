import json
import os
from datasets import load_dataset
from tqdm import tqdm


class DataProcessor:
    """
    Handles the downloading, extraction, and formatting of datasets from Hugging Face.
    """

    def __init__(self, dataset_name: str, subset_name: str, output_path: str):
        self.dataset_name = dataset_name
        self.subset_name = subset_name
        self.output_path = output_path

    def download_and_format_dataset(self) -> None:
        """
        Downloads the dataset, extracts fields, and saves to JSONL with a progress bar.
        """
        print(f"Loading {self.dataset_name} ({self.subset_name})...")

        dataset = load_dataset(
            self.dataset_name,
            self.subset_name,
            split="validation",
            trust_remote_code=True
        )

        # Ensure output directory exists
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        print(f"Writing data to {self.output_path}")

        with open(self.output_path, 'w', encoding='utf-8') as f:
            # Use tqdm for a clean progress bar
            for item in tqdm(dataset, desc="Processing rows", unit="rows"):
                question = item.get('question', '')
                match_behavior = item.get('answer_matching_behavior', '')
                not_match_behavior = item.get(
                    'answer_not_matching_behavior', '')

                formatted_data = {
                    "clean_text": question,
                    "sycophantic_token": match_behavior,
                    "factual_token": not_match_behavior
                }

                f.write(json.dumps(formatted_data) + '\n')
