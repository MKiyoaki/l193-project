# utils/data_processor/data_processor.py

import json
import os
from datasets import load_dataset


class DataProcessor:
    """
    Handles the downloading, extraction, and formatting of datasets from Hugging Face.
    """

    def __init__(self, dataset_name: str, subset_name: str, output_path: str):
        """
        Initializes the DataProcessor with target dataset details and output destination.
        """
        self.dataset_name = dataset_name
        self.subset_name = subset_name
        self.output_path = output_path

    def download_and_format_dataset(self) -> None:
        """
        Downloads the specified dataset subset, extracts relevant fields for the sycophancy task,
        and saves the formatted dictionaries into a local JSONL file.
        """
        dataset = load_dataset(
            self.dataset_name,
            self.subset_name,
            split="validation",
            trust_remote_code=True
        )

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        with open(self.output_path, 'w', encoding='utf-8') as f:
            for item in dataset:
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
