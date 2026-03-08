# tools/process_dataset.py

from utils.data_processor.data_processor import DataProcessor


if __name__ == "__main__":
    processor = DataProcessor(
        dataset_name="EleutherAI/sycophancy",
        subset_name="sycophancy_on_nlp_survey",
        output_path="/root/autodl-tmp/projects/feature-circuits/data/sycophancy_nlp.jsonl"
    )
    processor.download_and_format_dataset()
