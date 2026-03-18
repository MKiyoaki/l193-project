import argparse
from pathlib import Path
from utils.data_processor.data_processor import DataProcessor


def main():
    # 1. Initialize the parser
    parser = argparse.ArgumentParser(
        description="Download and format datasets with specific subsets.")

    # 2. Add arguments
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        required=True,
        help="The name of the dataset (e.g., 'EleutherAI/sycophancy')"
    )
    parser.add_argument(
        "--subset", "-s",
        type=str,
        required=True,
        help="The specific subset to process (e.g., 'sycophancy_on_nlp_survey')"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="output.jsonl",
        help="The name of the output file (e.g., 'sycophancy_nlp.jsonl')"
    )

    args = parser.parse_args()

    # 3. Construct the full output path using the config directory
    full_output_path = Path("data") / args.output

    print(f"Processing dataset: {args.dataset} ({args.subset})")
    print(f"Saving to: {full_output_path}")

    # 4. Initialize and run the processor
    processor = DataProcessor(
        dataset_name=args.dataset,
        subset_name=args.subset,
        output_path=full_output_path
    )

    processor.download_and_format_dataset()
    print("Dataset preprocess Done!")


if __name__ == "__main__":
    main()
