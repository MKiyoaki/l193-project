import os
import json
import argparse
import matplotlib.pyplot as plt
from pathlib import Path

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams["font.family"] = "serif"

def load_and_flatten_data(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing data file: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data_payload = json.load(f)
        raw_data = data_payload.get("data", {})
        metadata = data_payload.get("metadata", {})

    records = []
    clean_score = metadata.get("clean_score", None)
    empty_score = metadata.get("empty_score", 0.5) 

    for key, value in raw_data.items():
        nodes = int(key)
        rate = float(value)
        records.append((nodes, rate))

    records.sort(key=lambda item: item[0])
    
    x_vals = [r[0] for r in records]
    y_vals = [r[1] for r in records]
    
    return x_vals, y_vals, clean_score, empty_score

def plot_normalized_curve(x_sfc, y_sfc, x_dense, y_dense, clean_score, empty_score, experiment_name, save_path):
    plt.figure(figsize=(10, 6))

    def normalize(y_vals):
        if clean_score is None or empty_score is None:
            return y_vals
        denominator = clean_score - empty_score
        if denominator == 0:
            return [0 for _ in y_vals]
        return [(y - empty_score) / denominator for y in y_vals]

    norm_y_sfc = normalize(y_sfc) if x_sfc else []
    norm_y_dense = normalize(y_dense) if x_dense else []

    if x_sfc:
        plt.plot(x_sfc, norm_y_sfc, marker='o', color='#2980b9', linewidth=2.5, 
                 markersize=6, label='SFC (SAE Features Kept)')
    
    if x_dense:
        plt.plot(x_dense, norm_y_dense, marker='s', color='#8e44ad', linewidth=2.5, 
                 markersize=6, label='Dense Neurons Kept')

    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.8, linewidth=2, label='Clean Model (1.0)')
    plt.axhline(y=0.0, color='red', linestyle='--', alpha=0.3, linewidth=2, label='Empty Model (0.0)')

    plt.xscale('linear')
    plt.xlim(-100, 5000)
    plt.ylim(-0.05, 1.05)
    
    plt.title(f"Normalized Faithfulness ({experiment_name.upper()})", fontsize=16, fontweight='bold', pad=15)
    plt.xlabel("Top K Nodes Kept (Linear Scale, Cropped)", fontsize=13)
    plt.ylabel("Normalized Faithfulness of C", fontsize=13)
    
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend(frameon=True, loc='upper left', fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved successfully as '{save_path}'")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="nlp", choices=["nlp", "phi", "political"])
    parser.add_argument("--effects_dir", type=str, default="effects")
    args = parser.parse_args()

    sfc_file = Path(args.effects_dir) / f"global_faithfulness_sfc_{args.experiment_name}.json"
    if not sfc_file.exists():
        sfc_file = Path(args.effects_dir) / f"sfc_faithfulness_results_{args.experiment_name}.json"
    if not sfc_file.exists():
        sfc_file = Path(f"global_faithfulness_sfc_{args.experiment_name}.json")

    dense_file = Path(args.effects_dir) / f"global_faithfulness_dense_{args.experiment_name}.json"
    if not dense_file.exists():
        dense_file = Path(args.effects_dir) / f"dense_faithfulness_results_{args.experiment_name}.json"
    if not dense_file.exists():
        dense_file = Path(f"global_faithfulness_dense_{args.experiment_name}.json")
    
    x_sfc, y_sfc, clean_score_sfc, empty_sfc = [], [], None, 0.5
    if sfc_file.exists():
        x_sfc, y_sfc, clean_score_sfc, empty_sfc = load_and_flatten_data(sfc_file)

    x_dense, y_dense, clean_score_dense, empty_dense = [], [], None, 0.5
    if dense_file.exists():
        x_dense, y_dense, clean_score_dense, empty_dense = load_and_flatten_data(dense_file)

    final_clean = clean_score_sfc if clean_score_sfc is not None else clean_score_dense
    final_empty = empty_sfc if sfc_file.exists() else empty_dense

    save_path = Path("effects") / f"faithfulness_plot_{args.experiment_name}.png"
    save_path.parent.mkdir(exist_ok=True)
    
    plot_normalized_curve(x_sfc, y_sfc, x_dense, y_dense, final_clean, final_empty, args.experiment_name, save_path)

if __name__ == "__main__":
    main()