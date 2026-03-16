import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import interpolate

# Set global style for academic reporting
sns.set_theme(style="whitegrid")
plt.rcParams["font.family"] = "serif"


def load_data(filepath):
    """Load JSON results from local storage."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing data file: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_interpolated_values(x_raw, y_raw, x_target):
    """Linearly interpolate y values at target x coordinates."""
    # Ensure data is sorted by X for interpolation
    sorted_idx = np.argsort(x_raw)
    x_raw, y_raw = np.array(x_raw)[sorted_idx], np.array(y_raw)[sorted_idx]

    # Use linear interpolation with edge-value filling
    f = interpolate.interp1d(
        x_raw, y_raw, kind='linear', fill_value="extrapolate")
    return f(x_target)


def main():
    # 1. Load experimental results
    sfc_res = load_data("sfc_completeness_results.json")
    dense_res = load_data("dense_completeness_results.json")

    # Map data into the structure used in original paper's analysis
    # outs[method][chunk_name][k_nodes]
    outs = {
        'SAE Features (SFC)': sfc_res['data'],
        'Dense Neurons (Baseline)': {'Global': dense_res['data']}
    }

    colors = {
        'SAE Features (SFC)': '#3498db',    # Blue
        'Dense Neurons (Baseline)': '#9b59b6'  # Purple
    }

    plt.figure(figsize=(9, 5))

    for method, subouts in outs.items():
        all_chunks = list(subouts.keys())

        # Determine X-axis boundaries across all data points
        all_x_coords = []
        for chunk in all_chunks:
            all_x_coords.extend([int(k) for k in subouts[chunk].keys()])

        x_min, x_max = min(all_x_coords), max(all_x_coords)

        # Generate 100 points in log space for smooth mean curve (matching original paper)
        xs_smooth = np.logspace(np.log10(x_min), np.log10(x_max), 100)

        interpolated_matrix = []

        # 2. Plot individual traces (low opacity) and collect for mean calculation
        for chunk in all_chunks:
            x_raw = [int(k) for k in subouts[chunk].keys()]
            y_raw = [float(subouts[chunk][str(k)]) for k in x_raw]

            # Draw faint lines for individual data streams
            sns.lineplot(
                x=x_raw, y=y_raw, color=colors[method], alpha=0.15, linewidth=1, legend=False)

            # Interpolate for the aggregate mean
            y_interp = get_interpolated_values(x_raw, y_raw, xs_smooth)
            interpolated_matrix.append(y_interp)

        # 3. Calculate and plot the aggregate Mean Curve
        y_mean = np.mean(interpolated_matrix, axis=0)
        plt.plot(xs_smooth, y_mean,
                 color=colors[method], linewidth=2.5, label=method)

    # Final visual refinements
    plt.xscale('log')
    plt.ylim(0.5, 0.9)  # Focus on the relevant Sycophancy Rate range
    plt.title("Completeness Curve: SAE Features vs. Dense Neurons",
              fontsize=14, fontweight='bold')
    plt.xlabel("Nodes Ablated (Log Scale)", fontsize=12)
    plt.ylabel("Sycophancy Rate", fontsize=12)
    plt.legend(frameon=True, loc='upper right')

    plt.tight_layout()
    plt.savefig("completeness_report_plot.pdf", dpi=300)
    plt.show()
    print("Plot successfully saved as 'completeness_report_plot.pdf'")


if __name__ == "__main__":
    main()
