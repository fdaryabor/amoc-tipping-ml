#!/usr/bin/env python3
# coding: utf-8

"""
Plotting Sensitivity Analysis Results
=====================================

Overview
--------
This script visualizes the results from the AMOC sensitivity analysis 
(`ml_sensitivity.py`). It reads the summary table 
`sensitivity_results/sensitivity_summary.csv` and generates both 
individual metric plots and an overlay plot for direct comparison.

Each plot is styled to be publication-ready and includes a vertical 
line/annotation at the "best threshold," defined as the threshold that 
maximizes the F1-score.

Workflow
--------
1. Load Results
   - Reads the summary CSV file produced by `ml_sensitivity.py`.
   - Extracts thresholds and performance metrics.

2. Individual Plots
   - Generates one figure per metric (accuracy, precision, recall, F1-score).
   - Adds a dashed vertical line for the best threshold.
   - Annotates the best F1-score directly on the figure.

3. Overlay Plot
   - Plots all metrics on a single figure with distinct markers and colors.
   - Adds small jitter to overlapping lines for readability.
   - Highlights the best threshold with a vertical line and point marker.

Outputs
-------
Figures are saved in the `figures/` directory:
- `sensitivity_accuracy.png`
- `sensitivity_precision.png`
- `sensitivity_recall.png`
- `sensitivity_f1_score.png`
- `sensitivity_overlay.png`

Requirements
------------
- Python 3.x
- Pandas
- Matplotlib
- CSV file produced by `ml_sensitivity.py`:
    `sensitivity_results/sensitivity_summary.csv`

Usage
-----
Run the script after `ml_sensitivity.py` has completed:

    $ python plot_sensitivity_results.py

This will create individual and overlay plots in the `figures/` folder.

Author
------
Developed by Farshid Daryabor (2025)
"""


import os
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# USER PARAMETERS
# ----------------------------
INPUT_FILE = "sensitivity_results/sensitivity_summary.csv"
OUTPUT_DIR = "figures"

# ----------------------------
# STYLING
# ----------------------------
plt.rcParams.update({
    "font.size": 14,
    "axes.labelweight": "bold",
    "axes.titlesize": 16,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12
})

# ----------------------------
# MAIN SCRIPT
# ----------------------------
def main():
    # Create output directory if it doesn’t exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load sensitivity summary CSV
    df = pd.read_csv(INPUT_FILE)

    # Ensure Threshold column is numeric and sorted
    df["Threshold"] = df["Threshold"].astype(float)
    df = df.sort_values("Threshold")

    # Identify all metrics (exclude Threshold)
    metrics = [col for col in df.columns if col.lower() != "threshold"]

    print(f"[INFO] Available metrics: {metrics}")

    # --- Find best threshold based on max F1-score ---
    if "f1_score" in df.columns:
        best_idx = df["f1_score"].idxmax()
        best_threshold = df.loc[best_idx, "Threshold"]
        best_f1 = df.loc[best_idx, "f1_score"]
        print(f"[INFO] Best threshold = {best_threshold} (F1-score = {best_f1:.3f})")
    else:
        best_threshold, best_f1 = None, None
        print("[WARNING] No f1_score column found. Skipping best threshold line.")

    # --- Individual plots ---
    for metric in metrics:
        thresholds = df["Threshold"].values
        values = df[metric].values

        plt.figure(figsize=(7, 5))
        plt.plot(thresholds, values, marker="o", linestyle="-", linewidth=2, label=metric.capitalize())

        if best_threshold is not None:
            plt.axvline(best_threshold, color="red", linestyle="--", linewidth=1.5,
                        label=f"Best threshold = {best_threshold}")
            if metric == "f1_score":
                plt.scatter(best_threshold, best_f1, color="red", zorder=5)
                plt.text(best_threshold, best_f1, f"  {best_f1:.3f}",
                         va="bottom", ha="left", fontsize=12, color="red", fontweight="bold")

        plt.xlabel("Threshold")
        plt.ylabel(metric.capitalize())
        plt.title(f"Sensitivity Analysis – {metric.capitalize()}")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()

        out_file = os.path.join(OUTPUT_DIR, f"sensitivity_{metric}.png")
        plt.savefig(out_file, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"[INFO] Saved plot: {out_file}")

    # --- Overlay plot ---
    plt.figure(figsize=(8, 6))
    thresholds = df["Threshold"].values

    # Custom styles per metric (distinct markers & colors)
    styles = {
        "accuracy": {"color": "blue", "marker": "s", "zorder": 4, "linewidth": 2.5},
        "precision": {"color": "orange", "marker": "o", "zorder": 3, "alpha": 0.9},
        "recall": {"color": "green", "marker": "D", "zorder": 5, "linewidth": 2},
        "f1_score": {"color": "red", "marker": "^", "zorder": 6, "linewidth": 2},
    }

    # Add small jitter to avoid perfect overlaps (for visualization only)
    jitter_map = {"accuracy": 0.0003, "precision": 0.0002, "recall": -0.0002, "f1_score": 0}
    
    for metric in metrics:
        style = styles.get(metric.lower(), {})
        jitter = jitter_map.get(metric.lower(), 0)
        values = df[metric].values + jitter
        plt.plot(
            thresholds,
            values,
            label=metric.capitalize(),
            linestyle="-",
            markersize=6,
            **style
        )

    # Highlight best threshold
    if best_threshold is not None:
        plt.axvline(best_threshold, color="red", linestyle="--", linewidth=1.5,
                    label=f"Best threshold = {best_threshold}")
        plt.scatter(best_threshold, best_f1, color="red", zorder=7)
        plt.text(best_threshold, best_f1, f"  {best_f1:.3f}",
                 va="bottom", ha="left", fontsize=12, color="red", fontweight="bold")

    plt.xlabel("Threshold", fontweight="bold")
    plt.ylabel("Metric value", fontweight="bold")
    plt.title("Sensitivity Analysis – All Metrics", fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    # Expand y-limits slightly so flat lines don’t hug axes
    y_min, y_max = df[metrics].min().min(), df[metrics].max().max()
    plt.ylim(y_min - 0.001, y_max + 0.001)

    overlay_file = os.path.join(OUTPUT_DIR, "sensitivity_overlay.png")
    plt.savefig(overlay_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[INFO] Saved overlay plot: {overlay_file}")

if __name__ == "__main__":
    main()
