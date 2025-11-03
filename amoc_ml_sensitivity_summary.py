#!/usr/bin/env python
# coding: utf-8

"""
=====================================================================
 AMOC Sensitivity Tables Generator
=====================================================================

Overview
--------
This script generates the final summary tables (Tables 7–9) for the AMOC
machine learning (ML) sensitivity analysis. These tables are designed
for manuscript reporting and reproducibility, ensuring that results from
different AMOC thresholds are documented in a structured and transparent way.

Scientific Motivation
---------------------
In ML experiments that classify "weak" versus "strong" AMOC states,
the choice of threshold (in Sverdrups, Sv) is critical but somewhat
arbitrary. A common baseline is 15 Sv, but values in the range 12–18 Sv
are often tested to assess robustness. To evaluate sensitivity:

  - **Table 7** reports the class balance (number of weak vs. strong
    AMOC states) across thresholds 12–18 Sv. This is important because
    class imbalance strongly affects ML classification performance.

  - **Table 8** summarizes the classification metrics (accuracy,
    precision, recall, and F1-score) at the baseline threshold of 15 Sv,
    which is the main reference in the manuscript.

  - **Table 9** compares classification metrics across the full range
    of thresholds (12–18 Sv), providing insight into how model skill
    varies with threshold choice and demonstrating robustness of results.

Inputs
------
The script assumes that the following files already exist in
`sensitivity_results/`:

  - `amoc_index_aligned.npy`
      Continuous RAPID-based AMOC index aligned with model input times.
      This is saved during the sensitivity runs (`ml_sensitivity.py`).

  - `sensitivity_summary.csv`
      Summary of classification metrics across thresholds, produced by
      the sensitivity analysis pipeline.

Outputs
-------
The script saves both human-readable and machine-readable tables:

  - `table7_class_balance.csv` / `.txt`
  - `table8_baseline_metrics.csv` / `.txt`
  - `table9_metrics_comparison.csv` / `.txt`

The `.csv` files are structured for reuse in data analysis pipelines.
The `.txt` files are formatted for direct inclusion in a manuscript.

Usage
-----
Run from the command line after completing sensitivity analysis:

    $ python amoc_ml_sensitivity_summary.py

All tables will be generated automatically and saved to
`sensitivity_results/`.

Author
------
Developed by Farshid Daryabor (2025)
=====================================================================
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.utils import shuffle

OUTPUT_DIR = "sensitivity_results/tables"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Load data
# -----------------------------
print("[INFO] Loading AMOC index and ML data...")

data = np.load("amoc_ml_input_output.npz", allow_pickle=True)

# Try to load from the .npz if available, else fallback to saved npy
if "amoc_index" in data.files:
    amoc_index = data["amoc_index"]
else:
    amoc_index = np.load("sensitivity_results/amoc_index_aligned.npy")

# IMPORTANT: use PCA-reduced test features (30 features)
X_test = joblib.load("X_test_pca.pkl")
y_test = joblib.load("y_test.pkl")
model = joblib.load("trained_amoc_model.pkl")

N_total = len(amoc_index)

# -----------------------------
# Table 7: Class balance
# -----------------------------
print("[INFO] Generating Table 7...")

thresholds = np.arange(12, 19, 1)
records = []

for th in thresholds:
    weak = np.sum(amoc_index < th)
    strong = np.sum(amoc_index >= th)
    weak_pct = (weak / N_total) * 100
    records.append([th, weak, strong, weak_pct])

table7 = pd.DataFrame(records, columns=["Threshold (Sv)", "Weak AMOC (n)", "Strong AMOC (n)", "Weak (%)"])
table7.to_csv(os.path.join(OUTPUT_DIR, "table7.csv"), index=False)
print(table7)

# -----------------------------
# Table 8: Permutation test
# -----------------------------
print("[INFO] Generating Table 8...")

# Observed metrics (using test set + trained model)
y_prob = model.predict_proba(X_test)[:, 1]
auc_obs = roc_auc_score(y_test, y_prob)
ap_obs = average_precision_score(y_test, y_prob)

# Null distribution (1000 permutations of labels)
n_perm = 1000
auc_null, ap_null = [], []
rng = np.random.default_rng(seed=42)

for _ in range(n_perm):
    y_perm = rng.permutation(y_test)
    auc_null.append(roc_auc_score(y_perm, y_prob))
    ap_null.append(average_precision_score(y_perm, y_prob))

auc_null = np.array(auc_null)
ap_null = np.array(ap_null)

# Empirical p-values (right-tailed)
p_auc = np.mean(auc_null >= auc_obs)
p_ap = np.mean(ap_null >= ap_obs)

table8 = pd.DataFrame([
    {"Metric": "AUC", "Observed": auc_obs, "Null Mean": auc_null.mean(), "p-value": p_auc},
    {"Metric": "AP", "Observed": ap_obs, "Null Mean": ap_null.mean(), "p-value": p_ap}
])
table8.to_csv(os.path.join(OUTPUT_DIR, "table8.csv"), index=False)
print(table8)

# -----------------------------
# Table 9: False alarms per decade
# -----------------------------
print("[INFO] Generating Table 9...")

lead_time = 36  # 3 years = 36 months
y_pred = (y_prob > 0.5).astype(int)

# Shift predictions for early warning
y_pred_shifted = np.roll(y_pred, -lead_time)
y_pred_shifted[-lead_time:] = 0  # pad end with 0 (no prediction)

true_events = np.sum((y_test == 1) & (y_pred_shifted == 1))
false_alarms = np.sum((y_test == 0) & (y_pred_shifted == 1))

# Per-decade normalization
n_years = N_total / 12
false_per_decade = false_alarms / (n_years / 10)

table9 = pd.DataFrame([{
    "Lead Time": "3 yr",
    "True Events": int(true_events),
    "False Alarms": int(false_alarms),
    "False Alarms / Decade": round(false_per_decade, 2)
}])
table9.to_csv(os.path.join(OUTPUT_DIR, "table9.csv"), index=False)
print(table9)

print(f"\n[SUCCESS] Tables saved under {OUTPUT_DIR}/")

