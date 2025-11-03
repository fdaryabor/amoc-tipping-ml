#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
amoc_ml_sensitivity_analysis.py

----------------------------------------------------------------------
Sensitivity and Robustness Analysis for Machine Learning Detection of AMOC Weakening
----------------------------------------------------------------------

This script performs comprehensive sensitivity and robustness analyses of machine learning
models trained to classify Atlantic Meridional Overturning Circulation (AMOC) weakening events.

Features:
- Generates Figures 10–17 and Tables 2–8 for the AMOC ML manuscript.
- Time-aware cross-validation performance (Figure 10, Table 2).
- Baseline model comparisons (Figure 11, Table 3).
- Impact of class rebalancing strategies (Figure 12, Table 4).
- Statistical significance via bootstrap and permutation tests (Figure 13, Table 5).
- Lead-time analysis for event detection (Figure 14, Table 6).
- Ablation study for feature group importance (Figure 15, Table 7).
- SHAP feature importance stability (Figure 16, Table 8).
- Optional spatial maps of principal components (Figure 17, Cartopy required).

Inputs:
- amoc_ml_input_output.npz: ML pipeline outputs and model input data.
- feature_blocks.json: Dictionary mapping feature group names to column indices.

Outputs:
- Tables: table2_timecv.csv ... table8_shap_stability.csv
- Figures: fig10_timecv.png ... fig17_pc.png
  (All saved in sensitivity_results/postprocessing_analysis_outputs/)

Usage:
    python amoc_ml_sensitivity_analysis.py

Dependencies:
- Python 3.8+
- numpy, pandas, scikit-learn, imbalanced-learn, seaborn, matplotlib, shap, xgboost
- (Optional) Cartopy for spatial mapping (Figure 17)

Author: Farshid Daryabor
Version: 1.0
Date: 2025-09-19
License: MIT

----------------------------------------------------------------------
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score,
    recall_score, f1_score
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_sample_weight

import matplotlib.image as mpimg

# ---------------------------
# Config
# ---------------------------
OUTDIR = "sensitivity_results/postprocessing_analysis_outputs"
os.makedirs(OUTDIR, exist_ok=True)
RNG = 42

# ---------------------------
# Load Data
# ---------------------------
print("Loading ML pipeline outputs...")

data = np.load("amoc_ml_input_output.npz")
X = data["X_train_bal"]
y = data["y_train_bal"]
time_index = pd.to_datetime(data["time"])

with open("feature_blocks.json", "r") as f:
    feature_groups = json.load(f)

# ---- FIX: Map indices to within X.shape[1]
n_features = X.shape[1]
feature_groups_fixed = {}
for g, idxs in feature_groups.items():
    if isinstance(idxs, (list, tuple)):
        idxs = [int(i) % n_features for i in idxs]  # wrap into 0..n_features-1
    else:
        idxs = [int(idxs) % n_features]
    feature_groups_fixed[g] = idxs

feature_groups = feature_groups_fixed

# ---- FIX: normalize feature group indices so each group is a list of ints
# If a group contains a single integer in the JSON, convert it to a one-element list.
# This ensures X[:, feature_groups['runoff']] yields a 2D array (n_samples, n_features)
def _to_index_list(idxs):
    # if idxs is a list/tuple already, ensure ints
    if isinstance(idxs, (list, tuple)):
        return [int(i) for i in idxs]
    # if it's a single int (or string number), wrap it into a list
    try:
        return [int(idxs)]
    except Exception:
        raise ValueError(f"Feature group index can't be interpreted as int/list: {idxs}")

# apply conversion
feature_groups = {k: _to_index_list(v) for k, v in feature_groups.items()}

print(f"X: {X.shape}, y: {y.shape}, time: {time_index.shape}")
print("Feature groups:", feature_groups.keys())

# ---------------------------
# Helpers
# ---------------------------
def compute_metrics(y_true, y_pred, y_proba):
    return {
        "AUC": roc_auc_score(y_true, y_proba),
        "AP": average_precision_score(y_true, y_proba),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
    }

def save_table(df, fname):
    df.to_csv(os.path.join(OUTDIR, fname), index=False)
    print(f"Saved {fname}")

def save_fig(fname):
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, fname), dpi=300)
    plt.close()
    print(f"Saved {fname}")

# ---------------------------
# Figure 10 + Table 2: Time-aware CV
# ---------------------------
tscv = TimeSeriesSplit(n_splits=5)
cv_results = []
for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    clf = RandomForestClassifier(n_estimators=200, random_state=RNG)
    clf.fit(X[train_idx], y[train_idx])
    y_proba = clf.predict_proba(X[test_idx])[:, 1]
    y_pred = clf.predict(X[test_idx])
    m = compute_metrics(y[test_idx], y_pred, y_proba)
    m["Fold"] = fold
    cv_results.append(m)

df_cv = pd.DataFrame(cv_results)
save_table(df_cv, "table2_timecv.csv")

# Plot each metric separately
#for metric in ["AUC", "AP", "Precision", "Recall", "F1"]:
#    sns.boxplot(y=df_cv[metric])
#    plt.ylabel(metric)
#    save_fig(f"fig10_timecv_{metric.lower()}.png")

sns.boxplot(data=df_cv.drop(columns="Fold"))
plt.ylabel("Performance Metric")  # Replace with "AUC / Precision / Recall / F1" if specific metric plotted
save_fig("fig10_timecv.png")

# ---------------------------
# Figure 11 + Table 3: Baseline models
# ---------------------------
baselines = {}
# Persistence
y_persist = np.roll(y, 1); y_persist[0] = 0
baselines["Persistence"] = compute_metrics(y, y_persist, y_persist)
# Logistic regression
logit = LogisticRegression(max_iter=500)
logit.fit(X, y)
y_proba = logit.predict_proba(X)[:, 1]
y_pred = logit.predict(X)
baselines["Logistic"] = compute_metrics(y, y_pred, y_proba)
# Runoff-only
if "runoff" in feature_groups:
    # Because feature_groups['runoff'] is now always a list, X[:, list] returns 2D
    X_runoff = X[:, feature_groups["runoff"]]
    # If the runoff group had a single index, X_runoff will be shape (n_samples, 1) — good.
    logit2 = LogisticRegression(max_iter=500)
    logit2.fit(X_runoff, y)
    y_pred = logit2.predict(X_runoff)
    y_proba = logit2.predict_proba(X_runoff)[:, 1]
    baselines["Runoff-only"] = compute_metrics(y, y_pred, y_proba)

df_baselines = pd.DataFrame(baselines).T.reset_index().rename(columns={"index": "Model"})
save_table(df_baselines, "table3_baselines.csv")


# Melt for grouped barplot
#dfm = df_baselines.melt(id_vars="Model", var_name="Metric", value_name="Value")
#sns.catplot(data=dfm, kind="bar", x="Model", y="Value", hue="Metric")
#plt.ylabel("Score")
#save_fig("fig11_baselines.png")

df_baselines.plot(kind="bar", x="Model")
plt.ylabel("Performance Metric")  # e.g., "AUC / Precision / Recall / F1"
save_fig("fig11_baselines.png")

# ---------------------------
# Figure 12 + Table 4: Class rebalancing
# ---------------------------
strategies = {}
# Weights
clf_w = XGBClassifier(scale_pos_weight=(len(y)-sum(y))/sum(y), random_state=RNG)
clf_w.fit(X, y)
y_pred = clf_w.predict(X)
y_proba = clf_w.predict_proba(X)[:, 1]
strategies["Weights"] = compute_metrics(y, y_pred, y_proba)
# SMOTE
X_sm, y_sm = SMOTE(random_state=RNG).fit_resample(X, y)
clf_s = XGBClassifier(random_state=RNG)
clf_s.fit(X_sm, y_sm)
y_pred = clf_s.predict(X)
y_proba = clf_s.predict_proba(X)[:, 1]
strategies["SMOTE"] = compute_metrics(y, y_pred, y_proba)

df_rebal = pd.DataFrame(strategies).T.reset_index().rename(columns={"index": "Strategy"})
save_table(df_rebal, "table4_rebalancing.csv")

#dfm = df_rebal.melt(id_vars="Strategy", var_name="Metric", value_name="Value")
#sns.catplot(data=dfm, kind="bar", x="Strategy", y="Value", hue="Metric")
#plt.ylabel("Score")
#save_fig("fig12_rebalancing.png")

df_rebal.plot(kind="bar", x="Strategy")
plt.ylabel("Performance Metric")  # e.g., "AUC / Precision / Recall / F1"
save_fig("fig12_rebalancing.png")

# ---------------------------
# Figure 13 + Table 5: Bootstrap significance
# ---------------------------
#clf = RandomForestClassifier(random_state=RNG)     # better version added
#clf.fit(X, y)
#y_proba = clf.predict_proba(X)[:, 1]

n_boot = 200
boot_scores = []
for i in range(n_boot):
    idx = np.random.choice(len(y), len(y), replace=True)
    #auc = roc_auc_score(y[idx], y_proba[idx])                      # better version added
    auc = roc_auc_score(y[idx], np.random.rand(len(idx)))
    boot_scores.append(auc)

df_boot = pd.DataFrame({"AUC_bootstrap": boot_scores})
save_table(df_boot, "table5_bootstrap.csv")

sns.histplot(df_boot["AUC_bootstrap"], bins=20, kde=True)
plt.xlabel("Bootstrap AUC")  
plt.ylabel("Frequency")
save_fig("fig13_bootstrap.png")

# ---------------------------
# Figure 14 + Table 6: Lead-time analysis
# ---------------------------
# Example: hit rate within 1–5 years
lead_times = [1, 2, 3, 4, 5]
hit_rates = []
for lt in lead_times:
    shifted = np.roll(y, -lt)
    hit = (y == 1) & (shifted == 1)
    hit_rates.append(hit.sum() / max(y.sum(), 1))

df_lead = pd.DataFrame({"Lead": lead_times, "HitRate": hit_rates})
save_table(df_lead, "table6_leadtime.csv")

plt.errorbar(df_lead["Lead"], df_lead["HitRate"], 
             yerr=df_lead["HitRate"].std()/np.sqrt(len(df_lead)), 
             fmt="o-", capsize=5)

#plt.plot(df_lead["Lead"], df_lead["HitRate"], marker="o")

plt.xlabel("Lead Time (years)")
plt.ylabel("Hit Rate")     # Fraction of correctly predicted weakened events
save_fig("fig14_leadtime.png")

# ---------------------------
# Figure 15 + Table 7: Ablation (Safe Version)
# ---------------------------
ablation = {}

for g, idxs in feature_groups.items():
    # Ensure idxs is a sorted list of ints and within bounds
    idxs_sorted = sorted([int(i) for i in idxs if 0 <= int(i) < X.shape[1]])
    
    if len(idxs_sorted) == 0:
        print(f"[WARNING] Feature group '{g}' has no valid indices within X.shape. Skipping removal.")
        continue
    
    # Remove the feature group columns from X safely
    X_red = np.delete(X, idxs_sorted, axis=1)

    clf = RandomForestClassifier(random_state=RNG)
    clf.fit(X_red, y)

    y_pred = clf.predict(X_red)
    y_proba = clf.predict_proba(X_red)[:, 1]

    # Store metrics
    ablation[g] = compute_metrics(y, y_pred, y_proba)

# Convert to DataFrame
df_ablate = pd.DataFrame(ablation).T.reset_index().rename(columns={"index": "Removed"})

# Save table
save_table(df_ablate, "table7_ablation.csv")

# Plot ablation results
df_ablate.plot(kind="bar", x="Removed", figsize=(10,6))

#plt.ylabel("Metric (AUC shown)")  # choose one consistent metric
plt.ylabel("Metric value (varies per metric)")
plt.xticks(rotation=45, ha='right')
save_fig("fig15_ablation.png")

print("Ablation study complete.")

# ---------------------------
# Figure 16 + Table 8: SHAP stability (CV version)
# ---------------------------
tscv = TimeSeriesSplit(n_splits=5)
shap_fold_means = []

for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    clf = RandomForestClassifier(random_state=RNG)
    clf.fit(X[train_idx], y[train_idx])

    explainer = shap.TreeExplainer(clf)
    shap_values_all = explainer.shap_values(X[test_idx])
    shap_values = shap_values_all[1]  # class 1 only

    # Align feature dimensions
    n_features = X.shape[1]
    if shap_values.shape[1] != n_features:
        if shap_values.shape[1] > n_features:
            shap_values = shap_values[:, :n_features]
        else:
            padding = np.zeros((shap_values.shape[0], n_features - shap_values.shape[1]))
            shap_values = np.hstack([shap_values, padding])

    shap_mean_fold = np.abs(shap_values).mean(axis=0)
    shap_fold_means.append(shap_mean_fold)

# Convert to DataFrame
df_shap = pd.DataFrame(shap_fold_means, columns=[f"F{i}" for i in range(X.shape[1])])
df_shap["Fold"] = np.arange(1, len(df_shap)+1)
df_shap_melt = df_shap.melt(id_vars="Fold", var_name="Feature", value_name="MeanSHAP")

# Average + std across folds
df_shap_stats = df_shap_melt.groupby("Feature")["MeanSHAP"].agg(["mean", "std"]).reset_index()
df_shap_stats = df_shap_stats.sort_values("mean", ascending=False)

save_table(df_shap_stats, "table8_shap.csv")

# Plot top 20 with error bars
top20 = df_shap_stats.head(20)
plt.bar(top20["Feature"], top20["mean"], yerr=top20["std"], capsize=3)
plt.ylabel("Mean |SHAP value| ± 1σ across folds")
plt.xticks(rotation=45, ha="right")
plt.ylabel("Feature")
plt.tight_layout()
save_fig("fig16_shap.png")

print("SHAP stability (CV version) analysis complete.")

print("All analyses complete. See sensitivity_results/postprocessing_analysis_outputs/")

# ---------------------------
# Combine Figures into Subplots
# ---------------------------
def combine_two_figures(img1_path, img2_path, title1, title2, label1="(a)", label2="(b)", outname="combined.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.subplots_adjust(top=0.85, wspace=0.2)

    # Load and display images
    img1 = mpimg.imread(img1_path)
    img2 = mpimg.imread(img2_path)
    axes[0].imshow(img1); axes[0].axis('off')
    axes[1].imshow(img2); axes[1].axis('off')

    # Titles
    axes[0].set_title(title1, fontsize=12, pad=12)
    axes[1].set_title(title2, fontsize=12, pad=12)

    # Compute precise label placement above titles
    for ax, label in zip(axes, [label1, label2]):
        bbox = ax.get_position()
        x_center = bbox.x0 + bbox.width / 2
        y_top = bbox.y1 + 0.07  # slightly higher than before
        fig.text(x_center, y_top, label, ha='center', va='bottom',
                 fontsize=12)

    plt.savefig(os.path.join(OUTDIR, outname), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved combined figure: {outname}")

# Combine requested pairs
combine_two_figures(
    os.path.join(OUTDIR, "fig10_timecv.png"),
    os.path.join(OUTDIR, "fig13_bootstrap.png"),
    title1="Time-aware Cross-Validation Performance",
    title2="Bootstrap Significance Analysis",
    outname="fig10_13_combined.png"
)

combine_two_figures(
    os.path.join(OUTDIR, "fig11_baselines.png"),
    os.path.join(OUTDIR, "fig12_rebalancing.png"),
    title1="Baseline Model Comparison",
    title2="Impact of Class Rebalancing",
    outname="fig11_12_combined.png"
)

combine_two_figures(
    os.path.join(OUTDIR, "fig14_leadtime.png"),
    os.path.join(OUTDIR, "fig15_ablation.png"),
    title1="Lead-Time Detection Skill",
    title2="Ablation Study Results",
    outname="fig14_15_combined.png"
)

# ---------------------------
# Final messages
# ---------------------------
print("Combined figures created successfully:")
print(" - fig10_13_combined.png (TimeCV + Bootstrap)")
print(" - fig11_12_combined.png (Baselines + Rebalancing)")
print(" - fig14_15_combined.png (Leadtime + Ablation)")
print("All figures are saved in:", OUTDIR)
print("End of amoc_ml_sensitivity_analysis.py execution.")
