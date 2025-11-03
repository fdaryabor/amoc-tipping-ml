#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AMOC ML Post-Model Analysis Script

-------------------------------------------------------------------------------
Description:
This script performs comprehensive post-analysis of a machine learning model
trained to predict Atlantic Meridional Overturning Circulation (AMOC) weakening
using climate variables such as SST, SSS, SSH, and runoff. It includes:

- Model performance evaluation (confusion matrix, ROC, Precision-Recall curves)
- Prediction visualization over time
- Feature importance analysis via PCA components
- Spatial visualization of variable contributions on geographic maps
- Detection of AMOC tipping points based on probability thresholds

Designed to provide scientific understanding of physical drivers, temporal
patterns, and spatial features critical to AMOC state transitions.

-------------------------------------------------------------------------------
Workflow Summary:
1. Load trained ML model, PCA-transformed feature sets, labels, and time data.
2. Evaluate model using standard metrics and visualization plots.
3. Predict AMOC state probabilities on test data; plot time series of predictions.
4. Analyze and visualize PCA-based feature importances and spatial patterns.
5. Detect statistically relevant tipping points from model output probabilities.
6. Save all relevant figures to the 'figures/' directory.

-------------------------------------------------------------------------------
Usage:
Ensure prerequisite files, including the trained model and PCA data, are available.
Run from the command line:

    python amoc_ml_post_model_analysis.py

-------------------------------------------------------------------------------
Outputs:
- Evaluation plots (ROC, Precision-Recall, confusion matrix)
- Time series plots of predicted AMOC probabilities
- Spatial importance maps (low and high resolution)
- Tipping points detection visualization
- All outputs saved under 'post_analysis/figures/' directory

-------------------------------------------------------------------------------
Dependencies:
- Python 3.8+
- numpy, pandas, matplotlib, seaborn, xarray, scikit-learn
- joblib, cartopy for spatial plotting

-------------------------------------------------------------------------------
Author: Farshid Daryabor
Date: 2025-08-04
License: MIT License
-------------------------------------------------------------------------------
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score

# Config constants
FIGURE_DIR = 'post_analysis/figures'
os.makedirs(FIGURE_DIR, exist_ok=True)

MODEL_PATH = 'trained_amoc_model.pkl'
DATA_PATH = 'amoc_ml_input_output.npz'


def load_model_and_data():
    model = joblib.load(MODEL_PATH)
    data = np.load(DATA_PATH)
    return model, data


def add_subplot_label(ax, label):
    ax.text(0.5, 1.10, label, transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='bottom', ha='center')


def plot_combined_roc_pr(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    avg_precision = average_precision_score(y_true, y_prob)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # (a) ROC
    add_subplot_label(axes[0], '(a)')
    axes[0].plot(fpr, tpr, color='darkorange', label=f"AUC = {roc_auc:.2f}")
    axes[0].plot([0, 1], [0, 1], linestyle='--', color='gray')
    axes[0].set_title('ROC Curve')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].legend()

    # (b) Precision-Recall
    add_subplot_label(axes[1], '(b)')
    axes[1].plot(recall, precision, color='blue', label=f"Avg Precision = {avg_precision:.2f}")
    axes[1].set_title('Precision-Recall Curve')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'roc_pr_combined.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_pca_importances_combined(model):
    importances = model.feature_importances_
    feature_blocks = {'sst': 10, 'sss': 8, 'ssh': 6, 'runoff': 5}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    start = 0
    for idx, (name, size) in enumerate(feature_blocks.items()):
        block_imp = importances[start:start+size]
        start += size
        # Fix seaborn warning: assign hue to x values
        sns.barplot(x=np.arange(size), y=block_imp, hue=np.arange(size), palette='viridis', ax=axes[idx], legend=False)
        add_subplot_label(axes[idx], f'({chr(97+idx)})')
        axes[idx].set_title(f'PCA Component Importance: {name.upper()}')
        axes[idx].set_xlabel('PCA Component Index')
        axes[idx].set_ylabel('Importance')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'pca_importance_combined.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_tipping_and_probability(df, threshold=0.5):
    df['tipping_point'] = (df['y_prob'] > threshold) & (df['y_prob'].shift(1) <= threshold)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # (a) Tipping Points Detection
    add_subplot_label(axes[0], '(a)')
    axes[0].plot(df['time'], df['y_prob'], label='Predicted Probability (Weak AMOC)')
    axes[0].scatter(df.loc[df['tipping_point'], 'time'],
                    df.loc[df['tipping_point'], 'y_prob'], color='orange', s=50, label='Tipping Point')
    axes[0].set_title('AMOC Tipping Points Detection')
    axes[0].set_ylabel('Probability')
    axes[0].legend()

    # (b) Probability Over Time
    add_subplot_label(axes[1], '(b)')
    axes[1].plot(df['time'], df['y_prob'], label='Predicted Probability (Weak AMOC)', color='blue')
    axes[1].scatter(df['time'], df['y_true'], color='red', s=10, label='True Class')
    axes[1].set_title('AMOC Weakening Probability Over Time')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Probability / Class')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'tipping_probability_combined.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Load trained model and test data
    model, data = load_model_and_data()
    X_test = data['X_test_pca']
    y_test = data['y_test']
    time = data['time']

    # Model predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Strong', 'Weak'],
                yticklabels=['Strong', 'Weak'])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'confusion_matrix.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # Build dataframe for probability & time-based plots
    df = pd.DataFrame({
        'time': pd.to_datetime(time[-len(y_test):]),
        'y_true': y_test,
        'y_prob': y_prob
    })

    # Plot combined metrics and analyses
    plot_combined_roc_pr(y_test, y_prob)
    plot_pca_importances_combined(model)
    plot_tipping_and_probability(df)


if __name__ == '__main__':
    main()

