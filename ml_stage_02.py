#!/usr/bin/env python
# coding: utf-8

"""

ml_stage_02.py

This script continues the AMOC ML pipeline after preprocessing by:
1. Loading pre-saved scaled training/test data and applying a previously trained PCA model (`pca_model.pkl`)
   — ensure `apply_pca.py` was run prior to this step.
2. Saving PCA-transformed data and creating mini-batches.
3. Handling class imbalance via SMOTE (Synthetic Minority Over-sampling Technique).
4. Training Random Forest and XGBoost classifiers.
5. Evaluating performance with ROC and Precision-Recall curves.
6. Analyzing feature importance by oceanographic variable blocks based on PCA contributions.
7. Generating SHAP explanations for interpretability.
8. Saving final processed datasets and metadata for reproducibility.

Outputs:
- Trained models and datasets (`trained_amoc_model.pkl`, `X_train_bal.pkl`, etc.)
- Evaluation plots (`figures/`)
- Feature contribution scores (`pca_contrib_data.npz`)
- SHAP visualizations (`shap_summary.png`)
- Reproducible dataset bundle (`amoc_ml_input_output.npz`)

Author: Farshid Daryabor
Date: 2025-08-04

"""

import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

import shap
import joblib
import cftime

import json

from sklearn.impute import SimpleImputer  # optional if you choose imputation
from sklearn.decomposition import PCA

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.decomposition import IncrementalPCA
from xgboost import XGBClassifier
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
from sklearn.preprocessing import StandardScaler

import esmpy
import sys
import os
import glob

import subprocess

sys.modules['ESMF'] = esmpy

import xesmf as xe

sys.path.append('./scripts')
from data_utils import regrid_to_target

# Create 'figures' folder
os.makedirs("figures", exist_ok=True)


def run_pipeline_02():

    # In[1]:

    # -----------------------------------------------------------
    # Step 1: Apply PCA
    # Before running this section, make sure to:
    # 1. Run: $ python apply_pca.py (to create 'pca_model.pkl')
    # 2. Ensure 'X_train_scaled.pkl' and 'X_test_scaled.pkl' exist
    # -----------------------------------------------------------

    # Load scaled training and test data
    X_train_scaled = joblib.load("X_train_scaled.pkl")
    X_test_scaled = joblib.load("X_test_scaled.pkl")
    print("Scaled data loaded:")
    print("  - X_train_scaled shape:", X_train_scaled.shape)
    print("  - X_test_scaled shape :", X_test_scaled.shape)

    # Load previously saved PCA model
    ipca = joblib.load("pca_model.pkl")  # Ensure apply_pca.py has been run

    # Transform the scaled training and test data
    X_train_pca = ipca.transform(X_train_scaled)
    X_test_pca = ipca.transform(X_test_scaled)

    print("PCA transformation applied successfully.")
    print("  - X_train_pca shape:", X_train_pca.shape)
    print("  - X_test_pca shape :", X_test_pca.shape)

    # Save PCA-transformed data
    joblib.dump(X_train_pca, "X_train_pca.pkl")
    joblib.dump(X_test_pca, "X_test_pca.pkl")
    print("PCA-transformed data saved.")


    # In[2]:


    batch_size = 100  # or whatever you used in apply_pca.py

    os.makedirs("pca_batches/train", exist_ok=True)

    for i in range(0, X_train_scaled.shape[0], batch_size):
        X_batch_pca = ipca.transform(X_train_scaled[i:i+batch_size])
        np.save(f"pca_batches/train/X_train_pca_batch_{i}.npy", X_batch_pca)

    print("Transformed training batches saved.")


    # In[3]:


    os.makedirs("pca_batches/test", exist_ok=True)

    for i in range(0, X_test_scaled.shape[0], batch_size):
        X_batch_pca = ipca.transform(X_test_scaled[i:i+batch_size])
        np.save(f"pca_batches/test/X_test_pca_batch_{i}.npy", X_batch_pca)

    print("Transformed testing batches saved.")


    # In[4]:


    # Reconstruct full PCA-transformed train/test data
    train_batches = sorted(glob.glob("pca_batches/train/*.npy"))
    X_train_pca = np.vstack([np.load(f) for f in train_batches])

    test_batches = sorted(glob.glob("pca_batches/test/*.npy"))
    X_test_pca = np.vstack([np.load(f) for f in test_batches])

    print("Combined train and test PCA batches.")


    # In[5]:

    # ------------------------------------------
    # Step 2: SMOTE (balance only training set)
    # ------------------------------------------

    # Load training labels
    y_train = joblib.load("y_train.pkl")
    print("y_train loaded. Shape:", y_train.shape)

    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_pca, y_train)

    print("After SMOTE:")
    print("  - X_train_bal shape:", X_train_bal.shape)
    print("  - y_train_bal distribution:", np.bincount(y_train_bal))

    # Save balanced data
    joblib.dump(X_train_bal, "X_train_bal.pkl")
    joblib.dump(y_train_bal, "y_train_bal.pkl")
    print("Balanced training data saved.")


    # In[6]:

    # ----------------------------------
    # Step 3: Train Random Forest & Save
    # ----------------------------------

    # Load y_test
    y_test = joblib.load("y_test.pkl")
    print("y_test loaded. Shape:", y_test.shape)

    # Train model
    model = RandomForestClassifier(
        n_estimators=200,
        class_weight=None,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_bal, y_train_bal)

    # Save model and data
    joblib.dump(model, 'trained_amoc_model.pkl')
    print("Model trained and saved as 'trained_amoc_model.pkl'")

    joblib.dump(X_train_bal, "X_train_bal.pkl")
    joblib.dump(y_train_bal, "y_train_bal.pkl")

    # In[7]:


    # -------------------------------------------------
    # Step 4: Train Model & Feature Importance by Block
    # -------------------------------------------------

    # Load Train model
    X_train_bal = joblib.load("X_train_bal.pkl")
    y_train_bal = joblib.load("y_train_bal.pkl")
    X_test_pca = joblib.load("X_test_pca.pkl")
    y_test = joblib.load("y_test.pkl")

    from xgboost import XGBClassifier
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        roc_auc_score,
        RocCurveDisplay,
        PrecisionRecallDisplay
    )

    # - X_train_bal: SMOTE-balanced PCA-transformed training data
    # - X_test_pca: PCA-transformed test data
    # - y_train_bal: SMOTE-balanced labels
    # - y_test: test labels from original split

    # Double-check they are in memory
    print("Shapes check:")
    print("  - X_train_bal:", X_train_bal.shape)
    print("  - y_train_bal:", y_train_bal.shape)
    print("  - X_test_pca :", X_test_pca.shape)
    print("  - y_test     :", y_test.shape)

    # Train XGBoost model on balanced PCA data
    xgb_model = XGBClassifier(random_state=42, eval_metric='logloss')
    xgb_model.fit(X_train_bal, y_train_bal)

    # Predict on test set
    y_pred = xgb_model.predict(X_test_pca)

    # Classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Plot ROC Curve
    plt.figure()
    RocCurveDisplay.from_estimator(xgb_model, X_test_pca, y_test)
    plt.title("ROC Curve")
    plt.tight_layout()
    plt.savefig("figures/roc_curve.png", dpi=300)
    plt.close()

    # Plot Precision-Recall Curve
    plt.figure()
    PrecisionRecallDisplay.from_estimator(xgb_model, X_test_pca, y_test)
    plt.title("Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig("figures/precision_recall_curve.png", dpi=300)
    plt.close()

    print("Plots saved to 'figures/' directory.")


    # In[8]:


    # -----------------------------------
    # Step 5: Feature Importance by Block
    # -----------------------------------

    sst_clean = joblib.load("sst_clean.pkl")
    sss_clean = joblib.load("sss_clean.pkl")
    ssh_clean = joblib.load("ssh_clean.pkl")
    X_runoff  = joblib.load("X_runoff.pkl")

    print("Feature blocks loaded:")
    print("  - sst_clean shape:", sst_clean.shape)
    print("  - sss_clean shape:", sss_clean.shape)
    print("  - ssh_clean shape:", ssh_clean.shape)
    print("  - X_runoff shape :", X_runoff.shape)

    # Reconstruct original feature block sizes
    feature_blocks = {
        'sst': sst_clean.shape[1],
        'sss': sss_clean.shape[1],
        'ssh': ssh_clean.shape[1],
        'runoff': X_runoff.shape[1],  # Should be 4: sum, lag1, lag2, interaction
    }

    # Load full feature matrix to confirm total feature count
    X = joblib.load("X.pkl")
    print("X loaded. Shape:", X.shape)

    with open('feature_blocks.json', 'w') as f:
        json.dump(feature_blocks, f)

    # Confirm the total matches the original number of features before PCA
    total_features = sum(feature_blocks.values())
    assert total_features == X.shape[1], f"Feature count mismatch! Got {total_features}, expected {X.shape[1]}"

    # Project PCA components back to original feature space

    # Load saved PCA model from apply_pca.py
    pca = joblib.load("pca_model.pkl")

    pca_back = np.abs(pca.components_)  # shape: [n_components, original_features]
    mean_pca_contrib = pca_back.mean(axis=0)  # average contribution across components

    print("mean_pca_contrib.shape:", mean_pca_contrib.shape)
    print("feature_blocks:", feature_blocks)

    assert sum(feature_blocks.values()) == mean_pca_contrib.shape[0]
    # Compute contribution of each block
    block_scores = {}
    start = 0
    for block, size in feature_blocks.items():
        end = start + size
        block_score = mean_pca_contrib[start:end].sum()
        block_scores[block] = block_score
        start = end

    # Normalize to get relative importances
    total = sum(block_scores.values())
    for k in block_scores:
        block_scores[k] /= total

    # Save mean_pca_contrib and block_scores for later use
    np.savez('pca_contrib_data.npz', mean_pca_contrib=mean_pca_contrib, block_scores=block_scores)

    # Plot
    plt.figure(figsize=(6, 4))
    plt.bar(block_scores.keys(), block_scores.values(), color='steelblue')
    plt.ylabel("Relative Importance")
    plt.title("Feature Importance by Block (from PCA contributions)")
    plt.tight_layout()
    plt.grid(True, axis='y')
    plt.savefig('figures/Feature_Importance_by_Block.png', dpi=300)
    plt.close()

    # Print importance values
    for k, v in block_scores.items():
        print(f"{k} block importance: {v:.6f}")


    # In[9]:


    # ----------------------------
    # Step 6: SHAP Explainability
    # ----------------------------
    # Limit sample size due to SHAP memory requirements
    sample_size = 100
    X_train_sample = X_train_bal[:sample_size]
    X_test_sample = X_test_pca[:sample_size]

    # SHAP explainer for XGBoost
    explainer = shap.Explainer(xgb_model, X_train_sample, feature_names=[f'PC{i+1}' for i in range(X_train_bal.shape[1])])

    # Compute SHAP values
    shap_values = explainer(X_test_sample)

    # Plot summary
    shap.summary_plot(shap_values, X_test_sample, show=False)
    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/shap_summary.png", dpi=300)
    plt.close()

    print("SHAP summary plot saved as 'figures/shap_summary.png'.")


    # In[10]:


    # --------------------------------
    # Step 7: Save for Reproducibility
    # --------------------------------

    # Load saved dimensions
    dims = np.load("preprocessed_dims.npz")
    nlat = int(dims["nlat"])
    nlon = int(dims["nlon"])
    print(f"Loaded dimensions: nlat={nlat}, nlon={nlon}")

    # Load valid_mask_combined, sst_time_norm and valid_time_mask2 from saved file
    valid_mask_combined = np.load("valid_mask_combined.npy")
    sst_times_norm = np.load("sst_times_norm.npy")
    valid_time_mask2 = np.load('valid_time_mask2.npy')

    np.savez("amoc_ml_input_output.npz",
            X_train_bal=X_train_bal,
            y_train_bal=y_train_bal,
            X_test_pca=X_test_pca,
            y_test=y_test,
            sst_shape=(nlat, nlon),
            valid_mask_combined=valid_mask_combined,
            time=sst_times_norm[valid_time_mask2]  # aligned with masked data
    )

    print("Saved AMOC ML input/output to 'amoc_ml_input_output.npz'")


    # In[11]:

  
if __name__ == "__main__":
    run_pipeline_02()

