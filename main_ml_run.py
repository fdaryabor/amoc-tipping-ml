#!/usr/bin/env python
# coding: utf-8

"""
main_ml_run.py

This script orchestrates the full machine learning pipeline for predicting weakened AMOC states
by executing the following three modules in sequence:

1. `ml_stage_01.py` – Prepares input features and targets from gridded oceanographic datasets.
2. `apply_pca.py` – Applies Incremental PCA to the standardized training dataset.
3. `ml_stage_02.py` – Trains machine learning classifiers on PCA-reduced data and evaluates performance.

Ensure that each module exposes a callable function:
- run_pipeline_01() in ml_stage_01.py
- run_apply_pca() in apply_pca.py
- run_pipeline_02() in ml_stage_02.py

Author: Farshid Daryabor
Date: 2025-08-06
"""

from ml_stage_01 import run_pipeline_01
from apply_pca import run_apply_pca
from ml_stage_02 import run_pipeline_02


def main():
    print("\n==> Stage 1: Running ml_stage_01 (ML Model Preparation)...")
    run_pipeline_01()
    print("Stage 1 completed.\n")

    print("==> Stage 2: Running apply_pca (PCA Dimensionality Reduction)...")
    run_apply_pca()
    print("Stage 2 completed.\n")

    print("==> Stage 3: Running ml_stage_02 (Model Training and Evaluation)...")
    run_pipeline_02()
    print("Stage 3 completed.\n")

    print("=" * 60)
    print("All pipeline stages completed successfully!")
    print("Your AMOC ML workflow is complete.")
    print("Output files and models are saved in the working directory.")
    print("=" * 60)


if __name__ == "__main__":
    main()



