# AMOC Machine Learning Pipeline

This repository contains the full machine learning workflow to identify and classify weakened Atlantic Meridional Overturning Circulation (AMOC) states from gridded oceanographic datasets.

### Key Features
- Predicts AMOC weakening using gridded data: SST, SSH, SSS, and Greenland runoff.
- Applies PCA for dimensionality reduction.
- Balances dataset using SMOTE.
- Trains and evaluates a Random Forest classifier.
- Generates feature importance maps.
- Provides post-model analysis with visualization.
- Supports sensitivity analysis for AMOC threshold definition.

---

### Repository Structure

| File                              | Description                                           |
|-----------------------------------|-------------------------------------------------------|
| `ml_stage_01.py`                  | Loads, preprocesses, and flattens raw input datasets |
| `apply_pca.py`                    | Computes and saves PCA model based on training data  |
| `ml_stage_02.py`                  | Trains and evaluates the classifier, SHAP analysis   |
| `main_ml_run.py`                  | Executes Stages 1–3 of the ML pipeline sequentially  |
| `ml_sensitivity.py`               | Runs sensitivity tests for AMOC weak threshold; executes full ML pipeline for each threshold |
| `plot_sensitivity_results.py`     | Visualizes sensitivity analysis results across thresholds |
| `trained_amoc_map.py`             | Generates spatial PCA contribution maps              |
| `amoc_ml_post_model_analysis.py`  | Post-model visualization & ROC/PR curves             |
| `LICENSE`                         | Usage and distribution terms for the repository      |
| `README.md`                       | Overview and usage guide for the AMOC ML pipeline    |

---

### Input Data
- ERA5 SST, SSH, SSS (monthly means)
- Greenland Runoff (monthly)
- AMOC Index (for labels)
- All datasets are preprocessed and flattened to 2D format for ML use.

---

### ML Pipeline Overview

The entire AMOC ML workflow is orchestrated through the main runner script:

#### **`main_ml_run.py` – Main Pipeline Executor**
This script serves as the single entry point for executing the full ML workflow, consisting of three main stages:
1. Loading and preprocessing satellite datasets
2. Dimensionality reduction using PCA
3. Model training, evaluation, and feature interpretation

> All sub-steps are encapsulated within this file. Users are **not required to run other scripts individually**.
>
> **To change input file paths**, modify the `Step 1: Load & Decode Datasets` section within `ml_stage_01.py`.

---

### Pipeline Summary

#### **Step 1 – Load and Preprocess Data**
Functionally executed via `ml_stage_01.py`
- Loads and aligns gridded datasets: SST, SSS, SSH, and Runoff
- Reshapes and flattens each field into a 2D matrix (samples × features)
- Standardizes data using `StandardScaler`
- Splits the dataset into train/test groups and saves:
  - `X_train_scaled.pkl`, `X_test_scaled.pkl`, `scaler.pkl`
  - Labels: `y_train`, `y_test`

#### **Step 2 – Apply Dimensionality Reduction (PCA)**
Executed internally through `apply_pca.py`
- Reduces high-dimensional features into top N principal components
- Captures most variance while minimizing overfitting risk
- Saves the PCA model as `pca_model.pkl`

#### **Step 3 – Train and Evaluate Classifier**
Automated in `ml_stage_02.py`
- Applies SMOTE to address class imbalance in training data
- Trains a Random Forest classifier
- Evaluates test performance (accuracy, F1-score, ROC)
- Computes SHAP values and feature contributions
- Saves model and results:
  - `trained_amoc_model.pkl`, `amoc_ml_input_output.npz`
  - SHAP & evaluation plots in `figures/`

#### **Step 4 – Generate Contribution Maps (Optional)**
Run separately: `trained_amoc_map.py`
- Reconstructs 2D spatial maps for SST, SSS, SSH PCA contributions
- Visualizes where features most influence predictions

#### **Step 5 – Post-Model Analysis (Optional)**
Run separately: `amoc_ml_post_model_analysis.py`
- Visualizes model performance: ROC, PR, confusion matrix
- Shows predicted probabilities and temporal trends
- Summarizes feature importance using PCA space and SHAP

#### **Step 6 – Sensitivity Analysis (Optional but Recommended)**
Run separately: `ml_sensitivity.py`
- Performs sensitivity testing of the AMOC weak threshold definition
- Automatically generates binary labels for multiple thresholds (±20% around baseline of 15 Sv)
- Executes the **full ML pipeline (Stages 1–3)** for each threshold in separate subprocesses to avoid memory issues
- Saves results (metrics per threshold) into:
  - `sensitivity_results/metrics/metrics_thr_*.csv`
  - Summary table: `sensitivity_results/sensitivity_summary.csv`

**Note**: Users do **not** need to run `main_ml_run.py` when performing sensitivity tests. Simply run:

```bash
python ml_sensitivity.py
```

#### **Step 7 – Visualize Sensitivity Results**
Run separately: `plot_sensitivity_results.py`
- Reads the summary table from Step 6 (`sensitivity_results/sensitivity_summary.csv`)
- Generates publication-ready plots for each metric (accuracy, precision, recall, F1-score)
- Produces an overlay plot comparing all metrics across thresholds
- Highlights the "best threshold" (based on maximum F1-score) with a vertical line and annotation
- Outputs are saved in the `figures/` folder:
  - `sensitivity_accuracy.png`
  - `sensitivity_precision.png`
  - `sensitivity_recall.png`
  - `sensitivity_f1_score.png`
  - `sensitivity_overlay.png`

Usage:

```bash
python plot_sensitivity_results.py
```

---

### Directory Structure

```
.
├── main_ml_run.py
├── ml_stage_01.py
├── apply_pca.py
├── ml_stage_02.py
├── ml_sensitivity.py
├── plot_sensitivity_results.py
├── trained_amoc_map.py
├── amoc_ml_post_model_analysis.py
├── data/
│   ├── sst.nc
│   ├── sss.nc
│   ├── ssh.nc
│   ├── runoff.nc
│   └── amoc_index.nc
├── models/
│   ├── trained_amoc_model.pkl
│   └── pca_model.pkl
├── figures/
│   └── *.png
├── outputs/
│   └── amoc_ml_input_output.npz
├── sensitivity_results/
│   ├── metrics/
│   │   └── metrics_thr_*.csv
│   └── sensitivity_summary.csv
└── README.md
```

---

### Requirements
Python 3.8+  

Required packages:
- `scikit-learn`
- `xarray`
- `matplotlib`
- `joblib`
- `pandas`
- `numpy`
- `imbalanced-learn`
- `cartopy`

---

### Download Required Datasets

#### 1. AMOC Index
[RAPID Project](https://rapid.ac.uk/data/data-download)

#### 2. ERA5 Monthly SST and Runoff
[Copernicus Climate Data Store (CDS)](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-monthly-means)

#### 3. CMEMS Sea Surface Salinity (SSS)
[Copernicus Marine Service](https://data.marine.copernicus.eu/product/MULTIOBS_GLO_PHY_S_SURFACE_MYNRT_015_013/description)

#### 4. CMEMS Sea Surface Height (SSH)
[Copernicus Marine Service](https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030/description)

> **Note:** Users are responsible for acquiring, preprocessing, and organizing datasets according to pipeline input requirements.

---

### Contact

**Author**: Farshid Daryabor  
**Email**: farshiddaryabor7@gmail.com  

---

## License
Apache License 2.0 © 2025 Farshid Daryabor
