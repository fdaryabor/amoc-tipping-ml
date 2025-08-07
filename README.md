# AMOC Machine Learning Pipeline

This repository contains the full machine learning workflow to identify and classify weakened Atlantic Meridional Overturning Circulation (AMOC) states from gridded oceanographic datasets.

### Key Features
- Predicts AMOC weakening using gridded data: SST, SSH, SSS, and Greenland runoff.
- Applies PCA for dimensionality reduction.
- Balances dataset using SMOTE.
- Trains and evaluates a Random Forest classifier.
- Generates feature importance maps.
- Provides post-model analysis with visualization.

---

### Repository Structure

| File                              | Description                                           |
|-----------------------------------|-------------------------------------------------------|
| `ml_stage_01.py`                 | Loads, preprocesses, and flattens raw input datasets |
| `apply_pca.py`                   | Computes and saves PCA model based on training data  |
| `ml_stage_02.py`                 | Trains and evaluates the classifier, SHAP analysis   |
| `main_ml_run.py`                 | Executes Stages 1–3 of the ML pipeline sequentially  |
| `trained_amoc_map.py`            | Generates spatial PCA contribution maps              |
| `amoc_ml_post_model_analysis.py` | Post-model visualization & ROC/PR curves             |
| `LICENSE`                        | Usage and distribution terms for the repository      |
| `README.md`                      | Overview and usage guide for the AMOC ML pipeline    |

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

---

### Directory Structure

```
.
├── main_ml_run.py
├── ml_stage_01.py
├── apply_pca.py
├── ml_stage_02.py
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

To run the AMOC Tipping ML Pipeline, users must manually download and preprocess the required datasets. You may use any reliable data source; the recommended datasets are listed below:

#### 1. AMOC Index
Download from the RAPID project:
[https://rapid.ac.uk/data/data-download](https://rapid.ac.uk/data/data-download)

#### 2. ERA5 Monthly SST and Runoff
Copernicus Climate Data Store (CDS):
[https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-monthly-means](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-monthly-means?tab=overview)

#### 3. CMEMS Sea Surface Salinity (SSS)
Copernicus Marine Service:
[MULTIOBS_GLO_PHY_S_SURFACE_MYNRT_015_013](https://data.marine.copernicus.eu/product/MULTIOBS_GLO_PHY_S_SURFACE_MYNRT_015_013/description)

#### 4. CMEMS Sea Surface Height (SSH)
Copernicus Marine Service:
[GLOBAL_MULTIYEAR_PHY_001_030](https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030/description)

> **Note:** It is the user’s responsibility to acquire, preprocess, and organize these datasets according to the pipeline’s input requirements.

---

### Contact

**Author**: Farshid Daryabor  
**Email**: farshiddaryabor7@gmail.com  

Feel free to reach out for collaboration or questions regarding this work.

## License

This project is licensed under the [Apache License 2.0](LICENSE) © 2025 Farshid Daryabor.
