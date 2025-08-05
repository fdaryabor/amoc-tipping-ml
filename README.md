# AMOC Tipping ML Pipeline

This repository implements a machine learning (ML) pipeline to predict weakened states of the Atlantic Meridional Overturning Circulation (AMOC) using satellite-derived oceanographic datasets.

### Key Features
- Predicts AMOC weakening using gridded data: SST, SSH, SSS, and Greenland runoff.
- Applies PCA for dimensionality reduction.
- Balances dataset using SMOTE.
- Trains and evaluates Random Forest classifier.
- Generates feature importance maps.
- Provides post-model analysis with visualization.

### Repository Structure

| File                          | Description                                     |
|-------------------------------|-------------------------------------------------|
| `amoc_ml_model.py`            | Main ML pipeline script                         |
| `amoc_ml_model.ipynb`         | Jupyter version of the pipeline                 |
| `amoc_ml_post_model_analysis.ipynb` | Evaluates model and plots PR/ROC curves         |
| `figures/`                    | Folder for saving all generated plots/images    |

### Input Data
- ERA5 SST, SSH, SSS
- Greenland Runoff data
- Pre-processed and flattened to 2D format for ML use

### Pipeline Summary
1. Load & decode gridded datasets
2. Flatten spatial grids and standardize features
3. Apply PCA (saved in `pca_model.pkl`)
4. Handle label imbalance using SMOTE
5. Train Random Forest classifier
6. Evaluate and visualize results

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

---

### Download Required Datasets

To run the AMOC Tipping ML Pipeline, users must manually download and preprocess the required datasets. You may use any reliable data source; the recommended datasets are listed below:

#### 1. AMOC Index
Download from the RAPID project:  
[https://rapid.ac.uk/data/data-download](https://rapid.ac.uk/data/data-download)

#### 2. ERA5 Monthly SST and Runoff
Access through the Copernicus Climate Data Store (CDS):  
[https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-monthly-means](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-monthly-means?tab=overview)

#### 3. CMEMS Sea Surface Salinity (SSS)
Copernicus Marine Environment Monitoring Service (CMEMS):  
[MULTIOBS_GLO_PHY_S_SURFACE_MYNRT_015_013](https://data.marine.copernicus.eu/product/MULTIOBS_GLO_PHY_S_SURFACE_MYNRT_015_013/description)

#### 4. CMEMS Sea Surface Height (SSH)
Copernicus Marine Environment Monitoring Service (CMEMS):  
[GLOBAL_MULTIYEAR_PHY_001_030](https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030/description)

> **Note:** It is the user’s responsibility to acquire, preprocess, and organize these datasets according to the pipeline’s input requirements.

