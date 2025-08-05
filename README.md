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

| File | Description |
|------|-------------|
|  | Main ML pipeline script |
|  | Applies PCA to training data and saves PCA model |
|  | Evaluates model and plots PR/ROC curves |
|  | Visualizes spatial importance of features |

### Input Data
- ERA5 SST, SSH, SSS
- Greenland Runoff data
- Pre-processed and flattened to 2D format for ML use

### Pipeline Summary
1. Load & decode gridded datasets
2. Flatten spatial grids and standardize features
3. Apply PCA (saved in )
4. Handle label imbalance using SMOTE
5. Train Random Forest classifier
6. Evaluate and visualize results

### Requirements
Python 3.8+, scikit-learn, xarray, matplotlib, joblib, pandas, numpy, imbalanced-learn

