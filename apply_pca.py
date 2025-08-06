# apply_pca.py

"""
apply_pca.py

This script applies dimensionality reduction to the standardized training dataset
(`X_train_scaled.pkl`) generated in `01_ml_model.py` using Incremental Principal Component Analysis (IncrementalPCA).

Purpose:
- Efficiently fit a PCA model in mini-batches to reduce feature dimensionality.
- Save the trained PCA model (`pca_model.pkl`) for use in `02_ml_model.py`.

Workflow:
1. Load the preprocessed and scaled training dataset
2. Apply IncrementalPCA with the specified number of components and batch size
3. Save the trained PCA model to disk

Parameters:
- n_components: 30 principal components
- batch_size: 100 (used for partial fitting)

Important:
- Ensure that `batch_size >= n_components` to satisfy `IncrementalPCA` requirements
  and avoid runtime errors during partial fitting.

Output:
- Trained PCA model: `pca_model.pkl`

Author: Farshid Daryabor
Date: 2025-08-04
"""


import joblib
import numpy as np
from sklearn.decomposition import IncrementalPCA

# Load preprocessed X_train_scaled
X_train_scaled = joblib.load("X_train_scaled.pkl")

# Set parameters
n_components = 30
batch_size = 100
ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)

# Fit PCA in batches
for i in range(0, X_train_scaled.shape[0], batch_size):
    ipca.partial_fit(X_train_scaled[i:i+batch_size])

# Save the fitted PCA model
joblib.dump(ipca, "pca_model.pkl")
print("PCA model saved as 'pca_model.pkl'")

