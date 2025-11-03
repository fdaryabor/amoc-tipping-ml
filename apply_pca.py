"""
apply_pca.py
--------------------
This script performs Incremental Principal Component Analysis (IncrementalPCA) 
on a preprocessed and scaled training dataset (`X_train_scaled.pkl`). IncrementalPCA 
is particularly suited for large datasets that do not fit entirely into memory, 
as it processes data in mini-batches.

Workflow:
1. Loads the preprocessed feature matrix `X_train_scaled` using joblib.
2. Initializes an IncrementalPCA model with a specified number of components and batch size.
3. Fits the PCA model iteratively over data batches to compute principal components.
4. Saves the trained PCA model as `pca_model.pkl` for later transformation of training 
   and testing datasets or for model interpretation.

Output:
    - pca_model.pkl : Serialized IncrementalPCA model file

Author: Farshid Daryabor
Date: 2025-08-05
Dependencies:
    - numpy
    - scikit-learn
    - joblib
"""

import joblib
import numpy as np
from sklearn.decomposition import IncrementalPCA

def run_apply_pca():
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

if __name__ == "__main__":
    run_apply_pca()

