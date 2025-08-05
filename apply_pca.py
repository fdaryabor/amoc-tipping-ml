# apply_pca.py

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

