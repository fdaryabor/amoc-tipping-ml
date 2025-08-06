#!/usr/bin/env python
# coding: utf-8

"""
01_ml_model.py

This script is the first stage in a machine learning pipeline for predicting weakened states of the 
Atlantic Meridional Overturning Circulation (AMOC), using gridded oceanographic datasets:
- Sea Surface Temperature (SST)
- Sea Surface Salinity (SSS)
- Sea Surface Height (SSH)
- Greenland freshwater runoff

Workflow:
1. Load and decode gridded datasets (NetCDF format)
2. Align temporal coordinates and filter consistent spatial-temporal points
3. Flatten and clean spatial fields to construct input feature matrices
4. Generate engineered runoff features: lags and SST-runoff interaction
5. Apply spatial and temporal masks to define valid input data
6. Standardize features and save all preprocessed datasets

Important Notes:
- This script **does not apply PCA**. It prepares the scaled dataset (`X_train_scaled.pkl`) 
  required by `apply_pca.py`, which generates the PCA model (`pca_model.pkl`) used in `02_ml_model.py`.
- Outputs include scaler, cleaned features, runoff features, diagnostic plots, and binary AMOC labels.

Outputs:
- Feature matrix: `X.pkl`, `X_train_scaled.pkl`
- Labels: `y_train.pkl`, `y_test.pkl`
- Scaler: `scaler.pkl`
- Intermediate features: `sst_clean.pkl`, `sss_clean.pkl`, `ssh_clean.pkl`, `X_runoff.pkl`
- Diagnostic plots saved to: `figures/`

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


# In[1]:

# ----------------------------
# Function: Decode Time
# ----------------------------
def decode_time(ds, time_var='time'):
    time_num = ds[time_var].values
    time_units = ds[time_var].attrs.get('units', 'seconds since 1970-01-01')
    calendar = ds[time_var].attrs.get('calendar', 'proleptic_gregorian')

    if np.all(time_num > 1e20):
        print(f"All time values in {time_var} are invalid. Skipping decode.")
        full_times = np.full(len(time_num), np.datetime64('NaT'), dtype='datetime64[ns]')
        return ds.assign_coords({time_var: full_times})

    time_num = np.where(time_num > 1e20, np.nan, time_num)
    valid_mask = ~np.isnan(time_num)
    valid_times = time_num[valid_mask]

    decoded = cftime.num2date(valid_times, units=time_units, calendar=calendar)
    full_times = np.full(len(time_num), np.datetime64('NaT'), dtype='datetime64[ns]')
    full_times[valid_mask] = np.array(decoded, dtype='datetime64[ns]')
    return ds.assign_coords({time_var: full_times})


# In[2]:


# ----------------------------
# Step 1: Load & Decode Datasets
# ----------------------------

# Load SST dataset without decoding time
ds_sst = xr.open_dataset('/mnt/f/AMOC_Tipping_ML/data/ERA5_SST_fine_sst_trimmed_rename.nc', decode_times=False)

# Manually assign correct time values for SST (monthly from Feb 1993)
ds_sst['time'] = pd.date_range(start='1993-02-01', periods=370, freq='MS')
sst = ds_sst['sst'] - 273.15

print("SST time dtype:", ds_sst['time'].dtype)

# Load other datasets without decoding time
ds_runoff = xr.open_dataset('/mnt/f/AMOC_Tipping_ML/data/ERA5_runoff_remapped_Greenland_conservative.nc', decode_times=False)
runoff = ds_runoff['runoff']

ds_sss = xr.open_dataset('/mnt/f/AMOC_Tipping_ML/data/cmems_obs-mob_glo_phy-sss_my_multi_fine_sos.nc', decode_times=False)
sss = ds_sss['sos']

ds_ssh = xr.open_dataset('/mnt/f/AMOC_Tipping_ML/data/cmems_obs-sl_glo_phy-ssh_my_allsat-l4_fine_sla_trimmed.nc', decode_times=False)
ssh = ds_ssh['sla']

# Decode time for datasets that need it
runoff = decode_time(runoff)
sss = decode_time(sss)
ssh = decode_time(ssh)

# Note: SST time was manually assigned, so decoding is not needed here

print("All data loaded!")


# In[3]:


# ----------------------------
# Step 2: Align Time
# ----------------------------

def ensure_unique_time(ds):
    # Keep only unique times to avoid duplicates
    _, idx = np.unique(ds.time.values, return_index=True)
    return ds.isel(time=np.sort(idx))

def normalize_time(ds):
    # Convert to pandas datetime, normalize to midnight, assign back
    times = pd.to_datetime(ds.time.values)
    normalized_times = times.normalize()
    return ds.assign_coords(time=normalized_times)

# Remove duplicates from each dataset
sst = ensure_unique_time(sst)
runoff = ensure_unique_time(runoff)
sss = ensure_unique_time(sss).squeeze()
ssh = ensure_unique_time(ssh)

# Normalize time coordinates to midnight for all datasets
sst = normalize_time(sst)
runoff = normalize_time(runoff)
sss = normalize_time(sss)
ssh = normalize_time(ssh)

# Extract normalized times as pd.DatetimeIndex for intersection
sst_times_norm = pd.to_datetime(sst.time.values)
runoff_times_norm = pd.to_datetime(runoff.time.values)
sss_times_norm = pd.to_datetime(sss.time.values)
ssh_times_norm = pd.to_datetime(ssh.time.values)

# time_norm saved (e.g., sst_times_norm)
np.save("sst_times_norm.npy", sst_times_norm)

# Find common intersection of times across all datasets
common_time = sorted(set(sst_times_norm) & set(runoff_times_norm) & set(sss_times_norm) & set(ssh_times_norm))

# Select only the common times from each dataset
sst = sst.sel(time=common_time)
runoff = runoff.sel(time=common_time)
sss = sss.sel(time=common_time)
ssh = ssh.sel(time=common_time)

print(f"Aligned shapes - sst: {sst.shape}, runoff: {runoff.shape}, sss: {sss.shape}, ssh: {ssh.shape}")


# In[4]:


# ----------------------------
# Step 3: Load AMOC Index Labels
# ----------------------------

# Load AMOC (overturning transport in unit of Sv) dataset with automatic decoding of time units
amoc_ds = xr.open_dataset('/mnt/f/AMOC_Tipping_ML/data/moc_transports.nc', decode_times=True)

# Extract AMOC variable and mask invalid values
amoc_var = amoc_ds['moc_mar_hc10'].where(amoc_ds['moc_mar_hc10'] > -9999, np.nan)

# Extract and convert time coordinate to pandas datetime (should be already decoded)
amoc_time = pd.to_datetime(amoc_ds['time'].values)

# Model time from SST dataset
model_time = pd.to_datetime(sst.time.values)

# Align AMOC data to model time by nearest timestamp
amoc_index = pd.Series(amoc_var.values, index=amoc_time).reindex(model_time, method='nearest')

# Define binary target: 1 if AMOC < 15.0 (weakened), else 0
y = (amoc_index < 15.0).astype(int).values

print("Load AMOC Index Labels Done!")


# In[5]:


# ----------------------------
# Step 4: Flatten Gridded Data for ML (With Extended Runoff Features)
# ----------------------------

# 1. Flatten spatial grids
n_time, nlat, nlon = sst.shape
sst_flat = sst.values.reshape((n_time, -1))
sss_flat = sss.values.reshape((n_time, -1))
ssh_flat = ssh.values.reshape((n_time, -1))

# Save them (e.g., in preprocessing output)
np.savez("preprocessed_dims.npz", nlat=nlat, nlon=nlon)
print("Saved nlat and nlon to 'preprocessed_dims.npz'")

# 2. Consistent spatial mask (only valid grid points across all variables at time=0)
valid_mask_combined = ~np.isnan(sst_flat[0]) & ~np.isnan(sss_flat[0]) & ~np.isnan(ssh_flat[0])
sst_flat = sst_flat[:, valid_mask_combined]
sss_flat = sss_flat[:, valid_mask_combined]
ssh_flat = ssh_flat[:, valid_mask_combined]

# 3. Drop columns (grid cells) that have any NaN over time
valid_grid_mask = ~np.isnan(sst_flat).any(axis=0) & ~np.isnan(sss_flat).any(axis=0) & ~np.isnan(ssh_flat).any(axis=0)
sst_clean = sst_flat[:, valid_grid_mask]
sss_clean = sss_flat[:, valid_grid_mask]
ssh_clean = ssh_flat[:, valid_grid_mask]

# 4. Save final spatial mask for later reconstruction
np.save("valid_mask_combined.npy", valid_mask_combined)

# --- Runoff features ---
# Sum and normalize runoff to Sverdrups
runoff_sum = np.nansum(runoff.values.reshape(n_time, -1), axis=1).reshape(-1, 1) / 1e6

# Lagged versions
runoff_lag1 = np.roll(runoff_sum, 1)
runoff_lag2 = np.roll(runoff_sum, 2)
runoff_lag1[0] = np.nan
runoff_lag2[:2] = np.nan

# Runoff-SST interaction feature (optional)
sst_mean = np.mean(sst_clean, axis=1, keepdims=True)
runoff_sst_interaction = runoff_sum * sst_mean

# Combine all runoff-derived features
X_runoff = np.concatenate([runoff_sum, runoff_lag1, runoff_lag2, runoff_sst_interaction], axis=1)

# Time-based mask: drop time steps with any NaNs in runoff-derived features
valid_time_mask2 = ~np.isnan(X_runoff).any(axis=1)

# Save Time-based mask
np.save("valid_time_mask2.npy", valid_time_mask2)

# Fix mismatch in y vs valid_time_mask2 length
if len(valid_time_mask2) > len(y):
    valid_time_mask2 = valid_time_mask2[:len(y)]
elif len(y) > len(valid_time_mask2):
    y = y[:len(valid_time_mask2)]

# 5. Final ML input feature matrix
X = np.concatenate([
    sst_clean[valid_time_mask2],
    sss_clean[valid_time_mask2],
    ssh_clean[valid_time_mask2],
    X_runoff[valid_time_mask2]
], axis=1)

# 6. Apply same time mask to labels
y = y[valid_time_mask2]

# 7. Optional: Plot runoff components (for diagnostics)

plt.figure(figsize=(12, 5))

plt.plot(sst_times_norm[valid_time_mask2], X_runoff[valid_time_mask2][:, 0], label='Runoff', linewidth=2)
plt.plot(sst_times_norm[valid_time_mask2], X_runoff[valid_time_mask2][:, 1], label='Lag 1', linestyle='--')
plt.plot(sst_times_norm[valid_time_mask2], X_runoff[valid_time_mask2][:, 2], label='Lag 2', linestyle=':')

plt.title('Runoff Feature Variants (Sverdrups)')
plt.xlabel('Time')
plt.ylabel('Runoff (Sv)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/runoff_variants_plot.png", dpi=300)
plt.close()

print("Final input X shape:", X.shape)
print("Final label y shape:", y.shape)


joblib.dump(sst_clean, "sst_clean.pkl")
joblib.dump(sss_clean, "sss_clean.pkl")
joblib.dump(ssh_clean, "ssh_clean.pkl")
joblib.dump(X_runoff,  "X_runoff.pkl")
joblib.dump(X,  "X.pkl")

print("Cleaned feature blocks saved.")


# In[6]:


# ----------------------------
# Step 5: PCA + Train/Test Split + SMOTE
# ----------------------------

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = IncrementalPCA(n_components=50)
X_pca = pca.fit_transform(X_scaled)

print("Standardize features & PCA applied!")


# In[7]:


# ----------------------------
# Step 6: Train/Test Split, Handle Imbalance, Train & Save Model
# ----------------------------
# 6.1 - Train/Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print("Before SMOTE:")
print("  - Class distribution in y_train:", np.bincount(y_train))
print("  - Class distribution in y_test :", np.bincount(y_test))

# Save labels
joblib.dump(y_train, "y_train.pkl")
joblib.dump(y_test, "y_test.pkl")
print("y_train and y_test saved.")


# In[8]:


# 6.2 - Standardize (Scale)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, "scaler.pkl")
print("Scaler saved as 'scaler.pkl'")

joblib.dump(X_train_scaled, "X_train_scaled.pkl")
print("X train scaled saved as 'X_train_scaled.pkl'")


# In[9]:
