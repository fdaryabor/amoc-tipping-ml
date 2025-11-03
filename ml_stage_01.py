#!/usr/bin/env python
# coding: utf-8

"""
ml_stage_01.py

This script prepares the input dataset for predicting weakened states of the Atlantic Meridional Overturning Circulation (AMOC)
using gridded oceanographic data (SST, SSH, SSS, Greenland runoff) and the AMOC transport index.

Pipeline steps include:
1. Loading and decoding gridded datasets (NetCDF).
2. Time alignment and filtering of valid spatial-temporal data points.
3. Flattening and cleaning spatial grids to generate a consistent feature matrix.
4. Creating time-lagged runoff features and interaction terms.
5. Standardizing and applying PCA (IncrementalPCA) to reduce dimensionality.
6. Splitting into training/testing sets and saving preprocessed data for further modeling.

Notes:
- Saves scaler and scaled training set: 'scaler.pkl', 'X_train_scaled.pkl', and 'X_test_scaled.pkl'
- PCA is applied externally by running `apply_pca.py`, which must be run before proceeding to 'ml_stage_02.py'

Author: Farshid Daryabor
Date: 2025-08-04
"""

import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA

import joblib
import cftime
import shap
import os
import sys
import esmpy

# Local imports
sys.modules['ESMF'] = esmpy

import xesmf as xe

sys.path.append('./scripts')
from data_utils import regrid_to_target
from make_amoc_labels import make_labels

# Ensure figures directory exists
os.makedirs("figures", exist_ok=True)

# ============================================================
# Main pipeline function
# ============================================================
def run_pipeline_01(weak_threshold=15.0, custom_labels=None):
    """Main preprocessing and feature engineering pipeline."""

    if custom_labels is None:
        labels = None
        print(f"[INFO] No custom_labels provided; labels will be generated later using weak_threshold = {weak_threshold} Sv")
    else:
        labels = custom_labels
        print(f"[INFO] Using custom_labels provided with length = {len(labels)}")

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

    # ----------------------------
    # Step 1: Load & Decode Datasets
    # ----------------------------
    ds_sst = xr.open_dataset('data/ERA5_SST_fine_sst_trimmed_rename.nc', decode_times=False)
    ds_sst['time'] = pd.date_range(start='1993-02-01', periods=370, freq='MS')
    sst = ds_sst['sst'] - 273.15

    ds_runoff = xr.open_dataset('data/ERA5_runoff_remapped_Greenland_conservative.nc', decode_times=False)
    runoff = ds_runoff['runoff']

    ds_sss = xr.open_dataset('data/cmems_obs-mob_glo_phy-sss_my_multi_fine_sos.nc', decode_times=False)
    sss = ds_sss['sos']

    ds_ssh = xr.open_dataset('data/cmems_obs-sl_glo_phy-ssh_my_allsat-l4_fine_sla_trimmed.nc', decode_times=False)
    ssh = ds_ssh['sla']

    runoff = decode_time(runoff)
    sss = decode_time(sss)
    ssh = decode_time(ssh)

    print("All data loaded!")

    # ----------------------------
    # Step 2: Align Time
    # ----------------------------
    def ensure_unique_time(ds):
        _, idx = np.unique(ds.time.values, return_index=True)
        return ds.isel(time=np.sort(idx))

    def normalize_time(ds):
        times = pd.to_datetime(ds.time.values)
        normalized_times = times.normalize()
        return ds.assign_coords(time=normalized_times)

    sst = ensure_unique_time(sst)
    runoff = ensure_unique_time(runoff)
    sss = ensure_unique_time(sss).squeeze()
    ssh = ensure_unique_time(ssh)

    sst = normalize_time(sst)
    runoff = normalize_time(runoff)
    sss = normalize_time(sss)
    ssh = normalize_time(ssh)

    sst_times_norm = pd.to_datetime(sst.time.values)
    runoff_times_norm = pd.to_datetime(runoff.time.values)
    sss_times_norm = pd.to_datetime(sss.time.values)
    ssh_times_norm = pd.to_datetime(ssh.time.values)

    np.save("sst_times_norm.npy", sst_times_norm)

    common_time = sorted(set(sst_times_norm) & set(runoff_times_norm) & set(sss_times_norm) & set(ssh_times_norm))

    sst = sst.sel(time=common_time)
    runoff = runoff.sel(time=common_time)
    sss = sss.sel(time=common_time)
    ssh = ssh.sel(time=common_time)

    print(f"Aligned shapes - sst: {sst.shape}, runoff: {runoff.shape}, sss: {sss.shape}, ssh: {ssh.shape}")

    # ----------------------------
    # Step 3: Load and Label AMOC Index
    # ----------------------------
    amoc_ds = xr.open_dataset('data/RAPID_2004_2024/moc_transports.nc', decode_times=True)
    amoc_var = amoc_ds['moc_mar_hc10'].where(amoc_ds['moc_mar_hc10'] > -9999, np.nan)
    amoc_time = pd.to_datetime(amoc_ds['time'].values)
    model_time = pd.to_datetime(sst.time.values)
    amoc_index = pd.Series(amoc_var.values, index=amoc_time).reindex(model_time, method='nearest')

    if labels is None:
        labels = make_labels(amoc_index, threshold=weak_threshold, persistence=3, smoothing=3)
        print(f"[INFO] Labels generated using make_labels(threshold={weak_threshold}, persistence=3, smoothing=3)")

    y = labels.astype(int)
    print("AMOC labels generated successfully!")

    # ----------------------------
    # Step 4: Flatten Gridded Data for ML
    # ----------------------------
    n_time, nlat, nlon = sst.shape
    sst_flat = sst.values.reshape((n_time, -1))
    sss_flat = sss.values.reshape((n_time, -1))
    ssh_flat = ssh.values.reshape((n_time, -1))

    np.savez("preprocessed_dims.npz", nlat=nlat, nlon=nlon)

    valid_mask_combined = ~np.isnan(sst_flat[0]) & ~np.isnan(sss_flat[0]) & ~np.isnan(ssh_flat[0])
    sst_flat = sst_flat[:, valid_mask_combined]
    sss_flat = sss_flat[:, valid_mask_combined]
    ssh_flat = ssh_flat[:, valid_mask_combined]

    valid_grid_mask = ~np.isnan(sst_flat).any(axis=0) & ~np.isnan(sss_flat).any(axis=0) & ~np.isnan(ssh_flat).any(axis=0)
    sst_clean = sst_flat[:, valid_grid_mask]
    sss_clean = sss_flat[:, valid_grid_mask]
    ssh_clean = ssh_flat[:, valid_grid_mask]

    np.save("valid_mask_combined.npy", valid_mask_combined)

    runoff_sum = np.nansum(runoff.values.reshape(n_time, -1), axis=1).reshape(-1, 1) / 1e6
    runoff_lag1 = np.roll(runoff_sum, 1)
    runoff_lag2 = np.roll(runoff_sum, 2)
    runoff_lag1[0] = np.nan
    runoff_lag2[:2] = np.nan

    sst_mean = np.mean(sst_clean, axis=1, keepdims=True)
    runoff_sst_interaction = runoff_sum * sst_mean
    X_runoff = np.concatenate([runoff_sum, runoff_lag1, runoff_lag2, runoff_sst_interaction], axis=1)
    valid_time_mask2 = ~np.isnan(X_runoff).any(axis=1)
    np.save("valid_time_mask2.npy", valid_time_mask2)

    if len(valid_time_mask2) > len(y):
        valid_time_mask2 = valid_time_mask2[:len(y)]
    elif len(y) > len(valid_time_mask2):
        y = y[:len(valid_time_mask2)]

    X = np.concatenate([
        sst_clean[valid_time_mask2],
        sss_clean[valid_time_mask2],
        ssh_clean[valid_time_mask2],
        X_runoff[valid_time_mask2]
    ], axis=1)
    y = y[valid_time_mask2]

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
    joblib.dump(X_runoff, "X_runoff.pkl")
    joblib.dump(X, "X.pkl")

    print("Cleaned feature blocks saved.")

    # ----------------------------
    # Step 5: PCA + Train/Test Split + Scaling
    # ----------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = IncrementalPCA(n_components=50)
    X_pca = pca.fit_transform(X_scaled)

    print("Standardization and PCA applied successfully!")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(indices, test_size=0.3, random_state=42, stratify=y)
    np.save("train_idx.npy", train_idx)
    np.save("test_idx.npy", test_idx)

    joblib.dump(y_train, "y_train.pkl")
    joblib.dump(y_test, "y_test.pkl")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(X_train_scaled, "X_train_scaled.pkl")
    joblib.dump(X_test_scaled, "X_test_scaled.pkl")

    print("[INFO] Preprocessing complete. All intermediate files saved.")

if __name__ == "__main__":
    run_pipeline_01()
