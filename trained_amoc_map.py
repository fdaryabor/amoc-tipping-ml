#!/usr/bin/env python
# coding: utf-8

"""
=====================================================================
                          trained_amoc_map.py
=====================================================================

This script generates spatial contribution maps for variables used in AMOC 
(Atlantic Meridional Overturning Circulation) machine learning analysis. 
It uses pre-computed PCA contribution data and climate reanalysis datasets 
to highlight which regions and features are most influential in the model.

Workflow:
1. Load saved ML input/output data and PCA contributions from `.npz` files.
2. Load SST (ERA5), SSS (CMEMS), and SSH (CMEMS) gridded datasets with latitude/longitude.
3. Rebuild a spatial mask to keep only valid ocean grid cells without missing values.
4. Organize features into four blocks:
   - SST (Sea Surface Temperature, spatial)
   - SSS (Sea Surface Salinity, spatial)
   - SSH (Sea Surface Height, spatial)
   - Runoff (2×2 aggregated block, non-spatial)
5. Create plots:
   - Runoff: small heatmap with four labeled regions.
   - SST, SSS, SSH: global maps with coastlines, borders, and individual colorbars.
6. Save all figures into the `figures/` directory for further analysis or publication.

Purpose:
These maps provide a visual interpretation of feature contributions, 
helping identify key spatial drivers in PCA-reduced ML models for AMOC tipping point studies.

Author
------
Developed by Farshid Daryabor (2025)
=====================================================================

"""

import os
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr

# ------------------------- Ensure Figures Directory Exists -------------------------
fig_dir = "figures"
os.makedirs(fig_dir, exist_ok=True)
print(f"Saving figures to directory: {fig_dir}")

# ------------------------- Load Saved Data -------------------------
data = np.load("amoc_ml_input_output.npz", allow_pickle=True)
contrib_data = np.load("pca_contrib_data.npz")

mean_pca_contrib = contrib_data["mean_pca_contrib"]  # (total_features,)
sst_shape = data["sst_shape"]                        # (525, 601)

# ------------------------- Load lat/lon -------------------------
ds_sst = xr.open_dataset("/mnt/f/AMOC_Tipping_ML/data/ERA5_SST_fine_sst_trimmed_rename.nc", decode_times=False)
lon = ds_sst["lon"].values                           # (601,)
lat = ds_sst["lat"].values                           # (525,)
lon2d, lat2d = np.meshgrid(lon, lat)

sst = ds_sst['sst'].values - 273.15  # convert K to °C

# ------------------------- Load other datasets -------------------------
ds_sss = xr.open_dataset('/mnt/f/AMOC_Tipping_ML/data/cmems_obs-mob_glo_phy-sss_my_multi_fine_sos.nc', decode_times=False)
sss = ds_sss['sos'].values  # (time, lat, lon)

ds_ssh = xr.open_dataset('/mnt/f/AMOC_Tipping_ML/data/cmems_obs-sl_glo_phy-ssh_my_allsat-l4_fine_sla_trimmed.nc', decode_times=False)
ssh = ds_ssh['sla'].values  # (time, lat, lon)

# ------------------------- Rebuild final PCA valid mask -------------------------
valid_mask_combined = np.load("valid_mask_combined.npy")  # shape: (nlat*nlon,)

n_time, nlat, nlon = sst.shape

# Flatten spatial dimensions
sst_flat = sst.reshape(n_time, -1)
sss_flat = sss.reshape(n_time, -1)
ssh_flat = ssh.reshape(n_time, -1)

# Apply time=0 mask
sst_masked = sst_flat[:, valid_mask_combined]
sss_masked = sss_flat[:, valid_mask_combined]
ssh_masked = ssh_flat[:, valid_mask_combined]

# Filter out any grid cell with NaNs over time for all variables
valid_grid_mask = (~np.isnan(sst_masked).any(axis=0)) & \
                  (~np.isnan(sss_masked).any(axis=0)) & \
                  (~np.isnan(ssh_masked).any(axis=0))

# Build final mask over original flattened grid
final_valid_mask = np.zeros(nlat * nlon, dtype=bool)
final_valid_mask[np.where(valid_mask_combined)[0][valid_grid_mask]] = True

print(f"Initial valid points at time=0: {valid_mask_combined.sum()}")
print(f"Valid points after temporal filtering: {final_valid_mask.sum()}")
print(f"PCA contributions length: {mean_pca_contrib.shape[0]}")

# Validate PCA contributions length matches expected size (3 spatial blocks + runoff)
expected_pca_length = final_valid_mask.sum() * 3 + 4  # SST + SSS + SSH + runoff
assert mean_pca_contrib.shape[0] == expected_pca_length, \
    "PCA contributions length does not match expected total features!"

valid_indices = np.where(final_valid_mask)[0]

# ------------------------- Define Feature Blocks -------------------------
block_sizes = [
    final_valid_mask.sum(),  # SST
    final_valid_mask.sum(),  # SSS
    final_valid_mask.sum(),  # SSH
    4                       # runoff (non-spatial)
]
block_names = ["sst", "sss", "ssh", "runoff"]
block_starts = np.cumsum([0] + block_sizes[:-1])

# ------------------------- Process and Plot -------------------------
#for name, start, size in zip(block_names, block_starts, block_sizes):
#    print(f"Processing block: {name.upper()}")

#    if name == "runoff":
#        print("Skipping runoff (non-spatial).")
#        continue

# ------------------------- Process and Plot -------------------------

for name, start, size in zip(block_names, block_starts, block_sizes):
    print(f"Processing block: {name.upper()}")

    if name == "runoff":
        print("Processing runoff (non-spatial 2x2 block).")

        contrib = mean_pca_contrib[start:start + size]

        # Define shape (must match what was used during training)
        runoff_shape = (2, 2)
        if np.prod(runoff_shape) != size:
            print(f"Skipping runoff: shape {runoff_shape} incompatible with size {size}")
            continue

        contrib_map = contrib.reshape(runoff_shape)

        # Plot heatmap with custom labels
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(contrib_map, cmap="viridis", interpolation='nearest')

        ax.set_title("PCA Contribution Map: RUNOFF", fontsize=14)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Region 1", "Region 2"])
        ax.set_yticklabels(["Region 3", "Region 4"])
        ax.set_xlabel("Runoff Regions (X)")
        ax.set_ylabel("Runoff Regions (Y)")

        cbar = plt.colorbar(im, ax=ax, orientation='vertical', label="Relative Importance")
        plt.tight_layout()

        fig_path = os.path.join(fig_dir, "pca_contrib_runoff.png")
        plt.savefig(fig_path, dpi=300)
        print(f"Saved: {fig_path}")

        continue  # Skip the rest of the loop for 'runoff'

    # --- For spatial blocks (sst, sss, ssh) ---

    spatial_blocks = ["sst", "sss", "ssh"]

    fig, axes = plt.subplots(
        nrows=3, ncols=1, figsize=(8, 14),
        subplot_kw={"projection": ccrs.PlateCarree()}
    )

    panel_labels = ["(a)", "(b)", "(c)"]
    titles = ["PCA Contribution Map: SST",
              "PCA Contribution Map: SSS",
              "PCA Contribution Map: SSH"]

    for ax, name, label, title in zip(axes, spatial_blocks, panel_labels, titles):
        start = block_starts[block_names.index(name)]
        size = block_sizes[block_names.index(name)]
        contrib = mean_pca_contrib[start:start + size]

        contrib_map_flat = np.full(np.prod(sst_shape), np.nan)
        contrib_map_flat[valid_indices] = contrib
        contrib_map = contrib_map_flat.reshape(sst_shape)

        im = ax.pcolormesh(
            lon2d, lat2d, contrib_map,
            transform=ccrs.PlateCarree(),
            cmap="viridis"
        )

        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)

        # Two-line title
        ax.set_title(f"{label}\n{title}", fontsize=12, loc="center")

        # Gridlines
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--")
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {"size": 9}
        gl.ylabel_style = {"size": 9}

        # Individual colorbar
        cbar = fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.025, pad=0.04)
        cbar.set_label("Relative Importance")

    plt.subplots_adjust(hspace=0.25, top=0.92, bottom=0.05, left=0.08, right=0.90)
    fig_path = os.path.join(fig_dir, "pca_contrib_sst_sss_ssh_subplot.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {fig_path}")
 
