#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
amoc_ml_spatial_contributions.py

----------------------------------------------------------------------
Generate Spatial Contribution Maps for AMOC Machine Learning Features
----------------------------------------------------------------------

Description:
This script generates spatial maps visualizing the contribution of different 
variables used in the Atlantic Meridional Overturning Circulation (AMOC) 
machine learning analysis. It uses pre-computed PCA contribution data along 
with climate reanalysis datasets to identify key spatial drivers influencing 
the ML model.

Workflow:
1. Load ML input/output and PCA contribution data from `.npz` files.
2. Load gridded SST (ERA5), SSS (CMEMS), and SSH (CMEMS) datasets.
3. Generate and apply a spatial mask to retain valid ocean grid cells.
4. Segment features into blocks: SST, SSS, SSH (spatial), and runoff (non-spatial).
5. Generate and save global maps with coastlines, borders, and colorbars.
6. Save figures in the `figures/` directory for analysis or publication.

Purpose:
Visual interpretation of PCA-reduced ML feature importance, aiding identification
of spatial patterns related to AMOC tipping points.

Usage:
    python amoc_ml_spatial_contributions.py

Dependencies:
- Python 3.8+
- numpy, matplotlib, cartopy, xarray

Author:
Farshid Daryabor (2025)
Date: 26-09-2025 

License:
MIT License

----------------------------------------------------------------------
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
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
ds_sst = xr.open_dataset("data/ERA5_SST_fine_sst_trimmed_rename.nc", decode_times=False)
lon = ds_sst["lon"].values                           # (601,)
lat = ds_sst["lat"].values                           # (525,)
lon2d, lat2d = np.meshgrid(lon, lat)

sst = ds_sst['sst'].values - 273.15  # convert K to °C

# ------------------------- Load other datasets -------------------------
ds_sss = xr.open_dataset('data/cmems_obs-mob_glo_phy-sss_my_multi_fine_sos.nc', decode_times=False)
sss = ds_sss['sos'].values  # (time, lat, lon)

ds_ssh = xr.open_dataset('data/cmems_obs-sl_glo_phy-ssh_my_allsat-l4_fine_sla_trimmed.nc', decode_times=False)
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

spatial_blocks = ["sst", "sss", "ssh"]

fig, axes = plt.subplots(
    nrows=3, ncols=1, figsize=(8, 14),
    subplot_kw={"projection": ccrs.PlateCarree()}
)

panel_labels = ['(a)', '(b)', '(c)']
titles = ['PCA Contribution Map: SST',
          'PCA Contribution Map: SSS',
          'PCA Contribution Map: SSH']

for ax, name, label, title in zip(axes, spatial_blocks, panel_labels, titles):
    print(f"Processing block: {name.upper()}")
    start = block_starts[block_names.index(name)]
    size = block_sizes[block_names.index(name)]
    contrib = mean_pca_contrib[start:start + size]

    # Reconstruct full map
    contrib_map_flat = np.full(np.prod(sst_shape), np.nan)
    contrib_map_flat[valid_indices] = contrib
    contrib_map = contrib_map_flat.reshape(sst_shape)

    # Plot data
    im = ax.pcolormesh(
        lon2d, lat2d, contrib_map,
        transform=ccrs.PlateCarree(),
        cmap="viridis"
    )

    # Coastlines and borders
    ax.coastlines(resolution='50m', linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)

    # RAPID array latitude (26° N): dashed red line
    ax.plot(
        lon2d[0, :], np.full(lon2d.shape[1], 26.0),
        color='red', linewidth=1.3, linestyle='--',
        transform=ccrs.PlateCarree(), zorder=5
    )

    # White text label above red line (use PlateCarree so it's at data coords).
    # Use clip_on=False and a stroke for readability.
    ax.text(
        np.nanmax(lon2d) - 5, 26.8, "26° N (RAPID)",
        color='white', fontsize=10, fontweight='bold',
        transform=ccrs.PlateCarree(),
        ha='right', va='bottom', zorder=10,
        clip_on=False,
        path_effects=[path_effects.withStroke(linewidth=2, foreground="black")]
    )

    # Numeric ticks (data coordinates)
    ax.set_xticks(np.arange(np.floor(lon2d.min()), np.ceil(lon2d.max()) + 1, 10))
    ax.set_yticks(np.arange(np.floor(lat2d.min()), np.ceil(lat2d.max()) + 1, 10))
    ax.set_xticklabels([f"{x:.0f}" for x in np.arange(np.floor(lon2d.min()), np.ceil(lon2d.max()) + 1, 10)])
    ax.set_yticklabels([f"{y:.0f}" for y in np.arange(np.floor(lat2d.min()), np.ceil(lat2d.max()) + 1, 10)])
    ax.set_xlabel("Longitude", fontsize=10)
    ax.set_ylabel("Latitude", fontsize=10)

    # Gridlines
    ax.gridlines(linewidth=0.4, color='gray', alpha=0.5, linestyle='--')

    # ------------------ PANEL LETTER ABOVE TITLE (CENTERED) ------------------
    # Place the letter centered above the title using axes fraction coords
    ax.text(
        0.5, 1.12, label,
        transform=ax.transAxes,
        ha='center', va='bottom',
        fontsize=13
        #fontsize=13, fontweight='bold'
    )

    # Title centered under the letter; pad increased to leave room for panel letter
    ax.set_title(title, fontsize=12, loc="center", pad=18)

    # Individual colorbar
    cbar = fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.025, pad=0.04)
    cbar.set_label("Relative Importance")

# Layout adjustment (tweak hspace or top if letter overlaps)
plt.subplots_adjust(hspace=0.42, top=0.92, bottom=0.06, left=0.08, right=0.92)
fig_path = os.path.join(fig_dir, "pca_contrib_sst_sss_ssh_subplot_with_26N_corrected.png")
plt.savefig(fig_path, dpi=300, bbox_inches="tight")
print(f"Saved: {fig_path}")
 
