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

    # For spatial blocks (sst, sss, ssh)
    contrib = mean_pca_contrib[start:start + size]

    contrib_map_flat = np.full(np.prod(sst_shape), np.nan)
    contrib_map_flat[valid_indices] = contrib
    contrib_map = contrib_map_flat.reshape(sst_shape)

    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    im = ax.pcolormesh(lon2d, lat2d, contrib_map,
                       transform=ccrs.PlateCarree(),
                       cmap="viridis")

    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.set_title(f"PCA Contribution Map: {name.upper()}", fontsize=14)

    # Add lat/lon gridlines with labels
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False   # remove top labels
    gl.right_labels = False # remove right labels
    gl.xlabel_style = {"size": 10}
    gl.ylabel_style = {"size": 10}

    plt.colorbar(im, ax=ax, orientation='vertical', label="Relative Importance")
    plt.tight_layout()

    fig_path = os.path.join(fig_dir, f"pca_contrib_{name.lower()}.png")
    plt.savefig(fig_path, dpi=300)
    print(f"Saved: {fig_path}")

   
