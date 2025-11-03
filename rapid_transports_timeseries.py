#!/usr/bin/env python3
# coding: utf-8

"""
===============================================================================
RAPID Atlantic Meridional Overturning Circulation (AMOC) Transport Plot
===============================================================================

Author: Farshid Daryabor
Date: October 2025
Repository: https://github.com/fdaryabor/amoc-tipping-ml

Description:
------------
This script reproduces the characteristic RAPID-MOCHA time series plot of
meridional volume transports across the Atlantic at 26.5°N, as shown in the
official RAPID methodology documentation (https://rapid.ac.uk/methodology).

The figure shows time series (in Sverdrups, 1 Sv = 10⁶ m³/s) for:

    - Gulf Stream transport (Florida Straits, blue)
    - Meridional Overturning Circulation (MOC, red)
    - Ekman transport (green)
    - Upper Mid-Ocean transport (magenta)

Black curves indicate 90-day running means or monthly-mean smoothed versions
of the daily transport data. This visualization provides insight into both
short-term variability and long-term weakening trends in the AMOC system.

Data Source:
------------
The script reads data from `moc_transports.nc`, which contains the following
variables:
    - time (days since 2004-04-01)
    - t_gs10  : Gulf Stream transport [Sv]
    - t_ek10  : Ekman transport [Sv]
    - t_umo10 : Upper Mid-Ocean transport [Sv]
    - moc_mar_hc10 : Overturning (MOC) transport [Sv]

Output:
-------
Creates a folder named `figures_rapid/` containing:
    - rapid_transports.png        : Daily series with 90-day smoothing
    - rapid_transports_monthly.png: Monthly-mean time series (optional)

Dependencies:
-------------
    - Python ≥ 3.8
    - xarray
    - numpy
    - matplotlib
    - cftime

===============================================================================
"""

import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from cftime import num2date

# -----------------------------------------------------------------------------
# 1. Setup output directory
# -----------------------------------------------------------------------------
output_dir = "figures_rapid"
os.makedirs(output_dir, exist_ok=True)

# -----------------------------------------------------------------------------
# 2. Load data from RAPID file (moc_transports.nc)
# -----------------------------------------------------------------------------
file_path = "data/RAPID_2004_2024/moc_transports.nc"
ds = xr.open_dataset(file_path, decode_times=True)

# Robust time decoding
if "units" in ds["time"].attrs:
    time = num2date(ds["time"].values, ds["time"].attrs["units"])
else:
    time = np.array(ds["time"].values)

# Extract transports (replace missing values)
gs = ds["t_gs10"].where(ds["t_gs10"] > -99999).values
moc = ds["moc_mar_hc10"].where(ds["moc_mar_hc10"] > -99999).values
ek = ds["t_ek10"].where(ds["t_ek10"] > -99999).values
umo = ds["t_umo10"].where(ds["t_umo10"] > -99999).values

# -----------------------------------------------------------------------------
# 3. Define helper function for smoothing (running mean)
# -----------------------------------------------------------------------------
def running_mean(x, N=90):
    """Compute a simple running mean of length N (days)."""
    return np.convolve(np.nan_to_num(x, nan=np.nanmean(x)), np.ones(N)/N, mode='same')

gs_smooth = running_mean(gs)
moc_smooth = running_mean(moc)
ek_smooth = running_mean(ek)
umo_smooth = running_mean(umo)

# -----------------------------------------------------------------------------
# 4. Plot RAPID-style daily figure (with 90-day smoothing)
# -----------------------------------------------------------------------------
plt.figure(figsize=(12, 6))

plt.plot(time, gs, color='blue', linewidth=1, label='Gulf Stream')
plt.plot(time, gs_smooth, color='black', linewidth=2)

plt.plot(time, moc, color='red', linewidth=1, label='MOC')
plt.plot(time, moc_smooth, color='black', linewidth=2)

plt.plot(time, ek, color='green', linewidth=1, label='Ekman')
plt.plot(time, ek_smooth, color='black', linewidth=2)

plt.plot(time, umo, color='magenta', linewidth=1, label='Upper Mid-Ocean')
plt.plot(time, umo_smooth, color='black', linewidth=2)

plt.ylabel("Transport (Sv)", fontsize=12)
plt.ylim(-30, 40)
plt.grid(True, linestyle='--', alpha=0.3)
plt.legend(loc='upper center', ncol=4, frameon=False)
plt.xticks(rotation=45)
plt.tight_layout()

daily_fig_path = os.path.join(output_dir, "rapid_transports_daily.png")
plt.savefig(daily_fig_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Daily RAPID transport figure saved at: {daily_fig_path}")

# -----------------------------------------------------------------------------
# 5. Optional: Compute and plot monthly means
# -----------------------------------------------------------------------------
# Reconstruct dataset for time-based resampling
transport_ds = xr.Dataset(
    {
        "Gulf_Stream": (["time"], gs),
        "MOC": (["time"], moc),
        "Ekman": (["time"], ek),
        "Upper_Mid_Ocean": (["time"], umo),
    },
    coords={"time": ("time", time)},
)

# Convert to DataArray and resample to monthly means
monthly_ds = transport_ds.resample(time="1ME").mean()

plt.figure(figsize=(12, 6))

plt.plot(monthly_ds.time, monthly_ds.Gulf_Stream, color='blue', linewidth=1.5, label='Gulf Stream')
plt.plot(monthly_ds.time, monthly_ds.MOC, color='red', linewidth=1.5, label='MOC')
plt.plot(monthly_ds.time, monthly_ds.Ekman, color='green', linewidth=1.5, label='Ekman')
plt.plot(monthly_ds.time, monthly_ds.Upper_Mid_Ocean, color='magenta', linewidth=1.5, label='Upper Mid-Ocean')

plt.ylabel("Transport (Sv)", fontsize=12)
plt.ylim(-30, 40)
plt.grid(True, linestyle='--', alpha=0.3)
plt.legend(loc='upper center', ncol=4, frameon=False)
plt.xticks(rotation=45)
plt.tight_layout()

monthly_fig_path = os.path.join(output_dir, "rapid_transports_monthly.png")
plt.savefig(monthly_fig_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Monthly-mean RAPID transport figure saved at: {monthly_fig_path}")

