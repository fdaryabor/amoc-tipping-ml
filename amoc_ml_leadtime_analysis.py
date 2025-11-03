#!/usr/bin/env python3
# ==========================================================
#  amoc_ml_leadtime_analysis.py
# ==========================================================
"""
Purpose:
--------
Generates two supplementary figures (S1–S2) and a summary text file
to assess the predictive lead-time behavior of the machine-learning (ML)
framework developed to detect early-warning signals of AMOC weakening.

It compares model-predicted weak-AMOC probabilities with the RAPID
AMOC index and labels, quantifying temporal alignment between
surface-based precursors and observed AMOC slowdowns.

Inputs:
-------
Required files in the working directory:
    - trained_amoc_model.pkl     : Trained ML classifier (Logistic/Random Forest/XGBoost)
    - X_train_pca.pkl, X_test_pca.pkl
    - y_train.pkl, y_test.pkl
    - train_idx.npy, test_idx.npy
    - sst_times_norm.npy         : Corresponding monthly timestamps (datetime64 or numeric)
Optional:
    - moc_transports.nc          : RAPID AMOC index (if available for overlay)

Outputs:
--------
All results are saved in ./Supplementary_Figures/:

    1. Supplementary_Fig_S1_timeseries.png
       → Continuous time series showing predicted probability, RAPID weak months, and normalized AMOC index.

    2. Supplementary_Fig_S2_leadtime_histogram.png
       → Histogram showing distribution of predictive lead times (in months).

    3. leadtime_stats.txt
       → Summary of median and interquartile range (IQR) of lead times.

Author:
-------
Farshid Daryabor (2025)

Date: 2025-09-04
"""

# ==========================================================
#  Imports
# ==========================================================
import os
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import joblib
from scipy.signal import find_peaks

# ==========================================================
# 1. Output directory
# ==========================================================
outdir = "Supplementary_Figures"
os.makedirs(outdir, exist_ok=True)
print(f"[INFO] Output directory created: {outdir}/")

# ==========================================================
# 2. Load ML model, data, and reconstruct full dataset
# ==========================================================
print("[INFO] Loading trained model and data...")

model = joblib.load("trained_amoc_model.pkl")
X_train = joblib.load("X_train_pca.pkl")
X_test = joblib.load("X_test_pca.pkl")
y_train = joblib.load("y_train.pkl")
y_test = joblib.load("y_test.pkl")
train_idx = np.load("train_idx.npy")
test_idx = np.load("test_idx.npy")

# ----------------------------------------------------------
# Reconstruct full PCA dataset (X_all_pca, y_all)
# ----------------------------------------------------------
n_total = len(train_idx) + len(test_idx)
X_all = np.empty((n_total, X_train.shape[1]))
y_all = np.empty(n_total)

X_all[train_idx] = X_train
X_all[test_idx] = X_test
y_all[train_idx] = y_train
y_all[test_idx] = y_test

print(f"[INFO] Reconstructed full dataset: {X_all.shape[0]} samples, {X_all.shape[1]} PCA features")

# Load timestamps
time_all = np.load("sst_times_norm.npy")
time_all = pd.to_datetime(time_all)
print(f"[INFO] Loaded {len(time_all)} timestamps spanning {time_all.min().date()} → {time_all.max().date()}")

# ----------------------------------------------------------
# Handle small mismatch between time and data lengths
# ----------------------------------------------------------
if len(time_all) != len(X_all):
    diff = len(time_all) - len(X_all)
    print(f"[WARN] Length mismatch detected: time={len(time_all)}, X_all={len(X_all)} (diff={diff})")

    if diff > 0:
        # Trim extra timestamps from the beginning
        time_all = time_all[diff:]
        print(f"[INFO] Trimmed first {diff} timestamps → new length = {len(time_all)}")
    elif diff < 0:
        # Trim extra samples if X_all is longer
        X_all = X_all[:len(time_all)]
        y_all = y_all[:len(time_all)]
        print(f"[INFO] Trimmed extra samples → new length = {len(X_all)}")

# ==========================================================
# 3. Predict over full time domain
# ==========================================================
print("[INFO] Generating continuous predictions across full timeline...")
prob = model.predict_proba(X_all)[:, 1]
labels = np.array(y_all).astype(int)

# Sort by time for proper plotting
sort_idx = np.argsort(time_all)
time = time_all[sort_idx]
prob = prob[sort_idx]
labels = labels[sort_idx]

# ==========================================================
# 4. Load and align RAPID AMOC index (optional)
# ==========================================================
amoc_available = False
try:
    nc_path = "data/RAPID_2004_2024/moc_transports.nc"
    try:
        ds = xr.open_dataset(nc_path)
    except TypeError:
        ds = xr.open_dataset(nc_path, decode_times=True)

    rapid_time = pd.to_datetime(ds["time"].values)
    amoc_raw = ds["moc_mar_hc10"].where(ds["moc_mar_hc10"] > -9999, np.nan)
    ds.close()

    # Interpolate RAPID AMOC to ML timestamps
    amoc_index = pd.Series(amoc_raw.values, index=rapid_time).reindex(time, method="nearest")

    # Restrict to overlapping time range
    t_min, t_max = rapid_time[0], rapid_time[-1]
    ml_mask = (time >= (t_min - pd.Timedelta(days=31))) & (time <= (t_max + pd.Timedelta(days=31)))

    if not np.any(ml_mask):
        print(f"[WARN] No full overlap found — extending by ±1 month around RAPID range.")
        ml_mask = np.ones_like(time, dtype=bool)

    time, prob, labels = time[ml_mask], prob[ml_mask], labels[ml_mask]
    amoc_index = amoc_index[ml_mask]
    print(f"[INFO] Overlapping time window used: {time.min().date()} → {time.max().date()}")

    # Normalize AMOC
    amoc_index = amoc_index - np.nanmin(amoc_index)
    if np.nanmax(amoc_index) > 0:
        amoc_index /= np.nanmax(amoc_index)
    else:
        amoc_index[:] = 0.0

    amoc_available = True
    print(f"[INFO] RAPID AMOC index loaded and aligned: {t_min.date()} → {t_max.date()}")

except Exception as e:
    print(f"[WARN] Could not load RAPID AMOC index: {e}")
    amoc_index = np.zeros_like(prob)

# ==========================================================
# 5. Identify peaks and compute lead times
# ==========================================================
peak_indices, _ = find_peaks(prob, height=0.5, distance=2)
event_indices = np.where(labels == 1)[0]

lead_times = []
for p in peak_indices:
    next_events = event_indices[event_indices > p]
    if len(next_events) > 0:
        lead = next_events[0] - p
        if 0 < lead <= 12:
            lead_times.append(lead)

lead_times = np.array(lead_times)
if len(lead_times) == 0:
    raise RuntimeError("No valid lead times found — check threshold or labels.")

median_lt = np.median(lead_times)
iqr_low, iqr_high = np.percentile(lead_times, [25, 75])

# ==========================================================
# 6. Figures S1–S3: Predicted AMOC Probability and RAPID Overlay
# ==========================================================

legend_kwargs = dict(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3)

# ---------- Figure S1: Time Series Overlay ----------
fig, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(time, prob, color="black", lw=1.5, label="Predicted Weak-AMOC Probability")
ax1.scatter(time[labels == 1], prob[labels == 1], color="red", s=25,
            label="RAPID Weak-State Months", zorder=3)
ax1.set_xlabel("Time")
ax1.set_ylabel("Predicted Probability", color="black")
ax1.tick_params(axis="y", labelcolor="black")
ax1.grid(alpha=0.3)

# Format x-axis as dates
ax1.xaxis.set_major_locator(mdates.YearLocator())       # major ticks every year
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y')) # show year
ax1.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(1,7))) # minor ticks Jan & Jul
fig.autofmt_xdate(rotation=45)  # rotate labels for readability

if amoc_available:
    ax2 = ax1.twinx()
    ax2.plot(time, amoc_index, color="royalblue", lw=1.2, label="Normalized RAPID AMOC Index")
    ax2.set_ylabel("Normalized AMOC Index", color="royalblue")
    ax2.tick_params(axis="y", labelcolor="royalblue")
    ax2.set_ylim(0, 1)

# Combine legends below plot
legend_kwargs = dict(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3)
lines, labels_ = ax1.get_legend_handles_labels()
if amoc_available:
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines += lines2
    labels_ += labels2
ax1.legend(lines, labels_, **legend_kwargs)

#plt.title("Predicted Weak-AMOC Probability vs RAPID Observations (Supplementary Fig. S1)")
plt.title("Predicted Weak-AMOC Probability vs RAPID Observations")
plt.tight_layout()
plt.savefig(os.path.join(outdir, "Supplementary_Fig_S1_timeseries.png"), dpi=300, bbox_inches='tight')
plt.close()
print(f"[INFO] Figure S1 saved: {outdir}/Supplementary_Fig_S1_timeseries.png")

# ---------- Figure S2: Lead-Time Histogram ----------
plt.figure(figsize=(6, 4))
plt.hist(lead_times, bins=np.arange(0.5, 12.5, 1), edgecolor="black", color="steelblue")
plt.xlabel("Lead time before RAPID-defined weak-AMOC event (months)")
plt.ylabel("Frequency")
#plt.title("Distribution of Predictive Lead Times (Supplementary Fig. S2)")
plt.title("Distribution of Predictive Lead Times")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(outdir, "Supplementary_Fig_S2_leadtime_histogram.png"), dpi=300)
plt.close()
print(f"[INFO] Figure S2 saved: {outdir}/Supplementary_Fig_S2_leadtime_histogram.png")

# ---------- Figure S3: AMOC Weakening Probability Over Time ----------
if amoc_available:
    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(time, prob, color="black", lw=1.5, label="Predicted Weak-AMOC Probability")
    ax1.scatter(time[labels == 1], prob[labels == 1], color="royalblue", s=25,
                label="RAPID Weak-State Months", zorder=3)
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Predicted Probability", color="black")
    ax1.tick_params(axis="y", labelcolor="black")
    ax1.grid(alpha=0.3)

    # Format x-axis
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax1.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(1,7)))
    fig.autofmt_xdate(rotation=45)

    ax2 = ax1.twinx()
    ax2.plot(time, amoc_index, color="red", lw=1.2, label="Normalized RAPID AMOC Index")
    ax2.set_ylabel("Normalized AMOC Index", color="red")
    ax2.tick_params(axis="y", labelcolor="red")
    ax2.set_ylim(0, 1)

    # Combine legends below plot
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, **legend_kwargs)

    plt.title("AMOC Weakening Probability Over Time (Model vs RAPID Observations)")
    plt.tight_layout()
    fname = os.path.join(outdir, "Supplementary_Fig_S3_AMOC_weakening_probability.png")
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Figure S3 saved: {fname}")

else:
    print("[WARN] RAPID AMOC index not available — skipping Figure S3 overlay plot.")


# ==========================================================
# 7. Save summary statistics
# ==========================================================
summary_path = os.path.join(outdir, "leadtime_stats.txt")
with open(summary_path, "w") as f:
    f.write("Supplementary Figures S1–S2: Predictive Lead-Time Analysis\n")
    f.write("----------------------------------------------------------\n")
    f.write(f"Median lead time : {median_lt:.1f} months\n")
    f.write(f"IQR              : {iqr_low:.1f} – {iqr_high:.1f} months\n")
    f.write(f"Sample count     : {len(lead_times)} probability peaks\n")
    f.write(f"AMOC overlay     : {'included' if amoc_available else 'not available'}\n")
    f.write("Figures:\n")
    f.write("  - Supplementary_Fig_S1_timeseries.png\n")
    f.write("  - Supplementary_Fig_S2_leadtime_histogram.png\n")

print(f"[INFO] Median lead time = {median_lt:.1f} months (IQR {iqr_low:.1f}–{iqr_high:.1f})")
print(f"[INFO] Summary written to {summary_path}")

