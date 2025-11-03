#!/usr/bin/env python
# coding: utf-8

"""
=====================================================================
 AMOC Sensitivity Analysis Tool
=====================================================================

Overview
--------
This script (`ml_sensitivity.py`) performs a sensitivity analysis of the 
Atlantic Meridional Overturning Circulation (AMOC) classification by 
testing multiple thresholds for defining "weak AMOC" events. It reruns 
the full machine learning (ML) classification pipeline at each threshold 
and evaluates how classification metrics (accuracy, precision, recall, 
and F1-score) vary with respect to the chosen threshold.

Motivation
----------
The strength of AMOC is often summarized by its maximum overturning 
streamfunction at ~26°N. In ML-based prediction experiments, defining 
what constitutes a "weak AMOC" requires setting a threshold (e.g., 
15 Sv). Because this threshold is somewhat arbitrary, sensitivity 
analysis is necessary to assess robustness of results. 

This script automates that analysis by:
  1. Generating binary labels for weak/non-weak AMOC months at 
     multiple thresholds around a baseline (±20% of 15 Sv).
  2. Running the existing ML pipeline with those labels.
  3. Collecting evaluation metrics into a summary table for comparison.

Workflow
--------
For each threshold value:
  1. **Label generation**  
     - Reads AMOC time series from `data/moc_transports.nc`.  
     - Aligns AMOC with model input times (`sst_times_norm.npy`).  
     - Generates binary labels: 1 = weak AMOC, 0 = non-weak.  
     - Saves labels as `.npy` file in `sensitivity_results/labels/`.

  2. **ML pipeline execution** (in a separate subprocess)  
     - Stage 1: `run_pipeline_01(custom_labels=...)`  
     - Stage 2: `run_apply_pca()`  
     - Stage 3: `run_pipeline_02()`  
     - Saves metrics (`accuracy`, `precision`, `recall`, `f1_score`) 
       as CSV in `sensitivity_results/metrics/`.

  3. **Result aggregation**  
     - Reads all per-threshold metrics.  
     - Saves them into `sensitivity_results/sensitivity_summary.csv`.  

Key Features
------------
- Runs each threshold in an isolated subprocess (avoids memory buildup).  
- Automatically skips thresholds that produce only one class.  
- Produces per-threshold metrics and a final summary table.  
- Designed for robustness in HPC or long-running workflows.  

Inputs
------
- `data/moc_transports.nc` : NetCDF file containing AMOC transport time series.  
- `sst_times_norm.npy`     : Numpy array of model SST time indices (used for alignment).  

Outputs
-------
- `sensitivity_results/labels/*.npy` : Binary label arrays for each threshold.  
- `sensitivity_results/metrics/*.csv`: Classification metrics for each threshold.  
- `sensitivity_results/sensitivity_summary.csv`: Final summary table of all metrics.  

Usage
-----
Run from the command line:
    $ python ml_sensitivity.py

No arguments are required. Results will be saved under the 
`sensitivity_results/` directory. Ensure that prerequisite data files 
(`moc_transports.nc` and `sst_times_norm.npy`) exist in the correct 
locations.

Visualization
-------------
After running this script, the results can be visualized using the 
companion plotting script:

    $ python plot_sensitivity_results.py

This will generate publication-ready plots of each metric (accuracy, 
precision, recall, F1-score) versus the AMOC weak threshold, as well 
as an overlay plot comparing all metrics. Plots are saved in:

    sensitivity_results/sensitivity_plot_*.png

Dependencies
------------
- Python 3.8+
- numpy, pandas, xarray
- Custom modules: ml_stage_01, apply_pca, ml_stage_02

Author
------
Developed by Farshid Daryabor (2025)
Date: 2025-08-10
=====================================================================
"""

import os
# --- Limit threads to reduce memory footprint (optional but helpful) ---
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import xarray as xr
import multiprocessing as mp
import traceback
import gc
import time

# We import stage modules INSIDE the worker to keep the parent light.
# from ml_stage_01 import run_pipeline_01
# from apply_pca import run_apply_pca
# from ml_stage_02 import run_pipeline_02

# ----------------------------
# Configuration
# ----------------------------
BASE_THRESHOLD = 15.0
THRESHOLDS = [
    BASE_THRESHOLD * 0.8,  # -20%
    BASE_THRESHOLD * 0.9,  # -10%
    BASE_THRESHOLD,        # baseline
    BASE_THRESHOLD * 1.1,  # +10%
    BASE_THRESHOLD * 1.2   # +20%
]

OUTPUT_DIR = "sensitivity_results"
LABEL_DIR = os.path.join(OUTPUT_DIR, "labels")
METRIC_DIR = os.path.join(OUTPUT_DIR, "metrics")

# Ensure the main output folder exists first
os.makedirs(LABEL_DIR, exist_ok=True)
os.makedirs(METRIC_DIR, exist_ok=True)

def generate_labels(threshold):
    """Generate binary weak/non-weak labels for given AMOC threshold."""
    ds = xr.open_dataset('data/RAPID_2004_2024/moc_transports.nc', decode_times=True)
    amoc = ds['moc_mar_hc10'].where(ds['moc_mar_hc10'] > -9999, np.nan)
    amoc_time = pd.to_datetime(ds['time'].values)
    ds.close()

    # Load SST/model time index (assumes preprocessed npy exists)
    sst_time = pd.to_datetime(np.load("sst_times_norm.npy", allow_pickle=True))

    # Align AMOC to model time
    amoc_aligned = pd.Series(amoc.values, index=amoc_time).reindex(sst_time, method="nearest")
    
    # Save the continuous RAPID AMOC index aligned to model time axis 
    np.save("sensitivity_results/amoc_index_aligned.npy", amoc_aligned.values)
    print("[INFO] Saved aligned RAPID AMOC index to sensitivity_results/amoc_index_aligned.npy")

    # Weak month labels
    labels = (amoc_aligned < threshold).astype(int).values
    return labels

def _worker_per_threshold(th, labels_path, metric_path, return_queue=None):
    """
    Worker process that:
      - loads labels (np.load)
      - runs Stage 1 with custom_labels
      - runs Stage 2
      - runs Stage 3
      - saves metrics CSV to metric_path
    """
    try:
        # Import heavy modules here so they stay in the child process
        from ml_stage_01 import run_pipeline_01
        from apply_pca import run_apply_pca
        from ml_stage_02 import run_pipeline_02

        # Load labels saved by parent
        labels = np.load(labels_path, allow_pickle=True)

        # Run pipeline stages (unchanged flow)
        run_pipeline_01(custom_labels=labels)
        # free labels asap
        del labels
        gc.collect()

        run_apply_pca()

        metrics = run_pipeline_02()

        # Handle None (failed run)
        if metrics is None:
            # Write a stub failed file so the parent can continue
            fail_df = pd.DataFrame([{"status": "failed"}])
            fail_df.to_csv(metric_path.replace(".csv", "_FAILED.csv"), index=False)
            if return_queue is not None:
                return_queue.put(("error", th, "metrics returned None"))
            return

        # Ensure dict before saving
        if not isinstance(metrics, dict):
            metrics = dict(metrics)

        pd.DataFrame([metrics]).to_csv(metric_path, index=False)

        # Optional: send back metrics
        if return_queue is not None:
            return_queue.put(("ok", th, metrics))

    except Exception as e:
        # Capture traceback for debugging
        err = f"[ERROR] Threshold {th:.1f} Sv failed: {e}\n" + traceback.format_exc()
        print(err, flush=True)
        if return_queue is not None:
            return_queue.put(("error", th, str(e)))

def run_sensitivity():
    results = {}
    ctx = mp.get_context("spawn")  # ensures a fresh interpreter per process
    retq = ctx.Queue()

    for th in THRESHOLDS:
        print(f"\n=== Threshold: {th:.1f} Sv ===", flush=True)

        # Step 1: Generate & (temporarily) save labels
        labels = generate_labels(th)

        # Skip if all labels are the same (no weak events OR everything weak)
        unique = np.unique(labels)
        if len(unique) < 2:
            print(f"[WARNING] Threshold {th:.1f} Sv -> Only one class present ({unique[0]}). Skipping...")
            continue

        label_file = os.path.join(LABEL_DIR, f"labels_thr_{th:.1f}.npy")
        # Save small dtype to minimize disk/memory
        np.save(label_file, labels.astype(np.int8))
        print(f"[INFO] Labels saved at {label_file}", flush=True)

        # Step 2–4 inside a separate process
        metric_file = os.path.join(METRIC_DIR, f"metrics_thr_{th:.1f}.csv")

        p = ctx.Process(target=_worker_per_threshold, args=(th, label_file, metric_file, retq))
        p.start()

        # Wait for the child to finish; if OOM kill happens, exitcode != 0
        p.join()

        # Drain queue non-blocking (if any message)
        try:
            while True:
                status, th_out, payload = retq.get_nowait()
                if status == "ok":
                    print(f"[INFO] Threshold {th_out:.1f} Sv finished.", flush=True)
                else:
                    print(f"[WARN] Threshold {th_out:.1f} Sv reported error: {payload}", flush=True)
        except Exception:
            pass

        if p.exitcode is None:
            print(f"[WARN] Threshold {th:.1f} Sv process is still alive, terminating...", flush=True)
            p.terminate()
            p.join()
        elif p.exitcode != 0:
            print(f"[WARN] Threshold {th:.1f} Sv process exited with code {p.exitcode}. Skipping metrics merge.", flush=True)
            # Do not raise; continue with next threshold
            # Optionally write a stub row to indicate failure
            stub = pd.DataFrame([{"status": "failed"}])
            stub.to_csv(metric_file.replace(".csv", "_FAILED.csv"), index=False)
        else:
            # Read metric file and add to results
            try:
                m = pd.read_csv(metric_file).iloc[0].to_dict()
                results[f"{th:.1f}"] = m
                print(f"[INFO] Metrics saved at {metric_file}", flush=True)
            except Exception as e:
                print(f"[WARN] Could not read metrics for {th:.1f} Sv: {e}", flush=True)

        # Proactive cleanup
        del labels
        gc.collect()
        time.sleep(0.1)  # tiny breather for OS to reclaim memory

    # Step 5: Save summary table
    if len(results) == 0:
        print("\n[WARN] No successful thresholds to summarize.")
    else:
        summary_df = pd.DataFrame(results).T
        summary_df.index.name = "Threshold"   
        summary_file = os.path.join(OUTPUT_DIR, "sensitivity_summary.csv")
        summary_df.to_csv(summary_file)

        print(f"\n[SUCCESS] Sensitivity analysis complete. Summary saved at {summary_file}")


if __name__ == "__main__":
    run_sensitivity()

