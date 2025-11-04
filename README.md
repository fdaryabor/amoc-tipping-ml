# AMOC Machine Learning Pipeline: A Physically Interpretable Framework for Early-Warning Detection

## Overview

This repository provides a complete and reproducible **machine-learning pipeline** for identifying and classifying weakened states of the **Atlantic Meridional Overturning Circulation (AMOC)** using gridded oceanographic datasets.  
It supports both **operational prediction** and **scientific analysis**, combining physically interpretable methods and explainable artificial intelligence (AI).

> Developed for the study:  
> *“Physically Interpretable Machine-Learning Framework for Detecting Early-Warning Signals of AMOC Weakening from Surface Hydrographic Variability.”*  

---

## 1. Scientific Background

The Atlantic Meridional Overturning Circulation (AMOC) is a key component of the global climate system, governing large-scale heat and freshwater transport between the tropics and high latitudes. Its variability influences regional climate, sea-level patterns, and carbon uptake. Detecting **early-warning signals** of a potential AMOC weakening is therefore critical for anticipating abrupt climate transitions.

This framework integrates **satellite- and reanalysis-based surface hydrographic variables** as predictors of AMOC strength, representing the dominant modes of surface ocean variability linked to density-driven overturning processes:

- **Sea Surface Temperature (SST)** — monthly means from the ERA5 reanalysis, providing physically consistent thermal variability constrained by data assimilation of satellite and in situ observations.  
- **Sea Surface Salinity (SSS)** — from the Copernicus Marine Service product *MULTIOBS_GLO_PHY_S_SURFACE_MYNRT_015_013*, a 0.125° × 0.125° multi-observation Level-4 analysis combining satellite and in situ data through optimal interpolation.  
- **Sea Surface Height (SSH)** — from the CMEMS reprocessed product *GLOBAL_MULTIYEAR_PHY_001_030*, a 0.083° × 0.083° merged satellite altimetry dataset capturing dynamical sea level anomalies associated with geostrophic circulation and steric effects.  
- **Greenland Runoff** — derived from ERA5 hydrological components by spatially integrating total surface and subsurface freshwater fluxes over the Greenland domain, representing seasonal meltwater discharge into the North Atlantic.

All variables are **spatially and temporally remapped to the SSS reference grid (0.125° × 0.125°)** and restricted to the **2004–2023** period to ensure alignment across datasets and comparability with the observational record of AMOC.

The **target variable** is the **RAPID array AMOC index** at 26.5°N, obtained from the [RAPID-MOCHA-WBTS project](https://rapid.ac.uk/data/data-download). This index represents the maximum value of the meridional overturning streamfunction, derived from continuous mooring observations that resolve Gulf Stream, Ekman, and upper mid-ocean transports. It provides a direct observational measure of large-scale overturning strength and serves as the ground truth for supervised classification.

By combining these physically meaningful predictors with the observational AMOC index, the pipeline enables a **transparent, physically interpretable ML framework** to detect **early-warning signals of AMOC weakening** from surface hydrographic variability.


The AMOC plays a pivotal role in regulating global and regional climate.  
Detecting early-warning indicators of its potential weakening is critical for understanding ocean–climate feedbacks.  
This framework integrates **satellite- and reanalysis-based surface hydrographic variables**—sea surface temperature (SST), salinity (SSS), height (SSH), and Greenland runoff—to build a **physically grounded ML classification system** with transparent feature attribution.

---

## 2. Datasets

All variables are **spatially and temporally remapped to the SSS reference grid** and restricted to the **2004–2023** period to ensure temporal alignment and comparability among datasets.  
The **0.125° × 0.125° SSS grid** serves as the common spatial reference because it provides an optimal balance between the finer SSH (0.083°) and coarser SST and runoff (0.25°) resolutions.  
Using SSS as the base grid ensures consistent representation of **freshwater-driven density anomalies** while avoiding mesoscale noise amplification.

| Variable | Source | Description |
|-----------|---------|-------------|
| **AMOC Index** | [RAPID-MOCHA/WBTS Project](https://rapid.ac.uk/data/data-download) | Monthly meridional overturning strength at 26.5° N, derived from RAPID array observations (26°–27° N) combining cable, dynamic height, and Ekman components. |
| **ERA5 Monthly SST and Runoff** | [Copernicus Climate Data Store (CDS)](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-monthly-means?tab=overview) | ERA5 monthly SST and Greenland runoff fields (0.25° × 0.25°), generated via 4D-Var data assimilation, combining satellite and in-situ observations. |
| **CMEMS Sea Surface Salinity (SSS)** | [Copernicus Marine Service – MULTIOBS_GLO_PHY_S_SURFACE_MYNRT_015_013](https://data.marine.copernicus.eu/product/MULTIOBS_GLO_PHY_S_SURFACE_MYNRT_015_013/description) | Gap-free Level-4 salinity analyses (0.125° × 0.125°), merged from satellite and in-situ data using optimal interpolation methods. |
| **CMEMS Sea Surface Height (SSH)** | [Copernicus Marine Service – GLOBAL_MULTIYEAR_PHY_001_030](https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030/description) | Reprocessed sea level fields (0.083° × 0.083°) from ocean reanalyses assimilating altimetry and in-situ observations. |

> **Note:** Users are responsible for acquiring, preprocessing, and organizing datasets according to pipeline input requirements.

---

## 3. Pipeline Overview

The pipeline is modular and sequential, executed through the **main runner** script:

### `main_ml_run.py`
Executes all three core stages of the ML workflow:

1. **`ml_stage_01.py`** — Data loading, alignment, and preprocessing  
   - Reads SST, SSS, SSH, and Greenland runoff  
   - Aligns datasets in time and space  
   - Generates lagged features and standardizes inputs  
   - Outputs:  
     `X_train_scaled.pkl`, `X_test_scaled.pkl`, `scaler.pkl`, `y_train.npy`, `y_test.npy`

2. **`apply_pca.py`** — Dimensionality reduction using Incremental PCA  
   - Fits PCA on the training dataset  
   - Saves model as `pca_model.pkl`

3. **`ml_stage_02.py`** — Model training, evaluation, and interpretation  
   - Balances data via SMOTE  
   - Trains Random Forest and XGBoost classifiers  
   - Evaluates metrics (Accuracy, F1, ROC, PR curves)  
   - Computes SHAP explanations and PCA-space importance maps  
   - Outputs:  
     `trained_amoc_model.pkl`, `amoc_ml_input_output.npz`, plots in `/figures/`

---

## 4. Example Usage

### Run full ML pipeline
```bash
python main_ml_run.py
```
This executes:
1. `ml_stage_01.py` — Preprocessing  
2. `apply_pca.py` — PCA training  
3. `ml_stage_02.py` — Model training and SHAP evaluation  

All outputs are saved in `/models/`, `/outputs/`, and `/figures/`.

---

### Perform sensitivity analysis of AMOC thresholds
```bash
python ml_sensitivity.py
```
This script:
- Tests multiple thresholds (e.g., 12–18 Sv)  
- Automatically runs the full ML pipeline for each threshold  
- Saves:  
  - Metrics → `sensitivity_results/metrics/metrics_thr_*.csv`  
  - Summary → `sensitivity_results/sensitivity_summary.csv`

Visualize results:
```bash
python plot_sensitivity_results.py
```

---

### Post-model and diagnostic analysis

| Script | Purpose |
|--------|----------|
| `amoc_ml_post_model_analysis.py` | Detailed ROC, PR, confusion matrix, SHAP plots |
| `amoc_ml_spatial_contributions.py` | 2D spatial contribution maps of PCA/SHAP |
| `amoc_ml_sensitivity_analysis.py` | Recomputes or extends sensitivity runs |
| `amoc_ml_sensitivity_summary.py` | Aggregates metrics from multiple runs |
| `amoc_ml_sensitivity_plot.py` | Generates publication-ready sensitivity plots |
| `amoc_ml_leadtime_analysis.py` | Evaluates lead-time predictability of weak-AMOC events |
| `rapid_transports_timeseries.py` | Plots RAPID array AMOC transport time series (2004–2024) |

---

## 5. Dependencies

Python ≥ 3.8  
Install via:

```bash
pip install scikit-learn xgboost shap xarray pandas numpy matplotlib cartopy joblib imbalanced-learn
```

---

## 6. Directory Layout

```
.
├── main_ml_run.py
├── ml_stage_01.py
├── apply_pca.py
├── ml_stage_02.py
├── ml_sensitivity.py
├── plot_sensitivity_results.py
├── amoc_ml_post_model_analysis.py
├── amoc_ml_spatial_contributions.py
├── amoc_ml_sensitivity_analysis.py
├── amoc_ml_sensitivity_summary.py
├── amoc_ml_sensitivity_plot.py
├── amoc_ml_leadtime_analysis.py
├── rapid_transports_timeseries.py
├── scripts/
│   ├── data_utils.py
│   └── make_amoc_labels.py
├── data/
│   ├── ERA5_SST_fine_sst_trimmed_rename.nc
│   ├── ERA5_runoff_remapped_Greenland_conservative.nc
│   ├── cmems_obs-mob_glo_phy-sss_my_multi_fine_sos.nc
│   ├── cmems_obs-sl_glo_phy-ssh_my_allsat-l4_fine_sla_trimmed.nc
│   └── RAPID_2004_2024/
├── models/
│   ├── pca_model.pkl
│   └── trained_amoc_model.pkl
├── outputs/
│   └── amoc_ml_input_output.npz
├── sensitivity_results/
│   ├── metrics/
│   │   └── metrics_thr_*.csv
│   └── sensitivity_summary.csv
└── figures/
    └── *.png
```

---

## 7. References

- **Baker, A. et al. (2025)**. *Recent AMOC dynamics and stability.*  
- **Boot, A. & Dijkstra, H. (2025)**. *AMOC multidecadal variability and resilience.*  
- **Frajka-Williams, E. et al. (2023)**. *Recent AMOC observations and trends.*  
- **Demšar, U. et al. (2013)**. *Principal Component Analysis on Spatial Data: An Overview.* *Ann. Assoc. Am. Geogr.*, 103(1), 106–128.  
- **Zhang, X. et al. (2023)**. *Multiyear predictability of sea level along the U.S. East Coast.* *Nat. Commun.*

---

## 8. Citation

If you use this repository, please cite:

> **Daryabor, F. (2025).** *AMOC Machine Learning Pipeline: A Physically Interpretable Framework for Early-Warning Detection.*  
> Zenodo. DOI: [10.5281/zenodo.17082368](https://doi.org/10.5281/zenodo.17082368)

---

## 9. Contact

**Farshid Daryabor, Ph.D.**  
Physical Scientist,   
Email: *farshiddaryabor7@gmail.com*  
GitHub: [github.com/fdaryabor](https://github.com/fdaryabor)

---

## License
Apache License 2.0 © 2025 Farshid Daryabor
