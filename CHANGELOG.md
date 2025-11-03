# Changelog
All notable changes to this project will be documented in this file.  
This project adheres to [Semantic Versioning](https://semver.org/).

---

## [v2.0] – 2025-11-03
### Added
- `amoc_ml_leadtime_analysis.py` for model skill evaluation and lead-time prediction performance.
- `amoc_ml_sensitivity_analysis.py` for extended multi-factor sensitivity testing.
- `amoc_ml_spatial_contributions.py` for spatial feature importance quantification.
- `rapid_transports_timeseries.py` for integrated RAPID array comparison and validation.
- New `scripts/` directory for modular subfunctions.
  
### Changed
- `README.md` fully revised to reflect updated workflow, expanded figures, and data preparation details.
- Enhanced model evaluation workflow integrating ensemble performance metrics.
- Improved PCA integration in `apply_pca.py` and `ml_stage_02.py`.
- Standardized output directory structure for consistency with publication-ready figures.

### Removed
- Deprecated sensitivity plotting scripts replaced by consolidated `amoc_ml_sensitivity_plot.py` and `amoc_ml_sensitivity_summary.py`.

---

## [v1.1] – 2025-09-08
### Added
- `ml_sensitivity.py` for automated threshold sensitivity testing.  
- `plot_sensitivity_results.py` for visualization of sensitivity metrics.

### Changed
- `README.md` updated to include:  
  - Step 6 – Sensitivity Analysis  
  - Step 7 – Visualize Sensitivity Results  
- Updated Repository and Directory Structure sections.

---

## [v1.0] – 2025-08-01
### Added
Initial release of **AMOC Tipping ML Pipeline**.  
Core functionality:
- Data preprocessing (`ml_stage_01.py`)  
- PCA dimensionality reduction (`apply_pca.py`)  
- Model training & SHAP analysis (`ml_stage_02.py`)  
- Full pipeline executor (`main_ml_run.py`)  
- Contribution map generation (`trained_amoc_map.py`)  
- Post-model evaluation (`amoc_ml_post_model_analysis.py`)  
- Documentation: initial `README.md` with instructions.

