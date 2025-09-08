# Changelog

All notable changes to this project will be documented in this file.  
This project adheres to [Semantic Versioning](https://semver.org/).

---

## [v1.1] – 2025-09-08
### Added
- `ml_sensitivity.py` for automated threshold sensitivity testing.  
- `plot_sensitivity_results.py` for visualization of sensitivity metrics.  

### Changed
- **README.md** updated to include:  
  - Step 6 – Sensitivity Analysis  
  - Step 7 – Visualize Sensitivity Results  
  - Updated Repository and Directory Structure sections.  

---

## [v1.0] – 2025-08-01
### Added
- Initial release of AMOC Tipping ML Pipeline.  
- Core functionality:  
  - Data preprocessing (`ml_stage_01.py`)  
  - PCA dimensionality reduction (`apply_pca.py`)  
  - Model training & SHAP analysis (`ml_stage_02.py`)  
  - Full pipeline executor (`main_ml_run.py`)  
  - Contribution map generation (`trained_amoc_map.py`)  
  - Post-model evaluation (`amoc_ml_post_model_analysis.py`)  
- Documentation: initial **README.md** with instructions.  

---
