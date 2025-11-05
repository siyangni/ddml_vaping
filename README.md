# Double/Debiased Machine Learning Study: Causal Effects of Vaping on Smoking

## Project Overview

This repository contains a complete, reproducible implementation of a causal study examining the relationship between electronic nicotine delivery system (ENDS/vaping) use and combustible cigarette smoking in the U.S., using Double/Debiased Machine Learning (DML) with cross-fitting and survey-consistent inference.

## Research Questions

1. **Youth Smoking Initiation**: Among never-smoker youth at baseline, what is the causal effect of current/recent vaping on subsequent cigarette smoking initiation?
2. **Adult Smoking Cessation**: Among adult current smokers at baseline, what is the causal effect of vaping on smoking abstinence or sustained reduction?
3. **Bidirectionality & Substitution** (secondary): Effects of smoking on vaping uptake; evidence of substitution vs. complementarity.

## Data Sources

- **PATH Study Public-Use Files (PUF)**, Waves 1-6 (youth and adult cohorts)
- **National Youth Tobacco Survey (NYTS)** microdata for triangulation

## Project Structure

```
ddml_vaping/
├── README.md                          # This file
├── environment.yml                    # Conda environment specification
├── config/
│   └── config.yaml                   # Global configuration
├── data/
│   ├── raw/                          # Downloaded raw data
│   ├── processed/                    # Cleaned and prepared data
│   └── derived/                      # Analysis-ready datasets
├── code/
│   ├── data_prep/
│   │   ├── 00_download_and_verify_data.py
│   │   ├── 01_build_cohorts_and_variables.py
│   │   └── utils_data.py
│   ├── analysis/
│   │   ├── 02_dag_and_design_diagnostics.py
│   │   ├── 03_dml_estimation_point_and_replicate_variance.py
│   │   ├── 04_robustness_heterogeneity_iv.py
│   │   └── 05_reporting_tables_and_figures.py
│   └── utils/
│       ├── survey_utils.py           # Survey weight handling
│       ├── dml_utils.py              # DML estimation helpers
│       └── plotting_utils.py         # Visualization helpers
├── outputs/
│   ├── tables/                       # CSV and LaTeX tables
│   ├── figures/                      # PDF and PNG figures
│   └── logs/                         # Runtime logs
└── paper/
    ├── manuscript/
    │   ├── 06_manuscript_draft.ipynb
    │   ├── manuscript.docx
    │   └── manuscript.pdf
    └── supplements/
        ├── strobe_checklist.md
        └── policy_brief.md
```

## Installation & Setup

### 1. Create Conda Environment

```bash
conda env create -f environment.yml
conda activate ddml_vaping
```

### 2. Download Data

```bash
python code/data_prep/00_download_and_verify_data.py
```

**Note**: PATH Study data requires registration with ICPSR/NAHDAP. Follow instructions in the download script.

## Execution Pipeline

Run the complete analysis pipeline in order:

```bash
# 1. Download and verify data
python code/data_prep/00_download_and_verify_data.py

# 2. Build cohorts and variables
python code/data_prep/01_build_cohorts_and_variables.py

# 3. Generate DAG and design diagnostics
python code/analysis/02_dag_and_design_diagnostics.py

# 4. Run DML estimation with survey-consistent inference
python code/analysis/03_dml_estimation_point_and_replicate_variance.py

# 5. Conduct robustness and heterogeneity analyses
python code/analysis/04_robustness_heterogeneity_iv.py

# 6. Generate publication tables and figures
python code/analysis/05_reporting_tables_and_figures.py

# 7. Generate manuscript (Jupyter notebook)
jupyter notebook paper/manuscript/06_manuscript_draft.ipynb
```

## Methods Summary

### Identification Strategy

- **Framework**: Unconfoundedness/selection-on-observables with rich baseline covariates
- **Assumptions**: SUTVA, positivity, conditional independence
- **Estimand**: Average Treatment Effect (ATE) and Average Treatment Effect on the Treated (ATT)

### Double/Debiased Machine Learning (DML)

- **Cross-fitting**: K-fold (K=5) cross-validation to avoid overfitting bias
- **Nuisance models**: Gradient boosting, random forests, XGBoost with hyperparameter tuning
- **Orthogonalization**: Neyman-orthogonal moment conditions
- **Survey inference**: BRR replicate weights with Fay's factor 0.3 for design-consistent variance estimation

### Heterogeneous Treatment Effects

- **Method**: Causal Forest DML for CATE estimation
- **Subgroups**: Age, sex, baseline risk, policy environment
- **Validation**: Honest inference with separate splitting

### Robustness & Sensitivity

- DoWhy refutation tests (placebo outcomes/treatments)
- Sensitivity to unobserved confounding (partial R²)
- Alternative treatment/outcome definitions
- IV-DML with state ENDS taxes (exploratory)

## Citation & Data Availability

### Data Sources

- **PATH Study**: Available from ICPSR (https://doi.org/10.3886/ICPSR36498.v13)
- **NYTS**: Available from CDC (https://www.cdc.gov/tobacco/data_statistics/surveys/nyts/index.htm)

### Software

- Python 3.9+ with EconML, scikit-learn, XGBoost, DoWhy
- R 4.0+ with DoubleML, survey packages (for replication)

## Contact & Support

For questions about methodology or implementation, please open an issue in this repository.

## License

Code is provided under MIT License. Data are subject to terms of use from original sources.

## Reproducibility

- All random seeds are fixed (seed=42)
- Environment specifications frozen in `environment.yml`
- Data versions and checksums logged
- Complete pipeline can be re-run end-to-end with `bash run_all.sh`
