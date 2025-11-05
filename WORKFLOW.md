# DML Vaping Study - Complete Workflow Documentation

This document provides a comprehensive guide for running the DML causal analysis of vaping effects on smoking using PATH Study data.

## Table of Contents

1. [Overview](#overview)
2. [Data Preparation](#data-preparation)
3. [Analysis Pipeline](#analysis-pipeline)
4. [Output Structure](#output-structure)
5. [Troubleshooting](#troubleshooting)

## Overview

This project implements a rigorous causal analysis using Double/Debiased Machine Learning (DML) to estimate the effects of e-cigarette use on:

- **Youth**: Smoking initiation among never-smokers
- **Adults**: Smoking cessation among current smokers

### Key Features

- Real PATH Study data (Waves 1-2)
- Survey-weighted estimates using BRR replicates
- Cross-fitting with flexible machine learning models
- Comprehensive diagnostics and robustness checks

## Data Preparation

### 1. PATH Study Data Download

The PATH Study Public-Use Files must be obtained from ICPSR:

```bash
# Register and download from:
# https://doi.org/10.3886/ICPSR36498.v13

# Required datasets:
# - Wave 1 Adult (DS1001)
# - Wave 1 Youth (DS1002)
# - Wave 2 Adult (DS2001)
# - Wave 2 Youth (DS2002)
```

### 2. Data Directory Structure

Place downloaded .rda files in the following structure:

```
data/raw/path/
├── DS1001/
│   └── 36498-1001-Data.rda
├── DS1002/
│   └── 36498-1002-Data.rda
├── DS2001/
│   └── 36498-2001-Data.rda
└── DS2002/
    └── 36498-2002-Data.rda
```

### 3. PATH Data Loader

The `path_loader.py` module handles all data loading and preprocessing:

```python
from pathlib import Path
from code.utils.path_loader import PATHDataLoader

# Initialize loader
data_dir = Path("data/raw/path")
loader = PATHDataLoader(data_dir)

# Build cohorts
youth_cohort = loader.build_youth_initiation_cohort()
adult_cohort = loader.build_adult_cessation_cohort()
```

**What the loader does:**
- Loads Waves 1 and 2 from .rda files
- Converts PATH categorical variables to 0/1 binary
- Merges waves by PERSONID
- Applies inclusion/exclusion criteria
- Extracts demographics and survey weights (full + 100 BRR replicates)
- Creates standardized `treatment` and `outcome` variables

## Analysis Pipeline

### Step 1: Quick Pilot Analysis

Run a quick pilot to verify data loading and get preliminary estimates:

```bash
python code/analysis/pilot_dml_real_data.py
```

**Output:**
- `outputs/pilot/pilot_results.json` - Summary statistics and ATE estimate
- `outputs/pilot/pilot_youth_cohort_sample.csv` - Sample of processed data

**Expected results:**
- N ~ 8,000-10,000 youth participants
- Treatment prevalence: 0.5-1.5% (low at Wave 1)
- Smoking initiation prevalence: 3-5%

### Step 2: Full Cohort Construction

Build complete analysis cohorts with all covariates:

```bash
python code/data_prep/01_build_cohorts_and_variables.py
```

**What this does:**
- Loads youth and adult cohorts via PATH loader
- Engineers additional covariates (interactions, standardized variables)
- Saves processed cohorts to `data/processed/`

**Outputs:**
- `data/processed/youth_cohort.parquet`
- `data/processed/adult_cohort.parquet`
- `data/processed/youth_cohort.csv` (for inspection)
- `data/processed/adult_cohort.csv` (for inspection)

### Step 3: Design Diagnostics

Generate DAG and design diagnostics:

```bash
python code/analysis/02_dag_and_design_diagnostics.py
```

**Outputs:**
- `outputs/figures/dag_youth.pdf` - Causal diagram for youth
- `outputs/figures/dag_adult.pdf` - Causal diagram for adults
- Diagnostic tables and balance checks

### Step 4: Main DML Estimation

Run the primary DML analysis with survey-consistent variance estimation:

```bash
python code/analysis/03_dml_estimation_point_and_replicate_variance.py
```

**What this does:**
- Estimates ATE and ATT for both cohorts
- Uses 5-fold cross-fitting
- Nuisance models: XGBoost for propensity scores and outcome regressions
- Computes survey-weighted variance using 100 BRR replicates (Fay coefficient = 0.3)
- Generates diagnostic plots (propensity distributions, balance checks)

**Outputs:**
- `outputs/results/dml_results_youth.pkl` - Full youth results
- `outputs/results/dml_results_adult.pkl` - Full adult results
- `outputs/results/dml_results_youth_summary.json` - Youth summary
- `outputs/results/dml_results_adult_summary.json` - Adult summary
- `outputs/tables/main_results.csv` - Main results table
- `outputs/figures/propensity_distribution_youth.pdf`
- `outputs/figures/balance_youth.pdf`
- `outputs/figures/propensity_distribution_adult.pdf`
- `outputs/figures/balance_adult.pdf`

### Step 5: Robustness and Heterogeneity

Conduct sensitivity analyses and explore heterogeneous effects:

```bash
python code/analysis/04_robustness_heterogeneity_iv.py
```

**Analyses:**
- Sensitivity to unobserved confounding
- Alternative model specifications
- Heterogeneous treatment effects by age, sex, baseline risk
- Instrumental variable analysis (if applicable)

### Step 6: Generate Publication Outputs

Create final tables and figures:

```bash
python code/analysis/05_reporting_tables_and_figures.py
```

**Outputs:**
- Publication-ready tables (CSV and LaTeX)
- High-resolution figures (PDF and PNG)
- Supplementary materials

## Output Structure

After running the complete pipeline, you will have:

```
outputs/
├── pilot/
│   ├── pilot_results.json
│   └── pilot_youth_cohort_sample.csv
├── results/
│   ├── dml_results_youth.pkl
│   ├── dml_results_adult.pkl
│   ├── dml_results_youth_summary.json
│   └── dml_results_adult_summary.json
├── tables/
│   ├── main_results.csv
│   ├── main_results.tex
│   ├── balance_table.csv
│   └── robustness_results.csv
├── figures/
│   ├── propensity_distribution_youth.pdf
│   ├── balance_youth.pdf
│   ├── propensity_distribution_adult.pdf
│   ├── balance_adult.pdf
│   └── [additional diagnostic plots]
└── logs/
    ├── cohort_construction.log
    ├── dml_estimation.log
    └── [other log files]
```

## Key Results Interpretation

### Youth Cohort (Smoking Initiation)

**Estimand:** ATE of current e-cigarette use on smoking initiation over 1 year

**Interpretation:**
- Positive ATE: Vaping increases risk of subsequent smoking initiation
- Negative ATE: Vaping decreases risk of smoking initiation
- Null ATE: No causal effect

**Example output:**
```json
{
  "cohort_type": "youth",
  "ATE": {
    "estimate": 0.0850,
    "std_error": 0.0320,
    "ci_lower": 0.0223,
    "ci_upper": 0.1477
  }
}
```

Interpretation: Among youth who vape, smoking initiation risk is 8.5 percentage points higher (95% CI: 2.2% to 14.8%) compared to non-vapers with similar characteristics.

### Adult Cohort (Smoking Cessation)

**Estimand:** ATE of current e-cigarette use on 30-day smoking abstinence

**Interpretation:**
- Positive ATE: Vaping increases cessation success
- Negative ATE: Vaping decreases cessation success
- Null ATE: No causal effect

## Troubleshooting

### Common Issues

#### 1. PATH Data Not Found

```
Error: PATH data file not found: /path/to/data
```

**Solution:**
- Verify data is downloaded and placed in `data/raw/path/`
- Check that .rda files are in correct subdirectories (DS1001, DS1002, etc.)
- Ensure file names match: `36498-{wave}{cohort}-Data.rda`

#### 2. Missing Python Packages

```
ModuleNotFoundError: No module named 'package'
```

**Solution:**
```bash
pip install pandas numpy scikit-learn pyreadr pyyaml xgboost
```

Or use the provided environment file:
```bash
conda env create -f environment.yml
conda activate ddml_vaping
```

#### 3. Small Sample Sizes

If cohort sizes are unexpectedly small:

**Check:**
- Inclusion criteria in `path_loader.py:build_youth_initiation_cohort()` and `build_adult_cessation_cohort()`
- Missing value patterns in key variables
- Variable conversion from PATH labeled format to 0/1

**Debug:**
```python
# Add verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Inspect intermediate filtering steps
loader = PATHDataLoader(data_dir)
df_w1, df_w2 = loader.load_youth_waves_1_2()
print(f"W1 never-smokers: {(df_w1['R01R_Y_EVR_CIGS'] == 0).sum()}")
print(f"W2 with outcome: {df_w2['smoking_initiation_w2'].notna().sum()}")
```

#### 4. Positivity Violations

If diagnostic output shows:
```
Positivity violations: 250 (2.8%)
Status: FAIL
```

**Solutions:**
- Trim propensity scores (already implemented in DML utils)
- Consider subgroup analysis excluding overlap violations
- Check for extreme covariate values driving violations

#### 5. Memory Issues with Large Datasets

**Solutions:**
- Process cohorts separately
- Use data chunking for very large files
- Reduce number of BRR replicates for testing (not for final analysis)

## Performance Optimization

### Speed Up Analysis

1. **Use smaller cross-validation folds during testing:**
   ```python
   # In config/config.yaml
   dml:
     n_folds: 2  # Use 2 instead of 5 for testing
   ```

2. **Reduce BRR replicates for pilot runs:**
   ```python
   # Only for testing - NOT for final results
   replicate_weights = replicate_weights[:, :20]  # Use 20 instead of 100
   ```

3. **Use simpler models for initial testing:**
   ```python
   # In dml_utils.py
   model_t = LogisticRegression()  # Instead of XGBoost
   ```

## Reproducibility

All analyses use fixed random seeds (default: 42) for reproducibility.

To ensure exact replication:
1. Use the same PATH data version (check ICPSR version number)
2. Use the same package versions (see `environment.yml`)
3. Run scripts in order (dependencies between steps)
4. Do not modify survey weights or cohort definitions

## Advanced Usage

### Running Individual Cohorts

To analyze only youth or only adults:

```python
# In 03_dml_estimation_point_and_replicate_variance.py
# Comment out the adult cohort section to run youth only

if __name__ == "__main__":
    estimator = DMLEstimator()

    # Youth only
    youth_cohort = estimator.load_cohort('youth')
    # ... run youth analysis

    # Skip adult analysis
```

### Custom Covariate Sets

Modify `engineer_covariates()` in `01_build_cohorts_and_variables.py`:

```python
def engineer_covariates(self, cohort, cohort_type):
    # Add custom covariates here
    if cohort_type == 'youth':
        # Add youth-specific variables
        cohort['parent_education_high'] = ...

    return cohort
```

### Alternative Model Specifications

Modify nuisance models in `03_dml_estimation_point_and_replicate_variance.py`:

```python
from sklearn.ensemble import GradientBoostingClassifier

model_t = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.01
)
```

## Citation

If you use this workflow, please cite:

```
PATH Study:
Hyland et al. (2017). Design and methods of the Population Assessment
of Tobacco and Health (PATH) Study. Tobacco Control, 26(4), 371-378.

DML Methods:
Chernozhukov et al. (2018). Double/debiased machine learning for
treatment and structural parameters. The Econometrics Journal, 21(1), C1-C68.
```

## Support

For questions or issues:
1. Check this workflow documentation
2. Review log files in `outputs/logs/`
3. Inspect intermediate outputs in `data/processed/`
4. Open an issue in the GitHub repository

## Summary Checklist

Before running the full analysis:

- [ ] PATH data downloaded and placed in `data/raw/path/`
- [ ] Python environment set up with required packages
- [ ] Pilot analysis runs successfully
- [ ] Output directories created (`outputs/`, `data/processed/`)
- [ ] Configuration file reviewed (`config/config.yaml`)

During analysis:

- [ ] Step 1: Pilot analysis completed
- [ ] Step 2: Cohorts built and saved
- [ ] Step 3: Diagnostics reviewed
- [ ] Step 4: Main DML estimation completed
- [ ] Step 5: Robustness checks run
- [ ] Step 6: Publication outputs generated

After analysis:

- [ ] Review all log files for warnings/errors
- [ ] Check diagnostic plots for issues (positivity, balance)
- [ ] Verify sample sizes match expectations
- [ ] Inspect main results tables
- [ ] Document any deviations from standard workflow

## Next Steps

After completing this workflow:

1. Review results in `outputs/tables/main_results.csv`
2. Examine diagnostic plots in `outputs/figures/`
3. Consider heterogeneity analyses for subgroups of interest
4. Prepare manuscript using template in `paper/manuscript/`
5. Generate policy brief using template in `paper/supplements/`
