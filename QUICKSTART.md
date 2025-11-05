# Quick Start Guide

This guide will help you execute the complete DML vaping/smoking causal analysis from start to finish.

## Prerequisites

- Conda or Miniconda installed
- Git (for version control)
- ~50 GB free disk space for data
- ~16 GB RAM recommended for analysis
- Python 3.10+

## Installation (5 minutes)

### 1. Clone or navigate to the project directory

```bash
cd /path/to/ddml_vaping
```

### 2. Create conda environment

```bash
conda env create -f environment.yml
conda activate ddml_vaping
```

This installs all required packages including:
- econml (DML implementation)
- xgboost, scikit-learn (ML algorithms)
- pandas, numpy (data manipulation)
- matplotlib, seaborn (visualization)
- dowhy (causal inference)

### 3. Verify installation

```bash
python -c "import econml, xgboost, dowhy; print('All packages installed successfully!')"
```

## Data Download (10-30 minutes)

### Option A: Automated download (NYTS only)

```bash
python code/data_prep/00_download_and_verify_data.py
```

**Note:** PATH Study data requires ICPSR registration and manual download.

### Option B: Manual download (PATH Study)

1. Visit: https://www.icpsr.umich.edu/web/NAHDAP/studies/36498
2. Register for ICPSR account
3. Request access to PATH Study PUF
4. Download Waves 1-6 data files
5. Extract to: `data/raw/PATH/`
6. Re-run verification:

```bash
python code/data_prep/00_download_and_verify_data.py
```

## Run Complete Analysis (2-6 hours)

### Option 1: One-command execution

```bash
bash run_all.sh
```

This runs the entire pipeline:
1. Data download/verification ← **Manual intervention required for PATH**
2. Cohort construction (10-20 min)
3. DAG & diagnostics (5-10 min)
4. DML estimation (60-120 min) ← **Most computationally intensive**
5. Robustness & heterogeneity (30-60 min)
6. Tables & figures (5-10 min)

**To skip data download (if already completed):**

```bash
bash run_all.sh --skip-download
```

### Option 2: Step-by-step execution

Run each script individually to debug or inspect intermediate outputs:

```bash
# Step 0: Data (if needed)
python code/data_prep/00_download_and_verify_data.py

# Step 1: Cohort construction
python code/data_prep/01_build_cohorts_and_variables.py

# Step 2: Design diagnostics
python code/analysis/02_dag_and_design_diagnostics.py

# Step 3: Main DML estimation (takes longest!)
python code/analysis/03_dml_estimation_point_and_replicate_variance.py

# Step 4: Robustness
python code/analysis/04_robustness_heterogeneity_iv.py

# Step 5: Reporting
python code/analysis/05_reporting_tables_and_figures.py
```

Each script logs progress to:
- Console (stdout)
- Log file: `outputs/logs/<script_name>.log`

## Check Outputs

After successful completion:

```bash
# View summary report
cat outputs/summary_report.txt

# List all tables
ls -lh outputs/tables/

# List all figures
ls -lh outputs/figures/

# View main results
head outputs/tables/main_results.csv
```

## Generate Manuscript

The manuscript template is in `paper/manuscript/manuscript_template.md`. To fill in results:

1. Review: `outputs/summary_report.txt`
2. Open: `paper/manuscript/manuscript_template.md`
3. Replace placeholders `[X.XX]` with actual estimates from results files
4. Convert to Word/PDF as needed

## Troubleshooting

### Issue: "Module not found"

**Solution:**
```bash
conda activate ddml_vaping
pip install --upgrade econml xgboost dowhy
```

### Issue: "Out of memory" during DML estimation

**Solution:**
- Reduce number of BRR replicates in `config/config.yaml`:
  ```yaml
  dml:
    n_folds: 3  # Instead of 5
  ```
- Or increase system swap space

### Issue: PATH data not found

**Solution:**
PATH requires manual download. See "Data Download" section above.

### Issue: Very slow execution

**Expected:** Main DML estimation with 100 BRR replicates takes 1-2 hours on a modern laptop.

**To speed up (at cost of precision):**
- Edit `config/config.yaml`:
  ```yaml
  survey_design:
    path:
      n_replicates: 20  # Instead of 100
  ```

### Issue: Figures not displaying

**Solution:**
- Figures are saved to disk automatically (non-interactive backend)
- Open PDFs in `outputs/figures/` with a PDF viewer

## Key Files to Review

After running the pipeline, review these key outputs:

1. **Main results:** `outputs/tables/main_results.csv`
2. **Summary:** `outputs/summary_report.txt`
3. **Descriptive stats:** `outputs/tables/table1_youth.csv`, `outputs/tables/table1_adult.csv`
4. **Forest plot:** `outputs/figures/figure1_forest_plot.pdf`
5. **Diagnostic plots:** `outputs/figures/*_youth.pdf`, `outputs/figures/*_adult.pdf`
6. **Full results:** `outputs/results/dml_results_*.pkl` (Python pickle files)

## Next Steps

After completing the analysis:

1. **Review results:** Check `outputs/summary_report.txt` for key findings
2. **Validate diagnostics:**
   - Positivity check should PASS
   - Balance should improve after adjustment
   - Sensitivity analysis should show robustness
3. **Draft manuscript:** Fill in `paper/manuscript/manuscript_template.md` with results
4. **Prepare supplementary materials:** Review `paper/supplements/`
5. **Archive and share:**
   - Version control with git
   - Create OSF project for preregistration/data sharing
   - Prepare code repository for publication

## Getting Help

- **Documentation:** See full README.md
- **Config:** All parameters in `config/config.yaml`
- **Logs:** Check `outputs/logs/*.log` for detailed execution traces
- **Code:** All scripts have detailed docstrings and comments

## Citation

If you use this code, please cite:

```
[Author names]. (2024). Causal Effects of ENDS Use on Cigarette Smoking:
A Double/Debiased Machine Learning Analysis of the PATH Study.
[Journal name, volume, pages]. [DOI]
```

And cite the underlying data:

```
United States Department of Health and Human Services. National Institutes
of Health. National Institute on Drug Abuse, and United States Department
of Health and Human Services. Food and Drug Administration. Center for
Tobacco Products. Population Assessment of Tobacco and Health (PATH) Study
[United States] Public-Use Files. Inter-university Consortium for Political
and Social Research [distributor], 2023-10-27. https://doi.org/10.3886/ICPSR36498.v13
```

## License

Code: MIT License
Data: Subject to original data source terms of use

---

**Questions or Issues?**

Open an issue on the GitHub repository or contact [researcher email].

**Good luck with your analysis!**
