#!/bin/bash
#
# Master execution script for DML Vaping/Smoking Causal Study
#
# Runs complete analysis pipeline from data download to final outputs.
#
# Usage:
#   bash run_all.sh [--skip-download]
#
# Options:
#   --skip-download    Skip data download step (use if data already present)

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Parse arguments
SKIP_DOWNLOAD=false
if [[ "$1" == "--skip-download" ]]; then
    SKIP_DOWNLOAD=true
fi

# Print header
echo -e "${BLUE}================================================================================${NC}"
echo -e "${BLUE}DML VAPING/SMOKING CAUSAL STUDY - COMPLETE ANALYSIS PIPELINE${NC}"
echo -e "${BLUE}================================================================================${NC}"
echo ""
echo "Project root: $PROJECT_ROOT"
echo "Start time: $(date)"
echo ""

# Create log directory
mkdir -p outputs/logs

# Activate conda environment
echo -e "${YELLOW}→ Activating conda environment...${NC}"
if [ -f "environment.yml" ]; then
    # Check if environment exists
    if conda env list | grep -q "ddml_vaping"; then
        eval "$(conda shell.bash hook)"
        conda activate ddml_vaping
        echo -e "${GREEN}✓ Environment activated${NC}"
    else
        echo -e "${YELLOW}Environment not found. Creating...${NC}"
        conda env create -f environment.yml
        eval "$(conda shell.bash hook)"
        conda activate ddml_vaping
        echo -e "${GREEN}✓ Environment created and activated${NC}"
    fi
else
    echo -e "${RED}✗ environment.yml not found${NC}"
    exit 1
fi

# Step 0: Data download
if [ "$SKIP_DOWNLOAD" = false ]; then
    echo ""
    echo -e "${BLUE}================================================================================${NC}"
    echo -e "${BLUE}STEP 0: DATA DOWNLOAD${NC}"
    echo -e "${BLUE}================================================================================${NC}"
    echo ""

    python code/data_prep/00_download_and_verify_data.py 2>&1 | tee outputs/logs/00_download.log

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Data download complete${NC}"
    else
        echo -e "${RED}✗ Data download failed${NC}"
        exit 1
    fi
else
    echo ""
    echo -e "${YELLOW}→ Skipping data download (--skip-download flag)${NC}"
fi

# Step 1: Cohort construction
echo ""
echo -e "${BLUE}================================================================================${NC}"
echo -e "${BLUE}STEP 1: COHORT CONSTRUCTION & VARIABLE ENGINEERING${NC}"
echo -e "${BLUE}================================================================================${NC}"
echo ""

python code/data_prep/01_build_cohorts_and_variables.py 2>&1 | tee outputs/logs/01_cohorts.log

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Cohort construction complete${NC}"
else
    echo -e "${RED}✗ Cohort construction failed${NC}"
    exit 1
fi

# Step 2: DAG and design diagnostics
echo ""
echo -e "${BLUE}================================================================================${NC}"
echo -e "${BLUE}STEP 2: DAG & DESIGN DIAGNOSTICS${NC}"
echo -e "${BLUE}================================================================================${NC}"
echo ""

python code/analysis/02_dag_and_design_diagnostics.py 2>&1 | tee outputs/logs/02_diagnostics.log

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Design diagnostics complete${NC}"
else
    echo -e "${RED}✗ Design diagnostics failed${NC}"
    exit 1
fi

# Step 3: DML estimation
echo ""
echo -e "${BLUE}================================================================================${NC}"
echo -e "${BLUE}STEP 3: DML ESTIMATION WITH SURVEY VARIANCE${NC}"
echo -e "${BLUE}================================================================================${NC}"
echo ""

python code/analysis/03_dml_estimation_point_and_replicate_variance.py 2>&1 | tee outputs/logs/03_estimation.log

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ DML estimation complete${NC}"
else
    echo -e "${RED}✗ DML estimation failed${NC}"
    exit 1
fi

# Step 4: Robustness and heterogeneity
echo ""
echo -e "${BLUE}================================================================================${NC}"
echo -e "${BLUE}STEP 4: ROBUSTNESS & HETEROGENEITY ANALYSIS${NC}"
echo -e "${BLUE}================================================================================${NC}"
echo ""

python code/analysis/04_robustness_heterogeneity_iv.py 2>&1 | tee outputs/logs/04_robustness.log

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Robustness analysis complete${NC}"
else
    echo -e "${RED}✗ Robustness analysis failed${NC}"
    exit 1
fi

# Step 5: Reporting
echo ""
echo -e "${BLUE}================================================================================${NC}"
echo -e "${BLUE}STEP 5: GENERATE TABLES & FIGURES${NC}"
echo -e "${BLUE}================================================================================${NC}"
echo ""

python code/analysis/05_reporting_tables_and_figures.py 2>&1 | tee outputs/logs/05_reporting.log

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Reporting complete${NC}"
else
    echo -e "${RED}✗ Reporting failed${NC}"
    exit 1
fi

# Final summary
echo ""
echo -e "${BLUE}================================================================================${NC}"
echo -e "${BLUE}ANALYSIS PIPELINE COMPLETE${NC}"
echo -e "${BLUE}================================================================================${NC}"
echo ""
echo -e "${GREEN}✓ All steps completed successfully${NC}"
echo ""
echo "End time: $(date)"
echo ""
echo "Outputs:"
echo "  - Data: data/processed/"
echo "  - Results: outputs/results/"
echo "  - Tables: outputs/tables/"
echo "  - Figures: outputs/figures/"
echo "  - Logs: outputs/logs/"
echo ""
echo "Next step: Review outputs and generate manuscript"
echo "  jupyter notebook paper/manuscript/06_manuscript_draft.ipynb"
echo ""
echo -e "${BLUE}================================================================================${NC}"
