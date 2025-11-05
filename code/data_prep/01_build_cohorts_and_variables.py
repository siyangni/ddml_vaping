"""
Cohort construction and variable engineering for DML analysis.

Builds youth smoking initiation and adult smoking cessation cohorts
from PATH Study data with proper treatment, outcome, and covariate definitions.

Usage:
    python 01_build_cohorts_and_variables.py
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
import logging
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / 'code' / 'utils'))

from path_loader import PATHDataLoader
from survey_utils import (
    identify_weight_columns,
    normalize_weights,
    check_weight_validity,
    create_survey_design_vars,
    summarize_survey_design
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(project_root / 'outputs' / 'logs' / 'cohort_construction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PATHCohortBuilder:
    """
    Build analysis cohorts from PATH Study data.

    Implements cohort definitions for youth smoking initiation
    and adult smoking cessation analyses.
    """

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize cohort builder.

        Parameters
        ----------
        config_path : Path, optional
            Path to configuration file
        """
        if config_path is None:
            config_path = project_root / 'config' / 'config.yaml'

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.data_dir = project_root / 'data' / 'raw' / 'PATH'
        self.output_dir = project_root / 'data' / 'processed'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.random_state = self.config['random_seed']

        logger.info(f"PATHCohortBuilder initialized")
        logger.info(f"Random seed: {self.random_state}")

    def load_path_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load PATH Study data files using the PATHDataLoader.

        Returns
        -------
        data : dict
            Dictionary with 'youth' and 'adult' DataFrames
        """
        logger.info("=" * 80)
        logger.info("LOADING PATH STUDY DATA")
        logger.info("=" * 80)

        # Initialize PATH data loader
        path_data_dir = project_root / 'data' / 'raw' / 'path'
        loader = PATHDataLoader(path_data_dir)

        # Build cohorts using real PATH data
        logger.info("\nLoading real PATH Study data...")

        youth_cohort = loader.build_youth_initiation_cohort()
        adult_cohort = loader.build_adult_cessation_cohort()

        data = {
            'youth': youth_cohort,
            'adult': adult_cohort
        }

        logger.info(f"\nLoaded data:")
        logger.info(f"  Youth: {len(data['youth']):,} observations")
        logger.info(f"  Adult: {len(data['adult']):,} observations")

        return data

    def _create_synthetic_youth_data(self, n: int = 5000) -> pd.DataFrame:
        """
        Create synthetic PATH youth data for demonstration.

        In production, replace this with actual PATH data loading.

        Parameters
        ----------
        n : int, default=5000
            Sample size

        Returns
        -------
        df : pd.DataFrame
            Synthetic youth data
        """
        np.random.seed(self.random_state)

        # Demographics
        age = np.random.randint(12, 18, n)
        sex = np.random.choice(['Male', 'Female'], n)
        race = np.random.choice(
            ['White', 'Black', 'Hispanic', 'Other'],
            n,
            p=[0.55, 0.14, 0.24, 0.07]
        )

        # Baseline smoking status (never-smokers)
        never_smoker = np.ones(n, dtype=int)

        # Vaping (treatment) - varies by age
        vaping_prob = 0.1 + 0.05 * (age - 12) / 6
        current_vape = np.random.binomial(1, vaping_prob)

        # Covariates
        parental_tobacco = np.random.binomial(1, 0.3, n)
        peer_tobacco = np.random.binomial(1, 0.25, n)
        sensation_seeking = np.random.normal(0, 1, n)
        academic_performance = np.random.choice(['High', 'Medium', 'Low'], n, p=[0.3, 0.5, 0.2])
        household_income = np.random.choice(['<$25k', '$25k-$50k', '$50k-$100k', '>$100k'], n, p=[0.2, 0.25, 0.35, 0.2])

        # Outcome (smoking initiation at follow-up)
        # Influenced by vaping, age, peers, parents
        smoking_risk = (
            0.05 +
            0.15 * current_vape +
            0.02 * (age - 12) +
            0.10 * parental_tobacco +
            0.08 * peer_tobacco +
            0.05 * (sensation_seeking > 0).astype(int)
        )
        smoking_risk = np.clip(smoking_risk, 0, 1)
        smoking_initiation = np.random.binomial(1, smoking_risk)

        # Survey weights (inverse probability with some noise)
        base_weight = 1000.0
        weight_factor = np.random.lognormal(0, 0.3, n)
        full_weight = base_weight * weight_factor

        # Replicate weights (BRR with Fay 0.3)
        n_reps = 100
        replicate_weights = {}
        for i in range(n_reps):
            rep_factor = np.random.normal(1.0, 0.1 * (1 - 0.3), n)
            replicate_weights[f'rep_weight_{i+1}'] = full_weight * rep_factor

        # Construct DataFrame
        df = pd.DataFrame({
            'person_id': range(1, n + 1),
            'wave_baseline': 1,
            'wave_followup': 2,
            'age': age,
            'sex': sex,
            'race_ethnicity': race,
            'education': academic_performance,
            'household_income': household_income,
            'never_smoker_baseline': never_smoker,
            'current_vape_baseline': current_vape,
            'parental_tobacco_use': parental_tobacco,
            'peer_tobacco_use': peer_tobacco,
            'sensation_seeking_score': sensation_seeking,
            'smoking_initiation_followup': smoking_initiation,
            'full_weight': full_weight,
            **replicate_weights
        })

        return df

    def _create_synthetic_adult_data(self, n: int = 3000) -> pd.DataFrame:
        """
        Create synthetic PATH adult data for demonstration.

        Parameters
        ----------
        n : int, default=3000
            Sample size

        Returns
        -------
        df : pd.DataFrame
            Synthetic adult data
        """
        np.random.seed(self.random_state + 1)

        # Demographics
        age = np.random.randint(18, 75, n)
        sex = np.random.choice(['Male', 'Female'], n)
        race = np.random.choice(
            ['White', 'Black', 'Hispanic', 'Other'],
            n,
            p=[0.60, 0.13, 0.18, 0.09]
        )

        # Baseline smoking status (current smokers)
        current_smoker_baseline = np.ones(n, dtype=int)
        cpd_baseline = np.random.randint(5, 41, n)  # Cigarettes per day

        # Vaping (treatment)
        vaping_prob = 0.20  # Higher for smokers trying to quit
        current_vape = np.random.binomial(1, vaping_prob)

        # Covariates
        quit_attempts = np.random.poisson(2, n)
        nicotine_dependence = np.random.normal(0, 1, n)
        mental_health = np.random.choice(['Good', 'Fair', 'Poor'], n, p=[0.5, 0.3, 0.2])
        education_level = np.random.choice(['<HS', 'HS', 'Some college', 'College+'], n, p=[0.15, 0.30, 0.35, 0.20])

        # Outcome (smoking abstinence at follow-up)
        # Influenced by vaping, quit attempts, dependence
        abstinence_prob = (
            0.05 +
            0.10 * current_vape +
            0.02 * np.minimum(quit_attempts, 5) +
            -0.03 * (nicotine_dependence > 0).astype(int) +
            -0.01 * (cpd_baseline / 20)
        )
        abstinence_prob = np.clip(abstinence_prob, 0, 1)
        smoking_abstinence = np.random.binomial(1, abstinence_prob)

        # Survey weights
        base_weight = 1500.0
        weight_factor = np.random.lognormal(0, 0.3, n)
        full_weight = base_weight * weight_factor

        # Replicate weights
        n_reps = 100
        replicate_weights = {}
        for i in range(n_reps):
            rep_factor = np.random.normal(1.0, 0.1 * (1 - 0.3), n)
            replicate_weights[f'rep_weight_{i+1}'] = full_weight * rep_factor

        # Construct DataFrame
        df = pd.DataFrame({
            'person_id': range(10001, 10001 + n),
            'wave_baseline': 1,
            'wave_followup': 2,
            'age': age,
            'sex': sex,
            'race_ethnicity': race,
            'education': education_level,
            'current_smoker_baseline': current_smoker_baseline,
            'cpd_baseline': cpd_baseline,
            'current_vape_baseline': current_vape,
            'prior_quit_attempts': quit_attempts,
            'nicotine_dependence_score': nicotine_dependence,
            'mental_health_status': mental_health,
            'smoking_abstinence_30day_followup': smoking_abstinence,
            'full_weight': full_weight,
            **replicate_weights
        })

        return df

    def build_youth_cohort(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare youth smoking initiation cohort for analysis.

        The PATH loader already applies inclusion criteria and creates
        treatment/outcome variables. This method adds any additional
        preprocessing needed for DML analysis.

        Parameters
        ----------
        data : pd.DataFrame
            Youth cohort from PATH loader

        Returns
        -------
        cohort : pd.DataFrame
            Analysis-ready youth cohort
        """
        logger.info("\n" + "=" * 80)
        logger.info("PREPARING YOUTH COHORT FOR ANALYSIS")
        logger.info("=" * 80)

        cohort = data.copy()
        n_initial = len(cohort)

        logger.info(f"\nYouth cohort from PATH loader: {n_initial:,}")
        logger.info(f"Treatment prevalence: {cohort['treatment'].mean():.1%}")
        logger.info(f"Outcome prevalence: {cohort['outcome'].mean():.1%}")

        # Data already has treatment and outcome variables from PATH loader
        # Just verify required columns exist
        required_cols = ['treatment', 'outcome', 'age', 'sex_Male']
        missing_cols = [col for col in required_cols if col not in cohort.columns]

        if missing_cols:
            logger.warning(f"Missing required columns: {missing_cols}")

        logger.info(f"\nFinal youth cohort: {len(cohort):,}")

        return cohort

    def build_adult_cohort(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare adult smoking cessation cohort for analysis.

        The PATH loader already applies inclusion criteria and creates
        treatment/outcome variables. This method adds any additional
        preprocessing needed for DML analysis.

        Parameters
        ----------
        data : pd.DataFrame
            Adult cohort from PATH loader

        Returns
        -------
        cohort : pd.DataFrame
            Analysis-ready adult cohort
        """
        logger.info("\n" + "=" * 80)
        logger.info("PREPARING ADULT COHORT FOR ANALYSIS")
        logger.info("=" * 80)

        cohort = data.copy()
        n_initial = len(cohort)

        logger.info(f"\nAdult cohort from PATH loader: {n_initial:,}")
        logger.info(f"Treatment prevalence: {cohort['treatment'].mean():.1%}")
        logger.info(f"Outcome prevalence (abstinence): {cohort['outcome'].mean():.1%}")

        # Data already has treatment and outcome variables from PATH loader
        # Just verify required columns exist
        required_cols = ['treatment', 'outcome', 'age', 'sex_Male']
        missing_cols = [col for col in required_cols if col not in cohort.columns]

        if missing_cols:
            logger.warning(f"Missing required columns: {missing_cols}")

        logger.info(f"\nFinal adult cohort: {len(cohort):,}")

        return cohort

    def engineer_covariates(self, cohort: pd.DataFrame, cohort_type: str) -> pd.DataFrame:
        """
        Engineer covariate matrix for DML.

        The PATH loader already creates basic demographic dummy variables.
        This method adds interactions and standardized versions.

        Parameters
        ----------
        cohort : pd.DataFrame
            Cohort data from PATH loader
        cohort_type : str
            'youth' or 'adult'

        Returns
        -------
        cohort : pd.DataFrame
            Cohort with engineered covariates
        """
        logger.info(f"\nEngineering covariates for {cohort_type} cohort...")

        # PATH loader already created dummy variables (sex_Male, race_ethnicity_White, etc.)
        # Just create interactions and standardized versions

        # Create interaction terms (age x sex if both present)
        if 'age' in cohort.columns and 'sex_Male' in cohort.columns:
            cohort['age_x_sex_Male'] = cohort['age'] * cohort['sex_Male']
            logger.info("  Created age x sex interaction")

        # Standardize continuous variables
        continuous_vars = ['age']
        continuous_vars = [v for v in continuous_vars if v in cohort.columns]

        for var in continuous_vars:
            if cohort[var].std() > 0:
                cohort[f'{var}_std'] = (cohort[var] - cohort[var].mean()) / cohort[var].std()

        # Count available covariates
        covariate_cols = [c for c in cohort.columns if c not in [
            'treatment', 'outcome', 'PERSONID', 'person_id', 'full_weight'
        ] and not c.startswith('rep_weight_')]

        logger.info(f"  Available covariate columns: {len(covariate_cols)}")

        return cohort

    def save_cohorts(
        self,
        youth_cohort: pd.DataFrame,
        adult_cohort: pd.DataFrame
    ) -> None:
        """
        Save cohorts to disk.

        Parameters
        ----------
        youth_cohort : pd.DataFrame
            Youth cohort
        adult_cohort : pd.DataFrame
            Adult cohort
        """
        logger.info("\n" + "=" * 80)
        logger.info("SAVING COHORTS")
        logger.info("=" * 80)

        # Save as parquet (efficient compression)
        youth_path = self.output_dir / 'youth_cohort.parquet'
        adult_path = self.output_dir / 'adult_cohort.parquet'

        youth_cohort.to_parquet(youth_path, index=False)
        adult_cohort.to_parquet(adult_path, index=False)

        logger.info(f"\nSaved:")
        logger.info(f"  Youth cohort: {youth_path}")
        logger.info(f"  Adult cohort: {adult_path}")

        # Also save as CSV for inspection
        youth_cohort.to_csv(self.output_dir / 'youth_cohort.csv', index=False)
        adult_cohort.to_csv(self.output_dir / 'adult_cohort.csv', index=False)

        logger.info("\nCSV versions also saved for inspection")


def main():
    """Main execution."""
    logger.info("=" * 80)
    logger.info("DML VAPING/SMOKING STUDY - COHORT CONSTRUCTION")
    logger.info("=" * 80)

    builder = PATHCohortBuilder()

    # Load data
    data = builder.load_path_data()

    # Build cohorts
    youth_cohort = builder.build_youth_cohort(data['youth'])
    adult_cohort = builder.build_adult_cohort(data['adult'])

    # Engineer covariates
    youth_cohort = builder.engineer_covariates(youth_cohort, 'youth')
    adult_cohort = builder.engineer_covariates(adult_cohort, 'adult')

    # Save cohorts
    builder.save_cohorts(youth_cohort, adult_cohort)

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("COHORT CONSTRUCTION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nYouth cohort: {len(youth_cohort):,} observations")
    logger.info(f"Adult cohort: {len(adult_cohort):,} observations")
    logger.info("\nNext step: Run 02_dag_and_design_diagnostics.py")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
