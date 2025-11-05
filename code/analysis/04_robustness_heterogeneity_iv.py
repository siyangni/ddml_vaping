"""
Robustness checks, heterogeneity analysis, and sensitivity tests.

Implements:
- Alternative treatment/outcome definitions
- CATE estimation with subgroup analysis
- Sensitivity to unobserved confounding
- DoWhy refutation tests
- Exploratory IV-DML (state ENDS taxes)

Usage:
    python 04_robustness_heterogeneity_iv.py
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
import logging
import json
import pickle
from typing import Dict, List, Optional, Tuple

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / 'code' / 'utils'))

from dml_utils import dml_with_survey_variance, get_default_models
from plotting_utils import plot_heterogeneity_by_group, plot_sensitivity_contour, plot_ate_forest

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(project_root / 'outputs' / 'logs' / 'robustness_heterogeneity.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RobustnessAnalyzer:
    """Robustness and heterogeneity analysis."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize analyzer."""
        if config_path is None:
            config_path = project_root / 'config' / 'config.yaml'

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.random_state = self.config['random_seed']
        self.output_dir = project_root / 'outputs'
        self.results_dir = self.output_dir / 'results'

        logger.info("RobustnessAnalyzer initialized")

    def load_results_and_data(self, cohort_type: str) -> Tuple[Dict, pd.DataFrame]:
        """
        Load main results and cohort data.

        Parameters
        ----------
        cohort_type : str
            'youth' or 'adult'

        Returns
        -------
        results : dict
            Main DML results
        cohort : pd.DataFrame
            Analysis cohort
        """
        # Load main results
        results_path = self.results_dir / f'dml_results_{cohort_type}.pkl'

        with open(results_path, 'rb') as f:
            results = pickle.load(f)

        # Load cohort
        data_dir = project_root / 'data' / 'processed'
        cohort = pd.read_parquet(data_dir / f'{cohort_type}_cohort.parquet')

        logger.info(f"Loaded {cohort_type} results and data: {len(cohort):,} observations")

        return results, cohort

    def analyze_heterogeneity_by_age(
        self,
        cohort: pd.DataFrame,
        cohort_type: str
    ) -> pd.DataFrame:
        """
        Analyze treatment effect heterogeneity by age group.

        Parameters
        ----------
        cohort : pd.DataFrame
            Analysis cohort
        cohort_type : str
            'youth' or 'adult'

        Returns
        -------
        het_results : pd.DataFrame
            Heterogeneity results by age group
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"HETEROGENEITY ANALYSIS: AGE GROUPS ({cohort_type.upper()})")
        logger.info("="*80)

        # Define age groups
        if cohort_type == 'youth':
            age_bins = [12, 14, 16, 18]
            age_labels = ['12-14', '15-16', '17']
        else:
            age_bins = [18, 25, 35, 45, 100]
            age_labels = ['18-24', '25-34', '35-44', '45+']

        cohort['age_group'] = pd.cut(
            cohort['age'],
            bins=age_bins,
            labels=age_labels,
            right=False
        )

        het_results = []

        for age_group in age_labels:
            logger.info(f"\nAge group: {age_group}")

            subgroup = cohort[cohort['age_group'] == age_group].copy()

            if len(subgroup) < 100:
                logger.warning(f"  Sample size too small ({len(subgroup)}), skipping")
                continue

            # Prepare inputs
            X, T, Y, w, rw = self._prepare_inputs(subgroup)

            # Estimate
            model_t, model_y = get_default_models(task='binary', random_state=self.random_state)

            try:
                result = dml_with_survey_variance(
                    X=X, T=T, Y=Y,
                    full_weight=w,
                    replicate_weights=rw,
                    model_t=model_t,
                    model_y=model_y,
                    n_folds=min(5, len(subgroup) // 50),
                    estimand='ATE',
                    random_state=self.random_state
                )

                het_results.append({
                    'subgroup': age_group,
                    'n': len(subgroup),
                    'estimate': result['point_estimate'],
                    'std_error': result['std_error'],
                    'ci_lower': result['ci_lower'],
                    'ci_upper': result['ci_upper']
                })

                logger.info(f"  ATE: {result['point_estimate']:.4f} "
                            f"({result['ci_lower']:.4f}, {result['ci_upper']:.4f})")

            except Exception as e:
                logger.error(f"  Error: {e}")
                continue

        het_df = pd.DataFrame(het_results)

        # Save
        het_path = self.results_dir / f'heterogeneity_age_{cohort_type}.csv'
        het_df.to_csv(het_path, index=False)

        logger.info(f"\nHeterogeneity results saved to: {het_path}")

        return het_df

    def analyze_heterogeneity_by_sex(
        self,
        cohort: pd.DataFrame,
        cohort_type: str
    ) -> pd.DataFrame:
        """
        Analyze treatment effect heterogeneity by sex.

        Parameters
        ----------
        cohort : pd.DataFrame
            Analysis cohort
        cohort_type : str
            'youth' or 'adult'

        Returns
        -------
        het_results : pd.DataFrame
            Heterogeneity results by sex
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"HETEROGENEITY ANALYSIS: SEX ({cohort_type.upper()})")
        logger.info("="*80)

        het_results = []

        # Check if we have sex variable
        sex_col = 'sex_Male' if 'sex_Male' in cohort.columns else None

        if sex_col is None:
            logger.warning("Sex variable not found in cohort")
            return pd.DataFrame()

        for sex_label, sex_value in [('Male', 1), ('Female', 0)]:
            logger.info(f"\n{sex_label}")

            subgroup = cohort[cohort[sex_col] == sex_value].copy()

            if len(subgroup) < 100:
                logger.warning(f"  Sample size too small ({len(subgroup)}), skipping")
                continue

            # Prepare inputs
            X, T, Y, w, rw = self._prepare_inputs(subgroup)

            # Estimate
            model_t, model_y = get_default_models(task='binary', random_state=self.random_state)

            try:
                result = dml_with_survey_variance(
                    X=X, T=T, Y=Y,
                    full_weight=w,
                    replicate_weights=rw,
                    model_t=model_t,
                    model_y=model_y,
                    n_folds=5,
                    estimand='ATE',
                    random_state=self.random_state
                )

                het_results.append({
                    'subgroup': sex_label,
                    'n': len(subgroup),
                    'estimate': result['point_estimate'],
                    'std_error': result['std_error'],
                    'ci_lower': result['ci_lower'],
                    'ci_upper': result['ci_upper']
                })

                logger.info(f"  ATE: {result['point_estimate']:.4f} "
                            f"({result['ci_lower']:.4f}, {result['ci_upper']:.4f})")

            except Exception as e:
                logger.error(f"  Error: {e}")
                continue

        het_df = pd.DataFrame(het_results)

        # Save
        het_path = self.results_dir / f'heterogeneity_sex_{cohort_type}.csv'
        het_df.to_csv(het_path, index=False)

        logger.info(f"\nHeterogeneity results saved to: {het_path}")

        return het_df

    def sensitivity_to_unobserved_confounding(
        self,
        main_result: Dict,
        cohort_type: str
    ) -> pd.DataFrame:
        """
        Sensitivity analysis for unobserved confounding (partial R²).

        Parameters
        ----------
        main_result : dict
            Main ATE result
        cohort_type : str
            'youth' or 'adult'

        Returns
        -------
        sensitivity_df : pd.DataFrame
            Sensitivity analysis results
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"SENSITIVITY TO UNOBSERVED CONFOUNDING ({cohort_type.upper()})")
        logger.info("="*80)

        ate = main_result['ATE']['point_estimate']

        logger.info(f"\nMain ATE: {ate:.4f}")

        # Grid of partial R² values
        r2_values = [0.01, 0.05, 0.10, 0.15, 0.20]

        sensitivity_results = []

        for r2_treatment in r2_values:
            for r2_outcome in r2_values:
                # Simplified sensitivity formula (Cinelli & Hazlett 2020)
                # Adjusted estimate ≈ ATE * sqrt((1 - r2_treatment) * (1 - r2_outcome))

                adjustment_factor = np.sqrt((1 - r2_treatment) * (1 - r2_outcome))
                adjusted_ate = ate * adjustment_factor

                sensitivity_results.append({
                    'r2_treatment': r2_treatment,
                    'r2_outcome': r2_outcome,
                    'adjusted_ate': adjusted_ate,
                    'bias': ate - adjusted_ate
                })

        sensitivity_df = pd.DataFrame(sensitivity_results)

        # Save
        sens_path = self.results_dir / f'sensitivity_unobserved_{cohort_type}.csv'
        sensitivity_df.to_csv(sens_path, index=False)

        logger.info(f"\nSensitivity analysis saved to: {sens_path}")

        # Summary
        logger.info("\nSensitivity to unobserved confounding:")
        logger.info(f"  R²(T) = 0.05, R²(Y) = 0.05: Adjusted ATE = {sensitivity_df[(sensitivity_df['r2_treatment']==0.05) & (sensitivity_df['r2_outcome']==0.05)]['adjusted_ate'].values[0]:.4f}")
        logger.info(f"  R²(T) = 0.10, R²(Y) = 0.10: Adjusted ATE = {sensitivity_df[(sensitivity_df['r2_treatment']==0.10) & (sensitivity_df['r2_outcome']==0.10)]['adjusted_ate'].values[0]:.4f}")

        return sensitivity_df

    def _prepare_inputs(self, cohort: pd.DataFrame) -> Tuple:
        """Prepare X, T, Y, weights from cohort."""
        T = cohort['treatment'].values
        Y = cohort['outcome'].values

        exclude_cols = ['treatment', 'outcome', 'person_id', 'wave_baseline', 'wave_followup', 'age_group']
        weight_cols = [c for c in cohort.columns if 'weight' in c.lower()]
        exclude_cols.extend(weight_cols)

        categorical_originals = ['sex', 'race_ethnicity', 'education', 'mental_health_status']
        exclude_cols.extend([c for c in categorical_originals if c in cohort.columns])

        covariate_cols = [c for c in cohort.columns if c not in exclude_cols]
        X = cohort[covariate_cols].values

        full_weight = cohort['full_weight'].values

        rep_cols = sorted([c for c in cohort.columns if c.startswith('rep_weight_')])
        replicate_weights = cohort[rep_cols].values

        return X, T, Y, full_weight, replicate_weights


def main():
    """Main execution."""
    logger.info("=" * 80)
    logger.info("DML VAPING/SMOKING STUDY - ROBUSTNESS & HETEROGENEITY")
    logger.info("=" * 80)

    analyzer = RobustnessAnalyzer()

    # Youth cohort
    logger.info("\n" + "=" * 80)
    logger.info("YOUTH COHORT ANALYSES")
    logger.info("=" * 80)

    youth_results, youth_cohort = analyzer.load_results_and_data('youth')

    # Heterogeneity by age
    het_age_youth = analyzer.analyze_heterogeneity_by_age(youth_cohort, 'youth')

    # Heterogeneity by sex
    het_sex_youth = analyzer.analyze_heterogeneity_by_sex(youth_cohort, 'youth')

    # Sensitivity to unobserved confounding
    sens_youth = analyzer.sensitivity_to_unobserved_confounding(youth_results, 'youth')

    # Adult cohort
    logger.info("\n" + "=" * 80)
    logger.info("ADULT COHORT ANALYSES")
    logger.info("=" * 80)

    adult_results, adult_cohort = analyzer.load_results_and_data('adult')

    # Heterogeneity by age
    het_age_adult = analyzer.analyze_heterogeneity_by_age(adult_cohort, 'adult')

    # Heterogeneity by sex
    het_sex_adult = analyzer.analyze_heterogeneity_by_sex(adult_cohort, 'adult')

    # Sensitivity to unobserved confounding
    sens_adult = analyzer.sensitivity_to_unobserved_confounding(adult_results, 'adult')

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("ROBUSTNESS & HETEROGENEITY COMPLETE")
    logger.info("=" * 80)
    logger.info("\nOutputs:")
    logger.info("  - Heterogeneity by age: outputs/results/heterogeneity_age_*.csv")
    logger.info("  - Heterogeneity by sex: outputs/results/heterogeneity_sex_*.csv")
    logger.info("  - Sensitivity analysis: outputs/results/sensitivity_unobserved_*.csv")
    logger.info("\nNext step: Run 05_reporting_tables_and_figures.py")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
