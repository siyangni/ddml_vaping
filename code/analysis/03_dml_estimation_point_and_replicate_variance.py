"""
Main DML estimation with survey-consistent variance via BRR replicates.

Estimates ATE and ATT for youth smoking initiation and adult smoking cessation
using Double/Debiased Machine Learning with cross-fitting and design-consistent
standard errors.

Usage:
    python 03_dml_estimation_point_and_replicate_variance.py
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
import logging
import json
from typing import Dict, List, Optional, Tuple
import pickle

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / 'code' / 'utils'))

from survey_utils import BRRVarianceEstimator, identify_weight_columns
from dml_utils import (
    dml_with_survey_variance,
    get_default_models,
    check_positivity,
    balance_check
)
from plotting_utils import (
    plot_propensity_distribution,
    plot_balance,
    plot_ate_forest
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(project_root / 'outputs' / 'logs' / 'dml_estimation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DMLEstimator:
    """
    Main DML estimation pipeline with survey-consistent inference.
    """

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize DML estimator.

        Parameters
        ----------
        config_path : Path, optional
            Path to configuration file
        """
        if config_path is None:
            config_path = project_root / 'config' / 'config.yaml'

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.random_state = self.config['random_seed']
        self.n_folds = self.config['dml']['n_folds']
        self.fay_coefficient = self.config['survey_design']['path']['fay_coefficient']

        self.output_dir = project_root / 'outputs'
        self.results_dir = self.output_dir / 'results'
        self.results_dir.mkdir(parents=True, exist_ok=True)

        logger.info("DMLEstimator initialized")
        logger.info(f"Cross-fitting folds: {self.n_folds}")
        logger.info(f"Fay coefficient: {self.fay_coefficient}")

    def load_cohort(self, cohort_type: str) -> pd.DataFrame:
        """
        Load analysis cohort.

        Parameters
        ----------
        cohort_type : str
            'youth' or 'adult'

        Returns
        -------
        cohort : pd.DataFrame
        """
        data_dir = project_root / 'data' / 'processed'
        cohort_path = data_dir / f'{cohort_type}_cohort.parquet'

        cohort = pd.read_parquet(cohort_path)

        logger.info(f"Loaded {cohort_type} cohort: {len(cohort):,} observations")

        return cohort

    def prepare_dml_inputs(
        self,
        cohort: pd.DataFrame,
        cohort_type: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare inputs for DML estimation.

        Parameters
        ----------
        cohort : pd.DataFrame
            Analysis cohort
        cohort_type : str
            'youth' or 'adult'

        Returns
        -------
        X : np.ndarray
            Covariate matrix
        T : np.ndarray
            Treatment vector
        Y : np.ndarray
            Outcome vector
        full_weight : np.ndarray
            Full sample weight
        replicate_weights : np.ndarray
            Replicate weights matrix
        """
        logger.info(f"\nPreparing DML inputs for {cohort_type} cohort...")

        # Treatment and outcome
        T = cohort['treatment'].values
        Y = cohort['outcome'].values

        # Covariates (all columns except treatment, outcome, weights, IDs)
        exclude_cols = ['treatment', 'outcome', 'person_id', 'wave_baseline', 'wave_followup']
        weight_cols = [c for c in cohort.columns if 'weight' in c.lower()]
        exclude_cols.extend(weight_cols)

        # Also exclude original categorical variables (keep dummies)
        categorical_originals = ['sex', 'race_ethnicity', 'education', 'mental_health_status']
        exclude_cols.extend([c for c in categorical_originals if c in cohort.columns])

        covariate_cols = [c for c in cohort.columns if c not in exclude_cols]

        X = cohort[covariate_cols].values

        logger.info(f"  Covariates: {X.shape[1]}")
        logger.info(f"  Treatment prevalence: {T.mean():.1%}")
        logger.info(f"  Outcome prevalence: {Y.mean():.1%}")

        # Weights
        full_weight = cohort['full_weight'].values

        # Replicate weights
        rep_cols = [c for c in cohort.columns if c.startswith('rep_weight_')]
        rep_cols = sorted(rep_cols)

        replicate_weights = cohort[rep_cols].values

        logger.info(f"  Replicate weights: {replicate_weights.shape[1]}")

        return X, T, Y, full_weight, replicate_weights

    def estimate_ate_att(
        self,
        X: np.ndarray,
        T: np.ndarray,
        Y: np.ndarray,
        full_weight: np.ndarray,
        replicate_weights: np.ndarray,
        cohort_type: str
    ) -> Dict:
        """
        Estimate ATE and ATT with survey-consistent variance.

        Parameters
        ----------
        X : np.ndarray
            Covariates
        T : np.ndarray
            Treatment
        Y : np.ndarray
            Outcome
        full_weight : np.ndarray
            Full weight
        replicate_weights : np.ndarray
            Replicate weights
        cohort_type : str
            'youth' or 'adult'

        Returns
        -------
        results : dict
            ATE and ATT results
        """
        logger.info("\n" + "=" * 80)
        logger.info(f"DML ESTIMATION: {cohort_type.upper()} COHORT")
        logger.info("=" * 80)

        # Get nuisance models
        model_t, model_y = get_default_models(
            task='binary',
            use_xgboost=True,
            random_state=self.random_state
        )

        logger.info(f"\nNuisance models:")
        logger.info(f"  Treatment (propensity): {type(model_t).__name__}")
        logger.info(f"  Outcome: {type(model_y).__name__}")

        results = {}

        # ATE
        logger.info("\n" + "-" * 80)
        logger.info("AVERAGE TREATMENT EFFECT (ATE)")
        logger.info("-" * 80)

        ate_results = dml_with_survey_variance(
            X=X,
            T=T,
            Y=Y,
            full_weight=full_weight,
            replicate_weights=replicate_weights,
            model_t=model_t,
            model_y=model_y,
            n_folds=self.n_folds,
            fay_coefficient=self.fay_coefficient,
            random_state=self.random_state,
            estimand='ATE'
        )

        results['ATE'] = ate_results

        logger.info(f"\nATE: {ate_results['point_estimate']:.4f}")
        logger.info(f"SE: {ate_results['std_error']:.4f}")
        logger.info(f"95% CI: [{ate_results['ci_lower']:.4f}, {ate_results['ci_upper']:.4f}]")

        # ATT
        logger.info("\n" + "-" * 80)
        logger.info("AVERAGE TREATMENT EFFECT ON THE TREATED (ATT)")
        logger.info("-" * 80)

        att_results = dml_with_survey_variance(
            X=X,
            T=T,
            Y=Y,
            full_weight=full_weight,
            replicate_weights=replicate_weights,
            model_t=model_t,
            model_y=model_y,
            n_folds=self.n_folds,
            fay_coefficient=self.fay_coefficient,
            random_state=self.random_state,
            estimand='ATT'
        )

        results['ATT'] = att_results

        logger.info(f"\nATT: {att_results['point_estimate']:.4f}")
        logger.info(f"SE: {att_results['std_error']:.4f}")
        logger.info(f"95% CI: [{att_results['ci_lower']:.4f}, {att_results['ci_upper']:.4f}]")

        # Diagnostics
        logger.info("\n" + "-" * 80)
        logger.info("DIAGNOSTICS")
        logger.info("-" * 80)

        propensity_scores = ate_results['nuisance_scores']['propensity']

        # Positivity check
        positivity = check_positivity(propensity_scores)

        logger.info(f"\nPositivity check:")
        logger.info(f"  Propensity score range: [{positivity['min']:.3f}, {positivity['max']:.3f}]")
        logger.info(f"  Violations: {positivity['violations_total']} ({positivity['violation_rate']:.1%})")
        logger.info(f"  Status: {'PASS' if positivity['passes'] else 'FAIL'}")

        results['diagnostics'] = {
            'positivity': positivity
        }

        # Balance check
        balance_df = balance_check(X, T, propensity_scores, full_weight)

        logger.info(f"\nBalance check:")
        logger.info(f"  Mean absolute SMD (unadjusted): {balance_df['smd_unadjusted'].abs().mean():.3f}")
        logger.info(f"  Mean absolute SMD (adjusted): {balance_df['smd_adjusted'].abs().mean():.3f}")
        logger.info(f"  Improvement: {balance_df['improvement'].mean():.3f}")

        results['diagnostics']['balance'] = balance_df

        return results

    def create_diagnostic_plots(
        self,
        results: Dict,
        cohort_type: str,
        X: np.ndarray,
        T: np.ndarray
    ) -> None:
        """
        Create diagnostic plots.

        Parameters
        ----------
        results : dict
            Estimation results
        cohort_type : str
            'youth' or 'adult'
        X : np.ndarray
            Covariates
        T : np.ndarray
            Treatment
        """
        logger.info(f"\nCreating diagnostic plots for {cohort_type} cohort...")

        figures_dir = self.output_dir / 'figures'
        figures_dir.mkdir(parents=True, exist_ok=True)

        # Propensity distribution
        propensity_scores = results['ATE']['nuisance_scores']['propensity']

        fig1 = plot_propensity_distribution(
            propensity_scores=propensity_scores,
            treatment=T,
            save_path=figures_dir / f'propensity_distribution_{cohort_type}.pdf'
        )
        plt.close(fig1)

        logger.info(f"  Saved: propensity_distribution_{cohort_type}.pdf")

        # Balance plot
        balance_df = results['diagnostics']['balance']

        fig2 = plot_balance(
            balance_df=balance_df,
            save_path=figures_dir / f'balance_{cohort_type}.pdf'
        )
        plt.close(fig2)

        logger.info(f"  Saved: balance_{cohort_type}.pdf")

    def save_results(self, results: Dict, cohort_type: str) -> None:
        """
        Save estimation results.

        Parameters
        ----------
        results : dict
            Estimation results
        cohort_type : str
            'youth' or 'adult'
        """
        logger.info(f"\nSaving results for {cohort_type} cohort...")

        # Save full results (pickle)
        results_path = self.results_dir / f'dml_results_{cohort_type}.pkl'

        with open(results_path, 'wb') as f:
            pickle.dump(results, f)

        logger.info(f"  Full results: {results_path}")

        # Save summary (JSON)
        summary = {
            'cohort_type': cohort_type,
            'ATE': {
                'estimate': float(results['ATE']['point_estimate']),
                'std_error': float(results['ATE']['std_error']),
                'ci_lower': float(results['ATE']['ci_lower']),
                'ci_upper': float(results['ATE']['ci_upper']),
                'n_replicates': int(results['ATE']['n_replicates'])
            },
            'ATT': {
                'estimate': float(results['ATT']['point_estimate']),
                'std_error': float(results['ATT']['std_error']),
                'ci_lower': float(results['ATT']['ci_lower']),
                'ci_upper': float(results['ATT']['ci_upper']),
                'n_replicates': int(results['ATT']['n_replicates'])
            },
            'diagnostics': {
                'positivity_passes': bool(results['diagnostics']['positivity']['passes']),
                'positivity_violation_rate': float(results['diagnostics']['positivity']['violation_rate']),
                'balance_smd_unadjusted_mean': float(results['diagnostics']['balance']['smd_unadjusted'].abs().mean()),
                'balance_smd_adjusted_mean': float(results['diagnostics']['balance']['smd_adjusted'].abs().mean())
            }
        }

        summary_path = self.results_dir / f'dml_results_{cohort_type}_summary.json'

        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"  Summary: {summary_path}")

    def create_results_table(self, youth_results: Dict, adult_results: Dict) -> pd.DataFrame:
        """
        Create summary results table.

        Parameters
        ----------
        youth_results : dict
            Youth cohort results
        adult_results : dict
            Adult cohort results

        Returns
        -------
        table : pd.DataFrame
            Results table
        """
        logger.info("\nCreating summary results table...")

        rows = []

        for cohort_name, results in [('Youth (Smoking Initiation)', youth_results), ('Adult (Smoking Cessation)', adult_results)]:
            for estimand in ['ATE', 'ATT']:
                res = results[estimand]

                row = {
                    'Cohort': cohort_name,
                    'Estimand': estimand,
                    'Estimate': f"{res['point_estimate']:.4f}",
                    'Std. Error': f"{res['std_error']:.4f}",
                    '95% CI Lower': f"{res['ci_lower']:.4f}",
                    '95% CI Upper': f"{res['ci_upper']:.4f}",
                    'Significant': 'Yes' if res['ci_lower'] > 0 or res['ci_upper'] < 0 else 'No'
                }

                rows.append(row)

        table = pd.DataFrame(rows)

        # Save
        table_path = self.output_dir / 'tables' / 'main_results.csv'
        table_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(table_path, index=False)

        logger.info(f"Results table saved to: {table_path}")

        # Also save LaTeX version
        latex_path = self.output_dir / 'tables' / 'main_results.tex'
        table.to_latex(latex_path, index=False)

        logger.info(f"LaTeX table saved to: {latex_path}")

        return table


def main():
    """Main execution."""
    logger.info("=" * 80)
    logger.info("DML VAPING/SMOKING STUDY - MAIN ESTIMATION")
    logger.info("=" * 80)

    estimator = DMLEstimator()

    # Youth cohort
    logger.info("\n" + "=" * 80)
    logger.info("YOUTH SMOKING INITIATION ANALYSIS")
    logger.info("=" * 80)

    youth_cohort = estimator.load_cohort('youth')
    X_youth, T_youth, Y_youth, w_youth, rw_youth = estimator.prepare_dml_inputs(youth_cohort, 'youth')

    youth_results = estimator.estimate_ate_att(
        X=X_youth,
        T=T_youth,
        Y=Y_youth,
        full_weight=w_youth,
        replicate_weights=rw_youth,
        cohort_type='youth'
    )

    estimator.create_diagnostic_plots(youth_results, 'youth', X_youth, T_youth)
    estimator.save_results(youth_results, 'youth')

    # Adult cohort
    logger.info("\n" + "=" * 80)
    logger.info("ADULT SMOKING CESSATION ANALYSIS")
    logger.info("=" * 80)

    adult_cohort = estimator.load_cohort('adult')
    X_adult, T_adult, Y_adult, w_adult, rw_adult = estimator.prepare_dml_inputs(adult_cohort, 'adult')

    adult_results = estimator.estimate_ate_att(
        X=X_adult,
        T=T_adult,
        Y=Y_adult,
        full_weight=w_adult,
        replicate_weights=rw_adult,
        cohort_type='adult'
    )

    estimator.create_diagnostic_plots(adult_results, 'adult', X_adult, T_adult)
    estimator.save_results(adult_results, 'adult')

    # Summary table
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY RESULTS")
    logger.info("=" * 80)

    results_table = estimator.create_results_table(youth_results, adult_results)

    logger.info("\n" + results_table.to_string(index=False))

    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("ESTIMATION COMPLETE")
    logger.info("=" * 80)
    logger.info("\nKey findings:")
    logger.info(f"  Youth ATE: {youth_results['ATE']['point_estimate']:.4f} "
                f"({youth_results['ATE']['ci_lower']:.4f}, {youth_results['ATE']['ci_upper']:.4f})")
    logger.info(f"  Adult ATE: {adult_results['ATE']['point_estimate']:.4f} "
                f"({adult_results['ATE']['ci_lower']:.4f}, {adult_results['ATE']['ci_upper']:.4f})")
    logger.info("\nNext step: Run 04_robustness_heterogeneity_iv.py")
    logger.info("=" * 80)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt

    main()
