"""
Pilot DML analysis with real PATH Study data.

Quick test to validate the PATH data loader and run a minimal DML estimation
on the youth smoking initiation cohort.

Usage:
    python code/analysis/pilot_dml_real_data.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / 'code' / 'utils'))

from path_loader import PATHDataLoader
from dml_utils import get_default_models
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def prepare_covariates(df, cohort_type):
    """
    Extract and prepare basic covariates for DML.

    Parameters
    ----------
    df : pd.DataFrame
        Cohort data from PATH loader
    cohort_type : str
        'youth' or 'adult'

    Returns
    -------
    X : np.ndarray
        Covariate matrix
    feature_names : list
        List of feature names
    """
    logger.info("\nPreparing covariates...")

    # Start with basic demographics that are already processed
    features = []
    feature_names = []

    # Age
    if 'age' in df.columns:
        features.append(df['age'].values.reshape(-1, 1))
        feature_names.append('age')

    # Sex
    if 'sex_Male' in df.columns:
        features.append(df['sex_Male'].values.reshape(-1, 1))
        feature_names.append('sex_Male')

    # Race/ethnicity
    for race in ['race_ethnicity_White', 'race_ethnicity_Black', 'race_ethnicity_Hispanic']:
        if race in df.columns:
            features.append(df[race].values.reshape(-1, 1))
            feature_names.append(race)

    # Combine features
    if features:
        X = np.hstack(features)
    else:
        raise ValueError("No features found in dataframe")

    # Handle missing values
    X = np.nan_to_num(X, nan=0.0)

    logger.info(f"  Features: {len(feature_names)}")
    logger.info(f"  Feature names: {feature_names}")
    logger.info(f"  Feature matrix shape: {X.shape}")

    return X, feature_names


def simple_dml_ate(X, T, Y, n_folds=2):
    """
    Simplified DML ATE estimation with cross-fitting.

    Parameters
    ----------
    X : np.ndarray
        Covariates
    T : np.ndarray
        Treatment (0/1)
    Y : np.ndarray
        Outcome (0/1)
    n_folds : int
        Number of cross-fitting folds

    Returns
    -------
    ate : float
        Average treatment effect estimate
    """
    logger.info(f"\nRunning DML estimation with {n_folds}-fold cross-fitting...")

    # Models
    model_t = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    model_y = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)

    # Cross-fitted predictions for propensity score
    logger.info("  Estimating propensity scores...")
    g_hat = cross_val_predict(model_t, X, T, cv=n_folds, method='predict_proba')[:, 1]

    # Cross-fitted predictions for outcome
    logger.info("  Estimating outcome regressions...")
    # For treated
    mask_treated = T == 1
    if mask_treated.sum() > 0:
        m1_hat = np.zeros(len(T))
        m1_hat[mask_treated] = cross_val_predict(
            model_y, X[mask_treated], Y[mask_treated], cv=min(n_folds, mask_treated.sum()), method='predict_proba'
        )[:, 1]

    # For controls
    mask_control = T == 0
    if mask_control.sum() > 0:
        m0_hat = np.zeros(len(T))
        m0_hat[mask_control] = cross_val_predict(
            model_y, X[mask_control], Y[mask_control], cv=min(n_folds, mask_control.sum()), method='predict_proba'
        )[:, 1]

    # Fit full models for out-of-sample predictions
    logger.info("  Fitting full models for residualization...")
    model_y.fit(X, Y)
    m_hat = model_y.predict_proba(X)[:, 1]

    # Neyman orthogonal scores (simplified)
    logger.info("  Computing orthogonal scores...")

    # Clip propensity scores
    g_hat = np.clip(g_hat, 0.01, 0.99)

    # Compute double residuals
    score = (T / g_hat - (1 - T) / (1 - g_hat)) * (Y - m_hat)

    # ATE estimate
    ate = np.mean(score)

    # Naive standard error (not survey-adjusted)
    se_naive = np.std(score) / np.sqrt(len(score))

    logger.info(f"\n  ATE: {ate:.4f}")
    logger.info(f"  SE (naive): {se_naive:.4f}")
    logger.info(f"  95% CI: [{ate - 1.96*se_naive:.4f}, {ate + 1.96*se_naive:.4f}]")

    # Diagnostics
    logger.info(f"\n  Diagnostics:")
    logger.info(f"    Propensity score range: [{g_hat.min():.3f}, {g_hat.max():.3f}]")
    logger.info(f"    Mean propensity (treated): {g_hat[T==1].mean():.3f}")
    logger.info(f"    Mean propensity (control): {g_hat[T==0].mean():.3f}")

    return ate, se_naive


def main():
    """Main pilot analysis."""
    logger.info("=" * 80)
    logger.info("PILOT DML ANALYSIS WITH REAL PATH DATA")
    logger.info("=" * 80)

    # Initialize PATH loader
    data_dir = project_root / "data" / "raw" / "path"
    loader = PATHDataLoader(data_dir)

    # Build youth cohort
    logger.info("\n" + "=" * 80)
    logger.info("BUILDING YOUTH SMOKING INITIATION COHORT")
    logger.info("=" * 80)

    youth_cohort = loader.build_youth_initiation_cohort()

    logger.info(f"\nYouth cohort summary:")
    logger.info(f"  N = {len(youth_cohort):,}")
    logger.info(f"  Treatment (vaping) prevalence: {youth_cohort['treatment'].mean():.2%}")
    logger.info(f"  Outcome (smoking initiation) prevalence: {youth_cohort['outcome'].mean():.2%}")

    # Check available columns
    logger.info(f"\nAvailable columns: {youth_cohort.columns.tolist()[:20]}")

    # Prepare for DML
    X, feature_names = prepare_covariates(youth_cohort, 'youth')
    T = youth_cohort['treatment'].values
    Y = youth_cohort['outcome'].values

    # Run simplified DML
    logger.info("\n" + "=" * 80)
    logger.info("DML ESTIMATION")
    logger.info("=" * 80)

    ate, se = simple_dml_ate(X, T, Y, n_folds=2)

    # Save minimal results
    logger.info("\n" + "=" * 80)
    logger.info("SAVING PILOT RESULTS")
    logger.info("=" * 80)

    output_dir = project_root / "outputs" / "pilot"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save cohort summary
    youth_cohort.head(100).to_csv(output_dir / "pilot_youth_cohort_sample.csv", index=False)
    logger.info(f"  Saved sample cohort to: {output_dir / 'pilot_youth_cohort_sample.csv'}")

    # Save results summary
    results_summary = {
        'cohort': 'youth',
        'n': len(youth_cohort),
        'treatment_prevalence': youth_cohort['treatment'].mean(),
        'outcome_prevalence': youth_cohort['outcome'].mean(),
        'ate': ate,
        'se_naive': se,
        'features': feature_names
    }

    import json
    with open(output_dir / "pilot_results.json", 'w') as f:
        json.dump(results_summary, f, indent=2, default=float)

    logger.info(f"  Saved results summary to: {output_dir / 'pilot_results.json'}")

    logger.info("\n" + "=" * 80)
    logger.info("PILOT ANALYSIS COMPLETE")
    logger.info("=" * 80)
    logger.info("\nKey findings:")
    logger.info(f"  Successfully loaded {len(youth_cohort):,} youth participants from PATH data")
    logger.info(f"  ATE of vaping on smoking initiation: {ate:.4f} (SE: {se:.4f})")
    logger.info("\nNext steps:")
    logger.info("  1. Update full pipeline to use PATH loader")
    logger.info("  2. Run complete DML estimation with survey weights")
    logger.info("  3. Conduct heterogeneity and robustness analyses")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
