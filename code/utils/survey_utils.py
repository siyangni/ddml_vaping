"""
Survey weight utilities for design-consistent DML estimation.

Implements BRR variance estimation with Fay's method for PATH Study
and Taylor linearization helpers for NYTS.
"""

import numpy as np
import pandas as pd
from typing import Callable, Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class BRRVarianceEstimator:
    """
    Balanced Repeated Replication (BRR) variance estimator for survey data.

    Implements Fay's method for PATH Study replicate weights.

    Parameters
    ----------
    fay_coefficient : float, default=0.3
        Fay's adjustment coefficient (PATH Study uses 0.3)
    """

    def __init__(self, fay_coefficient: float = 0.3):
        self.fay_coefficient = fay_coefficient
        self.c = 1 / (100 * (1 - fay_coefficient) ** 2)
        logger.info(f"BRR variance estimator initialized with Fay coefficient = {fay_coefficient}")
        logger.info(f"Variance multiplier c = {self.c:.6f}")

    def compute_variance(
        self,
        point_estimate: float,
        replicate_estimates: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Compute design-consistent variance from replicate estimates.

        Parameters
        ----------
        point_estimate : float
            Point estimate from full sample weight
        replicate_estimates : np.ndarray
            Array of estimates from each replicate weight (length = n_replicates)

        Returns
        -------
        variance : float
            Design-consistent variance estimate
        std_error : float
            Standard error
        ci_half_width : float
            Half-width of 95% confidence interval (1.96 * SE)
        """
        n_replicates = len(replicate_estimates)

        # BRR variance formula with Fay's method
        squared_deviations = (replicate_estimates - point_estimate) ** 2
        variance = self.c * np.sum(squared_deviations)

        std_error = np.sqrt(variance)
        ci_half_width = 1.96 * std_error

        logger.debug(f"Computed variance from {n_replicates} replicates")
        logger.debug(f"Variance = {variance:.6f}, SE = {std_error:.6f}")

        return variance, std_error, ci_half_width

    def estimate_with_replicates(
        self,
        estimator_func: Callable,
        data: pd.DataFrame,
        full_weight_col: str,
        replicate_weight_cols: List[str],
        **estimator_kwargs
    ) -> Dict:
        """
        Run estimator with full weight and all replicate weights.

        Parameters
        ----------
        estimator_func : Callable
            Function that takes (data, weight_col, **kwargs) and returns point estimate
        data : pd.DataFrame
            Analysis dataset
        full_weight_col : str
            Column name for full sample weight
        replicate_weight_cols : List[str]
            List of replicate weight column names
        **estimator_kwargs : dict
            Additional arguments passed to estimator_func

        Returns
        -------
        results : dict
            Dictionary containing:
            - point_estimate: float
            - variance: float
            - std_error: float
            - ci_lower: float (95% CI)
            - ci_upper: float (95% CI)
            - replicate_estimates: np.ndarray
        """
        logger.info("Starting design-consistent estimation with BRR replicates")

        # Point estimate with full weight
        logger.info(f"Computing point estimate with {full_weight_col}")
        point_estimate = estimator_func(
            data=data,
            weight_col=full_weight_col,
            **estimator_kwargs
        )
        logger.info(f"Point estimate: {point_estimate:.6f}")

        # Replicate estimates
        replicate_estimates = []
        n_replicates = len(replicate_weight_cols)

        logger.info(f"Computing {n_replicates} replicate estimates")
        for i, rep_weight_col in enumerate(replicate_weight_cols):
            if (i + 1) % 10 == 0:
                logger.info(f"  Replicate {i+1}/{n_replicates}")

            rep_estimate = estimator_func(
                data=data,
                weight_col=rep_weight_col,
                **estimator_kwargs
            )
            replicate_estimates.append(rep_estimate)

        replicate_estimates = np.array(replicate_estimates)

        # Compute variance
        variance, std_error, ci_half_width = self.compute_variance(
            point_estimate=point_estimate,
            replicate_estimates=replicate_estimates
        )

        # Confidence intervals
        ci_lower = point_estimate - ci_half_width
        ci_upper = point_estimate + ci_half_width

        results = {
            'point_estimate': point_estimate,
            'variance': variance,
            'std_error': std_error,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'ci_level': 0.95,
            'replicate_estimates': replicate_estimates,
            'n_replicates': n_replicates
        }

        logger.info(f"Results: {point_estimate:.4f} (95% CI: [{ci_lower:.4f}, {ci_upper:.4f}])")

        return results


def identify_weight_columns(df: pd.DataFrame, wave: Optional[int] = None) -> Dict[str, any]:
    """
    Automatically identify weight columns in PATH data.

    Parameters
    ----------
    df : pd.DataFrame
        PATH dataset
    wave : int, optional
        Wave number for longitudinal weights

    Returns
    -------
    weight_info : dict
        Dictionary with 'full_weight', 'replicate_weights' keys
    """
    cols = df.columns.tolist()

    # PATH weight naming patterns
    # Full weights: R{wave}_{cohort}_A{weights}
    # Replicate weights: R{wave}_{cohort}_A{weights}_REP{num}

    weight_info = {
        'full_weight': None,
        'replicate_weights': []
    }

    # Look for full weight (no _REP suffix)
    full_weight_candidates = [
        c for c in cols
        if ('weight' in c.lower() or '_a' in c.lower())
        and '_rep' not in c.lower()
        and 'flag' not in c.lower()
    ]

    if full_weight_candidates:
        # Prefer longitudinal weights if wave specified
        if wave is not None:
            wave_specific = [c for c in full_weight_candidates if f'r{wave}_' in c.lower() or f'w{wave}_' in c.lower()]
            if wave_specific:
                weight_info['full_weight'] = wave_specific[0]

        if weight_info['full_weight'] is None:
            weight_info['full_weight'] = full_weight_candidates[0]

    # Look for replicate weights
    if weight_info['full_weight']:
        base_name = weight_info['full_weight']
        replicate_pattern = base_name.lower()

        replicate_weights = [
            c for c in cols
            if replicate_pattern in c.lower()
            and '_rep' in c.lower()
        ]
        replicate_weights = sorted(replicate_weights)
        weight_info['replicate_weights'] = replicate_weights

    logger.info(f"Identified full weight: {weight_info['full_weight']}")
    logger.info(f"Identified {len(weight_info['replicate_weights'])} replicate weights")

    return weight_info


def normalize_weights(
    data: pd.DataFrame,
    weight_col: str,
    target_n: Optional[int] = None
) -> pd.DataFrame:
    """
    Normalize survey weights to sum to sample size.

    Some ML algorithms work better with normalized weights.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset with weight column
    weight_col : str
        Name of weight column
    target_n : int, optional
        Target sum for weights (default: sample size)

    Returns
    -------
    data : pd.DataFrame
        Data with normalized weight column
    """
    if target_n is None:
        target_n = len(data)

    original_sum = data[weight_col].sum()
    data[weight_col + '_normalized'] = data[weight_col] * (target_n / original_sum)

    logger.debug(f"Normalized {weight_col}: original sum = {original_sum:.0f}, new sum = {target_n}")

    return data


def check_weight_validity(
    data: pd.DataFrame,
    weight_cols: List[str],
    min_weight: float = 0.0,
    max_weight_ratio: float = 1000.0
) -> Dict[str, bool]:
    """
    Check validity of survey weights.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset with weight columns
    weight_cols : List[str]
        List of weight column names to check
    min_weight : float
        Minimum acceptable weight value
    max_weight_ratio : float
        Maximum ratio of max/min weight

    Returns
    -------
    checks : dict
        Dictionary of check results
    """
    checks = {}

    for col in weight_cols:
        weights = data[col].dropna()

        # Check for negative or zero weights
        has_invalid = (weights <= min_weight).any()

        # Check for extreme weight ratios
        if len(weights) > 0 and weights.min() > 0:
            weight_ratio = weights.max() / weights.min()
            has_extreme_ratio = weight_ratio > max_weight_ratio
        else:
            has_extreme_ratio = True

        # Check for missing weights
        has_missing = data[col].isna().any()

        checks[col] = {
            'valid': not (has_invalid or has_extreme_ratio),
            'has_invalid_values': has_invalid,
            'has_extreme_ratio': has_extreme_ratio,
            'has_missing': has_missing,
            'n_observations': len(weights),
            'min': float(weights.min()) if len(weights) > 0 else None,
            'max': float(weights.max()) if len(weights) > 0 else None,
            'mean': float(weights.mean()) if len(weights) > 0 else None
        }

    return checks


def create_survey_design_vars(
    data: pd.DataFrame,
    strata_col: Optional[str] = None,
    psu_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Create or identify survey design variables (strata, PSUs).

    Parameters
    ----------
    data : pd.DataFrame
        Dataset
    strata_col : str, optional
        Column name for strata
    psu_col : str, optional
        Column name for PSUs

    Returns
    -------
    data : pd.DataFrame
        Data with standardized design variable names
    """
    # Try to auto-detect if not specified
    if strata_col is None:
        strata_candidates = [c for c in data.columns if 'strat' in c.lower()]
        if strata_candidates:
            strata_col = strata_candidates[0]
            logger.info(f"Auto-detected strata column: {strata_col}")

    if psu_col is None:
        psu_candidates = [c for c in data.columns if 'psu' in c.lower() or 'cluster' in c.lower()]
        if psu_candidates:
            psu_col = psu_candidates[0]
            logger.info(f"Auto-detected PSU column: {psu_col}")

    # Standardize names
    if strata_col:
        data['survey_strata'] = data[strata_col]

    if psu_col:
        data['survey_psu'] = data[psu_col]

    return data


def compute_effective_sample_size(
    weights: np.ndarray
) -> float:
    """
    Compute effective sample size (Kish's approximation).

    ESS = (sum of weights)^2 / sum of squared weights

    Parameters
    ----------
    weights : np.ndarray
        Survey weights

    Returns
    -------
    ess : float
        Effective sample size
    """
    weights = weights[~np.isnan(weights)]

    if len(weights) == 0:
        return 0.0

    ess = (weights.sum() ** 2) / (weights ** 2).sum()

    return ess


def summarize_survey_design(
    data: pd.DataFrame,
    weight_col: str,
    strata_col: Optional[str] = None,
    psu_col: Optional[str] = None
) -> Dict:
    """
    Summarize survey design characteristics.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset
    weight_col : str
        Weight column name
    strata_col : str, optional
        Strata column name
    psu_col : str, optional
        PSU column name

    Returns
    -------
    summary : dict
        Survey design summary statistics
    """
    summary = {
        'n_observations': len(data),
        'n_weighted': float(data[weight_col].sum()),
        'effective_sample_size': compute_effective_sample_size(data[weight_col].values),
        'design_effect': None
    }

    # Design effect = n / ESS
    if summary['effective_sample_size'] > 0:
        summary['design_effect'] = summary['n_observations'] / summary['effective_sample_size']

    if strata_col and strata_col in data.columns:
        summary['n_strata'] = data[strata_col].nunique()

    if psu_col and psu_col in data.columns:
        summary['n_psu'] = data[psu_col].nunique()

    return summary


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Simulate survey data
    np.random.seed(42)
    n = 1000

    data = pd.DataFrame({
        'treatment': np.random.binomial(1, 0.3, n),
        'outcome': np.random.binomial(1, 0.5, n),
        'weight': np.random.uniform(0.5, 2.0, n)
    })

    # Add replicate weights
    for i in range(100):
        data[f'weight_rep{i+1}'] = data['weight'] * np.random.normal(1.0, 0.1, n)

    # Check weights
    weight_cols = ['weight'] + [f'weight_rep{i+1}' for i in range(100)]
    checks = check_weight_validity(data, weight_cols)
    print(f"Weight validity: {checks['weight']['valid']}")

    # Compute effective sample size
    ess = compute_effective_sample_size(data['weight'].values)
    print(f"Effective sample size: {ess:.0f}")

    # Test BRR variance estimator
    def simple_ate(data, weight_col):
        weights = data[weight_col].values
        treatment = data['treatment'].values
        outcome = data['outcome'].values

        # Weighted means
        y1_mean = np.average(outcome[treatment == 1], weights=weights[treatment == 1])
        y0_mean = np.average(outcome[treatment == 0], weights=weights[treatment == 0])

        return y1_mean - y0_mean

    brr = BRRVarianceEstimator(fay_coefficient=0.3)
    results = brr.estimate_with_replicates(
        estimator_func=simple_ate,
        data=data,
        full_weight_col='weight',
        replicate_weight_cols=[f'weight_rep{i+1}' for i in range(100)]
    )

    print(f"\nATE: {results['point_estimate']:.4f} (SE: {results['std_error']:.4f})")
    print(f"95% CI: [{results['ci_lower']:.4f}, {results['ci_upper']:.4f}]")
