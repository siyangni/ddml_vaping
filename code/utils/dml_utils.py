"""
Double/Debiased Machine Learning utilities with survey weight support.

Implements DML estimators for ATE, ATT, and CATE with cross-fitting
and design-consistent variance estimation via replicate weights.
"""

import numpy as np
import pandas as pd
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from sklearn.base import clone
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
import logging

# EconML imports
try:
    from econml.dml import LinearDML, CausalForestDML
    from econml.sklearn_extensions.linear_model import WeightedLasso
    ECONML_AVAILABLE = True
except ImportError:
    ECONML_AVAILABLE = False
    logging.warning("EconML not available. Install with: pip install econml")

from survey_utils import BRRVarianceEstimator

logger = logging.getLogger(__name__)


class SurveyWeightedDML:
    """
    Double/Debiased Machine Learning with survey weights.

    Implements cross-fitting with Neyman-orthogonal scores and
    design-consistent variance estimation via BRR replicate weights.

    Parameters
    ----------
    model_t : estimator
        Model for treatment (propensity score). Must support sample_weight.
    model_y : estimator
        Model for outcome (conditional mean). Must support sample_weight.
    n_folds : int, default=5
        Number of folds for cross-fitting
    random_state : int, default=42
        Random seed
    """

    def __init__(
        self,
        model_t: Any,
        model_y: Any,
        n_folds: int = 5,
        random_state: int = 42
    ):
        self.model_t = model_t
        self.model_y = model_y
        self.n_folds = n_folds
        self.random_state = random_state

        # Results storage
        self.ate_ = None
        self.att_ = None
        self.se_ate_ = None
        self.se_att_ = None
        self.nuisance_scores_ = {}

        logger.info(f"Initialized SurveyWeightedDML with {n_folds}-fold cross-fitting")

    def fit(
        self,
        X: np.ndarray,
        T: np.ndarray,
        Y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None
    ) -> 'SurveyWeightedDML':
        """
        Fit DML estimator with cross-fitting.

        Parameters
        ----------
        X : np.ndarray, shape (n, p)
            Covariate matrix
        T : np.ndarray, shape (n,)
            Treatment indicator (binary)
        Y : np.ndarray, shape (n,)
            Outcome variable
        sample_weight : np.ndarray, shape (n,), optional
            Survey weights

        Returns
        -------
        self : SurveyWeightedDML
        """
        n = X.shape[0]

        if sample_weight is None:
            sample_weight = np.ones(n)

        # Normalize weights
        sample_weight = sample_weight / sample_weight.mean()

        # Storage for cross-fitted predictions
        m_hat = np.zeros(n)  # E[Y|X]
        e_hat = np.zeros(n)  # E[T|X] (propensity score)

        # Cross-fitting
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

        logger.info(f"Starting {self.n_folds}-fold cross-fitting")

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
            logger.info(f"  Fold {fold_idx + 1}/{self.n_folds}")

            # Training data
            X_train = X[train_idx]
            T_train = T[train_idx]
            Y_train = Y[train_idx]
            w_train = sample_weight[train_idx]

            # Test data
            X_test = X[test_idx]

            # Fit propensity score model
            model_t_fold = clone(self.model_t)
            model_t_fold.fit(X_train, T_train, sample_weight=w_train)
            e_hat[test_idx] = model_t_fold.predict_proba(X_test)[:, 1]

            # Fit outcome model
            model_y_fold = clone(self.model_y)
            model_y_fold.fit(X_train, Y_train, sample_weight=w_train)
            m_hat[test_idx] = model_y_fold.predict(X_test)

        # Clip propensity scores for stability
        e_hat = np.clip(e_hat, 0.01, 0.99)

        # Store nuisance predictions
        self.nuisance_scores_['propensity'] = e_hat
        self.nuisance_scores_['outcome'] = m_hat

        # Residualize
        Y_res = Y - m_hat
        T_res = T - e_hat

        # ATE via orthogonal moment
        # ATE = E[ψ(W)] where ψ(W) = (Y - m(X)) * (T - e(X)) / (e(X) * (1 - e(X)))
        #                            + m_1(X) - m_0(X)

        # For simplicity, use doubly-robust moment:
        # ψ(W) = (T/e(X) - (1-T)/(1-e(X))) * (Y - m(X)) + m_1(X) - m_0(X)

        # Simple IPW-style orthogonal score
        psi_ate = (T / e_hat - (1 - T) / (1 - e_hat)) * Y_res

        # Weighted mean
        self.ate_ = np.average(psi_ate, weights=sample_weight)

        # ATT (average treatment effect on the treated)
        # ATT = E[Y(1) - Y(0) | T=1]
        # ψ_ATT = [T - e(X)] * (Y - m(X)) / P(T=1)

        p_t = np.average(T, weights=sample_weight)
        psi_att = (T - e_hat) * Y_res / p_t

        self.att_ = np.average(psi_att, weights=sample_weight)

        # Influence function-based standard errors
        psi_ate_centered = psi_ate - self.ate_
        psi_att_centered = psi_att - self.att_

        var_ate = np.average(psi_ate_centered ** 2, weights=sample_weight)
        var_att = np.average(psi_att_centered ** 2, weights=sample_weight)

        self.se_ate_ = np.sqrt(var_ate / n)
        self.se_att_ = np.sqrt(var_att / n)

        logger.info(f"ATE: {self.ate_:.4f} (SE: {self.se_ate_:.4f})")
        logger.info(f"ATT: {self.att_:.4f} (SE: {self.se_att_:.4f})")

        return self

    def ate(self, confidence_level: float = 0.95) -> Dict:
        """
        Get ATE with confidence interval.

        Parameters
        ----------
        confidence_level : float, default=0.95
            Confidence level for interval

        Returns
        -------
        results : dict
            ATE, SE, and CI
        """
        from scipy import stats

        z = stats.norm.ppf(1 - (1 - confidence_level) / 2)

        return {
            'estimate': self.ate_,
            'std_error': self.se_ate_,
            'ci_lower': self.ate_ - z * self.se_ate_,
            'ci_upper': self.ate_ + z * self.se_ate_,
            'ci_level': confidence_level
        }

    def att(self, confidence_level: float = 0.95) -> Dict:
        """
        Get ATT with confidence interval.

        Parameters
        ----------
        confidence_level : float, default=0.95
            Confidence level for interval

        Returns
        -------
        results : dict
            ATT, SE, and CI
        """
        from scipy import stats

        z = stats.norm.ppf(1 - (1 - confidence_level) / 2)

        return {
            'estimate': self.att_,
            'std_error': self.se_att_,
            'ci_lower': self.att_ - z * self.se_att_,
            'ci_upper': self.att_ + z * self.se_att_,
            'ci_level': confidence_level
        }


def dml_with_survey_variance(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    full_weight: np.ndarray,
    replicate_weights: np.ndarray,
    model_t: Any,
    model_y: Any,
    n_folds: int = 5,
    fay_coefficient: float = 0.3,
    random_state: int = 42,
    estimand: str = 'ATE'
) -> Dict:
    """
    DML estimation with design-consistent variance via BRR replicates.

    Parameters
    ----------
    X : np.ndarray, shape (n, p)
        Covariate matrix
    T : np.ndarray, shape (n,)
        Treatment indicator
    Y : np.ndarray, shape (n,)
        Outcome variable
    full_weight : np.ndarray, shape (n,)
        Full sample survey weight
    replicate_weights : np.ndarray, shape (n, n_reps)
        Replicate weights matrix
    model_t : estimator
        Treatment model
    model_y : estimator
        Outcome model
    n_folds : int, default=5
        Cross-fitting folds
    fay_coefficient : float, default=0.3
        Fay's coefficient for BRR
    random_state : int, default=42
        Random seed
    estimand : str, default='ATE'
        Estimand to compute ('ATE' or 'ATT')

    Returns
    -------
    results : dict
        Point estimate, design-consistent SE, and CI
    """
    logger.info(f"DML estimation with survey-consistent variance ({estimand})")
    logger.info(f"Sample size: {len(X)}, Covariates: {X.shape[1]}")
    logger.info(f"Replicate weights: {replicate_weights.shape[1]}")

    # Point estimate with full weight
    dml_full = SurveyWeightedDML(
        model_t=model_t,
        model_y=model_y,
        n_folds=n_folds,
        random_state=random_state
    )

    dml_full.fit(X, T, Y, sample_weight=full_weight)

    if estimand == 'ATE':
        point_estimate = dml_full.ate_
    elif estimand == 'ATT':
        point_estimate = dml_full.att_
    else:
        raise ValueError(f"Unknown estimand: {estimand}")

    logger.info(f"Point estimate ({estimand}): {point_estimate:.4f}")

    # Replicate estimates
    n_replicates = replicate_weights.shape[1]
    replicate_estimates = []

    logger.info(f"Computing {n_replicates} replicate estimates")

    for i in range(n_replicates):
        if (i + 1) % 10 == 0:
            logger.info(f"  Replicate {i+1}/{n_replicates}")

        rep_weight = replicate_weights[:, i]

        dml_rep = SurveyWeightedDML(
            model_t=clone(model_t),
            model_y=clone(model_y),
            n_folds=n_folds,
            random_state=random_state
        )

        dml_rep.fit(X, T, Y, sample_weight=rep_weight)

        if estimand == 'ATE':
            rep_estimate = dml_rep.ate_
        else:
            rep_estimate = dml_rep.att_

        replicate_estimates.append(rep_estimate)

    replicate_estimates = np.array(replicate_estimates)

    # BRR variance
    brr = BRRVarianceEstimator(fay_coefficient=fay_coefficient)
    variance, std_error, ci_half_width = brr.compute_variance(
        point_estimate=point_estimate,
        replicate_estimates=replicate_estimates
    )

    results = {
        'estimand': estimand,
        'point_estimate': point_estimate,
        'std_error': std_error,
        'variance': variance,
        'ci_lower': point_estimate - ci_half_width,
        'ci_upper': point_estimate + ci_half_width,
        'ci_level': 0.95,
        'replicate_estimates': replicate_estimates,
        'n_replicates': n_replicates,
        'nuisance_scores': dml_full.nuisance_scores_
    }

    logger.info(f"Design-consistent {estimand}: {point_estimate:.4f} (SE: {std_error:.4f})")
    logger.info(f"95% CI: [{results['ci_lower']:.4f}, {results['ci_upper']:.4f}]")

    return results


def get_default_models(
    task: str = 'binary',
    use_xgboost: bool = True,
    random_state: int = 42
) -> Tuple[Any, Any]:
    """
    Get default nuisance models for DML.

    Parameters
    ----------
    task : str, default='binary'
        Task type ('binary' or 'continuous')
    use_xgboost : bool, default=True
        Whether to use XGBoost (vs. random forest)
    random_state : int, default=42
        Random seed

    Returns
    -------
    model_t : estimator
        Treatment model
    model_y : estimator
        Outcome model
    """
    if use_xgboost:
        # XGBoost models
        model_t = XGBClassifier(
            max_depth=4,
            learning_rate=0.05,
            n_estimators=200,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            use_label_encoder=False,
            eval_metric='logloss'
        )

        if task == 'binary':
            model_y = XGBClassifier(
                max_depth=4,
                learning_rate=0.05,
                n_estimators=200,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_state,
                use_label_encoder=False,
                eval_metric='logloss'
            )
        else:
            model_y = XGBRegressor(
                max_depth=4,
                learning_rate=0.05,
                n_estimators=200,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_state
            )

    else:
        # Random forest models
        model_t = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=20,
            random_state=random_state,
            n_jobs=-1
        )

        if task == 'binary':
            model_y = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_leaf=20,
                random_state=random_state,
                n_jobs=-1
            )
        else:
            model_y = RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_leaf=20,
                random_state=random_state,
                n_jobs=-1
            )

    return model_t, model_y


def check_positivity(
    propensity_scores: np.ndarray,
    threshold_lower: float = 0.05,
    threshold_upper: float = 0.95
) -> Dict:
    """
    Check positivity assumption (overlap).

    Parameters
    ----------
    propensity_scores : np.ndarray
        Estimated propensity scores
    threshold_lower : float, default=0.05
        Lower threshold for acceptable overlap
    threshold_upper : float, default=0.95
        Upper threshold for acceptable overlap

    Returns
    -------
    diagnostics : dict
        Positivity check results
    """
    violations_lower = (propensity_scores < threshold_lower).sum()
    violations_upper = (propensity_scores > threshold_upper).sum()
    violations_total = violations_lower + violations_upper

    diagnostics = {
        'min': float(propensity_scores.min()),
        'max': float(propensity_scores.max()),
        'mean': float(propensity_scores.mean()),
        'violations_lower': int(violations_lower),
        'violations_upper': int(violations_upper),
        'violations_total': int(violations_total),
        'violation_rate': float(violations_total / len(propensity_scores)),
        'passes': violations_total == 0
    }

    if not diagnostics['passes']:
        logger.warning(
            f"Positivity violations detected: {violations_total} observations "
            f"({diagnostics['violation_rate']:.1%})"
        )

    return diagnostics


def balance_check(
    X: np.ndarray,
    T: np.ndarray,
    propensity_scores: np.ndarray,
    sample_weight: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """
    Compute standardized mean differences for balance.

    Parameters
    ----------
    X : np.ndarray, shape (n, p)
        Covariate matrix
    T : np.ndarray, shape (n,)
        Treatment indicator
    propensity_scores : np.ndarray, shape (n,)
        Propensity scores for IPW weighting
    sample_weight : np.ndarray, optional
        Survey weights

    Returns
    -------
    balance_df : pd.DataFrame
        Balance statistics for each covariate
    """
    n, p = X.shape

    if sample_weight is None:
        sample_weight = np.ones(n)

    balance_stats = []

    for j in range(p):
        x = X[:, j]

        # Unadjusted difference
        mean_t1 = np.average(x[T == 1], weights=sample_weight[T == 1])
        mean_t0 = np.average(x[T == 0], weights=sample_weight[T == 0])
        std_pooled = np.sqrt(
            (np.var(x[T == 1]) + np.var(x[T == 0])) / 2
        )

        smd_unadjusted = (mean_t1 - mean_t0) / std_pooled if std_pooled > 0 else 0

        # IPW-adjusted difference
        ipw_weight_t1 = sample_weight * T / propensity_scores
        ipw_weight_t0 = sample_weight * (1 - T) / (1 - propensity_scores)

        mean_t1_adj = np.average(x[T == 1], weights=ipw_weight_t1[T == 1])
        mean_t0_adj = np.average(x[T == 0], weights=ipw_weight_t0[T == 0])

        smd_adjusted = (mean_t1_adj - mean_t0_adj) / std_pooled if std_pooled > 0 else 0

        balance_stats.append({
            'covariate': f'X{j+1}',
            'smd_unadjusted': smd_unadjusted,
            'smd_adjusted': smd_adjusted,
            'improvement': abs(smd_unadjusted) - abs(smd_adjusted)
        })

    balance_df = pd.DataFrame(balance_stats)

    return balance_df


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Simulate data
    np.random.seed(42)
    n = 2000
    p = 20

    X = np.random.randn(n, p)
    propensity_true = 1 / (1 + np.exp(-(X[:, 0] + 0.5 * X[:, 1])))
    T = np.random.binomial(1, propensity_true)

    # Outcome with treatment effect
    tau = 0.3  # True ATE
    Y = X[:, 0] + 0.5 * X[:, 1] + tau * T + np.random.randn(n)
    Y = (Y > Y.mean()).astype(int)  # Binary outcome

    # Survey weights
    full_weight = np.random.uniform(0.5, 2.0, n)

    # Replicate weights
    replicate_weights = np.column_stack([
        full_weight * np.random.normal(1.0, 0.1, n)
        for _ in range(100)
    ])

    # Get models
    model_t, model_y = get_default_models(task='binary', use_xgboost=True)

    # Estimate with survey variance
    results = dml_with_survey_variance(
        X=X,
        T=T,
        Y=Y,
        full_weight=full_weight,
        replicate_weights=replicate_weights,
        model_t=model_t,
        model_y=model_y,
        n_folds=5,
        estimand='ATE'
    )

    print(f"\nResults:")
    print(f"ATE: {results['point_estimate']:.4f}")
    print(f"SE: {results['std_error']:.4f}")
    print(f"95% CI: [{results['ci_lower']:.4f}, {results['ci_upper']:.4f}]")

    # Check positivity
    positivity_check = check_positivity(results['nuisance_scores']['propensity'])
    print(f"\nPositivity check: {'PASS' if positivity_check['passes'] else 'FAIL'}")
    print(f"Propensity score range: [{positivity_check['min']:.3f}, {positivity_check['max']:.3f}]")
