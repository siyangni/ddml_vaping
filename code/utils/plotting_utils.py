"""
Plotting utilities for DML causal study visualizations.

Publication-quality plots for overlap, balance, heterogeneity,
and sensitivity analyses.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Set publication-quality style
sns.set_style("whitegrid")
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans']
})


def plot_propensity_distribution(
    propensity_scores: np.ndarray,
    treatment: np.ndarray,
    save_path: Optional[str] = None,
    dpi: int = 300
) -> plt.Figure:
    """
    Plot propensity score distributions by treatment group (overlap diagnostic).

    Parameters
    ----------
    propensity_scores : np.ndarray
        Estimated propensity scores
    treatment : np.ndarray
        Treatment indicator
    save_path : str, optional
        Path to save figure
    dpi : int, default=300
        Resolution for saved figure

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Density plots
    axes[0].hist(
        propensity_scores[treatment == 0],
        bins=50,
        alpha=0.6,
        label='Control',
        density=True,
        color='steelblue'
    )
    axes[0].hist(
        propensity_scores[treatment == 1],
        bins=50,
        alpha=0.6,
        label='Treated',
        density=True,
        color='coral'
    )
    axes[0].set_ylabel('Density')
    axes[0].set_title('Propensity Score Distribution by Treatment Group')
    axes[0].legend()
    axes[0].axvline(0.05, color='red', linestyle='--', alpha=0.5, label='Positivity thresholds')
    axes[0].axvline(0.95, color='red', linestyle='--', alpha=0.5)

    # Box plots
    data_for_box = pd.DataFrame({
        'Propensity Score': propensity_scores,
        'Treatment': ['Treated' if t == 1 else 'Control' for t in treatment]
    })

    sns.boxplot(
        data=data_for_box,
        x='Propensity Score',
        y='Treatment',
        ax=axes[1],
        orient='h',
        palette=['steelblue', 'coral']
    )
    axes[1].set_title('Propensity Score Distribution (Box Plot)')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved propensity plot to {save_path}")

    return fig


def plot_balance(
    balance_df: pd.DataFrame,
    threshold: float = 0.1,
    save_path: Optional[str] = None,
    dpi: int = 300
) -> plt.Figure:
    """
    Plot covariate balance (Love plot).

    Parameters
    ----------
    balance_df : pd.DataFrame
        Balance statistics with columns: covariate, smd_unadjusted, smd_adjusted
    threshold : float, default=0.1
        SMD threshold for acceptable balance
    save_path : str, optional
        Path to save figure
    dpi : int, default=300
        Resolution

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(10, max(6, len(balance_df) * 0.3)))

    y_pos = np.arange(len(balance_df))

    # Plot unadjusted SMDs
    ax.scatter(
        balance_df['smd_unadjusted'],
        y_pos,
        alpha=0.6,
        s=80,
        label='Unadjusted',
        color='coral',
        marker='o'
    )

    # Plot adjusted SMDs
    ax.scatter(
        balance_df['smd_adjusted'],
        y_pos,
        alpha=0.6,
        s=80,
        label='IPW-adjusted',
        color='steelblue',
        marker='s'
    )

    # Connect with lines
    for i in range(len(balance_df)):
        ax.plot(
            [balance_df.iloc[i]['smd_unadjusted'], balance_df.iloc[i]['smd_adjusted']],
            [y_pos[i], y_pos[i]],
            'k-',
            alpha=0.3,
            linewidth=1
        )

    # Threshold lines
    ax.axvline(-threshold, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(threshold, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(balance_df['covariate'])
    ax.set_xlabel('Standardized Mean Difference')
    ax.set_title('Covariate Balance Before and After IPW Adjustment')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved balance plot to {save_path}")

    return fig


def plot_ate_forest(
    results_list: List[Dict],
    labels: List[str],
    save_path: Optional[str] = None,
    dpi: int = 300
) -> plt.Figure:
    """
    Forest plot of ATEs from multiple specifications.

    Parameters
    ----------
    results_list : List[Dict]
        List of result dictionaries with 'point_estimate', 'ci_lower', 'ci_upper'
    labels : List[str]
        Labels for each specification
    save_path : str, optional
        Path to save figure
    dpi : int, default=300
        Resolution

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(10, max(4, len(results_list) * 0.5)))

    y_pos = np.arange(len(results_list))

    for i, (result, label) in enumerate(zip(results_list, labels)):
        estimate = result['point_estimate']
        ci_lower = result['ci_lower']
        ci_upper = result['ci_upper']

        # Point estimate
        ax.scatter(estimate, i, s=100, color='steelblue', zorder=3)

        # Confidence interval
        ax.plot(
            [ci_lower, ci_upper],
            [i, i],
            'k-',
            linewidth=2,
            zorder=2
        )

    ax.axvline(0, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Average Treatment Effect')
    ax.set_title('Treatment Effect Estimates Across Specifications')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved forest plot to {save_path}")

    return fig


def plot_cate_distribution(
    cates: np.ndarray,
    save_path: Optional[str] = None,
    dpi: int = 300
) -> plt.Figure:
    """
    Plot distribution of conditional average treatment effects (CATEs).

    Parameters
    ----------
    cates : np.ndarray
        Estimated CATEs
    save_path : str, optional
        Path to save figure
    dpi : int, default=300
        Resolution

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram
    axes[0].hist(cates, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    axes[0].axvline(cates.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {cates.mean():.3f}')
    axes[0].axvline(0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    axes[0].set_xlabel('Conditional Average Treatment Effect')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of CATEs')
    axes[0].legend()

    # Box plot
    axes[1].boxplot(cates, vert=True)
    axes[1].axhline(0, color='red', linestyle='--', alpha=0.5, linewidth=1)
    axes[1].set_ylabel('Conditional Average Treatment Effect')
    axes[1].set_title('CATE Distribution (Box Plot)')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved CATE distribution plot to {save_path}")

    return fig


def plot_heterogeneity_by_group(
    subgroup_results: pd.DataFrame,
    group_var: str = 'subgroup',
    save_path: Optional[str] = None,
    dpi: int = 300
) -> plt.Figure:
    """
    Plot treatment effects by subgroup.

    Parameters
    ----------
    subgroup_results : pd.DataFrame
        DataFrame with columns: subgroup, estimate, ci_lower, ci_upper
    group_var : str, default='subgroup'
        Column name for subgroup labels
    save_path : str, optional
        Path to save figure
    dpi : int, default=300
        Resolution

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(10, max(5, len(subgroup_results) * 0.5)))

    y_pos = np.arange(len(subgroup_results))

    for i, row in subgroup_results.iterrows():
        estimate = row['estimate']
        ci_lower = row['ci_lower']
        ci_upper = row['ci_upper']

        ax.scatter(estimate, i, s=120, color='steelblue', zorder=3)
        ax.plot(
            [ci_lower, ci_upper],
            [i, i],
            'k-',
            linewidth=2.5,
            zorder=2
        )

    ax.axvline(0, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(subgroup_results[group_var])
    ax.set_xlabel('Treatment Effect Estimate')
    ax.set_title(f'Treatment Effect Heterogeneity by {group_var.replace("_", " ").title()}')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved heterogeneity plot to {save_path}")

    return fig


def plot_sensitivity_contour(
    r2_treatment_grid: np.ndarray,
    r2_outcome_grid: np.ndarray,
    estimate_grid: np.ndarray,
    null_value: float = 0.0,
    save_path: Optional[str] = None,
    dpi: int = 300
) -> plt.Figure:
    """
    Contour plot for sensitivity to unobserved confounding.

    Parameters
    ----------
    r2_treatment_grid : np.ndarray
        Grid of partial R² for treatment model
    r2_outcome_grid : np.ndarray
        Grid of partial R² for outcome model
    estimate_grid : np.ndarray
        Grid of adjusted estimates
    null_value : float, default=0.0
        Null hypothesis value
    save_path : str, optional
        Path to save figure
    dpi : int, default=300
        Resolution

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    contour = ax.contourf(
        r2_treatment_grid,
        r2_outcome_grid,
        estimate_grid,
        levels=20,
        cmap='RdYlBu_r',
        alpha=0.8
    )

    # Add contour lines
    contour_lines = ax.contour(
        r2_treatment_grid,
        r2_outcome_grid,
        estimate_grid,
        levels=10,
        colors='black',
        alpha=0.3,
        linewidths=0.5
    )

    ax.clabel(contour_lines, inline=True, fontsize=8)

    # Null value contour
    null_contour = ax.contour(
        r2_treatment_grid,
        r2_outcome_grid,
        estimate_grid,
        levels=[null_value],
        colors='red',
        linewidths=2.5,
        linestyles='--'
    )

    ax.set_xlabel('Partial R² (Treatment Model)')
    ax.set_ylabel('Partial R² (Outcome Model)')
    ax.set_title('Sensitivity to Unobserved Confounding')

    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('Adjusted Treatment Effect')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved sensitivity contour to {save_path}")

    return fig


def plot_cohort_flow(
    flow_data: Dict[str, int],
    save_path: Optional[str] = None,
    dpi: int = 300
) -> plt.Figure:
    """
    CONSORT-style flow diagram (simplified text version).

    Parameters
    ----------
    flow_data : dict
        Dictionary with stage names and sample sizes
    save_path : str, optional
        Path to save figure
    dpi : int, default=300
        Resolution

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(10, max(6, len(flow_data) * 0.8)))

    ax.axis('off')

    y_start = 0.95
    y_step = 0.85 / len(flow_data)

    for i, (stage, n) in enumerate(flow_data.items()):
        y_pos = y_start - i * y_step

        # Box
        box = plt.Rectangle(
            (0.15, y_pos - 0.04),
            0.7,
            0.08,
            facecolor='lightblue',
            edgecolor='black',
            linewidth=1.5
        )
        ax.add_patch(box)

        # Text
        ax.text(
            0.5,
            y_pos,
            f"{stage}\nn = {n:,}",
            ha='center',
            va='center',
            fontsize=11,
            weight='bold'
        )

        # Arrow to next stage
        if i < len(flow_data) - 1:
            ax.arrow(
                0.5,
                y_pos - 0.04,
                0,
                -y_step + 0.08,
                head_width=0.03,
                head_length=0.02,
                fc='black',
                ec='black'
            )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Cohort Sample Construction Flow', fontsize=14, weight='bold', pad=20)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved flow diagram to {save_path}")

    return fig


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)

    # Propensity distribution
    prop_scores = np.concatenate([
        np.random.beta(2, 5, 500),
        np.random.beta(5, 2, 500)
    ])
    treatment = np.array([0] * 500 + [1] * 500)

    fig1 = plot_propensity_distribution(prop_scores, treatment)
    plt.close()

    # Balance
    balance_df = pd.DataFrame({
        'covariate': [f'X{i}' for i in range(1, 11)],
        'smd_unadjusted': np.random.uniform(-0.3, 0.3, 10),
        'smd_adjusted': np.random.uniform(-0.05, 0.05, 10)
    })

    fig2 = plot_balance(balance_df)
    plt.close()

    # Forest plot
    results = [
        {'point_estimate': 0.15, 'ci_lower': 0.05, 'ci_upper': 0.25},
        {'point_estimate': 0.18, 'ci_lower': 0.08, 'ci_upper': 0.28},
        {'point_estimate': 0.12, 'ci_lower': 0.02, 'ci_upper': 0.22}
    ]
    labels = ['Main specification', 'Alternative model', 'Robustness check']

    fig3 = plot_ate_forest(results, labels)
    plt.close()

    print("Plotting utilities test completed successfully")
