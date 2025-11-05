"""
Generate publication-ready tables and figures.

Creates all tables and figures for manuscript and supplementary materials.

Usage:
    python 05_reporting_tables_and_figures.py
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
from typing import Dict, List, Optional

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / 'code' / 'utils'))

from plotting_utils import plot_ate_forest, plot_heterogeneity_by_group

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(project_root / 'outputs' / 'logs' / 'reporting.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate publication tables and figures."""

    def __init__(self):
        """Initialize report generator."""
        self.output_dir = project_root / 'outputs'
        self.results_dir = self.output_dir / 'results'
        self.tables_dir = self.output_dir / 'tables'
        self.figures_dir = self.output_dir / 'figures'

        logger.info("ReportGenerator initialized")

    def load_all_results(self) -> Dict:
        """
        Load all analysis results.

        Returns
        -------
        all_results : dict
            Dictionary with all results
        """
        logger.info("Loading all analysis results...")

        results = {}

        # Main DML results
        for cohort in ['youth', 'adult']:
            with open(self.results_dir / f'dml_results_{cohort}.pkl', 'rb') as f:
                results[f'main_{cohort}'] = pickle.load(f)

        # Heterogeneity results
        for cohort in ['youth', 'adult']:
            for het_type in ['age', 'sex']:
                path = self.results_dir / f'heterogeneity_{het_type}_{cohort}.csv'
                if path.exists():
                    results[f'het_{het_type}_{cohort}'] = pd.read_csv(path)

        # Sensitivity results
        for cohort in ['youth', 'adult']:
            path = self.results_dir / f'sensitivity_unobserved_{cohort}.csv'
            if path.exists():
                results[f'sens_{cohort}'] = pd.read_csv(path)

        logger.info(f"Loaded {len(results)} result sets")

        return results

    def create_main_results_table(self, results: Dict) -> pd.DataFrame:
        """
        Create main results table (Table 2).

        Parameters
        ----------
        results : dict
            All results

        Returns
        -------
        table : pd.DataFrame
        """
        logger.info("\nCreating Table 2: Main Results")

        rows = []

        for cohort_name, cohort_key in [
            ('Youth: Smoking Initiation', 'youth'),
            ('Adult: Smoking Cessation', 'adult')
        ]:
            main_res = results[f'main_{cohort_key}']

            for estimand in ['ATE', 'ATT']:
                res = main_res[estimand]

                row = {
                    'Outcome': cohort_name,
                    'Estimand': estimand,
                    'Estimate': f"{res['point_estimate']:.4f}",
                    'SE': f"{res['std_error']:.4f}",
                    '95% CI': f"[{res['ci_lower']:.4f}, {res['ci_upper']:.4f}]",
                    'p-value': self._compute_p_value(res['point_estimate'], res['std_error']),
                    'N': f"{res['n_replicates']} replicates"
                }

                rows.append(row)

        table = pd.DataFrame(rows)

        # Save
        table.to_csv(self.tables_dir / 'table2_main_results.csv', index=False)
        table.to_latex(self.tables_dir / 'table2_main_results.tex', index=False)

        logger.info("Table 2 saved")

        return table

    def create_heterogeneity_table(self, results: Dict) -> pd.DataFrame:
        """
        Create heterogeneity table (Table 3).

        Parameters
        ----------
        results : dict
            All results

        Returns
        -------
        table : pd.DataFrame
        """
        logger.info("\nCreating Table 3: Heterogeneity Analysis")

        rows = []

        for cohort_name, cohort_key in [
            ('Youth', 'youth'),
            ('Adult', 'adult')
        ]:
            for het_type in ['age', 'sex']:
                key = f'het_{het_type}_{cohort_key}'

                if key not in results:
                    continue

                het_df = results[key]

                for _, row_data in het_df.iterrows():
                    row = {
                        'Cohort': cohort_name,
                        'Dimension': het_type.capitalize(),
                        'Subgroup': row_data['subgroup'],
                        'N': int(row_data['n']),
                        'ATE': f"{row_data['estimate']:.4f}",
                        'SE': f"{row_data['std_error']:.4f}",
                        '95% CI': f"[{row_data['ci_lower']:.4f}, {row_data['ci_upper']:.4f}]"
                    }

                    rows.append(row)

        table = pd.DataFrame(rows)

        # Save
        table.to_csv(self.tables_dir / 'table3_heterogeneity.csv', index=False)
        table.to_latex(self.tables_dir / 'table3_heterogeneity.tex', index=False)

        logger.info("Table 3 saved")

        return table

    def create_sensitivity_table(self, results: Dict) -> pd.DataFrame:
        """
        Create sensitivity analysis table (Table 4).

        Parameters
        ----------
        results : dict
            All results

        Returns
        -------
        table : pd.DataFrame
        """
        logger.info("\nCreating Table 4: Sensitivity Analysis")

        rows = []

        for cohort_name, cohort_key in [
            ('Youth', 'youth'),
            ('Adult', 'adult')
        ]:
            sens_df = results.get(f'sens_{cohort_key}')

            if sens_df is None:
                continue

            # Select key scenarios
            scenarios = [
                (0.01, 0.01),
                (0.05, 0.05),
                (0.10, 0.10),
                (0.15, 0.15)
            ]

            for r2_t, r2_y in scenarios:
                subset = sens_df[
                    (sens_df['r2_treatment'] == r2_t) &
                    (sens_df['r2_outcome'] == r2_y)
                ]

                if len(subset) > 0:
                    row = {
                        'Cohort': cohort_name,
                        'R² Treatment': f"{r2_t:.2f}",
                        'R² Outcome': f"{r2_y:.2f}",
                        'Adjusted ATE': f"{subset.iloc[0]['adjusted_ate']:.4f}",
                        'Bias': f"{subset.iloc[0]['bias']:.4f}"
                    }

                    rows.append(row)

        table = pd.DataFrame(rows)

        # Save
        table.to_csv(self.tables_dir / 'table4_sensitivity.csv', index=False)
        table.to_latex(self.tables_dir / 'table4_sensitivity.tex', index=False)

        logger.info("Table 4 saved")

        return table

    def create_forest_plot(self, results: Dict) -> None:
        """
        Create forest plot of main results (Figure 1).

        Parameters
        ----------
        results : dict
            All results
        """
        logger.info("\nCreating Figure 1: Forest Plot")

        import matplotlib.pyplot as plt

        result_list = []
        labels = []

        for cohort_name, cohort_key, estimand in [
            ('Youth ATE', 'youth', 'ATE'),
            ('Youth ATT', 'youth', 'ATT'),
            ('Adult ATE', 'adult', 'ATE'),
            ('Adult ATT', 'adult', 'ATT')
        ]:
            res = results[f'main_{cohort_key}'][estimand]

            result_list.append({
                'point_estimate': res['point_estimate'],
                'ci_lower': res['ci_lower'],
                'ci_upper': res['ci_upper']
            })

            labels.append(cohort_name)

        fig = plot_ate_forest(
            results_list=result_list,
            labels=labels,
            save_path=self.figures_dir / 'figure1_forest_plot.pdf'
        )

        plt.close(fig)

        logger.info("Figure 1 saved")

    def create_heterogeneity_plots(self, results: Dict) -> None:
        """
        Create heterogeneity plots (Figures 2-3).

        Parameters
        ----------
        results : dict
            All results
        """
        logger.info("\nCreating Figures 2-3: Heterogeneity Plots")

        import matplotlib.pyplot as plt

        for cohort_name, cohort_key in [
            ('youth', 'youth'),
            ('adult', 'adult')
        ]:
            for het_type in ['age', 'sex']:
                key = f'het_{het_type}_{cohort_key}'

                if key not in results:
                    continue

                het_df = results[key]

                fig = plot_heterogeneity_by_group(
                    subgroup_results=het_df,
                    group_var='subgroup',
                    save_path=self.figures_dir / f'figure_het_{het_type}_{cohort_key}.pdf'
                )

                plt.close(fig)

        logger.info("Heterogeneity plots saved")

    def _compute_p_value(self, estimate: float, se: float) -> str:
        """
        Compute p-value from estimate and SE.

        Parameters
        ----------
        estimate : float
            Point estimate
        se : float
            Standard error

        Returns
        -------
        p_value_str : str
            Formatted p-value
        """
        from scipy import stats

        z = estimate / se
        p = 2 * (1 - stats.norm.cdf(abs(z)))

        if p < 0.001:
            return "<0.001"
        elif p < 0.01:
            return f"{p:.3f}"
        else:
            return f"{p:.2f}"

    def create_summary_report(self, results: Dict) -> None:
        """
        Create summary report text file.

        Parameters
        ----------
        results : dict
            All results
        """
        logger.info("\nCreating summary report")

        report_lines = [
            "=" * 80,
            "DML VAPING/SMOKING CAUSAL STUDY - RESULTS SUMMARY",
            "=" * 80,
            "",
            "MAIN FINDINGS",
            "-" * 80,
            ""
        ]

        # Youth results
        youth_ate = results['main_youth']['ATE']
        report_lines.extend([
            "Youth Smoking Initiation:",
            f"  ATE: {youth_ate['point_estimate']:.4f}",
            f"  95% CI: [{youth_ate['ci_lower']:.4f}, {youth_ate['ci_upper']:.4f}]",
            f"  SE: {youth_ate['std_error']:.4f}",
            ""
        ])

        # Adult results
        adult_ate = results['main_adult']['ATE']
        report_lines.extend([
            "Adult Smoking Cessation:",
            f"  ATE: {adult_ate['point_estimate']:.4f}",
            f"  95% CI: [{adult_ate['ci_lower']:.4f}, {adult_ate['ci_upper']:.4f}]",
            f"  SE: {adult_ate['std_error']:.4f}",
            "",
            "=" * 80
        ])

        # Write to file
        report_path = self.output_dir / 'summary_report.txt'

        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))

        logger.info(f"Summary report saved to: {report_path}")


def main():
    """Main execution."""
    logger.info("=" * 80)
    logger.info("DML VAPING/SMOKING STUDY - REPORTING")
    logger.info("=" * 80)

    generator = ReportGenerator()

    # Load all results
    results = generator.load_all_results()

    # Create tables
    logger.info("\n" + "=" * 80)
    logger.info("GENERATING TABLES")
    logger.info("=" * 80)

    table2 = generator.create_main_results_table(results)
    table3 = generator.create_heterogeneity_table(results)
    table4 = generator.create_sensitivity_table(results)

    # Create figures
    logger.info("\n" + "=" * 80)
    logger.info("GENERATING FIGURES")
    logger.info("=" * 80)

    generator.create_forest_plot(results)
    generator.create_heterogeneity_plots(results)

    # Create summary report
    generator.create_summary_report(results)

    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("REPORTING COMPLETE")
    logger.info("=" * 80)
    logger.info("\nOutputs:")
    logger.info("  - Tables: outputs/tables/")
    logger.info("  - Figures: outputs/figures/")
    logger.info("  - Summary: outputs/summary_report.txt")
    logger.info("\nNext step: Generate manuscript (Jupyter notebook)")
    logger.info("=" * 80)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    main()
