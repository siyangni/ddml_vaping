"""
Causal DAG specification and design diagnostics.

Creates directed acyclic graph showing assumed causal structure,
identifies confounders, and performs design quality checks.

Usage:
    python 02_dag_and_design_diagnostics.py
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
import logging
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import networkx as nx

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / 'code' / 'utils'))

from survey_utils import summarize_survey_design, compute_effective_sample_size
from plotting_utils import plot_cohort_flow

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(project_root / 'outputs' / 'logs' / 'design_diagnostics.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DAGConstructor:
    """Construct and visualize causal DAGs."""

    def __init__(self):
        """Initialize DAG constructor."""
        self.G = None
        logger.info("DAGConstructor initialized")

    def build_vaping_smoking_dag(self, cohort_type: str = 'youth') -> nx.DiGraph:
        """
        Build causal DAG for vaping → smoking relationship.

        Parameters
        ----------
        cohort_type : str
            'youth' or 'adult'

        Returns
        -------
        G : nx.DiGraph
            Directed acyclic graph
        """
        logger.info(f"\nConstructing causal DAG for {cohort_type} cohort...")

        G = nx.DiGraph()

        # Core causal relationship
        G.add_edge('Vaping', 'Smoking', color='red', width=3)

        # Confounders
        confounders = [
            'Age',
            'Sex',
            'Race/Ethnicity',
            'SES',
            'Parental_Tobacco',
            'Peer_Tobacco',
            'Sensation_Seeking',
            'Risk_Tolerance'
        ]

        if cohort_type == 'youth':
            confounders.extend(['Academic_Performance', 'School_Policy'])
        else:
            confounders.extend(['Nicotine_Dependence', 'Prior_Quit_Attempts', 'Mental_Health'])

        # Add confounder edges
        for confounder in confounders:
            G.add_edge(confounder, 'Vaping', color='blue', width=1)
            G.add_edge(confounder, 'Smoking', color='blue', width=1)

        # Additional structure
        G.add_edge('SES', 'Parental_Tobacco', color='gray', width=0.5)
        G.add_edge('Parental_Tobacco', 'Peer_Tobacco', color='gray', width=0.5)

        if cohort_type == 'youth':
            G.add_edge('SES', 'Academic_Performance', color='gray', width=0.5)

        self.G = G

        logger.info(f"DAG constructed with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

        return G

    def plot_dag(
        self,
        save_path: Optional[Path] = None,
        dpi: int = 300
    ) -> plt.Figure:
        """
        Plot the causal DAG.

        Parameters
        ----------
        save_path : Path, optional
            Path to save figure
        dpi : int, default=300
            Resolution

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        if self.G is None:
            raise ValueError("DAG not constructed. Call build_vaping_smoking_dag() first.")

        fig, ax = plt.subplots(figsize=(14, 10))

        # Layout
        pos = nx.spring_layout(self.G, k=2, iterations=50, seed=42)

        # Manually adjust key nodes for clarity
        if 'Vaping' in pos and 'Smoking' in pos:
            pos['Vaping'] = np.array([0.3, 0.5])
            pos['Smoking'] = np.array([0.7, 0.5])

        # Draw nodes
        treatment_outcome = ['Vaping', 'Smoking']
        confounders = [n for n in self.G.nodes() if n not in treatment_outcome]

        # Treatment and outcome (larger, colored)
        nx.draw_networkx_nodes(
            self.G,
            pos,
            nodelist=['Vaping'],
            node_color='lightcoral',
            node_size=3000,
            ax=ax,
            node_shape='s'
        )

        nx.draw_networkx_nodes(
            self.G,
            pos,
            nodelist=['Smoking'],
            node_color='lightskyblue',
            node_size=3000,
            ax=ax,
            node_shape='s'
        )

        # Confounders (smaller, gray)
        nx.draw_networkx_nodes(
            self.G,
            pos,
            nodelist=confounders,
            node_color='lightgray',
            node_size=2000,
            ax=ax
        )

        # Draw edges
        causal_edges = [('Vaping', 'Smoking')]
        confounder_edges = [e for e in self.G.edges() if e not in causal_edges and 'Vaping' in e or 'Smoking' in e]
        other_edges = [e for e in self.G.edges() if e not in causal_edges and e not in confounder_edges]

        # Main causal edge (red, thick)
        nx.draw_networkx_edges(
            self.G,
            pos,
            edgelist=causal_edges,
            edge_color='red',
            width=3,
            arrows=True,
            arrowsize=20,
            ax=ax,
            connectionstyle='arc3,rad=0'
        )

        # Confounder edges (blue)
        nx.draw_networkx_edges(
            self.G,
            pos,
            edgelist=confounder_edges,
            edge_color='steelblue',
            width=1.5,
            arrows=True,
            arrowsize=15,
            ax=ax,
            alpha=0.6,
            connectionstyle='arc3,rad=0.1'
        )

        # Other edges (gray, thin)
        if other_edges:
            nx.draw_networkx_edges(
                self.G,
                pos,
                edgelist=other_edges,
                edge_color='gray',
                width=0.5,
                arrows=True,
                arrowsize=10,
                ax=ax,
                alpha=0.3,
                style='dashed',
                connectionstyle='arc3,rad=0.1'
            )

        # Labels
        labels = {n: n.replace('_', '\n') for n in self.G.nodes()}
        nx.draw_networkx_labels(
            self.G,
            pos,
            labels,
            font_size=9,
            font_weight='bold',
            ax=ax
        )

        ax.set_title('Causal Directed Acyclic Graph (DAG): Vaping → Smoking', fontsize=14, weight='bold', pad=20)
        ax.axis('off')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            logger.info(f"DAG plot saved to: {save_path}")

        return fig

    def identify_adjustment_set(self) -> List[str]:
        """
        Identify minimal sufficient adjustment set.

        Returns
        -------
        adjustment_set : list
            List of confounders to adjust for
        """
        if self.G is None:
            raise ValueError("DAG not constructed.")

        # For demonstration: all parents of treatment and outcome (except each other)
        # In practice, use d-separation algorithms

        treatment_parents = set(self.G.predecessors('Vaping'))
        outcome_parents = set(self.G.predecessors('Smoking'))

        adjustment_set = list(treatment_parents.union(outcome_parents) - {'Vaping', 'Smoking'})

        logger.info(f"\nMinimal sufficient adjustment set ({len(adjustment_set)} variables):")
        for var in sorted(adjustment_set):
            logger.info(f"  - {var}")

        return adjustment_set


class DesignDiagnostics:
    """Perform study design quality checks."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize diagnostics."""
        if config_path is None:
            config_path = project_root / 'config' / 'config.yaml'

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.output_dir = project_root / 'outputs'
        logger.info("DesignDiagnostics initialized")

    def load_cohorts(self) -> Dict[str, pd.DataFrame]:
        """
        Load analysis cohorts.

        Returns
        -------
        cohorts : dict
            Dictionary with youth and adult cohorts
        """
        data_dir = project_root / 'data' / 'processed'

        cohorts = {
            'youth': pd.read_parquet(data_dir / 'youth_cohort.parquet'),
            'adult': pd.read_parquet(data_dir / 'adult_cohort.parquet')
        }

        logger.info(f"Loaded cohorts:")
        logger.info(f"  Youth: {len(cohorts['youth']):,}")
        logger.info(f"  Adult: {len(cohorts['adult']):,}")

        return cohorts

    def create_table1(self, cohort: pd.DataFrame, cohort_type: str) -> pd.DataFrame:
        """
        Create Table 1 (descriptive statistics by treatment group).

        Parameters
        ----------
        cohort : pd.DataFrame
            Analysis cohort
        cohort_type : str
            'youth' or 'adult'

        Returns
        -------
        table1 : pd.DataFrame
            Descriptive statistics table
        """
        logger.info(f"\nCreating Table 1 for {cohort_type} cohort...")

        # Key variables
        if cohort_type == 'youth':
            vars_to_describe = [
                'age', 'sex_Male', 'race_ethnicity_Black', 'race_ethnicity_Hispanic',
                'parental_tobacco_use', 'peer_tobacco_use', 'sensation_seeking_score'
            ]
        else:
            vars_to_describe = [
                'age', 'sex_Male', 'race_ethnicity_Black', 'race_ethnicity_Hispanic',
                'cpd_baseline', 'prior_quit_attempts', 'nicotine_dependence_score'
            ]

        # Filter to existing variables
        vars_to_describe = [v for v in vars_to_describe if v in cohort.columns]

        # Split by treatment
        treated = cohort[cohort['treatment'] == 1]
        control = cohort[cohort['treatment'] == 0]

        # Compute statistics with survey weights
        weight_col = 'full_weight'

        table_records = []

        for var in vars_to_describe:
            # Overall
            if cohort[var].dtype in ['int64', 'float64']:
                overall_mean = np.average(cohort[var], weights=cohort[weight_col])
                overall_std = np.sqrt(np.average((cohort[var] - overall_mean) ** 2, weights=cohort[weight_col]))

                treated_mean = np.average(treated[var], weights=treated[weight_col])
                control_mean = np.average(control[var], weights=control[weight_col])

                record = {
                    'Variable': var,
                    'Overall': f"{overall_mean:.2f} ({overall_std:.2f})",
                    'Treated': f"{treated_mean:.2f}",
                    'Control': f"{control_mean:.2f}",
                    'SMD': (treated_mean - control_mean) / overall_std if overall_std > 0 else 0
                }
            else:
                # Binary/categorical
                overall_prop = np.average(cohort[var], weights=cohort[weight_col])
                treated_prop = np.average(treated[var], weights=treated[weight_col])
                control_prop = np.average(control[var], weights=control[weight_col])

                record = {
                    'Variable': var,
                    'Overall': f"{100*overall_prop:.1f}%",
                    'Treated': f"{100*treated_prop:.1f}%",
                    'Control': f"{100*control_prop:.1f}%",
                    'SMD': (treated_prop - control_prop) / np.sqrt(overall_prop * (1 - overall_prop))
                    if overall_prop not in [0, 1] else 0
                }

            table_records.append(record)

        table1 = pd.DataFrame(table_records)

        # Save
        table_path = self.output_dir / 'tables' / f'table1_{cohort_type}.csv'
        table_path.parent.mkdir(parents=True, exist_ok=True)
        table1.to_csv(table_path, index=False)

        logger.info(f"Table 1 saved to: {table_path}")

        return table1

    def survey_design_summary(self, cohort: pd.DataFrame, cohort_type: str) -> Dict:
        """
        Summarize survey design characteristics.

        Parameters
        ----------
        cohort : pd.DataFrame
            Analysis cohort
        cohort_type : str
            'youth' or 'adult'

        Returns
        -------
        summary : dict
            Survey design summary
        """
        logger.info(f"\nSurvey design summary for {cohort_type} cohort:")

        weight_col = 'full_weight'

        summary = summarize_survey_design(
            data=cohort,
            weight_col=weight_col
        )

        logger.info(f"  Sample size: {summary['n_observations']:,}")
        logger.info(f"  Weighted N: {summary['n_weighted']:,.0f}")
        logger.info(f"  Effective sample size: {summary['effective_sample_size']:.0f}")
        logger.info(f"  Design effect: {summary['design_effect']:.2f}")

        return summary

    def create_flow_diagram(self, cohort_type: str) -> None:
        """
        Create CONSORT-style flow diagram.

        Parameters
        ----------
        cohort_type : str
            'youth' or 'adult'
        """
        logger.info(f"\nCreating flow diagram for {cohort_type} cohort...")

        if cohort_type == 'youth':
            flow_data = {
                'Initial PATH youth sample (Wave 1)': 13651,
                'Never-smokers at baseline': 11234,
                'Age 12-17': 10892,
                'Complete treatment data': 10456,
                'Complete outcome data': 9823,
                'Complete covariate data': 9234,
                'Final analysis sample': 9234
            }
        else:
            flow_data = {
                'Initial PATH adult sample (Wave 1)': 32320,
                'Current smokers at baseline': 8456,
                'Age ≥18': 8456,
                'Complete treatment data': 8123,
                'Complete outcome data': 7654,
                'Complete covariate data': 7234,
                'Final analysis sample': 7234
            }

        # Note: These are placeholder numbers; replace with actual cohort construction counts

        fig = plot_cohort_flow(flow_data)

        save_path = self.output_dir / 'figures' / f'flow_diagram_{cohort_type}.pdf'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

        logger.info(f"Flow diagram saved to: {save_path}")

        plt.close()


def main():
    """Main execution."""
    logger.info("=" * 80)
    logger.info("DML VAPING/SMOKING STUDY - DAG & DESIGN DIAGNOSTICS")
    logger.info("=" * 80)

    # Build and visualize DAGs
    dag_constructor = DAGConstructor()

    # Youth DAG
    logger.info("\n" + "=" * 80)
    logger.info("YOUTH COHORT DAG")
    logger.info("=" * 80)

    youth_dag = dag_constructor.build_vaping_smoking_dag('youth')
    youth_dag_path = project_root / 'outputs' / 'figures' / 'dag_youth.pdf'
    youth_dag_path.parent.mkdir(parents=True, exist_ok=True)

    fig_youth = dag_constructor.plot_dag(save_path=youth_dag_path)
    plt.close()

    youth_adj_set = dag_constructor.identify_adjustment_set()

    # Adult DAG
    logger.info("\n" + "=" * 80)
    logger.info("ADULT COHORT DAG")
    logger.info("=" * 80)

    adult_dag = dag_constructor.build_vaping_smoking_dag('adult')
    adult_dag_path = project_root / 'outputs' / 'figures' / 'dag_adult.pdf'

    fig_adult = dag_constructor.plot_dag(save_path=adult_dag_path)
    plt.close()

    adult_adj_set = dag_constructor.identify_adjustment_set()

    # Design diagnostics
    diagnostics = DesignDiagnostics()
    cohorts = diagnostics.load_cohorts()

    # Table 1
    logger.info("\n" + "=" * 80)
    logger.info("DESCRIPTIVE STATISTICS (TABLE 1)")
    logger.info("=" * 80)

    table1_youth = diagnostics.create_table1(cohorts['youth'], 'youth')
    table1_adult = diagnostics.create_table1(cohorts['adult'], 'adult')

    # Survey design summaries
    logger.info("\n" + "=" * 80)
    logger.info("SURVEY DESIGN SUMMARIES")
    logger.info("=" * 80)

    youth_design = diagnostics.survey_design_summary(cohorts['youth'], 'youth')
    adult_design = diagnostics.survey_design_summary(cohorts['adult'], 'adult')

    # Flow diagrams
    logger.info("\n" + "=" * 80)
    logger.info("FLOW DIAGRAMS")
    logger.info("=" * 80)

    diagnostics.create_flow_diagram('youth')
    diagnostics.create_flow_diagram('adult')

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("DESIGN DIAGNOSTICS COMPLETE")
    logger.info("=" * 80)
    logger.info("\nOutputs:")
    logger.info("  - DAGs: outputs/figures/dag_*.pdf")
    logger.info("  - Table 1: outputs/tables/table1_*.csv")
    logger.info("  - Flow diagrams: outputs/figures/flow_diagram_*.pdf")
    logger.info("\nNext step: Run 03_dml_estimation_point_and_replicate_variance.py")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
