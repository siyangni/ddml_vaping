"""
PATH Study data loader for DML analysis.

Loads and preprocesses PATH Public-Use Files (PUF) Waves 1-2 for
youth smoking initiation and adult smoking cessation cohorts.

Author: Generated for DML vaping/smoking causal study
Date: 2024
"""

import pandas as pd
import numpy as np
import pyreadr
from pathlib import Path
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class PATHDataLoader:
    """Load and preprocess PATH Study data."""

    def __init__(self, data_dir: Path):
        """
        Initialize PATH data loader.

        Parameters
        ----------
        data_dir : Path
            Path to raw PATH data directory (e.g., data/raw/path)
        """
        self.data_dir = Path(data_dir)
        logger.info(f"PATH data directory: {self.data_dir}")

    def load_wave(self, wave: int, cohort: str) -> pd.DataFrame:
        """
        Load a single PATH wave.

        Parameters
        ----------
        wave : int
            Wave number (1-7)
        cohort : str
            'youth' or 'adult'

        Returns
        -------
        df : pd.DataFrame
            Wave data
        """
        # Determine dataset number
        if cohort == 'youth':
            ds_num = f"DS{wave}002"
        elif cohort == 'adult':
            ds_num = f"DS{wave}001"
        else:
            raise ValueError(f"Unknown cohort: {cohort}")

        # Load R data file
        data_file = self.data_dir / ds_num / f"36498-{wave}002-Data.rda" if cohort == 'youth' else \
                   self.data_dir / ds_num / f"36498-{wave}001-Data.rda"

        if not data_file.exists():
            raise FileNotFoundError(f"PATH data file not found: {data_file}")

        logger.info(f"Loading {cohort} Wave {wave} from {data_file.name}")

        try:
            result = pyreadr.read_r(str(data_file), use_objects=False)
            df_name = list(result.keys())[0]
            df = result[df_name]
            logger.info(f"  Loaded {len(df):,} observations, {len(df.columns)} variables")
            return df
        except Exception as e:
            logger.error(f"Error loading {data_file}: {e}")
            raise

    def convert_labeled_to_numeric(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Convert PATH labeled categorical variables to numeric (1=Yes, 0=No).

        PATH uses format: "(1) 1 = Yes", "(2) 2 = No"
        We convert to: 1=Yes, 0=No for easier analysis

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with labeled columns
        columns : list
            List of column names to convert

        Returns
        -------
        df : pd.DataFrame
            DataFrame with converted columns
        """
        for col in columns:
            if col in df.columns:
                # Convert "(1) 1 = Yes" -> 1, "(2) 2 = No" -> 0
                df[col] = df[col].astype(str).apply(lambda x:
                    1 if '1 = Yes' in x else (0 if '2 = No' in x else np.nan)
                )
        return df

    def load_youth_waves_1_2(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load youth Waves 1 and 2 with key variables.

        Returns
        -------
        df_w1 : pd.DataFrame
            Wave 1 youth data
        df_w2 : pd.DataFrame
            Wave 2 youth data
        """
        # Load waves
        df_w1 = self.load_wave(wave=1, cohort='youth')
        df_w2 = self.load_wave(wave=2, cohort='youth')

        # Key variables to convert to 0/1
        binary_vars_w1 = [
            'R01R_Y_EVR_ECIG',     # Ever e-cigarette
            'R01R_Y_CUR_ECIG',     # Current e-cigarette (treatment)
            'R01R_Y_EVR_CIGS',     # Ever cigarette (baseline exclusion)
            'R01R_Y_CUR_CIGS',     # Current cigarette
        ]

        binary_vars_w2 = [
            'R02R_Y_EVR_CIGS',     # Ever cigarette at W2 (outcome)
            'R02R_Y_CUR_CIGS',     # Current cigarette at W2
            'R02R_Y_EVR_ECIG',     # Ever e-cigarette at W2
            'R02R_Y_CUR_ECIG',     # Current e-cigarette at W2
        ]

        # Convert to 0/1
        df_w1 = self.convert_labeled_to_numeric(df_w1, binary_vars_w1)
        df_w2 = self.convert_labeled_to_numeric(df_w2, binary_vars_w2)

        # Rename W2 variables for merging
        rename_map = {
            'R02R_Y_EVR_CIGS': 'smoking_initiation_w2',
            'R02R_Y_CUR_CIGS': 'current_smoking_w2',
            'R02R_Y_EVR_ECIG': 'ever_ecig_w2',
            'R02R_Y_CUR_ECIG': 'current_ecig_w2',
        }
        df_w2 = df_w2.rename(columns=rename_map)

        # Keep only needed W2 columns for merge
        w2_keep_cols = ['PERSONID'] + list(rename_map.values())
        df_w2 = df_w2[[c for c in w2_keep_cols if c in df_w2.columns]]

        logger.info(f"✓ Youth W1: {len(df_w1):,}, W2: {len(df_w2):,}")

        return df_w1, df_w2

    def load_adult_waves_1_2(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load adult Waves 1 and 2 with key variables.

        Returns
        -------
        df_w1 : pd.DataFrame
            Wave 1 adult data
        df_w2 : pd.DataFrame
            Wave 2 adult data
        """
        # Load waves
        df_w1 = self.load_wave(wave=1, cohort='adult')
        df_w2 = self.load_wave(wave=2, cohort='adult')

        # Key variables to convert
        binary_vars_w1 = [
            'R01R_A_EVR_ECIG',     # Ever e-cigarette
            'R01R_A_CUR_ECIG',     # Current e-cigarette (treatment)
            'R01R_A_EVR_CIGS',     # Ever cigarette
            'R01R_A_CUR_CIGS',     # Current cigarette (baseline inclusion)
        ]

        binary_vars_w2 = [
            'R02R_A_CUR_CIGS',     # Current cigarette at W2 (for abstinence)
            'R02R_A_EVR_ECIG',
            'R02R_A_CUR_ECIG',
        ]

        # Convert
        df_w1 = self.convert_labeled_to_numeric(df_w1, binary_vars_w1)
        df_w2 = self.convert_labeled_to_numeric(df_w2, binary_vars_w2)

        # Rename W2 variables
        rename_map = {
            'R02R_A_CUR_CIGS': 'current_smoking_w2',
            'R02R_A_EVR_ECIG': 'ever_ecig_w2',
            'R02R_A_CUR_ECIG': 'current_ecig_w2',
        }
        df_w2 = df_w2.rename(columns=rename_map)

        # Keep only needed W2 columns
        w2_keep_cols = ['PERSONID'] + list(rename_map.values())
        df_w2 = df_w2[[c for c in w2_keep_cols if c in df_w2.columns]]

        logger.info(f"✓ Adult W1: {len(df_w1):,}, W2: {len(df_w2):,}")

        return df_w1, df_w2

    def build_youth_initiation_cohort(self) -> pd.DataFrame:
        """
        Build youth smoking initiation cohort.

        Baseline (W1): Never-smoker youth
        Follow-up (W2): Smoking initiation status
        Treatment: Current e-cigarette use at W1

        Returns
        -------
        cohort : pd.DataFrame
            Analysis-ready youth cohort
        """
        logger.info("\n" + "="*70)
        logger.info("BUILDING YOUTH SMOKING INITIATION COHORT")
        logger.info("="*70)

        # Load data
        df_w1, df_w2 = self.load_youth_waves_1_2()

        # Merge waves
        df = df_w1.merge(df_w2, on='PERSONID', how='inner')
        logger.info(f"\n1. Merged W1 and W2: {len(df):,} youth tracked")

        # Apply inclusion criteria
        n_start = len(df)

        # Never-smokers at baseline
        df = df[df['R01R_Y_EVR_CIGS'] == 0]  # 0 = No (after conversion)
        logger.info(f"2. Never-smokers at baseline: {len(df):,} ({100*len(df)/n_start:.1f}%)")

        # Age 12-17
        # (Already filtered by PATH - youth cohort is 12-17)

        # Non-missing treatment
        df = df[df['R01R_Y_CUR_ECIG'].notna()]
        logger.info(f"3. Non-missing treatment: {len(df):,}")

        # Non-missing outcome
        df = df[df['smoking_initiation_w2'].notna()]
        logger.info(f"4. Non-missing outcome: {len(df):,}")

        # Define treatment and outcome
        df['treatment'] = df['R01R_Y_CUR_ECIG']  # Already 0/1
        df['outcome'] = df['smoking_initiation_w2']  # Already 0/1

        # Extract demographic variables
        # These need to be converted too
        df = self._extract_demographics(df, cohort='youth')

        # Extract survey weights
        df = self._extract_weights(df, cohort='youth', wave=1)

        logger.info(f"\n✓ Final youth cohort: {len(df):,}")
        logger.info(f"  Treatment prevalence: {df['treatment'].mean():.2%}")
        logger.info(f"  Outcome prevalence: {df['outcome'].mean():.2%}")

        return df

    def build_adult_cessation_cohort(self) -> pd.DataFrame:
        """
        Build adult smoking cessation cohort.

        Baseline (W1): Current smokers
        Follow-up (W2): 30-day abstinence
        Treatment: Current e-cigarette use at W1

        Returns
        -------
        cohort : pd.DataFrame
            Analysis-ready adult cohort
        """
        logger.info("\n" + "="*70)
        logger.info("BUILDING ADULT SMOKING CESSATION COHORT")
        logger.info("="*70)

        # Load data
        df_w1, df_w2 = self.load_adult_waves_1_2()

        # Merge waves
        df = df_w1.merge(df_w2, on='PERSONID', how='inner')
        logger.info(f"\n1. Merged W1 and W2: {len(df):,} adults tracked")

        # Apply inclusion criteria
        n_start = len(df)

        # Current smokers at baseline
        df = df[df['R01R_A_CUR_CIGS'] == 1]  # 1 = Yes
        logger.info(f"2. Current smokers at baseline: {len(df):,} ({100*len(df)/n_start:.1f}%)")

        # Non-missing treatment
        df = df[df['R01R_A_CUR_ECIG'].notna()]
        logger.info(f"3. Non-missing treatment: {len(df):,}")

        # Non-missing outcome
        df = df[df['current_smoking_w2'].notna()]
        logger.info(f"4. Non-missing outcome: {len(df):,}")

        # Define treatment and outcome
        df['treatment'] = df['R01R_A_CUR_ECIG']  # 1 = vaping at baseline
        df['outcome'] = 1 - df['current_smoking_w2']  # Abstinence = NOT smoking at W2

        # Extract demographics
        df = self._extract_demographics(df, cohort='adult')

        # Extract weights
        df = self._extract_weights(df, cohort='adult', wave=1)

        logger.info(f"\n✓ Final adult cohort: {len(df):,}")
        logger.info(f"  Treatment prevalence: {df['treatment'].mean():.2%}")
        logger.info(f"  Outcome prevalence (abstinence): {df['outcome'].mean():.2%}")

        return df

    def _extract_demographics(self, df: pd.DataFrame, cohort: str) -> pd.DataFrame:
        """Extract and process demographic variables."""
        prefix = 'R01R_Y_' if cohort == 'youth' else 'R01R_A_'

        # Age - extract numeric from categorical
        age_var = f'{prefix}AGECAT2' if cohort == 'youth' else f'{prefix}AGE'
        if age_var in df.columns:
            if cohort == 'youth':
                # Youth age categories: "(1) 1 = 12 to 14", "(2) 2 = 15 to 17"
                df['age'] = df[age_var].astype(str).apply(lambda x:
                    13.5 if '12 to 14' in x else (16 if '15 to 17' in x else np.nan)
                )
            else:
                # Adult age is numeric
                df['age'] = pd.to_numeric(df[age_var], errors='coerce')

        # Sex
        sex_var = f'{prefix}SEX'
        if sex_var in df.columns:
            df['sex_Male'] = df[sex_var].astype(str).apply(lambda x: 1 if 'Male' in x else 0)

        # Race/ethnicity
        race_var = f'{prefix}RACECAT3'
        if race_var in df.columns:
            df['race_ethnicity_White'] = df[race_var].astype(str).str.contains('White', na=False).astype(int)
            df['race_ethnicity_Black'] = df[race_var].astype(str).str.contains('Black', na=False).astype(int)
            df['race_ethnicity_Hispanic'] = df[race_var].astype(str).str.contains('Hispanic', na=False).astype(int)

        return df

    def _extract_weights(self, df: pd.DataFrame, cohort: str, wave: int) -> pd.DataFrame:
        """Extract survey weights and BRR replicates."""
        prefix = f'R0{wave}_Y_' if cohort == 'youth' else f'R0{wave}_A_'

        # Main weight
        main_weight = f'{prefix}PWGT'
        if main_weight in df.columns:
            df['full_weight'] = df[main_weight]

        # BRR replicates
        for i in range(1, 101):
            rep_weight = f'{prefix}PWGT{i}'
            if rep_weight in df.columns:
                df[f'rep_weight_{i}'] = df[rep_weight]

        return df


if __name__ == "__main__":
    # Test the loader
    logging.basicConfig(level=logging.INFO)

    data_dir = Path("data/raw/path")
    loader = PATHDataLoader(data_dir)

    # Test youth cohort
    youth_cohort = loader.build_youth_initiation_cohort()
    print(f"\nYouth cohort shape: {youth_cohort.shape}")
    print(f"Youth cohort columns: {youth_cohort.columns.tolist()[:20]}")

    # Test adult cohort
    adult_cohort = loader.build_adult_cessation_cohort()
    print(f"\nAdult cohort shape: {adult_cohort.shape}")
    print(f"Adult cohort columns: {adult_cohort.columns.tolist()[:20]}")
