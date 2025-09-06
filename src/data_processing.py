"""
Data processing module for Churn Risk Intelligence project.
Handles data loading, cleaning, and exploratory data analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Tuple, Dict, Any

from config import DATA_CONFIG, VIZ_CONFIG, CATEGORICAL_COLUMNS, DATA_DIR

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Handles data loading and initial inspection."""

    def __init__(self, data_path: Path = None):
        self.data_path = data_path or DATA_DIR / DATA_CONFIG['file_name']
        self.df = None

    def load_data(self) -> pd.DataFrame:
        try:
            logger.info(f"Loading data from {self.data_path}")
            self.df = pd.read_csv(self.data_path)
            logger.info(f"Data loaded successfully. Shape: {self.df.shape}")
            return self.df
        except FileNotFoundError:
            logger.error(f"Data file not found at {self.data_path}")
            raise

    def print_data_overview(self):
        print("=" * 50)
        print("DATASET OVERVIEW")
        print("=" * 50)
        print(f"Shape: {self.df.shape}")
        print(f"\nFirst 5 rows:")
        print(self.df.head())
        print(f"\nDataset Description:")
        print(self.df.describe())


class DataCleaner:
    """Handles data cleaning operations."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        logger.info("DataCleaner initialized")

    def remove_columns(self, columns: list) -> pd.DataFrame:
        columns_to_drop = [col for col in columns if col in self.df.columns]
        if columns_to_drop:
            self.df = self.df.drop(columns_to_drop, axis=1)
            logger.info(f"Dropped columns: {columns_to_drop}")
        return self.df

    def convert_total_charges(self) -> pd.DataFrame:
        logger.info("Converting TotalCharges to numeric")
        self.df['TotalCharges'] = pd.to_numeric(self.df['TotalCharges'], errors='coerce')

        missing_count = self.df['TotalCharges'].isnull().sum()
        if missing_count > 0:
            mean_charges = self.df['TotalCharges'].mean()
            self.df['TotalCharges'] = self.df['TotalCharges'].fillna(mean_charges)
            logger.info(f"Imputed {missing_count} missing values with mean: {mean_charges:.2f}")

        return self.df

    def encode_target_variable(self, target_col: str = 'Churn') -> pd.DataFrame:
        logger.info(f"Encoding target variable: {target_col}")
        self.df[target_col] = self.df[target_col].replace({'Yes': 1, 'No': 0})
        self.df[target_col] = self.df[target_col].astype(int)
        return self.df

    def get_clean_data(self) -> pd.DataFrame:
        """Return the cleaned dataframe."""
        return self.df

class EDAAnalyzer:
    """Handles Exploratory Data Analysis."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.target_col = DATA_CONFIG['target_column']

    def plot_target_distribution(self) -> None:
        """Plot distribution of target variable."""
        plt.figure(figsize=VIZ_CONFIG['figure_size'])
        churn_counts = self.df[self.target_col].value_counts()
        churn_counts.plot(kind='bar', color=['skyblue', 'salmon'])
        plt.title('Distribution of Customer Churn', fontsize=16, pad=20)
        plt.xlabel("Churn Status (0=No, 1=Yes)", fontsize=12)
        plt.ylabel('Number of Customers', fontsize=12)
        plt.xticks(rotation=0)
        plt.tight_layout()

        # Save plot instead of showing (to avoid hanging)
        from config import RESULTS_DIR
        plt.savefig(RESULTS_DIR / "churn_distribution.png", dpi=100, bbox_inches='tight')
        plt.close()  # Close instead of show
        print("Churn distribution plot saved to results/churn_distribution.png")

        total = len(self.df)
        churn_rate = (churn_counts[1] / total) * 100
        print(f"\nChurn Statistics:")
        print(f"Total customers: {total:,}")
        print(f"Churned: {churn_counts[1]:,} ({churn_rate:.1f}%)")
        print(f"Retained: {churn_counts[0]:,} ({100 - churn_rate:.1f}%)")

def main_data_processing(data_path: Path = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Main function to execute complete data processing pipeline."""
    logger.info("Starting data processing pipeline")

    # Load data
    loader = DataLoader(data_path)
    df = loader.load_data()
    loader.print_data_overview()

    # Clean data
    cleaner = DataCleaner(df)
    cleaner.remove_columns([DATA_CONFIG['id_column']])
    cleaner.convert_total_charges()
    cleaner.encode_target_variable()
    cleaned_df = cleaner.get_clean_data()

    # EDA
    eda = EDAAnalyzer(cleaned_df)
    eda.plot_target_distribution()

    # One-hot encode for correlation analysis
    categorical_cols = [col for col in CATEGORICAL_COLUMNS if col in cleaned_df.columns]
    encoded_df = pd.get_dummies(cleaned_df, columns=categorical_cols, drop_first=True)

    logger.info("Data processing pipeline completed")
    return cleaned_df, encoded_df


if __name__ == "__main__":
    cleaned_data, encoded_data = main_data_processing()
    print(f"Processing complete! Cleaned: {cleaned_data.shape}, Encoded: {encoded_data.shape}")