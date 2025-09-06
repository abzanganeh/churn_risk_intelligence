"""
Feature engineering module for Churn Risk Intelligence project.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import logging
from typing import Tuple, List

from config import DATA_CONFIG, CATEGORICAL_COLUMNS, NUMERICAL_COLUMNS, LEAKY_FEATURES, MODEL_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Handles feature engineering operations."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.target_col = DATA_CONFIG['target_column']
        self.scaler = StandardScaler()
        logger.info("FeatureEngineer initialized")

    def encode_categorical_features(self, drop_first: bool = True) -> pd.DataFrame:
        """One-hot encode categorical features."""
        logger.info("One-hot encoding categorical features")
        categorical_cols = [col for col in CATEGORICAL_COLUMNS if col in self.df.columns]
        encoded_df = pd.get_dummies(self.df, columns=categorical_cols, drop_first=drop_first)
        logger.info(f"Encoded {len(categorical_cols)} categorical columns")
        return encoded_df

    def detect_data_leakage(self, df_encoded: pd.DataFrame) -> List[str]:
        """Detect potential data leakage features."""
        logger.info("Detecting potential data leakage")
        leaky_features = []

        if self.target_col not in df_encoded.columns:
            return leaky_features

        y = df_encoded[self.target_col]

        for col in df_encoded.columns:
            if col != self.target_col:
                correlation = df_encoded[col].corr(y)
                if abs(correlation) > 0.95:
                    leaky_features.append(col)
                    logger.warning(f"High correlation detected: {col} = {correlation:.3f}")

        # Add predefined leaky features
        for feature in LEAKY_FEATURES:
            if feature in df_encoded.columns and feature not in leaky_features:
                leaky_features.append(feature)

        return leaky_features

    def remove_leaky_features(self, df_encoded: pd.DataFrame) -> pd.DataFrame:
        """Remove features that cause data leakage."""
        leaky_features = self.detect_data_leakage(df_encoded)

        if leaky_features:
            logger.info(f"Removing leaky features: {leaky_features}")
            df_cleaned = df_encoded.drop(columns=leaky_features)
            return df_cleaned

        return df_encoded

    def prepare_features_and_target(self, df_encoded: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Separate features and target variable."""
        logger.info("Preparing features and target")
        df_clean = self.remove_leaky_features(df_encoded)

        X = df_clean.drop(columns=[self.target_col])
        y = df_clean[self.target_col]

        logger.info(f"Features shape: {X.shape}")
        logger.info(f"Target distribution:\n{y.value_counts()}")

        return X, y

    def split_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into train and test sets."""
        test_size = DATA_CONFIG['test_size']
        random_state = DATA_CONFIG['random_state']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        logger.info(f"Train set - X: {X_train.shape}, y: {y_train.shape}")
        logger.info(f"Test set - X: {X_test.shape}, y: {y_test.shape}")

        return X_train, X_test, y_train, y_test

    def scale_numerical_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[
        pd.DataFrame, pd.DataFrame]:
        """Scale numerical features using StandardScaler."""
        logger.info("Scaling numerical features")

        num_cols = [col for col in NUMERICAL_COLUMNS if col in X_train.columns]

        if not num_cols:
            logger.warning("No numerical columns found for scaling")
            return X_train, X_test

        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()

        X_train_scaled[num_cols] = self.scaler.fit_transform(X_train[num_cols])
        X_test_scaled[num_cols] = self.scaler.transform(X_test[num_cols])

        return X_train_scaled, X_test_scaled

    def apply_smote(self, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Apply SMOTE to handle class imbalance."""
        logger.info("Applying SMOTE for class imbalance")

        smote = SMOTE(random_state=MODEL_CONFIG['smote']['random_state'])
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

        logger.info(f"Original training set distribution:\n{y_train.value_counts()}")
        logger.info(f"SMOTE training set distribution:\n{pd.Series(y_train_smote).value_counts()}")

        return X_train_smote, y_train_smote


def main_feature_engineering(df: pd.DataFrame, apply_smote: bool = True) -> dict:
    """Main function to execute complete feature engineering pipeline."""
    logger.info("Starting feature engineering pipeline")

    engineer = FeatureEngineer(df)

    # 1. Encode categorical features
    df_encoded = engineer.encode_categorical_features()

    # 2. Prepare features and target
    X, y = engineer.prepare_features_and_target(df_encoded)

    # 3. Split data
    X_train, X_test, y_train, y_test = engineer.split_data(X, y)

    # 4. Scale numerical features
    X_train_scaled, X_test_scaled = engineer.scale_numerical_features(X_train, X_test)

    # 5. Apply SMOTE if requested
    X_train_smote, y_train_smote = None, None
    if apply_smote:
        X_train_smote, y_train_smote = engineer.apply_smote(X_train_scaled, y_train)

    results = {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'X_train_smote': X_train_smote,
        'y_train_smote': y_train_smote,
        'scaler': engineer.scaler,
        'encoded_df': df_encoded
    }

    logger.info("Feature engineering pipeline completed")
    return results


if __name__ == "__main__":
    from data_processing import main_data_processing

    cleaned_df, _ = main_data_processing()
    processed_data = main_feature_engineering(cleaned_df, apply_smote=True)
    print("Feature engineering completed!")