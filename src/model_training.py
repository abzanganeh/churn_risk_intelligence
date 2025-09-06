"""
Model training module for Churn Risk Intelligence project.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import joblib
import logging
from pathlib import Path
from typing import Dict, Any

from config import MODEL_CONFIG, MODELS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Handles training of individual models."""

    def __init__(self):
        self.models = {}
        self.model_configs = MODEL_CONFIG
        logger.info("ModelTrainer initialized")

    def train_logistic_regression(self, X_train: pd.DataFrame, y_train: pd.Series,
                                  model_name: str = "logistic") -> LogisticRegression:
        """Train Logistic Regression model."""
        logger.info("Training Logistic Regression model")

        config = self.model_configs['logistic_regression']
        model = LogisticRegression(**config)
        model.fit(X_train, y_train)

        self.models[model_name] = model
        logger.info(f"Logistic Regression model '{model_name}' trained successfully")
        return model

    def train_knn(self, X_train: pd.DataFrame, y_train: pd.Series,
                  model_name: str = "knn", n_neighbors: int = None) -> KNeighborsClassifier:
        """Train K-Nearest Neighbors model."""
        logger.info("Training KNN model")

        k = n_neighbors or self.model_configs['knn']['n_neighbors']
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)

        self.models[model_name] = model
        logger.info(f"KNN model '{model_name}' trained successfully with k={k}")
        return model

    def optimize_knn_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """Optimize KNN hyperparameters using GridSearchCV."""
        logger.info("Optimizing KNN hyperparameters")

        param_grid = self.model_configs['knn_grid_search']['param_grid']
        cv = self.model_configs['knn_grid_search']['cv']

        knn = KNeighborsClassifier()
        grid_search = GridSearchCV(knn, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        logger.info(f"Best KNN parameters: {grid_search.best_params_}")
        self.models['knn_optimized'] = grid_search.best_estimator_

        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'best_estimator': grid_search.best_estimator_
        }

    def save_model(self, model_name: str, file_path: Path = None) -> Path:
        """Save trained model to disk."""
        model = self.models[model_name]

        if file_path is None:
            file_path = MODELS_DIR / f"{model_name}_model.pkl"

        joblib.dump(model, file_path)
        logger.info(f"Model '{model_name}' saved to {file_path}")
        return file_path


def main_model_training(processed_data: Dict[str, Any]) -> Dict[str, Any]:
    """Main function to execute complete model training pipeline."""
    logger.info("Starting model training pipeline")

    trainer = ModelTrainer()

    X_train = processed_data['X_train']
    y_train = processed_data['y_train']
    X_train_smote = processed_data['X_train_smote']
    y_train_smote = processed_data['y_train_smote']

    results = {}

    # Train models on regular data
    lr_model = trainer.train_logistic_regression(X_train, y_train, "logistic_regular")
    results['logistic_regular'] = {'model': lr_model}

    # Optimize KNN and train
    knn_optimization = trainer.optimize_knn_hyperparameters(X_train, y_train)
    best_k = knn_optimization['best_params']['n_neighbors']

    knn_model = trainer.train_knn(X_train, y_train, "knn_regular", n_neighbors=best_k)
    results['knn_regular'] = {'model': knn_model}

    # Train models on SMOTE data if available
    if X_train_smote is not None:
        lr_smote_model = trainer.train_logistic_regression(X_train_smote, y_train_smote, "logistic_smote")
        results['logistic_smote'] = {'model': lr_smote_model}

        knn_smote_model = trainer.train_knn(X_train_smote, y_train_smote, "knn_smote", n_neighbors=best_k)
        results['knn_smote'] = {'model': knn_smote_model}

    # Save all models
    saved_paths = {}
    for model_name in trainer.models.keys():
        saved_paths[model_name] = trainer.save_model(model_name)

    results['_metadata'] = {
        'saved_model_paths': saved_paths
    }

    logger.info("Model training pipeline completed")
    print(f"\n✅ Trained {len(trainer.models)} models successfully!")
    print(f"Models: {list(trainer.models.keys())}")

    return results


if __name__ == "__main__":
    from data_processing import main_data_processing
    from feature_engineering import main_feature_engineering

    cleaned_df, _ = main_data_processing()
    processed_data = main_feature_engineering(cleaned_df, apply_smote=True)
    training_results = main_model_training(processed_data)
    print("Model training completed!")