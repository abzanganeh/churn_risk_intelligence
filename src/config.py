"""
Configuration file for Churn Risk Intelligence project.
Contains all constants, paths, and hyperparameters.
"""

import os
from pathlib import Path

# Project Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Ensure directories exist
for directory in [DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    directory.mkdir(exist_ok=True)

# Data Configuration
DATA_CONFIG = {
    'file_name': 'Customer-Churn.csv',
    'target_column': 'Churn',
    'id_column': 'customerID',
    'test_size': 0.2,
    'random_state': 42
}

# Feature Configuration
CATEGORICAL_COLUMNS = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod'
]

NUMERICAL_COLUMNS = ['tenure', 'MonthlyCharges', 'TotalCharges']

# Columns to drop
DROP_COLUMNS = ['customerID']

# Problematic columns (data leakage)
LEAKY_FEATURES = ['PaymentMethod_Yes']

# Model Configuration
MODEL_CONFIG = {
    'logistic_regression': {
        'random_state': 42,
        'max_iter': 1000
    },
    'knn': {
        'n_neighbors': 15,
        'random_state': 42
    },
    'knn_grid_search': {
        'param_grid': {'n_neighbors': [3, 5, 7, 9, 11, 15]},
        'cv': 5
    },
    'smote': {
        'random_state': 42
    }
}

# Visualization Configuration
VIZ_CONFIG = {
    'figure_size': (12, 8),
    'dpi': 100,
    'style': 'whitegrid',
    'color_palette': 'Set2'
}