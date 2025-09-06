"""
Model evaluation module for Churn Risk Intelligence project.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report
import logging
from typing import Dict, Any
from config import RESULTS_DIR

from config import VIZ_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Handles comprehensive model evaluation."""

    def __init__(self):
        self.evaluation_results = {}
        self.predictions = {}
        logger.info("ModelEvaluator initialized")

    def evaluate_single_model(self, model, X_test: pd.DataFrame, y_test: pd.Series, model_name: str) -> Dict[str, Any]:
        """Evaluate a single model comprehensively."""
        logger.info(f"Evaluating model: {model_name}")

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }

        # Store results
        self.evaluation_results[model_name] = metrics
        self.predictions[model_name] = {'y_pred': y_pred, 'y_true': y_test}

        logger.info(f"Model {model_name} evaluated - Accuracy: {metrics['accuracy']:.4f}")
        return metrics

    def evaluate_multiple_models(self, models_dict: Dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series) -> Dict[
        str, Dict]:
        """Evaluate multiple models."""
        logger.info(f"Evaluating {len(models_dict)} models")

        all_results = {}

        for model_name, model_info in models_dict.items():
            if model_name.startswith('_') or model_name == 'saved_model_paths':
                continue

            if isinstance(model_info, dict) and 'model' in model_info:
                model = model_info['model']
            else:
                model = model_info

            try:
                results = self.evaluate_single_model(model, X_test, y_test, model_name)
                all_results[model_name] = results
            except Exception as e:
                logger.error(f"Error evaluating model {model_name}: {e}")
                continue

        return all_results

    def create_comparison_table(self) -> pd.DataFrame:
        """Create comparison table of all evaluated models."""
        if not self.evaluation_results:
            return pd.DataFrame()

        comparison_data = []

        for model_name, metrics in self.evaluation_results.items():
            row = {
                'Model': model_name,
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1_score']:.4f}"
            }
            comparison_data.append(row)

        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Accuracy', ascending=False)

        return comparison_df

    def plot_confusion_matrices(self) -> None:
        """Plot confusion matrices for all evaluated models."""

        if not self.evaluation_results:
            return

        n_models = len(self.evaluation_results)
        n_cols = 2
        n_rows = (n_models + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 6 * n_rows))
        if n_models == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)

        axes = axes.flatten()

        for i, (model_name, metrics) in enumerate(self.evaluation_results.items()):
            cm = metrics['confusion_matrix']

            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                        xticklabels=['No Churn', 'Churn'],
                        yticklabels=['No Churn', 'Churn'])

            axes[i].set_title(f'{model_name}\nAccuracy: {metrics["accuracy"]:.3f}')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')

        # Hide empty subplots
        for i in range(n_models, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        # Save instead of show
        save_path = RESULTS_DIR / "confusion_matrices.png"
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrices saved to {save_path}")

    def print_evaluation_summary(self) -> None:
        """Print comprehensive evaluation summary."""
        comparison_df = self.create_comparison_table()

        print("=" * 80)
        print("MODEL EVALUATION SUMMARY")
        print("=" * 80)

        if comparison_df.empty:
            print("No evaluation results available.")
            return

        print("\nPerformance Comparison:")
        print(comparison_df.to_string(index=False))

        # Best model analysis
        best_accuracy = comparison_df.iloc[0]
        best_recall = comparison_df.loc[comparison_df['Recall'].astype(float).idxmax()]
        best_precision = comparison_df.loc[comparison_df['Precision'].astype(float).idxmax()]

        print(f"\n🏆 BEST PERFORMERS:")
        print(f"  Best Accuracy:  {best_accuracy['Model']} ({best_accuracy['Accuracy']})")
        print(f"  Best Recall:    {best_recall['Model']} ({best_recall['Recall']})")
        print(f"  Best Precision: {best_precision['Model']} ({best_precision['Precision']})")


def main_evaluation(models_dict: Dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series) -> ModelEvaluator:
    """Main function to execute complete model evaluation pipeline."""
    logger.info("Starting evaluation pipeline")

    evaluator = ModelEvaluator()
    evaluator.evaluate_multiple_models(models_dict, X_test, y_test)
    evaluator.print_evaluation_summary()
    # Save results to files
    from config import RESULTS_DIR

    # Save comparison table
    comparison_df = evaluator.create_comparison_table()
    if not comparison_df.empty:
        csv_path = RESULTS_DIR / "model_comparison.csv"
        comparison_df.to_csv(csv_path, index=False)
        print(f"Model comparison saved to {csv_path}")

    print(f"\nResults saved to: {RESULTS_DIR}")
    evaluator.plot_confusion_matrices()

    logger.info("Evaluation pipeline completed")
    return evaluator


if __name__ == "__main__":
    from data_processing import main_data_processing
    from feature_engineering import main_feature_engineering
    from model_training import main_model_training

    cleaned_df, _ = main_data_processing()
    processed_data = main_feature_engineering(cleaned_df, apply_smote=True)
    training_results = main_model_training(processed_data)
    evaluator = main_evaluation(training_results, processed_data['X_test'], processed_data['y_test'])
    print("Evaluation completed!")