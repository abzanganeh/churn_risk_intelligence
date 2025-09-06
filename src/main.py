"""
Main pipeline orchestrator for Churn Risk Intelligence project.
"""

import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

from config import DATA_CONFIG, PROJECT_ROOT, DATA_DIR
from data_processing import main_data_processing
from feature_engineering import main_feature_engineering
from model_training import main_model_training
from evaluation import main_evaluation

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChurnPredictionPipeline:
    """Main pipeline class for the Churn Risk Intelligence project."""

    def __init__(self, data_path: Optional[Path] = None):
        self.data_path = data_path or DATA_DIR / DATA_CONFIG['file_name']
        self.results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info("ChurnPredictionPipeline initialized")

    def run_complete_pipeline(self, apply_smote: bool = True) -> Dict[str, Any]:
        """Execute the complete machine learning pipeline."""
        logger.info("🚀 STARTING COMPLETE CHURN PREDICTION PIPELINE")

        try:
            # Step 1: Data Processing
            logger.info("STEP 1: DATA PROCESSING")
            cleaned_df, encoded_df = main_data_processing(self.data_path)

            # Step 2: Feature Engineering
            logger.info("STEP 2: FEATURE ENGINEERING")
            processed_data = main_feature_engineering(cleaned_df, apply_smote=apply_smote)

            # Step 3: Model Training
            logger.info("STEP 3: MODEL TRAINING")
            training_results = main_model_training(processed_data)

            # Step 4: Model Evaluation
            logger.info("STEP 4: MODEL EVALUATION")
            evaluator = main_evaluation(
                training_results,
                processed_data['X_test'],
                processed_data['y_test']
            )

            self.print_pipeline_summary()
            logger.info("✅ COMPLETE PIPELINE EXECUTED SUCCESSFULLY")

            return {
                'cleaned_df': cleaned_df,
                'processed_data': processed_data,
                'training_results': training_results,
                'evaluator': evaluator
            }

        except Exception as e:
            logger.error(f"❌ PIPELINE FAILED: {e}")
            raise

    def print_pipeline_summary(self):
        """Print pipeline execution summary."""
        print("\n" + "=" * 80)
        print("🎯 CHURN RISK INTELLIGENCE - PIPELINE COMPLETED")
        print("=" * 80)
        print(f"📅 Execution Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📁 Data Source: {self.data_path}")
        print("✅ All steps completed successfully!")


def main():
    """Main function to run the pipeline."""
    pipeline = ChurnPredictionPipeline()
    results = pipeline.run_complete_pipeline()
    return results


if __name__ == "__main__":
    results = main()