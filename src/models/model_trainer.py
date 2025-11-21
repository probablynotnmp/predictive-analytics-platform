"""
Unified model training pipeline.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json

sys.path.append(str(Path(__file__).parent.parent))

from data.data_generator import CustomerDataGenerator
from data.feature_engineering import FeatureEngineer
from data.data_preprocessor import DataPreprocessor
from models.clv_predictor import CLVPredictor
from models.churn_predictor import ChurnPredictor
from models.segmentation_model import CustomerSegmentation
from utils.config import CLV_TARGET, CHURN_TARGET, MODELS_DIR, DATA_DIR
from utils.logger import logger
from utils.metrics import business_impact_metrics
import joblib


class ModelTrainer:
    """Unified training pipeline for all models."""
    
    def __init__(self, n_customers: int = 10000):
        """
        Initialize model trainer.
        
        Args:
            n_customers: Number of customers to generate for training
        """
        self.n_customers = n_customers
        self.data_generator = CustomerDataGenerator(n_customers=n_customers)
        self.feature_engineer = FeatureEngineer()
        self.preprocessor = DataPreprocessor()
        
        self.df = None
        self.df_processed = None
        self.feature_names = []
        
        self.clv_model = None
        self.churn_model = None
        self.segmentation_model = None
        
        self.training_report = {}
        
    def generate_data(self):
        """Generate and save customer data."""
        logger.info("="*60)
        logger.info("STEP 1: GENERATING CUSTOMER DATA")
        logger.info("="*60)
        
        self.df = self.data_generator.generate_and_save()
        
        logger.info(f"Generated {len(self.df)} customer records")
        
    def engineer_features(self):
        """Engineer features from raw data."""
        logger.info("\n" + "="*60)
        logger.info("STEP 2: FEATURE ENGINEERING")
        logger.info("="*60)
        
        self.df_processed, self.feature_names = self.feature_engineer.prepare_features(self.df)
        
        logger.info(f"Created {len(self.feature_names)} features")
        
    def train_clv_model(self):
        """Train CLV prediction model."""
        logger.info("\n" + "="*60)
        logger.info("STEP 3: TRAINING CLV PREDICTION MODEL")
        logger.info("="*60)
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.preprocessor.prepare_for_modeling(
            self.df_processed, CLV_TARGET, self.feature_names
        )
        
        # Scale features
        X_train_scaled = self.feature_engineer.scale_features(X_train, fit=True)
        X_test_scaled = self.feature_engineer.scale_features(X_test, fit=False)
        
        # Train
        self.clv_model = CLVPredictor()
        self.clv_model.train(X_train_scaled, y_train, self.feature_names)
        
        # Evaluate
        metrics = self.clv_model.evaluate(X_test_scaled, y_test)
        
        # Save
        self.clv_model.save('clv_model.joblib')
        
        self.training_report['clv_model'] = {
            'metrics': metrics,
            'feature_importance': self.clv_model.get_feature_importance().head(10).to_dict('records')
        }
        
        logger.info("CLV model training complete")
        
    def train_churn_model(self):
        """Train churn prediction model."""
        logger.info("\n" + "="*60)
        logger.info("STEP 4: TRAINING CHURN PREDICTION MODEL")
        logger.info("="*60)
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.preprocessor.prepare_for_modeling(
            self.df_processed, CHURN_TARGET, self.feature_names
        )
        
        # Scale features
        X_train_scaled = self.feature_engineer.scale_features(X_train, fit=True)
        X_test_scaled = self.feature_engineer.scale_features(X_test, fit=False)
        
        # Train
        self.churn_model = ChurnPredictor()
        self.churn_model.train(X_train_scaled, y_train, self.feature_names)
        
        # Evaluate
        metrics = self.churn_model.evaluate(X_test_scaled, y_test)
        
        # Save
        self.churn_model.save('churn_model.joblib')
        
        self.training_report['churn_model'] = {
            'metrics': metrics,
            'churn_factors': self.churn_model.get_churn_factors(5).to_dict('records')
        }
        
        logger.info("Churn model training complete")
        
    def train_segmentation_model(self):
        """Train customer segmentation model."""
        logger.info("\n" + "="*60)
        logger.info("STEP 5: TRAINING SEGMENTATION MODEL")
        logger.info("="*60)
        
        # Prepare data
        X = self.df_processed[self.feature_names]
        X_scaled = self.feature_engineer.scale_features(X, fit=True)
        
        # Train
        self.segmentation_model = CustomerSegmentation()
        self.segmentation_model.train(X_scaled, feature_names=self.feature_names)
        
        # Evaluate
        metrics = self.segmentation_model.evaluate(X_scaled)
        
        # Save
        self.segmentation_model.save('segmentation_model.joblib')
        
        self.training_report['segmentation_model'] = {
            'metrics': metrics,
            'segment_profiles': self.segmentation_model.get_segment_profiles().to_dict('records')
        }
        
        logger.info("Segmentation model training complete")
        
    def calculate_business_impact(self):
        """Calculate business impact metrics."""
        logger.info("\n" + "="*60)
        logger.info("STEP 6: CALCULATING BUSINESS IMPACT")
        logger.info("="*60)
        
        # Prepare test data
        X_test_clv, _, y_test_clv, _ = self.preprocessor.prepare_for_modeling(
            self.df_processed, CLV_TARGET, self.feature_names
        )
        X_test_churn, _, y_test_churn, _ = self.preprocessor.prepare_for_modeling(
            self.df_processed, CHURN_TARGET, self.feature_names
        )
        
        X_test_clv_scaled = self.feature_engineer.scale_features(X_test_clv, fit=False)
        X_test_churn_scaled = self.feature_engineer.scale_features(X_test_churn, fit=False)
        
        # Predictions
        y_pred_clv = self.clv_model.predict(X_test_clv_scaled)
        y_pred_churn = self.churn_model.predict(X_test_churn_scaled)
        
        # Business metrics
        business_metrics = business_impact_metrics(
            y_test_clv.values, y_pred_clv,
            y_test_churn.values, y_pred_churn
        )
        
        self.training_report['business_impact'] = business_metrics
        
        logger.info("Business impact analysis complete")
        
    def save_training_report(self):
        """Save comprehensive training report."""
        report_path = MODELS_DIR / 'training_report.json'
        
        with open(report_path, 'w') as f:
            json.dump(self.training_report, f, indent=2)
        
        logger.info(f"Saved training report to {report_path}")
        
        # Also save feature scaler
        scaler_path = MODELS_DIR / 'feature_scaler.joblib'
        joblib.dump(self.feature_engineer.scaler, scaler_path)
        logger.info(f"Saved feature scaler to {scaler_path}")
        
        # Save feature names
        features_path = MODELS_DIR / 'feature_names.json'
        with open(features_path, 'w') as f:
            json.dump(self.feature_names, f, indent=2)
        logger.info(f"Saved feature names to {features_path}")
        
    def print_summary(self):
        """Print training summary."""
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        
        print("\n📊 CLV Prediction Model:")
        clv_metrics = self.training_report['clv_model']['metrics']
        print(f"  RMSE: ${clv_metrics['rmse']:.2f}")
        print(f"  R² Score: {clv_metrics['r2']:.4f}")
        
        print("\n🎯 Churn Prediction Model:")
        churn_metrics = self.training_report['churn_model']['metrics']
        print(f"  Accuracy: {churn_metrics['accuracy']:.4f}")
        print(f"  AUC-ROC: {churn_metrics['auc_roc']:.4f}")
        
        print("\n👥 Segmentation Model:")
        seg_metrics = self.training_report['segmentation_model']['metrics']
        print(f"  Silhouette Score: {seg_metrics['silhouette_score']:.4f}")
        print(f"  Number of Segments: {seg_metrics['n_clusters']}")
        
        print("\n💼 Business Impact:")
        business = self.training_report['business_impact']
        print(f"  Retention Lift: {business['retention_lift_pct']:.2f}%")
        print(f"  At-Risk Customers: {business['at_risk_customers_count']}")
        print(f"  Potential Saved Revenue: ${business['potential_saved_revenue']:,.2f}")
        
        print("\n" + "="*60)
        
    def run_full_pipeline(self):
        """Run the complete training pipeline."""
        logger.info("Starting full model training pipeline...")
        
        self.generate_data()
        self.engineer_features()
        self.train_clv_model()
        self.train_churn_model()
        self.train_segmentation_model()
        self.calculate_business_impact()
        self.save_training_report()
        self.print_summary()
        
        logger.info("\n✅ All models trained successfully!")


def main():
    """Main execution."""
    trainer = ModelTrainer(n_customers=10000)
    trainer.run_full_pipeline()


if __name__ == "__main__":
    main()
