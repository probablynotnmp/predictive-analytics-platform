"""
Churn prediction model using Random Forest.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from typing import Dict
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from models.base_model import BaseModel
from utils.metrics import classification_metrics
from utils.config import CHURN_MODEL_PARAMS, CHURN_THRESHOLD
from utils.logger import logger


class ChurnPredictor(BaseModel):
    """Churn risk prediction using Random Forest."""
    
    def __init__(self, **kwargs):
        """
        Initialize churn predictor.
        
        Args:
            **kwargs: Model hyperparameters (overrides defaults)
        """
        super().__init__(model_name="Churn Predictor")
        
        # Merge custom params with defaults
        self.params = {**CHURN_MODEL_PARAMS, **kwargs}
        self.threshold = CHURN_THRESHOLD
        
    def build_model(self) -> RandomForestClassifier:
        """
        Build Random Forest classifier.
        
        Returns:
            RandomForestClassifier instance
        """
        return RandomForestClassifier(**self.params)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate churn prediction performance.
        
        Args:
            X_test: Test features
            y_test: True churn labels
            
        Returns:
            Dictionary of classification metrics
        """
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)[:, 1]
        
        metrics = classification_metrics(y_test, y_pred, y_pred_proba)
        self.metrics = metrics
        
        logger.info(f"Churn Prediction Metrics:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}")
        logger.info(f"  F1 Score: {metrics['f1']:.4f}")
        logger.info(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
        
        return metrics
    
    def predict_churn_risk(self, X: np.ndarray) -> pd.DataFrame:
        """
        Predict churn risk with probability scores.
        
        Args:
            X: Feature array
            
        Returns:
            DataFrame with churn predictions and risk scores
        """
        churn_proba = self.predict_proba(X)[:, 1]
        churn_pred = (churn_proba >= self.threshold).astype(int)
        
        # Risk categories
        risk_categories = pd.cut(
            churn_proba,
            bins=[0, 0.25, 0.5, 0.75, 1.0],
            labels=['Low Risk', 'Medium Risk', 'High Risk', 'Critical Risk']
        )
        
        results = pd.DataFrame({
            'churn_probability': churn_proba,
            'churn_prediction': churn_pred,
            'risk_category': risk_categories
        })
        
        return results
    
    def identify_at_risk_customers(self, X: np.ndarray, 
                                   risk_threshold: float = 0.7) -> np.ndarray:
        """
        Identify customers at high risk of churning.
        
        Args:
            X: Feature array
            risk_threshold: Probability threshold for at-risk classification
            
        Returns:
            Boolean array indicating at-risk customers
        """
        churn_proba = self.predict_proba(X)[:, 1]
        at_risk = churn_proba >= risk_threshold
        
        logger.info(f"Identified {at_risk.sum()} at-risk customers "
                   f"(churn probability >= {risk_threshold:.0%})")
        
        return at_risk
    
    def get_churn_factors(self, top_n: int = 10) -> pd.DataFrame:
        """
        Get top factors contributing to churn.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with top churn factors
        """
        importance = self.get_feature_importance()
        
        top_factors = importance.head(top_n).copy()
        top_factors.columns = ['churn_factor', 'importance_score']
        
        return top_factors
    
    def analyze_churn_segments(self, X: np.ndarray) -> Dict[str, int]:
        """
        Analyze distribution of churn risk across segments.
        
        Args:
            X: Feature array
            
        Returns:
            Dictionary with segment counts
        """
        risk_df = self.predict_churn_risk(X)
        
        segment_counts = {
            'low_risk': int((risk_df['risk_category'] == 'Low Risk').sum()),
            'medium_risk': int((risk_df['risk_category'] == 'Medium Risk').sum()),
            'high_risk': int((risk_df['risk_category'] == 'High Risk').sum()),
            'critical_risk': int((risk_df['risk_category'] == 'Critical Risk').sum()),
            'total_at_risk': int((risk_df['churn_prediction'] == 1).sum()),
            'avg_churn_probability': float(risk_df['churn_probability'].mean())
        }
        
        return segment_counts


def main():
    """Main execution for testing."""
    from data.data_generator import CustomerDataGenerator
    from data.feature_engineering import FeatureEngineer
    from data.data_preprocessor import DataPreprocessor
    from utils.config import CHURN_TARGET
    
    # Generate and prepare data
    logger.info("Generating customer data...")
    generator = CustomerDataGenerator(n_customers=5000)
    df = generator.generate()
    
    logger.info("Engineering features...")
    engineer = FeatureEngineer()
    df_processed, feature_names = engineer.prepare_features(df)
    
    logger.info("Preprocessing data...")
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.prepare_for_modeling(
        df_processed, CHURN_TARGET, feature_names
    )
    
    # Scale features
    X_train_scaled = engineer.scale_features(X_train, fit=True)
    X_test_scaled = engineer.scale_features(X_test, fit=False)
    
    # Train model
    logger.info("Training churn prediction model...")
    churn_model = ChurnPredictor()
    churn_model.train(X_train_scaled, y_train, feature_names)
    
    # Evaluate
    metrics = churn_model.evaluate(X_test_scaled, y_test)
    
    # Feature importance
    importance = churn_model.get_churn_factors(top_n=5)
    
    # Segment analysis
    segments = churn_model.analyze_churn_segments(X_test_scaled)
    
    print("\n" + "="*60)
    print("CHURN PREDICTION MODEL RESULTS")
    print("="*60)
    print(f"\nModel Performance:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1 Score: {metrics['f1']:.4f}")
    print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
    print(f"\nTop 5 Churn Factors:")
    for idx, row in importance.iterrows():
        print(f"  {row['churn_factor']}: {row['importance_score']:.4f}")
    print(f"\nRisk Segment Distribution:")
    print(f"  Low Risk: {segments['low_risk']}")
    print(f"  Medium Risk: {segments['medium_risk']}")
    print(f"  High Risk: {segments['high_risk']}")
    print(f"  Critical Risk: {segments['critical_risk']}")
    print("\n" + "="*60)
    
    # Save model
    churn_model.save()


if __name__ == "__main__":
    main()
