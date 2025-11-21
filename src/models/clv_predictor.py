"""
Customer Lifetime Value (CLV) prediction model using Gradient Boosting.
"""

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from typing import Dict
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from models.base_model import BaseModel
from utils.metrics import regression_metrics
from utils.config import CLV_MODEL_PARAMS
from utils.logger import logger


class CLVPredictor(BaseModel):
    """Customer Lifetime Value prediction using XGBoost."""
    
    def __init__(self, **kwargs):
        """
        Initialize CLV predictor.
        
        Args:
            **kwargs: Model hyperparameters (overrides defaults)
        """
        super().__init__(model_name="CLV Predictor")
        
        # Merge custom params with defaults
        self.params = {**CLV_MODEL_PARAMS, **kwargs}
        
    def build_model(self) -> XGBRegressor:
        """
        Build XGBoost regression model.
        
        Returns:
            XGBRegressor instance
        """
        return XGBRegressor(**self.params)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate CLV prediction performance.
        
        Args:
            X_test: Test features
            y_test: True CLV values
            
        Returns:
            Dictionary of regression metrics
        """
        y_pred = self.predict(X_test)
        
        metrics = regression_metrics(y_test, y_pred)
        self.metrics = metrics
        
        logger.info(f"CLV Prediction Metrics:")
        logger.info(f"  RMSE: ${metrics['rmse']:.2f}")
        logger.info(f"  MAE: ${metrics['mae']:.2f}")
        logger.info(f"  R²: {metrics['r2']:.4f}")
        logger.info(f"  MAPE: {metrics['mape']:.2f}%")
        
        return metrics
    
    def predict_clv_segments(self, X: np.ndarray) -> pd.DataFrame:
        """
        Predict CLV and assign value segments.
        
        Args:
            X: Feature array
            
        Returns:
            DataFrame with CLV predictions and segments
        """
        clv_predictions = self.predict(X)
        
        # Assign segments based on CLV quartiles
        segments = pd.qcut(
            clv_predictions,
            q=4,
            labels=['Low Value', 'Medium Value', 'High Value', 'Premium Value']
        )
        
        results = pd.DataFrame({
            'predicted_clv': clv_predictions,
            'clv_segment': segments
        })
        
        return results
    
    def identify_high_value_customers(self, X: np.ndarray, 
                                     threshold_percentile: float = 75) -> np.ndarray:
        """
        Identify high-value customers based on predicted CLV.
        
        Args:
            X: Feature array
            threshold_percentile: Percentile threshold for high-value classification
            
        Returns:
            Boolean array indicating high-value customers
        """
        clv_predictions = self.predict(X)
        threshold = np.percentile(clv_predictions, threshold_percentile)
        
        high_value = clv_predictions >= threshold
        
        logger.info(f"Identified {high_value.sum()} high-value customers "
                   f"(CLV >= ${threshold:.2f})")
        
        return high_value
    
    def get_clv_distribution_stats(self, X: np.ndarray) -> Dict[str, float]:
        """
        Get statistical distribution of predicted CLV.
        
        Args:
            X: Feature array
            
        Returns:
            Dictionary of distribution statistics
        """
        clv_predictions = self.predict(X)
        
        stats = {
            'mean_clv': float(np.mean(clv_predictions)),
            'median_clv': float(np.median(clv_predictions)),
            'std_clv': float(np.std(clv_predictions)),
            'min_clv': float(np.min(clv_predictions)),
            'max_clv': float(np.max(clv_predictions)),
            'q25_clv': float(np.percentile(clv_predictions, 25)),
            'q75_clv': float(np.percentile(clv_predictions, 75)),
            'total_predicted_value': float(np.sum(clv_predictions))
        }
        
        return stats


def main():
    """Main execution for testing."""
    from data.data_generator import CustomerDataGenerator
    from data.feature_engineering import FeatureEngineer
    from data.data_preprocessor import DataPreprocessor
    from utils.config import CLV_TARGET
    
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
        df_processed, CLV_TARGET, feature_names
    )
    
    # Scale features
    X_train_scaled = engineer.scale_features(X_train, fit=True)
    X_test_scaled = engineer.scale_features(X_test, fit=False)
    
    # Train model
    logger.info("Training CLV prediction model...")
    clv_model = CLVPredictor()
    clv_model.train(X_train_scaled, y_train, feature_names)
    
    # Evaluate
    metrics = clv_model.evaluate(X_test_scaled, y_test)
    
    # Feature importance
    importance = clv_model.get_feature_importance()
    
    print("\n" + "="*60)
    print("CLV PREDICTION MODEL RESULTS")
    print("="*60)
    print(f"\nModel Performance:")
    print(f"  RMSE: ${metrics['rmse']:.2f}")
    print(f"  MAE: ${metrics['mae']:.2f}")
    print(f"  R² Score: {metrics['r2']:.4f}")
    print(f"  MAPE: {metrics['mape']:.2f}%")
    print(f"\nTop 5 Important Features:")
    for idx, row in importance.head(5).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    print("\n" + "="*60)
    
    # Save model
    clv_model.save()


if __name__ == "__main__":
    main()
