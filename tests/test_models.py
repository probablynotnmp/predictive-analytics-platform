"""
Unit tests for ML models.
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.data.data_generator import CustomerDataGenerator
from src.data.feature_engineering import FeatureEngineer
from src.data.data_preprocessor import DataPreprocessor
from src.models.clv_predictor import CLVPredictor
from src.models.churn_predictor import ChurnPredictor
from src.models.segmentation_model import CustomerSegmentation
from src.utils.config import CLV_TARGET, CHURN_TARGET


@pytest.fixture
def sample_data():
    """Generate sample customer data for testing."""
    generator = CustomerDataGenerator(n_customers=500)
    df = generator.generate()
    return df


@pytest.fixture
def processed_data(sample_data):
    """Generate processed features."""
    engineer = FeatureEngineer()
    df_processed, feature_names = engineer.prepare_features(sample_data)
    return df_processed, feature_names, engineer


class TestDataGeneration:
    """Test data generation."""
    
    def test_data_generator_creates_correct_shape(self, sample_data):
        """Test that data generator creates expected number of rows."""
        assert len(sample_data) == 500
        
    def test_data_generator_has_required_columns(self, sample_data):
        """Test that all required columns are present."""
        required_cols = ['customer_id', 'age', 'account_type', 'tenure_days',
                        'customer_lifetime_value', 'churned']
        for col in required_cols:
            assert col in sample_data.columns
            
    def test_data_values_are_valid(self, sample_data):
        """Test that generated data has valid ranges."""
        assert sample_data['age'].min() >= 18
        assert sample_data['age'].max() <= 80
        assert sample_data['customer_lifetime_value'].min() >= 0
        assert sample_data['churned'].isin([0, 1]).all()


class TestFeatureEngineering:
    """Test feature engineering."""
    
    def test_feature_engineer_creates_features(self, processed_data):
        """Test that feature engineering creates expected features."""
        df_processed, feature_names, _ = processed_data
        assert len(feature_names) > 10
        
    def test_rfm_scores_created(self, processed_data):
        """Test that RFM scores are created."""
        df_processed, _, _ = processed_data
        assert 'rfm_score' in df_processed.columns
        assert 'rfm_segment' in df_processed.columns
        
    def test_engagement_features_created(self, processed_data):
        """Test that engagement features are created."""
        df_processed, _, _ = processed_data
        assert 'engagement_score' in df_processed.columns
        assert df_processed['engagement_score'].between(0, 100).all()


class TestCLVPredictor:
    """Test CLV prediction model."""
    
    def test_clv_model_training(self, processed_data):
        """Test that CLV model can be trained."""
        df_processed, feature_names, engineer = processed_data
        
        preprocessor = DataPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.prepare_for_modeling(
            df_processed, CLV_TARGET, feature_names
        )
        
        X_train_scaled = engineer.scale_features(X_train, fit=True)
        
        model = CLVPredictor()
        model.train(X_train_scaled, y_train, feature_names)
        
        assert model.is_trained
        assert model.model is not None
        
    def test_clv_model_prediction(self, processed_data):
        """Test that CLV model can make predictions."""
        df_processed, feature_names, engineer = processed_data
        
        preprocessor = DataPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.prepare_for_modeling(
            df_processed, CLV_TARGET, feature_names
        )
        
        X_train_scaled = engineer.scale_features(X_train, fit=True)
        X_test_scaled = engineer.scale_features(X_test, fit=False)
        
        model = CLVPredictor()
        model.train(X_train_scaled, y_train, feature_names)
        
        predictions = model.predict(X_test_scaled)
        
        assert len(predictions) == len(X_test)
        assert all(predictions >= 0)  # CLV should be non-negative
        
    def test_clv_model_evaluation(self, processed_data):
        """Test that CLV model evaluation returns metrics."""
        df_processed, feature_names, engineer = processed_data
        
        preprocessor = DataPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.prepare_for_modeling(
            df_processed, CLV_TARGET, feature_names
        )
        
        X_train_scaled = engineer.scale_features(X_train, fit=True)
        X_test_scaled = engineer.scale_features(X_test, fit=False)
        
        model = CLVPredictor()
        model.train(X_train_scaled, y_train, feature_names)
        
        metrics = model.evaluate(X_test_scaled, y_test)
        
        assert 'rmse' in metrics
        assert 'r2' in metrics
        assert metrics['r2'] > 0  # Model should have some predictive power


class TestChurnPredictor:
    """Test churn prediction model."""
    
    def test_churn_model_training(self, processed_data):
        """Test that churn model can be trained."""
        df_processed, feature_names, engineer = processed_data
        
        preprocessor = DataPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.prepare_for_modeling(
            df_processed, CHURN_TARGET, feature_names
        )
        
        X_train_scaled = engineer.scale_features(X_train, fit=True)
        
        model = ChurnPredictor()
        model.train(X_train_scaled, y_train, feature_names)
        
        assert model.is_trained
        assert model.model is not None
        
    def test_churn_model_prediction(self, processed_data):
        """Test that churn model can make predictions."""
        df_processed, feature_names, engineer = processed_data
        
        preprocessor = DataPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.prepare_for_modeling(
            df_processed, CHURN_TARGET, feature_names
        )
        
        X_train_scaled = engineer.scale_features(X_train, fit=True)
        X_test_scaled = engineer.scale_features(X_test, fit=False)
        
        model = ChurnPredictor()
        model.train(X_train_scaled, y_train, feature_names)
        
        predictions = model.predict(X_test_scaled)
        probabilities = model.predict_proba(X_test_scaled)
        
        assert len(predictions) == len(X_test)
        assert all(p in [0, 1] for p in predictions)
        assert probabilities.shape == (len(X_test), 2)
        assert all(probabilities[:, 1] >= 0) and all(probabilities[:, 1] <= 1)


class TestSegmentationModel:
    """Test customer segmentation model."""
    
    def test_segmentation_model_training(self, processed_data):
        """Test that segmentation model can be trained."""
        df_processed, feature_names, engineer = processed_data
        
        X = df_processed[feature_names]
        X_scaled = engineer.scale_features(X, fit=True)
        
        model = CustomerSegmentation()
        model.train(X_scaled, feature_names=feature_names)
        
        assert model.is_trained
        assert model.model is not None
        
    def test_segmentation_model_prediction(self, processed_data):
        """Test that segmentation model can assign segments."""
        df_processed, feature_names, engineer = processed_data
        
        X = df_processed[feature_names]
        X_scaled = engineer.scale_features(X, fit=True)
        
        model = CustomerSegmentation()
        model.train(X_scaled, feature_names=feature_names)
        
        segments = model.predict(X_scaled)
        
        assert len(segments) == len(X)
        assert all(s >= 0 and s < 5 for s in segments)  # 5 clusters


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
