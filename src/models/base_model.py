"""
Base model class for all ML models in the platform.
"""

import joblib
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Tuple
from sklearn.model_selection import cross_val_score
import sys

sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import logger
from utils.config import MODELS_DIR


class BaseModel(ABC):
    """Abstract base class for all ML models."""
    
    def __init__(self, model_name: str):
        """
        Initialize base model.
        
        Args:
            model_name: Name of the model
        """
        self.model_name = model_name
        self.model = None
        self.is_trained = False
        self.feature_names = []
        self.metrics = {}
        
    @abstractmethod
    def build_model(self) -> Any:
        """
        Build and return the model instance.
        
        Returns:
            Model instance
        """
        pass
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
             feature_names: list = None) -> 'BaseModel':
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training target
            feature_names: List of feature names
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Training {self.model_name}...")
        
        if self.model is None:
            self.model = self.build_model()
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        if feature_names:
            self.feature_names = feature_names
        
        logger.info(f"{self.model_name} training complete")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Feature array
            
        Returns:
            Predictions
        """
        if not self.is_trained:
            raise ValueError(f"{self.model_name} must be trained before prediction")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities (for classifiers).
        
        Args:
            X: Feature array
            
        Returns:
            Probability predictions
        """
        if not self.is_trained:
            raise ValueError(f"{self.model_name} must be trained before prediction")
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise NotImplementedError(f"{self.model_name} does not support probability predictions")
    
    @abstractmethod
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary of metrics
        """
        pass
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, 
                      cv: int = 5, scoring: str = 'r2') -> Dict[str, float]:
        """
        Perform cross-validation.
        
        Args:
            X: Feature array
            y: Target array
            cv: Number of folds
            scoring: Scoring metric
            
        Returns:
            Cross-validation results
        """
        if self.model is None:
            self.model = self.build_model()
        
        scores = cross_val_score(self.model, X, y, cv=cv, scoring=scoring)
        
        results = {
            f'cv_{scoring}_mean': scores.mean(),
            f'cv_{scoring}_std': scores.std(),
            f'cv_{scoring}_min': scores.min(),
            f'cv_{scoring}_max': scores.max()
        }
        
        logger.info(f"Cross-validation {scoring}: {scores.mean():.4f} (+/- {scores.std():.4f})")
        
        return results
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance scores.
        
        Returns:
            DataFrame with feature importance
        """
        if not self.is_trained:
            raise ValueError(f"{self.model_name} must be trained first")
        
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names if self.feature_names else 
                          [f'feature_{i}' for i in range(len(self.model.feature_importances_))],
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df
        else:
            raise NotImplementedError(f"{self.model_name} does not support feature importance")
    
    def save(self, filename: str = None) -> Path:
        """
        Save model to disk.
        
        Args:
            filename: Optional custom filename
            
        Returns:
            Path to saved model
        """
        if not self.is_trained:
            raise ValueError(f"{self.model_name} must be trained before saving")
        
        if filename is None:
            filename = f"{self.model_name.lower().replace(' ', '_')}.joblib"
        
        filepath = MODELS_DIR / filename
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'metrics': self.metrics,
            'model_name': self.model_name
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Saved {self.model_name} to {filepath}")
        
        return filepath
    
    def load(self, filename: str) -> 'BaseModel':
        """
        Load model from disk.
        
        Args:
            filename: Model filename
            
        Returns:
            Self for method chaining
        """
        filepath = MODELS_DIR / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.metrics = model_data['metrics']
        self.model_name = model_data['model_name']
        self.is_trained = True
        
        logger.info(f"Loaded {self.model_name} from {filepath}")
        
        return self
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information and metadata.
        
        Returns:
            Dictionary with model info
        """
        return {
            'model_name': self.model_name,
            'is_trained': self.is_trained,
            'n_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'metrics': self.metrics
        }
