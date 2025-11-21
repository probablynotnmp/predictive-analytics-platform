"""
Model loader and caching for API service.
"""

import joblib
import json
from pathlib import Path
from typing import Optional, Dict, Any
import sys

sys.path.append(str(Path(__file__).parent.parent))

from models.clv_predictor import CLVPredictor
from models.churn_predictor import ChurnPredictor
from models.segmentation_model import CustomerSegmentation
from utils.config import MODELS_DIR
from utils.logger import logger


class ModelLoader:
    """Lazy loading and caching of trained models."""
    
    def __init__(self):
        """Initialize model loader."""
        self._clv_model: Optional[CLVPredictor] = None
        self._churn_model: Optional[ChurnPredictor] = None
        self._segmentation_model: Optional[CustomerSegmentation] = None
        self._feature_scaler = None
        self._feature_names = None
        
    @property
    def clv_model(self) -> CLVPredictor:
        """
        Get CLV model (lazy loaded).
        
        Returns:
            Loaded CLV model
        """
        if self._clv_model is None:
            logger.info("Loading CLV model...")
            self._clv_model = CLVPredictor()
            self._clv_model.load('clv_model.joblib')
            logger.info("CLV model loaded successfully")
        return self._clv_model
    
    @property
    def churn_model(self) -> ChurnPredictor:
        """
        Get churn model (lazy loaded).
        
        Returns:
            Loaded churn model
        """
        if self._churn_model is None:
            logger.info("Loading churn model...")
            self._churn_model = ChurnPredictor()
            self._churn_model.load('churn_model.joblib')
            logger.info("Churn model loaded successfully")
        return self._churn_model
    
    @property
    def segmentation_model(self) -> CustomerSegmentation:
        """
        Get segmentation model (lazy loaded).
        
        Returns:
            Loaded segmentation model
        """
        if self._segmentation_model is None:
            logger.info("Loading segmentation model...")
            self._segmentation_model = CustomerSegmentation()
            self._segmentation_model.load('segmentation_model.joblib')
            logger.info("Segmentation model loaded successfully")
        return self._segmentation_model
    
    @property
    def feature_scaler(self):
        """
        Get feature scaler (lazy loaded).
        
        Returns:
            Loaded feature scaler
        """
        if self._feature_scaler is None:
            scaler_path = MODELS_DIR / 'feature_scaler.joblib'
            if scaler_path.exists():
                logger.info("Loading feature scaler...")
                self._feature_scaler = joblib.load(scaler_path)
                logger.info("Feature scaler loaded successfully")
            else:
                raise FileNotFoundError(f"Feature scaler not found: {scaler_path}")
        return self._feature_scaler
    
    @property
    def feature_names(self):
        """
        Get feature names (lazy loaded).
        
        Returns:
            List of feature names
        """
        if self._feature_names is None:
            features_path = MODELS_DIR / 'feature_names.json'
            if features_path.exists():
                logger.info("Loading feature names...")
                with open(features_path, 'r') as f:
                    self._feature_names = json.load(f)
                logger.info(f"Loaded {len(self._feature_names)} feature names")
            else:
                raise FileNotFoundError(f"Feature names not found: {features_path}")
        return self._feature_names
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get information about a specific model.
        
        Args:
            model_name: Name of the model ('clv', 'churn', or 'segmentation')
            
        Returns:
            Model information dictionary
        """
        model_map = {
            'clv': self.clv_model,
            'churn': self.churn_model,
            'segmentation': self.segmentation_model
        }
        
        if model_name not in model_map:
            raise ValueError(f"Unknown model: {model_name}")
        
        model = model_map[model_name]
        return model.get_model_info()
    
    def is_model_loaded(self, model_name: str) -> bool:
        """
        Check if a model is loaded.
        
        Args:
            model_name: Name of the model
            
        Returns:
            True if loaded, False otherwise
        """
        model_map = {
            'clv': self._clv_model,
            'churn': self._churn_model,
            'segmentation': self._segmentation_model
        }
        
        return model_map.get(model_name) is not None
    
    def preload_all_models(self):
        """Preload all models into memory."""
        logger.info("Preloading all models...")
        _ = self.clv_model
        _ = self.churn_model
        _ = self.segmentation_model
        _ = self.feature_scaler
        _ = self.feature_names
        logger.info("All models preloaded successfully")
    
    def get_models_status(self) -> Dict[str, bool]:
        """
        Get loading status of all models.
        
        Returns:
            Dictionary with model loading status
        """
        return {
            'clv': self.is_model_loaded('clv'),
            'churn': self.is_model_loaded('churn'),
            'segmentation': self.is_model_loaded('segmentation'),
            'feature_scaler': self._feature_scaler is not None,
            'feature_names': self._feature_names is not None
        }


# Global model loader instance
model_loader = ModelLoader()
