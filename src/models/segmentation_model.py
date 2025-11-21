"""
Customer segmentation model using K-Means clustering.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from typing import Dict, List
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from models.base_model import BaseModel
from utils.metrics import segment_quality_metrics
from utils.config import SEGMENTATION_PARAMS
from utils.logger import logger


class CustomerSegmentation(BaseModel):
    """Customer segmentation using K-Means clustering."""
    
    def __init__(self, **kwargs):
        """
        Initialize segmentation model.
        
        Args:
            **kwargs: Model hyperparameters (overrides defaults)
        """
        super().__init__(model_name="Customer Segmentation")
        
        # Merge custom params with defaults
        self.params = {**SEGMENTATION_PARAMS, **kwargs}
        self.segment_profiles = {}
        
    def build_model(self) -> KMeans:
        """
        Build K-Means clustering model.
        
        Returns:
            KMeans instance
        """
        return KMeans(**self.params)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray = None,
             feature_names: list = None) -> 'CustomerSegmentation':
        """
        Train clustering model (y_train ignored for unsupervised learning).
        
        Args:
            X_train: Training features
            y_train: Ignored (clustering is unsupervised)
            feature_names: List of feature names
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Training {self.model_name}...")
        
        if self.model is None:
            self.model = self.build_model()
        
        self.model.fit(X_train)
        self.is_trained = True
        
        if feature_names:
            self.feature_names = feature_names
        
        # Create segment profiles
        self._create_segment_profiles(X_train)
        
        logger.info(f"{self.model_name} training complete")
        return self
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray = None) -> Dict[str, float]:
        """
        Evaluate clustering quality.
        
        Args:
            X_test: Test features
            y_test: Ignored
            
        Returns:
            Dictionary of clustering metrics
        """
        labels = self.predict(X_test)
        
        metrics = segment_quality_metrics(X_test, labels)
        self.metrics = metrics
        
        logger.info(f"Segmentation Quality Metrics:")
        logger.info(f"  Silhouette Score: {metrics['silhouette_score']:.4f}")
        logger.info(f"  Calinski-Harabasz Score: {metrics['calinski_harabasz_score']:.2f}")
        logger.info(f"  Davies-Bouldin Score: {metrics['davies_bouldin_score']:.4f}")
        
        return metrics
    
    def _create_segment_profiles(self, X: np.ndarray):
        """
        Create profiles for each segment based on cluster centers.
        
        Args:
            X: Feature array
        """
        labels = self.model.labels_
        
        for segment_id in range(self.params['n_clusters']):
            segment_mask = labels == segment_id
            segment_data = X[segment_mask]
            
            self.segment_profiles[segment_id] = {
                'size': int(segment_mask.sum()),
                'percentage': float(segment_mask.mean() * 100),
                'center': self.model.cluster_centers_[segment_id].tolist()
            }
        
        logger.info(f"Created profiles for {len(self.segment_profiles)} segments")
    
    def predict_segments(self, X: np.ndarray) -> pd.DataFrame:
        """
        Predict customer segments with metadata.
        
        Args:
            X: Feature array
            
        Returns:
            DataFrame with segment assignments and distances
        """
        segments = self.predict(X)
        distances = self.model.transform(X)
        
        # Distance to assigned cluster center
        min_distances = np.min(distances, axis=1)
        
        # Segment names (can be customized based on business logic)
        segment_names = [
            'High-Value Engaged',
            'Loyal Regulars',
            'At-Risk',
            'New Prospects',
            'Dormant'
        ]
        
        # Map segment IDs to names
        segment_labels = [segment_names[s] if s < len(segment_names) 
                         else f'Segment {s}' for s in segments]
        
        results = pd.DataFrame({
            'segment_id': segments,
            'segment_name': segment_labels,
            'distance_to_center': min_distances
        })
        
        return results
    
    def get_segment_profiles(self) -> pd.DataFrame:
        """
        Get detailed profiles for all segments.
        
        Returns:
            DataFrame with segment profiles
        """
        profiles = []
        
        for segment_id, profile in self.segment_profiles.items():
            profiles.append({
                'segment_id': segment_id,
                'size': profile['size'],
                'percentage': profile['percentage']
            })
        
        return pd.DataFrame(profiles)
    
    def get_segment_characteristics(self, X: np.ndarray, 
                                   segment_id: int) -> Dict[str, float]:
        """
        Get statistical characteristics of a specific segment.
        
        Args:
            X: Feature array
            segment_id: Segment ID to analyze
            
        Returns:
            Dictionary of segment characteristics
        """
        labels = self.predict(X)
        segment_mask = labels == segment_id
        segment_data = X[segment_mask]
        
        characteristics = {
            'segment_id': segment_id,
            'size': int(segment_mask.sum()),
            'percentage': float(segment_mask.mean() * 100)
        }
        
        # Add feature-wise statistics
        for i, feature_name in enumerate(self.feature_names):
            if i < segment_data.shape[1]:
                characteristics[f'{feature_name}_mean'] = float(segment_data[:, i].mean())
                characteristics[f'{feature_name}_std'] = float(segment_data[:, i].std())
        
        return characteristics
    
    def recommend_segment_actions(self, segment_id: int) -> List[str]:
        """
        Recommend marketing actions for a specific segment.
        
        Args:
            segment_id: Segment ID
            
        Returns:
            List of recommended actions
        """
        # Simplified recommendation logic (can be enhanced with business rules)
        recommendations = {
            0: [  # High-Value Engaged
                "Offer exclusive VIP benefits and early access to new products",
                "Implement loyalty rewards program with premium tiers",
                "Provide personalized concierge service"
            ],
            1: [  # Loyal Regulars
                "Send personalized product recommendations based on purchase history",
                "Offer bundle deals and cross-sell opportunities",
                "Implement referral incentive programs"
            ],
            2: [  # At-Risk
                "Launch win-back campaigns with special offers",
                "Conduct satisfaction surveys to identify pain points",
                "Provide re-engagement incentives and discounts"
            ],
            3: [  # New Prospects
                "Offer onboarding tutorials and welcome bonuses",
                "Send educational content about product benefits",
                "Provide first-purchase discounts"
            ],
            4: [  # Dormant
                "Send reactivation emails with compelling offers",
                "Highlight new features and improvements",
                "Offer time-limited comeback discounts"
            ]
        }
        
        return recommendations.get(segment_id, ["Analyze segment characteristics for custom strategy"])


def main():
    """Main execution for testing."""
    from data.data_generator import CustomerDataGenerator
    from data.feature_engineering import FeatureEngineer
    
    # Generate and prepare data
    logger.info("Generating customer data...")
    generator = CustomerDataGenerator(n_customers=5000)
    df = generator.generate()
    
    logger.info("Engineering features...")
    engineer = FeatureEngineer()
    df_processed, feature_names = engineer.prepare_features(df)
    
    # Select features for clustering
    X = df_processed[feature_names]
    X_scaled = engineer.scale_features(X, fit=True)
    
    # Train model
    logger.info("Training segmentation model...")
    seg_model = CustomerSegmentation()
    seg_model.train(X_scaled, feature_names=feature_names)
    
    # Evaluate
    metrics = seg_model.evaluate(X_scaled)
    
    # Get profiles
    profiles = seg_model.get_segment_profiles()
    
    print("\n" + "="*60)
    print("CUSTOMER SEGMENTATION RESULTS")
    print("="*60)
    print(f"\nClustering Quality:")
    print(f"  Silhouette Score: {metrics['silhouette_score']:.4f}")
    print(f"  Calinski-Harabasz: {metrics['calinski_harabasz_score']:.2f}")
    print(f"  Davies-Bouldin: {metrics['davies_bouldin_score']:.4f}")
    print(f"\nSegment Distribution:")
    for idx, row in profiles.iterrows():
        print(f"  Segment {row['segment_id']}: {row['size']} customers ({row['percentage']:.1f}%)")
    print("\n" + "="*60)
    
    # Save model
    seg_model.save()


if __name__ == "__main__":
    main()
