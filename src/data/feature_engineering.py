"""
Feature engineering module for customer analytics.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, List
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from utils.config import RFM_QUANTILES, ALL_FEATURES
from utils.logger import logger


class FeatureEngineer:
    """Feature engineering for customer data."""
    
    def __init__(self):
        """Initialize feature engineer."""
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        
    def create_rfm_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create RFM (Recency, Frequency, Monetary) scores.
        
        Args:
            df: Customer DataFrame
            
        Returns:
            DataFrame with RFM scores added
        """
        df = df.copy()
        
        # RFM scores (1-5, where 5 is best)
        df['recency_score'] = pd.qcut(df['recency'], q=RFM_QUANTILES, labels=False, duplicates='drop')
        df['recency_score'] = RFM_QUANTILES - df['recency_score']  # Invert: lower recency is better
        
        df['frequency_score'] = pd.qcut(df['frequency'], q=RFM_QUANTILES, labels=False, duplicates='drop')
        
        df['monetary_score'] = pd.qcut(df['monetary'], q=RFM_QUANTILES, labels=False, duplicates='drop')
        
        # Combined RFM score
        df['rfm_score'] = df['recency_score'] + df['frequency_score'] + df['monetary_score']
        
        # RFM segment
        df['rfm_segment'] = pd.cut(
            df['rfm_score'],
            bins=[0, 5, 9, 12, 15],
            labels=['At Risk', 'Developing', 'Established', 'Champions']
        )
        
        logger.info("Created RFM scores and segments")
        return df
    
    def create_engagement_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engagement-based features.
        
        Args:
            df: Customer DataFrame
            
        Returns:
            DataFrame with engagement features
        """
        df = df.copy()
        
        # Engagement score (0-100)
        df['engagement_score'] = (
            df['email_open_rate'] * 0.3 +
            (df['website_visits_per_month'] / df['website_visits_per_month'].max() * 100) * 0.5 +
            (1 - df['support_tickets'] / df['support_tickets'].max()) * 100 * 0.2
        ).clip(0, 100)
        
        # Activity level
        df['activity_level'] = pd.cut(
            df['engagement_score'],
            bins=[0, 25, 50, 75, 100],
            labels=['Low', 'Medium', 'High', 'Very High']
        )
        
        # Purchase velocity (purchases per day)
        df['purchase_velocity'] = df['total_purchases'] / df['tenure_days']
        
        # Value per visit
        df['value_per_visit'] = df['monetary'] / (df['website_visits_per_month'] * (df['tenure_days'] / 30))
        df['value_per_visit'] = df['value_per_visit'].replace([np.inf, -np.inf], 0).fillna(0)
        
        logger.info("Created engagement features")
        return df
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features.
        
        Args:
            df: Customer DataFrame
            
        Returns:
            DataFrame with temporal features
        """
        df = df.copy()
        
        # Tenure segments
        df['tenure_segment'] = pd.cut(
            df['tenure_days'],
            bins=[0, 90, 365, 730, 1825],
            labels=['New', 'Regular', 'Loyal', 'VIP']
        )
        
        # Recency segments
        df['recency_segment'] = pd.cut(
            df['recency'],
            bins=[0, 30, 90, 180, 365],
            labels=['Very Recent', 'Recent', 'Moderate', 'Dormant']
        )
        
        # Purchase trend (recent vs historical)
        df['recent_purchase_ratio'] = np.where(
            df['total_purchases'] > 0,
            (df['frequency'] * 3) / df['total_purchases'],  # Last 3 months vs total
            0
        ).clip(0, 2)
        
        logger.info("Created temporal features")
        return df
    
    def create_value_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create customer value features.
        
        Args:
            df: Customer DataFrame
            
        Returns:
            DataFrame with value features
        """
        df = df.copy()
        
        # CLV per tenure year
        df['clv_per_year'] = df['customer_lifetime_value'] / (df['tenure_days'] / 365)
        df['clv_per_year'] = df['clv_per_year'].replace([np.inf, -np.inf], 0).fillna(0)
        
        # Value tier
        df['value_tier'] = pd.qcut(
            df['customer_lifetime_value'],
            q=4,
            labels=['Bronze', 'Silver', 'Gold', 'Platinum'],
            duplicates='drop'
        )
        
        # Lifetime value to cost ratio (assuming acquisition cost)
        assumed_acquisition_cost = 100
        df['ltv_to_cac'] = df['customer_lifetime_value'] / assumed_acquisition_cost
        
        logger.info("Created value features")
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical features.
        
        Args:
            df: Customer DataFrame
            fit: Whether to fit encoders (True for training, False for inference)
            
        Returns:
            DataFrame with encoded features
        """
        df = df.copy()
        
        categorical_cols = ['account_type']
        
        for col in categorical_cols:
            if col in df.columns:
                if fit:
                    self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col])
                else:
                    if col in self.label_encoders:
                        df[col] = self.label_encoders[col].transform(df[col])
        
        logger.info(f"Encoded {len(categorical_cols)} categorical features")
        return df
    
    def prepare_features(self, df: pd.DataFrame, fit: bool = True) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepare all features for modeling.
        
        Args:
            df: Customer DataFrame
            fit: Whether to fit transformers
            
        Returns:
            Tuple of (processed DataFrame, feature names)
        """
        logger.info("Starting feature engineering pipeline...")
        
        # Create all engineered features
        df = self.create_rfm_scores(df)
        df = self.create_engagement_features(df)
        df = self.create_temporal_features(df)
        df = self.create_value_features(df)
        
        # Encode categorical features
        df = self.encode_categorical_features(df, fit=fit)
        
        # Select features for modeling
        feature_cols = [
            'age', 'account_type', 'tenure_days',
            'recency', 'frequency', 'monetary',
            'email_open_rate', 'website_visits_per_month', 'support_tickets',
            'avg_purchase_amount', 'purchase_frequency', 'days_since_last_purchase',
            'rfm_score', 'engagement_score', 'purchase_velocity',
            'clv_per_year', 'ltv_to_cac', 'recent_purchase_ratio'
        ]
        
        # Ensure all features exist
        available_features = [f for f in feature_cols if f in df.columns]
        self.feature_names = available_features
        
        logger.info(f"Prepared {len(available_features)} features for modeling")
        
        return df, available_features
    
    def scale_features(self, X: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """
        Scale features using StandardScaler.
        
        Args:
            X: Feature DataFrame
            fit: Whether to fit the scaler
            
        Returns:
            Scaled feature array
        """
        if fit:
            X_scaled = self.scaler.fit_transform(X)
            logger.info("Fitted and transformed features")
        else:
            X_scaled = self.scaler.transform(X)
            logger.info("Transformed features using existing scaler")
        
        return X_scaled


def main():
    """Main execution for testing."""
    from data_generator import CustomerDataGenerator
    
    # Generate sample data
    generator = CustomerDataGenerator(n_customers=1000)
    df = generator.generate()
    
    # Engineer features
    engineer = FeatureEngineer()
    df_processed, feature_names = engineer.prepare_features(df)
    
    print("\n" + "="*60)
    print("FEATURE ENGINEERING SUMMARY")
    print("="*60)
    print(f"\nTotal Features: {len(feature_names)}")
    print(f"\nFeature List:")
    for i, feat in enumerate(feature_names, 1):
        print(f"  {i}. {feat}")
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
