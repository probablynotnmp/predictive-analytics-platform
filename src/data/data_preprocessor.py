"""
Data preprocessing and validation module.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from utils.config import RANDOM_SEED, CLV_TARGET, CHURN_TARGET
from utils.logger import logger


class DataPreprocessor:
    """Data preprocessing and validation for customer analytics."""
    
    def __init__(self, test_size: float = 0.2, random_state: int = RANDOM_SEED):
        """
        Initialize preprocessor.
        
        Args:
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
        """
        self.test_size = test_size
        self.random_state = random_state
        
    def validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean customer data.
        
        Args:
            df: Customer DataFrame
            
        Returns:
            Validated DataFrame
        """
        logger.info("Validating customer data...")
        
        initial_rows = len(df)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['customer_id'])
        
        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        # Handle outliers in CLV (cap at 99th percentile)
        if CLV_TARGET in df.columns:
            clv_99 = df[CLV_TARGET].quantile(0.99)
            df[CLV_TARGET] = df[CLV_TARGET].clip(upper=clv_99)
        
        # Ensure non-negative values for certain columns
        non_negative_cols = ['tenure_days', 'total_purchases', 'monetary', 
                            'customer_lifetime_value', 'avg_purchase_amount']
        for col in non_negative_cols:
            if col in df.columns:
                df[col] = df[col].clip(lower=0)
        
        final_rows = len(df)
        logger.info(f"Validation complete: {initial_rows} -> {final_rows} rows")
        
        return df
    
    def split_data(self, df: pd.DataFrame, target_col: str, 
                   feature_cols: list) -> Tuple[pd.DataFrame, pd.DataFrame, 
                                                 pd.Series, pd.Series]:
        """
        Split data into training and testing sets.
        
        Args:
            df: Customer DataFrame
            target_col: Target column name
            feature_cols: List of feature column names
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        X = df[feature_cols]
        y = df[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y if target_col == CHURN_TARGET else None
        )
        
        logger.info(f"Split data: {len(X_train)} train, {len(X_test)} test samples")
        
        return X_train, X_test, y_train, y_test
    
    def prepare_for_modeling(self, df: pd.DataFrame, target_col: str,
                            feature_cols: list) -> Tuple:
        """
        Complete preprocessing pipeline for modeling.
        
        Args:
            df: Customer DataFrame
            target_col: Target column name
            feature_cols: List of feature column names
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Validate data
        df = self.validate_data(df)
        
        # Split data
        return self.split_data(df, target_col, feature_cols)


def main():
    """Main execution for testing."""
    from data_generator import CustomerDataGenerator
    from feature_engineering import FeatureEngineer
    
    # Generate and prepare data
    generator = CustomerDataGenerator(n_customers=1000)
    df = generator.generate()
    
    engineer = FeatureEngineer()
    df_processed, feature_names = engineer.prepare_features(df)
    
    # Preprocess
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.prepare_for_modeling(
        df_processed, CLV_TARGET, feature_names
    )
    
    print("\n" + "="*60)
    print("DATA PREPROCESSING SUMMARY")
    print("="*60)
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Features: {len(feature_names)}")
    print(f"\nTarget statistics (CLV):")
    print(f"  Train mean: ${y_train.mean():.2f}")
    print(f"  Test mean: ${y_test.mean():.2f}")
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
