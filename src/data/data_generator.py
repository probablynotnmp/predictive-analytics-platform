"""
Synthetic customer data generator for demonstration purposes.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.config import RANDOM_SEED, NUM_CUSTOMERS, DATA_DIR
from utils.logger import logger


class CustomerDataGenerator:
    """Generate realistic synthetic customer data."""
    
    def __init__(self, n_customers: int = NUM_CUSTOMERS, random_seed: int = RANDOM_SEED):
        """
        Initialize the data generator.
        
        Args:
            n_customers: Number of customers to generate
            random_seed: Random seed for reproducibility
        """
        self.n_customers = n_customers
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
    def generate(self) -> pd.DataFrame:
        """
        Generate complete customer dataset with realistic patterns.
        
        Returns:
            DataFrame with customer data
        """
        logger.info(f"Generating {self.n_customers} synthetic customer records...")
        
        # Generate customer IDs
        customer_ids = [f"CUST_{i:06d}" for i in range(self.n_customers)]
        
        # Demographics
        ages = np.random.normal(40, 15, self.n_customers).clip(18, 80).astype(int)
        
        # Account types with different value profiles
        account_types = np.random.choice(
            ['Basic', 'Premium', 'Enterprise'],
            size=self.n_customers,
            p=[0.6, 0.3, 0.1]
        )
        
        # Tenure (days since joining)
        max_tenure = 1825  # 5 years
        tenure_days = np.random.exponential(365, self.n_customers).clip(1, max_tenure).astype(int)
        
        # Purchase behavior (influenced by account type and tenure)
        base_purchases = np.random.poisson(5, self.n_customers)
        type_multiplier = np.where(account_types == 'Basic', 1.0,
                                   np.where(account_types == 'Premium', 2.0, 3.5))
        total_purchases = (base_purchases * type_multiplier * (1 + tenure_days / max_tenure)).astype(int)
        
        # Purchase amounts
        base_amount = np.random.gamma(2, 50, self.n_customers)
        purchase_amounts = base_amount * type_multiplier
        avg_purchase_amount = purchase_amounts
        
        # Recency (days since last purchase)
        recency = np.random.exponential(30, self.n_customers).clip(0, 365).astype(int)
        
        # Frequency (purchases per month)
        frequency = (total_purchases / (tenure_days / 30)).clip(0, 20)
        
        # Monetary (total spend)
        monetary = total_purchases * avg_purchase_amount
        
        # Engagement metrics
        email_open_rate = np.random.beta(5, 2, self.n_customers) * 100  # Skewed towards higher rates
        website_visits_per_month = np.random.poisson(10, self.n_customers) * type_multiplier
        support_tickets = np.random.poisson(2, self.n_customers)
        
        # Calculate Customer Lifetime Value (CLV)
        # CLV = (avg_purchase_amount * frequency * 12) * (tenure_years) * retention_factor
        retention_factor = 1 - (recency / 365) * 0.3  # Recent customers more likely to stay
        clv = (avg_purchase_amount * frequency * 12 * (tenure_days / 365) * retention_factor).clip(0, 50000)
        
        # Churn prediction (influenced by recency, engagement, support tickets)
        churn_score = (
            (recency / 365) * 0.4 +
            (1 - email_open_rate / 100) * 0.3 +
            (support_tickets / 10) * 0.2 +
            np.random.random(self.n_customers) * 0.1
        )
        churned = (churn_score > 0.5).astype(int)
        
        # Days since last purchase
        days_since_last_purchase = recency
        
        # Purchase frequency (purchases per year)
        purchase_frequency = frequency * 12
        
        # Create DataFrame
        df = pd.DataFrame({
            'customer_id': customer_ids,
            'age': ages,
            'account_type': account_types,
            'tenure_days': tenure_days,
            'total_purchases': total_purchases,
            'avg_purchase_amount': avg_purchase_amount.round(2),
            'recency': recency,
            'frequency': frequency.round(2),
            'monetary': monetary.round(2),
            'email_open_rate': email_open_rate.round(2),
            'website_visits_per_month': website_visits_per_month.round(2),
            'support_tickets': support_tickets,
            'customer_lifetime_value': clv.round(2),
            'churned': churned,
            'days_since_last_purchase': days_since_last_purchase,
            'purchase_frequency': purchase_frequency.round(2)
        })
        
        logger.info(f"Generated {len(df)} customer records")
        logger.info(f"Churn rate: {churned.mean():.2%}")
        logger.info(f"Average CLV: ${clv.mean():.2f}")
        
        return df
    
    def save_data(self, df: pd.DataFrame, filename: str = "customer_data.csv"):
        """
        Save generated data to CSV file.
        
        Args:
            df: Customer DataFrame
            filename: Output filename
        """
        filepath = DATA_DIR / filename
        df.to_csv(filepath, index=False)
        logger.info(f"Saved customer data to {filepath}")
        
    def generate_and_save(self) -> pd.DataFrame:
        """
        Generate and save customer data.
        
        Returns:
            Generated DataFrame
        """
        df = self.generate()
        self.save_data(df)
        return df


def main():
    """Main execution function."""
    generator = CustomerDataGenerator()
    df = generator.generate_and_save()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("CUSTOMER DATA SUMMARY")
    print("="*60)
    print(f"\nTotal Customers: {len(df):,}")
    print(f"\nAccount Type Distribution:")
    print(df['account_type'].value_counts())
    print(f"\nChurn Rate: {df['churned'].mean():.2%}")
    print(f"\nCLV Statistics:")
    print(df['customer_lifetime_value'].describe())
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
