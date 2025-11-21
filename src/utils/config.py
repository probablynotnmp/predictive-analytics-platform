"""
Configuration settings for the predictive analytics platform.
"""

import os
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

# Data generation settings
RANDOM_SEED = 42
NUM_CUSTOMERS = 10000

# Feature engineering settings
RFM_QUANTILES = 5  # Number of quantiles for RFM scoring

# Model hyperparameters
CLV_MODEL_PARAMS = {
    'n_estimators': 200,
    'learning_rate': 0.05,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': RANDOM_SEED
}

CHURN_MODEL_PARAMS = {
    'n_estimators': 150,
    'max_depth': 10,
    'min_samples_split': 20,
    'min_samples_leaf': 10,
    'max_features': 'sqrt',
    'random_state': RANDOM_SEED,
    'n_jobs': -1
}

SEGMENTATION_PARAMS = {
    'n_clusters': 5,
    'random_state': RANDOM_SEED,
    'n_init': 10,
    'max_iter': 300
}

# API settings
API_HOST = "0.0.0.0"
API_PORT = 8000
API_TITLE = "Customer Analytics API"
API_VERSION = "1.0.0"

# Dashboard settings
DASHBOARD_HOST = "0.0.0.0"
DASHBOARD_PORT = 8050
DASHBOARD_DEBUG = True

# Model file names
CLV_MODEL_FILE = "clv_model.joblib"
CHURN_MODEL_FILE = "churn_model.joblib"
SEGMENTATION_MODEL_FILE = "segmentation_model.joblib"
FEATURE_SCALER_FILE = "feature_scaler.joblib"

# Feature columns
DEMOGRAPHIC_FEATURES = ['age', 'tenure_days', 'account_type']
RFM_FEATURES = ['recency', 'frequency', 'monetary']
ENGAGEMENT_FEATURES = ['email_open_rate', 'website_visits_per_month', 'support_tickets']
BEHAVIORAL_FEATURES = ['avg_purchase_amount', 'purchase_frequency', 'days_since_last_purchase']

ALL_FEATURES = DEMOGRAPHIC_FEATURES + RFM_FEATURES + ENGAGEMENT_FEATURES + BEHAVIORAL_FEATURES

# Target variables
CLV_TARGET = 'customer_lifetime_value'
CHURN_TARGET = 'churned'

# Business metrics
RETENTION_BASELINE = 0.75  # 75% baseline retention rate
CLV_BASELINE = 1000  # $1000 baseline CLV
CHURN_THRESHOLD = 0.5  # Probability threshold for churn classification
