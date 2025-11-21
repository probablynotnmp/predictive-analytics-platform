"""
Interactive Plotly Dash dashboard for customer analytics.
"""

import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from data.data_generator import CustomerDataGenerator
from data.feature_engineering import FeatureEngineer
from models.clv_predictor import CLVPredictor
from models.churn_predictor import ChurnPredictor
from models.segmentation_model import CustomerSegmentation
from visualization.charts import (
    create_clv_distribution_chart, create_churn_risk_pie_chart,
    create_segment_bar_chart, create_feature_importance_chart,
    create_clv_vs_churn_scatter, create_kpi_card
)
from utils.config import DASHBOARD_HOST, DASHBOARD_PORT, DASHBOARD_DEBUG, MODELS_DIR, DATA_DIR
from utils.logger import logger
import joblib
import json

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Customer Analytics Dashboard"

# Load data and models
def load_dashboard_data():
    """Load customer data and predictions."""
    try:
        # Load customer data
        data_path = DATA_DIR / "customer_data.csv"
        if data_path.exists():
            df = pd.read_csv(data_path)
            logger.info(f"Loaded {len(df)} customer records")
        else:
            logger.warning("Customer data not found, generating sample data...")
            generator = CustomerDataGenerator(n_customers=1000)
            df = generator.generate_and_save()
        
        # Load models
        clv_model = CLVPredictor()
        churn_model = ChurnPredictor()
        seg_model = CustomerSegmentation()
        
        clv_model.load('clv_model.joblib')
        churn_model.load('churn_model.joblib')
        seg_model.load('segmentation_model.joblib')
        
        # Load feature engineer
        engineer = FeatureEngineer()
        df_processed, feature_names = engineer.prepare_features(df)
        
        # Load scaler
        scaler = joblib.load(MODELS_DIR / 'feature_scaler.joblib')
        X = df_processed[feature_names]
        X_scaled = scaler.transform(X)
        
        # Make predictions
        df['predicted_clv'] = clv_model.predict(X_scaled)
        df['churn_probability'] = churn_model.predict_proba(X_scaled)[:, 1]
        df['segment_id'] = seg_model.predict(X_scaled)
        
        # Add risk categories
        df['risk_category'] = pd.cut(
            df['churn_probability'],
            bins=[0, 0.25, 0.5, 0.75, 1.0],
            labels=['Low Risk', 'Medium Risk', 'High Risk', 'Critical Risk']
        )
        
        # Add segment names
        segment_names = ['High-Value Engaged', 'Loyal Regulars', 'At-Risk', 'New Prospects', 'Dormant']
        df['segment_name'] = df['segment_id'].apply(
            lambda x: segment_names[x] if x < len(segment_names) else f'Segment {x}'
        )
        
        return df, clv_model, churn_model, seg_model
        
    except Exception as e:
        logger.error(f"Error loading dashboard data: {str(e)}")
        raise

# Load data
df, clv_model, churn_model, seg_model = load_dashboard_data()

# Calculate KPIs
total_customers = len(df)
avg_clv = df['predicted_clv'].mean()
churn_rate = (df['churn_probability'] > 0.5).mean()
high_value_customers = (df['predicted_clv'] > df['predicted_clv'].quantile(0.75)).sum()

# Dashboard layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("🎯 Customer Analytics Dashboard", className="text-center mb-4 mt-4"),
            html.P("AI-Powered Predictive Analytics Platform", className="text-center text-muted")
        ])
    ]),
    
    # KPI Cards
    dbc.Row([
        dbc.Col([
            dcc.Graph(
                figure=create_kpi_card(
                    total_customers,
                    "Total Customers",
                    suffix=""
                )
            )
        ], width=3),
        dbc.Col([
            dcc.Graph(
                figure=create_kpi_card(
                    avg_clv,
                    "Average CLV",
                    prefix="$"
                )
            )
        ], width=3),
        dbc.Col([
            dcc.Graph(
                figure=create_kpi_card(
                    churn_rate * 100,
                    "Churn Rate",
                    suffix="%"
                )
            )
        ], width=3),
        dbc.Col([
            dcc.Graph(
                figure=create_kpi_card(
                    high_value_customers,
                    "High-Value Customers",
                    suffix=""
                )
            )
        ], width=3)
    ], className="mb-4"),
    
    # Main Charts Row 1
    dbc.Row([
        dbc.Col([
            dcc.Graph(
                id='clv-distribution',
                figure=create_clv_distribution_chart(
                    df['predicted_clv'].values,
                    "Customer Lifetime Value Distribution"
                )
            )
        ], width=6),
        dbc.Col([
            dcc.Graph(
                id='churn-risk-pie',
                figure=create_churn_risk_pie_chart(
                    df['risk_category'],
                    "Churn Risk Distribution"
                )
            )
        ], width=6)
    ], className="mb-4"),
    
    # Main Charts Row 2
    dbc.Row([
        dbc.Col([
            dcc.Graph(
                id='segment-bar',
                figure=create_segment_bar_chart(
                    pd.DataFrame({
                        'segment_name': df['segment_name'].value_counts().index,
                        'count': df['segment_name'].value_counts().values,
                        'percentage': df['segment_name'].value_counts(normalize=True).values * 100
                    }),
                    "Customer Segmentation"
                )
            )
        ], width=6),
        dbc.Col([
            dcc.Graph(
                id='clv-churn-scatter',
                figure=create_clv_vs_churn_scatter(
                    df['predicted_clv'].values,
                    df['churn_probability'].values,
                    "CLV vs Churn Risk Analysis"
                )
            )
        ], width=6)
    ], className="mb-4"),
    
    # Feature Importance Row
    dbc.Row([
        dbc.Col([
            dcc.Graph(
                id='clv-feature-importance',
                figure=create_feature_importance_chart(
                    clv_model.get_feature_importance(),
                    "CLV Prediction - Top Features",
                    top_n=10
                )
            )
        ], width=6),
        dbc.Col([
            dcc.Graph(
                id='churn-feature-importance',
                figure=create_feature_importance_chart(
                    churn_model.get_feature_importance(),
                    "Churn Prediction - Top Risk Factors",
                    top_n=10
                )
            )
        ], width=6)
    ], className="mb-4"),
    
    # Data Table
    dbc.Row([
        dbc.Col([
            html.H4("Customer Data Sample", className="mb-3"),
            html.Div([
                html.Table([
                    html.Thead(
                        html.Tr([
                            html.Th("Customer ID"),
                            html.Th("Account Type"),
                            html.Th("Predicted CLV"),
                            html.Th("Churn Risk"),
                            html.Th("Segment")
                        ])
                    ),
                    html.Tbody([
                        html.Tr([
                            html.Td(row['customer_id']),
                            html.Td(row['account_type']),
                            html.Td(f"${row['predicted_clv']:.2f}"),
                            html.Td(row['risk_category']),
                            html.Td(row['segment_name'])
                        ]) for _, row in df.head(10).iterrows()
                    ])
                ], className="table table-striped table-hover")
            ])
        ])
    ], className="mb-4"),
    
    # Footer
    dbc.Row([
        dbc.Col([
            html.Hr(),
            html.P("© 2025 Customer Analytics Platform | Powered by AI & Machine Learning",
                  className="text-center text-muted")
        ])
    ])
], fluid=True)


def main():
    """Run the dashboard."""
    logger.info(f"Starting dashboard on {DASHBOARD_HOST}:{DASHBOARD_PORT}")
    app.run_server(host=DASHBOARD_HOST, port=DASHBOARD_PORT, debug=DASHBOARD_DEBUG)


if __name__ == "__main__":
    main()
