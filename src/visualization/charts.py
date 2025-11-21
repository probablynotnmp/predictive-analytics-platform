"""
Reusable chart components for visualizations.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import List, Dict, Any


def create_clv_distribution_chart(clv_values: np.ndarray, title: str = "CLV Distribution") -> go.Figure:
    """
    Create CLV distribution histogram.
    
    Args:
        clv_values: Array of CLV values
        title: Chart title
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=clv_values,
        nbinsx=50,
        name='CLV',
        marker_color='#3498db',
        opacity=0.75
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Customer Lifetime Value ($)",
        yaxis_title="Number of Customers",
        template="plotly_white",
        hovermode='x'
    )
    
    return fig


def create_churn_risk_pie_chart(risk_categories: pd.Series, title: str = "Churn Risk Distribution") -> go.Figure:
    """
    Create pie chart for churn risk categories.
    
    Args:
        risk_categories: Series of risk categories
        title: Chart title
        
    Returns:
        Plotly figure
    """
    risk_counts = risk_categories.value_counts()
    
    colors = {
        'Low Risk': '#2ecc71',
        'Medium Risk': '#f39c12',
        'High Risk': '#e67e22',
        'Critical Risk': '#e74c3c'
    }
    
    fig = go.Figure(data=[go.Pie(
        labels=risk_counts.index,
        values=risk_counts.values,
        marker=dict(colors=[colors.get(cat, '#95a5a6') for cat in risk_counts.index]),
        hole=0.4
    )])
    
    fig.update_layout(
        title=title,
        template="plotly_white"
    )
    
    return fig


def create_segment_bar_chart(segment_data: pd.DataFrame, title: str = "Customer Segments") -> go.Figure:
    """
    Create bar chart for customer segments.
    
    Args:
        segment_data: DataFrame with segment information
        title: Chart title
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=segment_data['segment_name'],
        y=segment_data['count'],
        marker_color='#9b59b6',
        text=segment_data['percentage'].apply(lambda x: f'{x:.1f}%'),
        textposition='auto'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Segment",
        yaxis_title="Number of Customers",
        template="plotly_white"
    )
    
    return fig


def create_feature_importance_chart(importance_df: pd.DataFrame, 
                                   title: str = "Feature Importance",
                                   top_n: int = 10) -> go.Figure:
    """
    Create horizontal bar chart for feature importance.
    
    Args:
        importance_df: DataFrame with feature and importance columns
        title: Chart title
        top_n: Number of top features to show
        
    Returns:
        Plotly figure
    """
    top_features = importance_df.head(top_n).sort_values('importance')
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=top_features['importance'],
        y=top_features['feature'],
        orientation='h',
        marker_color='#1abc9c'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Importance Score",
        yaxis_title="Feature",
        template="plotly_white",
        height=400
    )
    
    return fig


def create_clv_vs_churn_scatter(clv_values: np.ndarray, 
                                churn_proba: np.ndarray,
                                title: str = "CLV vs Churn Risk") -> go.Figure:
    """
    Create scatter plot of CLV vs churn probability.
    
    Args:
        clv_values: Array of CLV values
        churn_proba: Array of churn probabilities
        title: Chart title
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=clv_values,
        y=churn_proba,
        mode='markers',
        marker=dict(
            size=5,
            color=churn_proba,
            colorscale='RdYlGn_r',
            showscale=True,
            colorbar=dict(title="Churn<br>Probability")
        ),
        text=[f'CLV: ${v:.2f}<br>Churn: {p:.2%}' for v, p in zip(clv_values, churn_proba)],
        hovertemplate='%{text}<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Customer Lifetime Value ($)",
        yaxis_title="Churn Probability",
        template="plotly_white"
    )
    
    return fig


def create_kpi_card(value: float, title: str, prefix: str = "", suffix: str = "", 
                   delta: float = None, delta_suffix: str = "%") -> go.Figure:
    """
    Create KPI indicator card.
    
    Args:
        value: KPI value
        title: KPI title
        prefix: Value prefix (e.g., "$")
        suffix: Value suffix (e.g., "%")
        delta: Change value
        delta_suffix: Delta suffix
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    indicator_args = {
        'mode': "number+delta" if delta is not None else "number",
        'value': value,
        'title': {'text': title},
        'number': {'prefix': prefix, 'suffix': suffix}
    }
    
    if delta is not None:
        indicator_args['delta'] = {
            'reference': value - delta,
            'suffix': delta_suffix,
            'relative': False
        }
    
    fig.add_trace(go.Indicator(**indicator_args))
    
    fig.update_layout(
        height=200,
        template="plotly_white"
    )
    
    return fig


def create_time_series_chart(dates: List, values: List, 
                            title: str = "Trend Over Time",
                            y_label: str = "Value") -> go.Figure:
    """
    Create time series line chart.
    
    Args:
        dates: List of dates
        values: List of values
        title: Chart title
        y_label: Y-axis label
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=values,
        mode='lines+markers',
        line=dict(color='#3498db', width=2),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=y_label,
        template="plotly_white",
        hovermode='x unified'
    )
    
    return fig


def create_correlation_heatmap(correlation_matrix: pd.DataFrame, 
                               title: str = "Feature Correlation") -> go.Figure:
    """
    Create correlation heatmap.
    
    Args:
        correlation_matrix: Correlation matrix DataFrame
        title: Chart title
        
    Returns:
        Plotly figure
    """
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.index,
        colorscale='RdBu',
        zmid=0,
        text=correlation_matrix.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=600
    )
    
    return fig


def create_segment_profile_radar(segment_profiles: Dict[str, float], 
                                 title: str = "Segment Profile") -> go.Figure:
    """
    Create radar chart for segment profile.
    
    Args:
        segment_profiles: Dictionary of feature: value pairs
        title: Chart title
        
    Returns:
        Plotly figure
    """
    features = list(segment_profiles.keys())
    values = list(segment_profiles.values())
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=features,
        fill='toself',
        marker_color='#9b59b6'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True)
        ),
        title=title,
        template="plotly_white"
    )
    
    return fig
