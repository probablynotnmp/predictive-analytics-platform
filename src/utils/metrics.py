"""
Custom metrics for model evaluation and business impact analysis.
"""

import numpy as np
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from typing import Dict, Any


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive regression metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary of metric names and values
    """
    return {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    }


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                          y_pred_proba: np.ndarray = None) -> Dict[str, float]:
    """
    Calculate comprehensive classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (optional)
        
    Returns:
        Dictionary of metric names and values
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }
    
    if y_pred_proba is not None:
        metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba)
    
    # Confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['true_negatives'] = int(tn)
    metrics['false_positives'] = int(fp)
    metrics['false_negatives'] = int(fn)
    metrics['true_positives'] = int(tp)
    
    return metrics


def business_impact_metrics(y_true_clv: np.ndarray, y_pred_clv: np.ndarray,
                           y_true_churn: np.ndarray, y_pred_churn: np.ndarray,
                           baseline_retention: float = 0.75) -> Dict[str, Any]:
    """
    Calculate business-focused metrics for marketing impact.
    
    Args:
        y_true_clv: True CLV values
        y_pred_clv: Predicted CLV values
        y_true_churn: True churn labels
        y_pred_churn: Predicted churn labels
        baseline_retention: Baseline retention rate
        
    Returns:
        Dictionary of business metrics
    """
    # CLV prediction accuracy
    clv_accuracy = 1 - (mean_absolute_error(y_true_clv, y_pred_clv) / np.mean(y_true_clv))
    
    # Retention improvement
    actual_retention = 1 - np.mean(y_true_churn)
    predicted_retention = 1 - np.mean(y_pred_churn)
    retention_lift = (predicted_retention - baseline_retention) / baseline_retention
    
    # High-value customer identification
    high_value_threshold = np.percentile(y_true_clv, 75)
    high_value_precision = precision_score(
        y_true_clv >= high_value_threshold,
        y_pred_clv >= high_value_threshold,
        zero_division=0
    )
    
    # Churn prevention potential
    at_risk_customers = np.sum(y_pred_churn == 1)
    potential_saved_revenue = np.sum(y_true_clv[y_pred_churn == 1])
    
    return {
        'clv_prediction_accuracy': float(clv_accuracy),
        'actual_retention_rate': float(actual_retention),
        'predicted_retention_rate': float(predicted_retention),
        'retention_lift_pct': float(retention_lift * 100),
        'high_value_identification_precision': float(high_value_precision),
        'at_risk_customers_count': int(at_risk_customers),
        'potential_saved_revenue': float(potential_saved_revenue),
        'avg_clv_prediction_error': float(mean_absolute_error(y_true_clv, y_pred_clv))
    }


def segment_quality_metrics(X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """
    Calculate clustering quality metrics.
    
    Args:
        X: Feature matrix
        labels: Cluster labels
        
    Returns:
        Dictionary of clustering metrics
    """
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    
    return {
        'silhouette_score': silhouette_score(X, labels),
        'calinski_harabasz_score': calinski_harabasz_score(X, labels),
        'davies_bouldin_score': davies_bouldin_score(X, labels),
        'n_clusters': len(np.unique(labels))
    }
