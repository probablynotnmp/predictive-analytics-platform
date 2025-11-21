"""
Pydantic schemas for API request/response validation.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


class AccountType(str, Enum):
    """Account type enumeration."""
    BASIC = "Basic"
    PREMIUM = "Premium"
    ENTERPRISE = "Enterprise"


class CustomerInput(BaseModel):
    """Input schema for customer data."""
    age: int = Field(..., ge=18, le=100, description="Customer age")
    account_type: AccountType = Field(..., description="Account type")
    tenure_days: int = Field(..., ge=0, description="Days since account creation")
    recency: int = Field(..., ge=0, description="Days since last purchase")
    frequency: float = Field(..., ge=0, description="Purchases per month")
    monetary: float = Field(..., ge=0, description="Total spend")
    email_open_rate: float = Field(..., ge=0, le=100, description="Email open rate percentage")
    website_visits_per_month: float = Field(..., ge=0, description="Website visits per month")
    support_tickets: int = Field(..., ge=0, description="Number of support tickets")
    avg_purchase_amount: float = Field(..., ge=0, description="Average purchase amount")
    purchase_frequency: float = Field(..., ge=0, description="Annual purchase frequency")
    days_since_last_purchase: int = Field(..., ge=0, description="Days since last purchase")
    
    class Config:
        schema_extra = {
            "example": {
                "age": 35,
                "account_type": "Premium",
                "tenure_days": 730,
                "recency": 15,
                "frequency": 2.5,
                "monetary": 5000.0,
                "email_open_rate": 75.0,
                "website_visits_per_month": 12.0,
                "support_tickets": 1,
                "avg_purchase_amount": 150.0,
                "purchase_frequency": 30.0,
                "days_since_last_purchase": 15
            }
        }


class CLVPredictionResponse(BaseModel):
    """Response schema for CLV prediction."""
    customer_lifetime_value: float = Field(..., description="Predicted CLV")
    clv_segment: str = Field(..., description="Value segment")
    confidence: str = Field(..., description="Prediction confidence level")
    
    class Config:
        schema_extra = {
            "example": {
                "customer_lifetime_value": 8500.50,
                "clv_segment": "High Value",
                "confidence": "high"
            }
        }


class ChurnPredictionResponse(BaseModel):
    """Response schema for churn prediction."""
    churn_probability: float = Field(..., ge=0, le=1, description="Churn probability")
    churn_prediction: int = Field(..., ge=0, le=1, description="Binary churn prediction")
    risk_category: str = Field(..., description="Risk category")
    top_risk_factors: List[str] = Field(..., description="Top factors contributing to churn risk")
    
    class Config:
        schema_extra = {
            "example": {
                "churn_probability": 0.35,
                "churn_prediction": 0,
                "risk_category": "Medium Risk",
                "top_risk_factors": ["recency", "engagement_score", "support_tickets"]
            }
        }


class SegmentPredictionResponse(BaseModel):
    """Response schema for segment prediction."""
    segment_id: int = Field(..., description="Segment ID")
    segment_name: str = Field(..., description="Segment name")
    distance_to_center: float = Field(..., description="Distance to segment center")
    recommended_actions: List[str] = Field(..., description="Recommended marketing actions")
    
    class Config:
        schema_extra = {
            "example": {
                "segment_id": 1,
                "segment_name": "Loyal Regulars",
                "distance_to_center": 0.45,
                "recommended_actions": [
                    "Send personalized product recommendations",
                    "Offer bundle deals"
                ]
            }
        }


class ComprehensivePredictionResponse(BaseModel):
    """Comprehensive prediction response with all models."""
    clv: CLVPredictionResponse
    churn: ChurnPredictionResponse
    segment: SegmentPredictionResponse
    
    class Config:
        schema_extra = {
            "example": {
                "clv": {
                    "customer_lifetime_value": 8500.50,
                    "clv_segment": "High Value",
                    "confidence": "high"
                },
                "churn": {
                    "churn_probability": 0.35,
                    "churn_prediction": 0,
                    "risk_category": "Medium Risk",
                    "top_risk_factors": ["recency", "engagement_score"]
                },
                "segment": {
                    "segment_id": 1,
                    "segment_name": "Loyal Regulars",
                    "distance_to_center": 0.45,
                    "recommended_actions": ["Send personalized recommendations"]
                }
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request schema for batch predictions."""
    customers: List[CustomerInput] = Field(..., description="List of customers")
    
    class Config:
        schema_extra = {
            "example": {
                "customers": [
                    {
                        "age": 35,
                        "account_type": "Premium",
                        "tenure_days": 730,
                        "recency": 15,
                        "frequency": 2.5,
                        "monetary": 5000.0,
                        "email_open_rate": 75.0,
                        "website_visits_per_month": 12.0,
                        "support_tickets": 1,
                        "avg_purchase_amount": 150.0,
                        "purchase_frequency": 30.0,
                        "days_since_last_purchase": 15
                    }
                ]
            }
        }


class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions."""
    predictions: List[ComprehensivePredictionResponse]
    total_customers: int
    processing_time_seconds: float
    
    class Config:
        schema_extra = {
            "example": {
                "predictions": [],
                "total_customers": 100,
                "processing_time_seconds": 1.25
            }
        }


class ModelInfo(BaseModel):
    """Model information schema."""
    model_name: str
    is_trained: bool
    n_features: int
    metrics: Dict[str, Any]
    
    class Config:
        schema_extra = {
            "example": {
                "model_name": "CLV Predictor",
                "is_trained": True,
                "n_features": 18,
                "metrics": {
                    "rmse": 450.25,
                    "r2": 0.85
                }
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    models_loaded: Dict[str, bool] = Field(..., description="Model loading status")
    version: str = Field(..., description="API version")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "models_loaded": {
                    "clv": True,
                    "churn": True,
                    "segmentation": True
                },
                "version": "1.0.0"
            }
        }


class ErrorResponse(BaseModel):
    """Error response schema."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    
    class Config:
        schema_extra = {
            "example": {
                "error": "Model not found",
                "detail": "CLV model file does not exist"
            }
        }
