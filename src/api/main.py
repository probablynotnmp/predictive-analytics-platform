"""
FastAPI application for customer analytics predictions.
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import pandas as pd
from typing import List
import time
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from api.schemas import (
    CustomerInput, CLVPredictionResponse, ChurnPredictionResponse,
    SegmentPredictionResponse, ComprehensivePredictionResponse,
    BatchPredictionRequest, BatchPredictionResponse,
    ModelInfo, HealthResponse, ErrorResponse
)
from api.model_loader import model_loader
from data.feature_engineering import FeatureEngineer
from utils.config import API_TITLE, API_VERSION
from utils.logger import logger

# Initialize FastAPI app
app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description="AI-powered customer analytics API for CLV prediction, churn detection, and segmentation"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize feature engineer
feature_engineer = FeatureEngineer()


def prepare_customer_features(customer: CustomerInput) -> np.ndarray:
    """
    Prepare customer features for prediction.
    
    Args:
        customer: Customer input data
        
    Returns:
        Scaled feature array
    """
    # Convert to DataFrame
    customer_dict = customer.dict()
    
    # Encode account type
    account_type_map = {'Basic': 0, 'Premium': 1, 'Enterprise': 2}
    customer_dict['account_type'] = account_type_map[customer_dict['account_type']]
    
    # Create additional engineered features
    df = pd.DataFrame([customer_dict])
    
    # Add engineered features
    df['rfm_score'] = (
        (5 - pd.qcut(df['recency'], q=5, labels=False, duplicates='drop').fillna(2)) +
        pd.qcut(df['frequency'], q=5, labels=False, duplicates='drop').fillna(2) +
        pd.qcut(df['monetary'], q=5, labels=False, duplicates='drop').fillna(2)
    )
    
    df['engagement_score'] = (
        df['email_open_rate'] * 0.3 +
        (df['website_visits_per_month'] / 50 * 100) * 0.5 +
        (1 - df['support_tickets'] / 10) * 100 * 0.2
    ).clip(0, 100)
    
    df['purchase_velocity'] = df['purchase_frequency'] / (df['tenure_days'] + 1)
    df['clv_per_year'] = (df['monetary'] / (df['tenure_days'] / 365 + 1))
    df['ltv_to_cac'] = df['monetary'] / 100  # Assuming $100 acquisition cost
    df['recent_purchase_ratio'] = (df['frequency'] * 3) / (df['purchase_frequency'] + 1)
    
    # Select features in correct order
    feature_names = model_loader.feature_names
    X = df[feature_names].values
    
    # Scale features
    X_scaled = model_loader.feature_scaler.transform(X)
    
    return X_scaled


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "message": "Customer Analytics API",
        "version": API_VERSION,
        "endpoints": {
            "health": "/health",
            "clv_prediction": "/predict/clv",
            "churn_prediction": "/predict/churn",
            "segment_prediction": "/predict/segment",
            "comprehensive": "/predict/comprehensive",
            "batch": "/batch/predict",
            "models_info": "/models/info"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    try:
        models_status = model_loader.get_models_status()
        
        return HealthResponse(
            status="healthy",
            models_loaded=models_status,
            version=API_VERSION
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e)
        )


@app.post("/predict/clv", response_model=CLVPredictionResponse, tags=["Predictions"])
async def predict_clv(customer: CustomerInput):
    """
    Predict Customer Lifetime Value.
    
    Args:
        customer: Customer data
        
    Returns:
        CLV prediction with segment
    """
    try:
        # Prepare features
        X = prepare_customer_features(customer)
        
        # Predict
        clv = float(model_loader.clv_model.predict(X)[0])
        
        # Determine segment
        if clv < 2000:
            segment = "Low Value"
        elif clv < 5000:
            segment = "Medium Value"
        elif clv < 10000:
            segment = "High Value"
        else:
            segment = "Premium Value"
        
        # Confidence (simplified)
        confidence = "high" if customer.tenure_days > 180 else "medium"
        
        return CLVPredictionResponse(
            customer_lifetime_value=round(clv, 2),
            clv_segment=segment,
            confidence=confidence
        )
    
    except Exception as e:
        logger.error(f"CLV prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/churn", response_model=ChurnPredictionResponse, tags=["Predictions"])
async def predict_churn(customer: CustomerInput):
    """
    Predict churn probability and risk.
    
    Args:
        customer: Customer data
        
    Returns:
        Churn prediction with risk category
    """
    try:
        # Prepare features
        X = prepare_customer_features(customer)
        
        # Predict
        churn_proba = float(model_loader.churn_model.predict_proba(X)[0, 1])
        churn_pred = int(churn_proba >= 0.5)
        
        # Risk category
        if churn_proba < 0.25:
            risk = "Low Risk"
        elif churn_proba < 0.5:
            risk = "Medium Risk"
        elif churn_proba < 0.75:
            risk = "High Risk"
        else:
            risk = "Critical Risk"
        
        # Top risk factors (from feature importance)
        importance = model_loader.churn_model.get_feature_importance()
        top_factors = importance.head(3)['feature'].tolist()
        
        return ChurnPredictionResponse(
            churn_probability=round(churn_proba, 4),
            churn_prediction=churn_pred,
            risk_category=risk,
            top_risk_factors=top_factors
        )
    
    except Exception as e:
        logger.error(f"Churn prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/segment", response_model=SegmentPredictionResponse, tags=["Predictions"])
async def predict_segment(customer: CustomerInput):
    """
    Predict customer segment.
    
    Args:
        customer: Customer data
        
    Returns:
        Segment prediction with recommendations
    """
    try:
        # Prepare features
        X = prepare_customer_features(customer)
        
        # Predict
        segment_id = int(model_loader.segmentation_model.predict(X)[0])
        
        # Get segment info
        segment_names = [
            'High-Value Engaged',
            'Loyal Regulars',
            'At-Risk',
            'New Prospects',
            'Dormant'
        ]
        segment_name = segment_names[segment_id] if segment_id < len(segment_names) else f'Segment {segment_id}'
        
        # Distance to center
        distances = model_loader.segmentation_model.model.transform(X)
        distance = float(distances[0, segment_id])
        
        # Recommendations
        recommendations = model_loader.segmentation_model.recommend_segment_actions(segment_id)
        
        return SegmentPredictionResponse(
            segment_id=segment_id,
            segment_name=segment_name,
            distance_to_center=round(distance, 4),
            recommended_actions=recommendations
        )
    
    except Exception as e:
        logger.error(f"Segmentation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/comprehensive", response_model=ComprehensivePredictionResponse, tags=["Predictions"])
async def predict_comprehensive(customer: CustomerInput):
    """
    Get comprehensive predictions from all models.
    
    Args:
        customer: Customer data
        
    Returns:
        Comprehensive prediction response
    """
    try:
        clv = await predict_clv(customer)
        churn = await predict_churn(customer)
        segment = await predict_segment(customer)
        
        return ComprehensivePredictionResponse(
            clv=clv,
            churn=churn,
            segment=segment
        )
    
    except Exception as e:
        logger.error(f"Comprehensive prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/batch/predict", response_model=BatchPredictionResponse, tags=["Batch"])
async def batch_predict(request: BatchPredictionRequest):
    """
    Batch prediction for multiple customers.
    
    Args:
        request: Batch prediction request
        
    Returns:
        Batch prediction results
    """
    try:
        start_time = time.time()
        
        predictions = []
        for customer in request.customers:
            pred = await predict_comprehensive(customer)
            predictions.append(pred)
        
        processing_time = time.time() - start_time
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_customers=len(predictions),
            processing_time_seconds=round(processing_time, 2)
        )
    
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.get("/models/info", response_model=List[ModelInfo], tags=["Models"])
async def get_models_info():
    """Get information about all loaded models."""
    try:
        models_info = []
        
        for model_name in ['clv', 'churn', 'segmentation']:
            info = model_loader.get_model_info(model_name)
            models_info.append(ModelInfo(**info))
        
        return models_info
    
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.on_event("startup")
async def startup_event():
    """Startup event handler."""
    logger.info("Starting Customer Analytics API...")
    logger.info(f"API Version: {API_VERSION}")
    
    # Preload models
    try:
        model_loader.preload_all_models()
        logger.info("All models loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load models: {str(e)}")
        logger.warning("Models will be loaded on first request")


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler."""
    logger.info("Shutting down Customer Analytics API...")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
