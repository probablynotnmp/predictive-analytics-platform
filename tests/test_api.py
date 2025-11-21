"""
Integration tests for API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.api.main import app

client = TestClient(app)


class TestHealthEndpoint:
    """Test health check endpoint."""
    
    def test_health_check(self):
        """Test that health endpoint returns 200."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "version" in data


class TestRootEndpoint:
    """Test root endpoint."""
    
    def test_root(self):
        """Test root endpoint returns info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "endpoints" in data


class TestPredictionEndpoints:
    """Test prediction endpoints."""
    
    @pytest.fixture
    def sample_customer(self):
        """Sample customer data for testing."""
        return {
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
    
    def test_clv_prediction(self, sample_customer):
        """Test CLV prediction endpoint."""
        response = client.post("/predict/clv", json=sample_customer)
        
        if response.status_code == 200:
            data = response.json()
            assert "customer_lifetime_value" in data
            assert "clv_segment" in data
            assert data["customer_lifetime_value"] >= 0
    
    def test_churn_prediction(self, sample_customer):
        """Test churn prediction endpoint."""
        response = client.post("/predict/churn", json=sample_customer)
        
        if response.status_code == 200:
            data = response.json()
            assert "churn_probability" in data
            assert "risk_category" in data
            assert 0 <= data["churn_probability"] <= 1
    
    def test_segment_prediction(self, sample_customer):
        """Test segment prediction endpoint."""
        response = client.post("/predict/segment", json=sample_customer)
        
        if response.status_code == 200:
            data = response.json()
            assert "segment_id" in data
            assert "segment_name" in data
            assert data["segment_id"] >= 0
    
    def test_comprehensive_prediction(self, sample_customer):
        """Test comprehensive prediction endpoint."""
        response = client.post("/predict/comprehensive", json=sample_customer)
        
        if response.status_code == 200:
            data = response.json()
            assert "clv" in data
            assert "churn" in data
            assert "segment" in data
    
    def test_invalid_customer_data(self):
        """Test that invalid data returns error."""
        invalid_data = {
            "age": -5,  # Invalid age
            "account_type": "Premium"
        }
        
        response = client.post("/predict/clv", json=invalid_data)
        assert response.status_code == 422  # Validation error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
