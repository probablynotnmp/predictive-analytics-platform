# 📘 Comprehensive Project Instructions

Welcome to **Nexus**, the AI-Powered Predictive Analytics Platform. This document provides step-by-step instructions for setting up, running, and deploying the project.

## 📋 Table of Contents
1. [System Requirements](#system-requirements)
2. [Installation & Setup](#installation--setup)
3. [Model Training Pipeline](#model-training-pipeline)
4. [Running the Application](#running-the-application)
5. [Using the API](#using-the-api)
6. [Using the Dashboard](#using-the-dashboard)
7. [Testing & Verification](#testing--verification)
8. [Docker Deployment](#docker-deployment)

---

## 1. System Requirements

Before you begin, ensure you have the following installed:
*   **Python**: Version 3.10 or higher
*   **pip**: Python package manager
*   **Git**: Version control system
*   **Docker** (Optional): For containerized deployment

## 2. Installation & Setup

### Clone the Repository
```bash
git clone https://github.com/probablynotnmp/predictive-analytics-platform.git
cd predictive-analytics-platform
```

### Create a Virtual Environment (Recommended)
```bash
# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
.\venv\Scripts\activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 3. Model Training Pipeline

The platform comes with a built-in pipeline to generate synthetic data and train the machine learning models.

**Run the training script:**
```bash
python src/models/model_trainer.py
```

**What this does:**
1.  **Generates Data**: Creates 10,000 synthetic customer records in `data/`.
2.  **Engineers Features**: Calculates RFM scores, engagement metrics, and more.
3.  **Trains Models**:
    *   `CLVPredictor` (XGBoost)
    *   `ChurnPredictor` (Random Forest)
    *   `CustomerSegmentation` (K-Means)
4.  **Saves Artifacts**: Models are saved to `models/` as `.joblib` files.
5.  **Generates Report**: A performance report is saved to `models/training_report.json`.

---

## 4. Running the Application

You need to run two separate processes: the API backend and the Dashboard frontend.

### Step 1: Start the API Server
Open a terminal and run:
```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```
*   **Status**: The API is now running at `http://localhost:8000`.
*   **Docs**: Swagger UI is available at `http://localhost:8000/docs`.

### Step 2: Start the Dashboard
Open a **new** terminal window (keep the API running!) and run:
```bash
python src/visualization/dashboard.py
```
*   **Status**: The Dashboard is now running at `http://localhost:8050`.

---

## 5. Using the API

You can interact with the API using `curl`, Postman, or the built-in Swagger UI.

### Example: Predict for a Single Customer
**Endpoint**: `POST /predict/comprehensive`

**Request Body**:
```json
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
```

**Response**:
```json
{
  "clv": {
    "customer_lifetime_value": 6250.50,
    "clv_segment": "High Value"
  },
  "churn": {
    "churn_probability": 0.12,
    "risk_category": "Low Risk"
  },
  "segment": {
    "segment_name": "Loyal Regulars"
  }
}
```

---

## 6. Using the Dashboard

Navigate to `http://localhost:8050` in your browser.

*   **KPI Cards**: View high-level metrics like Average CLV and Churn Rate.
*   **Charts**: Explore the "CLV Distribution" and "Churn Risk" pie charts.
*   **Segmentation**: See how your customer base is divided into clusters.
*   **Feature Importance**: Understand which factors (e.g., Recency, Frequency) drive the model predictions.

---

## 7. Testing & Verification

To ensure everything is working correctly, run the test suite:

```bash
# Run all tests
pytest tests/ -v

# Run specific tests
pytest tests/test_models.py
pytest tests/test_api.py
```

---

## 8. Docker Deployment

To run the entire platform in a container (simulating a production environment):

### Build the Image
```bash
docker build -t customer-analytics .
```

### Run the Container
```bash
docker run -p 8000:8000 -p 8050:8050 customer-analytics
```

The API will be available at port `8000` and the Dashboard at port `8050`.

---

**Enjoy building with Nexus!** 🚀
