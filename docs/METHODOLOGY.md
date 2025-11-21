# 📐 Methodology & Data Science Approach

## 1. Problem Statement
In the modern digital economy, data is abundant but insights are scarce. The goal of this project was to move beyond simple descriptive analytics ("what happened") to predictive analytics ("what will happen") and prescriptive analytics ("what should we do").

## 2. Data Generation & Synthesis
Since real-world customer data is sensitive (PII), I built a robust **Synthetic Data Generator** (`src/data/data_generator.py`) to mimic realistic customer behaviors.

I ensured the data wasn't just random noise by implementing:
*   **Pareto Distributions**: To simulate the "80/20 rule" where a small % of customers generate most revenue.
*   **Correlation Injection**: Ensuring logical relationships (e.g., higher `tenure` + high `engagement` = lower `churn_risk`).
*   **Seasonality**: Adding time-based patterns to purchase histories.

## 3. Feature Engineering Strategy
Raw data is rarely model-ready. I implemented a feature engineering pipeline (`src/data/feature_engineering.py`) focusing on:

### RFM Analysis (Recency, Frequency, Monetary)
I didn't just use raw values; I converted them into **quantile scores (1-5)**. This normalizes the data and makes the model more robust to outliers.

### Behavioral Features
*   `purchase_velocity`: How fast are they buying relative to their tenure?
*   `engagement_score`: A weighted index of email opens, site visits, and support interactions.
*   `ltv_to_cac`: A derived metric estimating profitability.

## 4. Model Selection & Validation

### CLV Prediction (Regression)
*   **Model**: XGBoost Regressor
*   **Rationale**: Customer value data is typically non-normal (long tail). Linear regression fails here. Tree-based boosting methods like XGBoost handle these non-linearities and interactions (e.g., Age vs. Spend) exceptionally well.
*   **Metric**: RMSE (Root Mean Squared Error) to penalize large errors.

### Churn Prediction (Classification)
*   **Model**: Random Forest Classifier
*   **Rationale**: Churn is complex. Random Forest provides excellent out-of-the-box performance and, crucially, **feature importance**. It allows us to tell the business *why* customers are leaving (e.g., "High support tickets" is the #1 predictor).
*   **Metric**: ROC-AUC (Area Under Curve) because classes are likely imbalanced (fewer churners than retainers).

### Segmentation (Clustering)
*   **Model**: K-Means
*   **Rationale**: Efficient and interpretable. I used the **Elbow Method** (pre-analysis) to determine that `k=5` was the optimal number of clusters for this dataset.

## 5. Production Considerations
I designed the code to be modular. The `ModelLoader` class uses **Singleton patterns** and **Lazy Loading** to ensure the API starts up quickly and doesn't waste memory until a prediction is actually requested.
