# Grocery Sales Forecasting using Machine Learning

## 1. Project Overview

This project implements an end-to-end machine learning pipeline to forecast daily grocery sales for different store–item combinations using historical time-series data.

The objective is to build a scalable, modular forecasting system that:

* Handles large retail datasets efficiently
* Applies structured time-series feature engineering
* Trains and compares multiple machine learning models
* Uses time-based validation to prevent data leakage
* Selects the best-performing model based on evaluation metrics

The dataset is based on the Corporación Favorita Grocery Sales Forecasting competition.

---

## 2. Business Problem

Retail businesses require accurate sales forecasts to:

* Optimize inventory management
* Reduce stockouts
* Prevent overstocking
* Improve supply chain efficiency
* Plan promotional campaigns

This project predicts:

**Target Variable:**
`unit_sales` – Daily sales quantity for each store and item.

---

## 3. Dataset Description

The project uses the following datasets:

| File Name           | Description                                      |
| ------------------- | ------------------------------------------------ |
| train.csv           | Historical sales data (includes target variable) |
| test.csv            | Future dates for prediction                      |
| stores.csv          | Store metadata (city, state, type, cluster)      |
| items.csv           | Product metadata (family, class, perishable)     |
| holidays_events.csv | Holiday information                              |
| oil.csv             | Oil price data                                   |

### Key Columns

**train.csv**

* date
* store_nbr
* item_nbr
* unit_sales (target)
* onpromotion

**stores.csv**

* city
* state
* type
* cluster

**items.csv**

* family
* class
* perishable

---

## 4. Engineering Constraints and Design Decisions

The full dataset contains approximately 71 million rows.

Since the local development environment has 8GB RAM, processing the full dataset caused memory overflow issues.

To ensure stable execution:

* Limited training data to 5,000,000 rows
* Preserved time-series structure
* Maintained full feature engineering logic

In a production environment, the system would scale using distributed processing frameworks such as Spark or Dask.

---

## 5. Project Structure

```
Grocery-Sales-Forecasting/
│
├── data/
│   └── raw/
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── encoding.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── config.py
│
├── main.py
└── README.md
```

The project follows a modular architecture similar to real-world ML production systems.

---

## 6. Data Processing Pipeline

### 6.1 Data Loading

* Loaded 5 million rows using pandas
* Optimized data types to reduce memory usage
* Converted date columns to datetime

---

### 6.2 Target Transformation

Applied log transformation:

```
unit_sales = log1p(unit_sales)
```

Reason:

* Sales distribution is highly skewed
* Log transformation stabilizes variance
* Improves RMSLE performance

---

### 6.3 Dataset Merging

Merged:

* Store metadata
* Item metadata
* Oil price data
* Holiday indicator

Key engineering considerations:

* Prevented many-to-many join explosion
* Ensured unique merge keys
* Used `.isin()` for holiday feature instead of full merge

---

## 7. Feature Engineering

Feature engineering plays a critical role in time-series forecasting.

### 7.1 Date-Based Features

Extracted:

* year
* month
* day
* dayofweek
* weekofyear
* quarter
* is_weekend

Purpose:

* Capture seasonality
* Capture weekly patterns
* Capture monthly cycles
* Capture weekend demand variations

---

### 7.2 Promotion Feature

Converted `onpromotion` to numeric format.

Promotions strongly influence short-term demand spikes.

---

### 7.3 Lag Features

Created:

* lag_7
* lag_14

Purpose:

* Capture autocorrelation
* Model influence of recent sales history

---

### 7.4 Rolling Mean Feature

Created:

* rolling_mean_7

Purpose:

* Smooth demand fluctuations
* Capture short-term sales trend

---

## 8. Time-Based Validation Strategy

Used time-based split instead of random split.

Procedure:

* Training data: Historical period
* Validation data: Most recent 30 days

Reason:

Random split causes data leakage in time-series problems.
Time-based validation simulates real-world forecasting.

---

## 9. Models Trained

The following models were trained and evaluated:

| Model             | Type                       |
| ----------------- | -------------------------- |
| Linear Regression | Linear baseline model      |
| Ridge Regression  | Regularized linear model   |
| Random Forest     | Bagging ensemble           |
| XGBoost           | Gradient boosting ensemble |

---

## 10. Evaluation Metrics

Used the following metrics:

* MAE (Mean Absolute Error)
* RMSE (Root Mean Squared Error)
* RMSLE (Root Mean Squared Log Error)

RMSLE is most important because:

* Sales data is skewed
* Penalizes underprediction
* Was the official Kaggle competition metric

---

## 11. Model Performance

| Model             | Validation MAE | Validation RMSE | Validation RMSLE |
| ----------------- | -------------- | --------------- | ---------------- |
| Linear Regression | 4.19           | 16.75           | 0.70             |
| Ridge Regression  | 4.15           | 16.70           | 0.69             |
| Random Forest     | 3.25           | 15.61           | 0.50             |
| XGBoost           | 3.19           | 15.66           | 0.495            |

---

## 12. Best Model

XGBoost achieved the best validation performance.

Reasons:

* Captures nonlinear relationships
* Handles feature interactions automatically
* Strong bias–variance balance
* Robust for structured tabular data

---

## 13. Key Learnings

* Importance of lag-based time-series features
* Avoiding many-to-many merge issues
* Preventing data leakage using time-based validation
* Handling large datasets under memory constraints
* Comparing multiple ML models for structured data

---

## 14. How to Run

### Step 1: Install Dependencies

```
pip install -r requirements.txt
```

### Step 2: Run the Pipeline

```
python main.py
```

---

## 15. Future Improvements

* Hyperparameter tuning for XGBoost
* Early stopping implementation
* Feature importance analysis
* SHAP-based interpretability
* Store-level or item-level specialized models
* Deep learning approaches (LSTM, Temporal Fusion Transformer)

---

## 16. Project Summary

This project demonstrates:

* End-to-end ML pipeline development
* Time-series feature engineering
* Memory-efficient data handling
* Proper validation methodology
* Model benchmarking and selection
