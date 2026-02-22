🛒 Grocery Sales Forecasting using Machine Learning
📌 Project Overview

This project focuses on forecasting daily grocery sales for different store-item combinations using historical time-series data.

The goal is to build an end-to-end machine learning pipeline that:

Processes large retail datasets efficiently

Engineers time-series features

Trains multiple ML models

Compares model performance

Selects the best-performing model

The project is based on the Corporación Favorita Grocery Sales dataset and is implemented using Python.

🎯 Business Problem

Retail companies need accurate sales forecasting to:

Optimize inventory

Reduce stockouts

Prevent overstocking

Improve supply chain efficiency

Plan promotions effectively

This project predicts daily unit_sales for each store-item pair.

🗂 Dataset Used

The following datasets were used:

train.csv – Historical sales data (target: unit_sales)

test.csv – Future prediction dates

stores.csv – Store metadata (city, state, type, cluster)

items.csv – Product metadata (family, class, perishable)

holidays_events.csv – Holiday information

oil.csv – Oil prices (economic indicator)

⚙️ Engineering Decision (Memory Optimization)

The full dataset contains ~71 million rows, which is too large for local processing on 8GB RAM.

To ensure stable training:

We limited training data to 5 million rows

Maintained time-series integrity

Preserved feature engineering pipeline

In production, this would scale using distributed systems like Spark or Dask.

🏗 Project Structure
Grocery-Sales-Forecasting/
│
├── data/
│   ├── raw/
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

This modular structure follows real-world ML project standards.

🧠 Project Workflow
1️⃣ Data Loading

Loaded 5 million rows for memory efficiency

Converted date columns to datetime

Optimized dtypes

2️⃣ Target Cleaning
unit_sales = log1p(unit_sales)

Why?

Sales data is highly skewed

Log transformation stabilizes variance

Improves RMSLE performance

3️⃣ Dataset Merging

Merged:

Store metadata

Item metadata

Oil prices

Holiday indicator

Avoided many-to-many merge explosions by:

Ensuring unique merge keys

Using .isin() for holiday feature

4️⃣ Feature Engineering
📅 Date Features

Year

Month

Day

Day of week

Week of year

Quarter

Weekend indicator

🏷 Promotion Feature

Converted onpromotion to numeric

⏳ Lag Features

lag_7

lag_14

Captures past demand influence.

📊 Rolling Features

rolling_mean_7

Captures short-term demand trend.

5️⃣ Time-Based Validation

Used time split instead of random split:

Train: Past data
Validation: Most recent 30 days

Prevents data leakage.

🤖 Models Trained

Linear Regression

Ridge Regression

Random Forest

XGBoost

📊 Evaluation Metrics

We evaluated using:

MAE (Mean Absolute Error)

RMSE (Root Mean Squared Error)

RMSLE (Root Mean Squared Log Error)

RMSLE is most important because:

Sales are skewed

Penalizes underprediction

Used in Kaggle competition

🏆 Model Performance
Model	Valid MAE	Valid RMSE	Valid RMSLE
Linear Regression	4.19	16.75	0.70
Ridge Regression	4.15	16.70	0.69
Random Forest	3.25	15.61	0.50
XGBoost	3.19	15.66	0.495
🥇 Best Model: XGBoost

XGBoost outperformed all models because:

Captures nonlinear relationships

Handles feature interactions

Strong bias-variance tradeoff

Robust to skewed distributions

📌 Key Learnings

Handling large datasets with memory constraints

Avoiding many-to-many merge issues

Preventing time-series data leakage

Importance of lag-based features

Comparing multiple ML algorithms

Proper validation strategy

🚀 How to Run
1️⃣ Install dependencies
pip install -r requirements.txt
2️⃣ Run the pipeline
python main.py
🔮 Future Improvements

Hyperparameter tuning

Early stopping for XGBoost

Feature importance analysis

SHAP interpretation

Store-level specialized models

Deep learning approaches (LSTM)

🎓 Interview Summary

If asked to explain this project:

Built an end-to-end time-series forecasting pipeline for grocery sales using lag features and boosting models. Implemented memory-efficient data handling, time-based validation, and model benchmarking, achieving best results with XGBoost.

📜 License

This project is for educational and portfolio purposes.
