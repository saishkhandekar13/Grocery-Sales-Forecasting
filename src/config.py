# Project Configuration File

import os
from datetime import timedelta

# DATA PATHS

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed")

TRAIN_FILE = os.path.join(RAW_DATA_PATH, "train.csv")
TEST_FILE = os.path.join(RAW_DATA_PATH, "test.csv")
STORES_FILE = os.path.join(RAW_DATA_PATH, "stores.csv")
ITEMS_FILE = os.path.join(RAW_DATA_PATH, "items.csv")
HOLIDAYS_FILE = os.path.join(RAW_DATA_PATH, "holidays_events.csv")
OIL_FILE = os.path.join(RAW_DATA_PATH, "oil.csv")
TRANSACTIONS_FILE = os.path.join(RAW_DATA_PATH, "transactions.csv")


# TIME SERIES SETTINGS

TRAIN_YEARS = 2              # Last 2 years for modeling
VALIDATION_DAYS = 60         # Last 60 days for validation


# FEATURE ENGINEERING SETTINGS

LAG_DAYS = [7, 14]
ROLLING_WINDOWS = [7]


# MODEL SETTINGS

RANDOM_STATE = 42