import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings("ignore")
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

WINDOW_DAYS = 7
THEFT_FRAC  = 0.072
SMOTE_RATIO = 0.3

RAW_COLS = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]

print("Loading daily_aggregated.csv")

daily = pd.read_csv("daily_aggregated.csv", index_col="Date", parse_dates=True)
print(f"  Daily rows       : {len(daily):,}")
print(f"  Features per day : {daily.shape[1]}")
print(f"  Date range       : {daily.index.min().date()}  →  {daily.index.max().date()}")
print()
print("STEP 5b — Exploratory Data Analysis (EDA)")
print()