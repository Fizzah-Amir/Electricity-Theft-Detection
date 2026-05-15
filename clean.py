import pandas as pd
import numpy as np
from scipy import stats
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings("ignore")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

RAW_COLS = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]

WINDOW_DAYS = 7
THEFT_FRAC  = 0.072
SMOTE_RATIO = 0.3

df = pd.read_csv(
    "household_power_consumption.txt",
    sep = ";",
    na_values = ["?", "", " "],
    low_memory= False
)
df.columns = df.columns.str.replace(r'^\d+', '', regex = True).str.strip()

print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Total missing vals: {df.isnull().sum().sum()}")
print(f"Missing per column \n{df.isnull().sum().to_string()}")

df["Datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], format = "%d/%m/%Y %H:%M:%S", errors = "coerce")
df.drop(columns=["Date", "Time"], inplace = True)
df.sort_values("Datetime", inplace = True)
df.set_index("Datetime", inplace = True)

dupes = df.index.duplicated().sum()
if dupes > 0:
    print(f"Removing {dupes} duplicate timestamps")
    df = df[~df.index.duplicated(keep = "first")]

full_range = pd.date_range(start = df.index.min(), end = df.index.max(), freq = "1min")
missing_ts = len(full_range) - len(df)
print(f"Date range: {df.index.min()} => {df.index.max()}")
print(f"Missing timestamps: {missing_ts:,} (gaps in the grid)")

df = df.reindex(full_range)
df = df.apply(pd.to_numeric, errors = "coerce")
print(f"Shape after reindex: {df.shape}")

print(f"Missing before fill: {df.isnull().sum().sum():,}")
df[RAW_COLS] = df[RAW_COLS].interpolate(method = "time", limit_direction = "both", limit = 60)
df[RAW_COLS] = df[RAW_COLS].ffill().bfill()
print(f"Missing after fill : {df.isnull().sum().sum():,}")

