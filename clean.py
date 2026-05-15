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
print(f"Missing after fill: {df.isnull().sum().sum():,}")

for col in RAW_COLS:
    n_neg = (df[col] < 0).sum()
    if n_neg > 0:
        print(f"Clipping {n_neg:,} negative values in '{col}'")
        df[col] = df[col].clip(lower = 0)

print("\nIterative Z-score outlier removal (threshold Z > 4):")
for col in RAW_COLS:
    iteration = 0
    while True:
        z_scores = np.abs(stats.zscore(df[col].dropna()))
        outlier_mask = pd.Series(False, index = df.index)
        outlier_mask[df[col].dropna().index] = z_scores > 4.0
        n_out = outlier_mask.sum()
        if n_out == 0:
            break
        iteration = iteration + 1
        print(f"[{col}] iter {iteration}: replacing {n_out:,} outliers")
        df.loc[outlier_mask, col] = np.nan
        df[col] = df[col].interpolate(method = "time", limit_direction = "both")
        df[col] = df[col].ffill().bfill()
    print(f"[{col}] clean after {iteration} iteration(s)")

print(f"Shape: {df.shape}")
print(f"Remaining NaN: {df.isnull().sum().sum()}")
print(f"Negative values: {sum((df[col] < 0).sum() for col in RAW_COLS)}")

all_clear = True
print("Outlier check (Z > 4):")
for col in RAW_COLS:
    z = np.abs(stats.zscore(df[col].dropna()))
    rem = (z > 4).sum()
    print(f"{col:<30}: {rem}")
    if rem > 0:
        all_clear = False

if all_clear:
    print("\nAll checks passed")
else:
    print("\nSome outliers remain. We need to check manually.")

df["Hour"] = df.index.hour
df["IsWeekend"] = (df.index.dayofweek >= 5).astype(int)
df["IsPeak"] = ((df["Hour"] >= 7) & (df["Hour"] < 22)).astype(int)

daily_parts = {}

for col in RAW_COLS:
    daily_parts[f"{col}_mean"] = df[col].resample("D").mean()
    daily_parts[f"{col}_max"] = df[col].resample("D").max()
    daily_parts[f"{col}_min"] = df[col].resample("D").min()
    daily_parts[f"{col}_std"] = df[col].resample("D").std()
    daily_parts[f"{col}_sum"] = df[col].resample("D").sum()

peak_mean = df[df["IsPeak"] == 1]["Global_active_power"].resample("D").mean()
offpeak_mean = df[df["IsPeak"] == 0]["Global_active_power"].resample("D").mean()
