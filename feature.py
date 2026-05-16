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


os.makedirs("eda_plots", exist_ok=True)
sns.set_theme(style="darkgrid", palette="muted")

FIGSIZE_WIDE = (14, 5)
FIGSIZE_SQ   = (10, 8)
ACCENT       = "#2196F3"
ACCENT2      = "#FF5722"

def sample_df(df, n=200_000):
    return df.sample(n, random_state=RANDOM_SEED) if len(df) > n else df

print("  Building EDA plots")

mean_cols = [c for c in daily.columns if c.endswith("_mean") and
             any(r in c for r in RAW_COLS)][:7]
desc = daily[mean_cols].describe().round(4)
fig, ax = plt.subplots(figsize=(14, 3))
ax.axis("off")
tbl = ax.table(cellText=desc.values, rowLabels=desc.index,
               colLabels=desc.columns, cellLoc="center", loc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(7)
tbl.scale(1, 1.6)
ax.set_title("EDA-1  Descriptive Statistics (Daily Aggregated)",
             fontsize=12, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig("eda_plots/EDA1_descriptive_statistics.png", dpi=150, bbox_inches="tight")
plt.close()
print("    EDA-1  Descriptive statistics          saved")

fig, ax = plt.subplots(figsize=(8, 5))
daily["Global_active_power_mean"].hist(
    bins=60, color=ACCENT, alpha=0.7, ax=ax, density=True)
daily["Global_active_power_mean"].plot.kde(
    ax=ax, color=ACCENT2, linewidth=2)
ax.set_title("EDA-2  Daily Mean Active Power Distribution",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Active Power (kW)")
ax.set_ylabel("Density")
plt.tight_layout()
plt.savefig("eda_plots/EDA2_feature_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print("    EDA-2  Feature distributions saved")

mean_cols_corr = [c for c in daily.columns if c.endswith("_mean")][:7]
corr = daily[mean_cols_corr].corr()
fig, ax = plt.subplots(figsize=FIGSIZE_SQ)
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, linewidths=0.5, annot_kws={"size": 10}, ax=ax)
ax.set_title("EDA-3  Feature Correlation Heatmap",
             fontsize=13, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig("eda_plots/EDA3_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("    EDA-3  Correlation heatmap saved")

fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
ax.plot(daily.index, daily["Global_active_power_mean"],
        color=ACCENT, linewidth=0.7, alpha=0.8, label="Daily mean")
ax.set_title("EDA-4  Daily Mean Active Power Over Time",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Date")
ax.set_ylabel("Active Power (kW)")
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.xticks(rotation=45)
ax.legend()
plt.tight_layout()
plt.savefig("eda_plots/EDA4_daily_power_timeseries.png", dpi=150, bbox_inches="tight")
plt.close()
print("    EDA-4  Time-series daily power saved")