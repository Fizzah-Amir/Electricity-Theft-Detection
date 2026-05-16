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

daily["Month"] = daily.index.month
month_order    = ["Jan","Feb","Mar","Apr","May","Jun",
                  "Jul","Aug","Sep","Oct","Nov","Dec"]
monthly_avg    = daily.groupby("Month")["Global_active_power_mean"].mean()
monthly_avg.index = month_order
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(monthly_avg.index, monthly_avg.values,
              color=sns.color_palette("muted", 12),
              edgecolor="white", linewidth=0.5)
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.01,
            f"{bar.get_height():.2f}",
            ha="center", va="bottom", fontsize=8)
ax.set_title("EDA-5  Monthly Average Active Power",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Month")
ax.set_ylabel("Avg Active Power (kW)")
plt.tight_layout()
plt.savefig("eda_plots/EDA5_monthly_avg_consumption.png", dpi=150, bbox_inches="tight")
plt.close()
print("    EDA-5  Monthly average consumption saved")

hour_cols = [f"hour_{h:02d}_mean_power" for h in range(24)]
hourly_avg = daily[hour_cols].mean()
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(range(24), hourly_avg.values,
        color=ACCENT, marker="o", linewidth=2, markersize=5)
ax.fill_between(range(24), hourly_avg.values, alpha=0.15, color=ACCENT)
ax.axvspan(7, 22, alpha=0.07, color="orange", label="Peak hours (07-22)")
ax.set_title("EDA-6  Hourly Average Active Power",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Avg Active Power (kW)")
ax.set_xticks(range(0, 24))
ax.legend()
plt.tight_layout()
plt.savefig("eda_plots/EDA6_hourly_avg_consumption.png", dpi=150, bbox_inches="tight")
plt.close()
print("    EDA-6  Hourly average consumption saved")

daily["DayType"] = np.where(daily.index.dayofweek >= 5, "Weekend", "Weekday")
fig, ax = plt.subplots(figsize=(7, 5))
weekday_v = daily[daily["DayType"]=="Weekday"]["Global_active_power_mean"]
weekend_v = daily[daily["DayType"]=="Weekend"]["Global_active_power_mean"]
ax.boxplot([weekday_v, weekend_v], labels=["Weekday", "Weekend"],
           patch_artist=True,
           boxprops=dict(facecolor=ACCENT, alpha=0.6),
           medianprops=dict(color=ACCENT2, linewidth=2))
ax.set_title("EDA-7  Weekday vs Weekend Consumption",
             fontsize=12, fontweight="bold")
ax.set_ylabel("Active Power (kW)")
plt.tight_layout()
plt.savefig("eda_plots/EDA7_weekday_vs_weekend.png", dpi=150, bbox_inches="tight")
plt.close()
print("    EDA-7  Weekday vs weekend               saved")

sm_cols = ["Sub_metering_1_mean", "Sub_metering_2_mean", "Sub_metering_3_mean"]
sm_avgs = daily[sm_cols].mean()
fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(["Kitchen\n(SM1)", "Laundry\n(SM2)", "Water Heater/AC\n(SM3)"],
       sm_avgs.values,
       color=["#42A5F5", "#EF5350", "#66BB6A"], alpha=0.85)
ax.set_title("EDA-8  Average Sub-metering Consumption",
             fontsize=12, fontweight="bold")
ax.set_ylabel("Energy (Wh)")
plt.tight_layout()
plt.savefig("eda_plots/EDA8_submetering_breakdown.png", dpi=150, bbox_inches="tight")
plt.close()
print("    EDA-8  Sub-metering breakdown saved")

rolling7  = daily["Global_active_power_mean"].rolling(7, min_periods=1).mean()
deviation = daily["Global_active_power_mean"] - rolling7
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
ax1.plot(daily.index, daily["Global_active_power_mean"],
         color=ACCENT, linewidth=0.8, alpha=0.6, label="Daily mean")
ax1.plot(daily.index, rolling7,
         color=ACCENT2, linewidth=2, label="7-day rolling avg")
ax1.set_ylabel("Active Power (kW)")
ax1.set_title("EDA-9  Rolling 7-Day Average vs Raw Daily Power",
              fontsize=12, fontweight="bold")
ax1.legend()
ax2.fill_between(daily.index, deviation,
                 where=deviation >= 0, color=ACCENT2,
                 alpha=0.5, label="Above avg")
ax2.fill_between(daily.index, deviation,
                 where=deviation < 0, color=ACCENT,
                 alpha=0.5, label="Below avg")
ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax2.set_ylabel("Deviation (kW)")
ax2.set_xlabel("Date")
ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.xticks(rotation=45)
ax2.legend(fontsize=9)
plt.tight_layout()
plt.savefig("eda_plots/EDA9_rolling_avg_vs_raw.png", dpi=150, bbox_inches="tight")
plt.close()
print("    EDA-9  Rolling avg vs raw daily power   saved")

day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
daily["DayOfWeek"] = daily.index.dayofweek
dow_groups = [
    daily[daily["DayOfWeek"]==d]["Global_active_power_mean"].values
    for d in range(7)
]
fig, ax = plt.subplots(figsize=(11, 5))
bp = ax.boxplot(dow_groups, labels=day_names,
                patch_artist=True, showfliers=False,
                medianprops=dict(color="white", linewidth=2))
colors = sns.color_palette("muted", 7)
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
ax.set_title("EDA-10  Active Power by Day of Week",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Day of Week")
ax.set_ylabel("Active Power (kW)")
plt.tight_layout()
plt.savefig("eda_plots/EDA10_power_by_day_of_week.png", dpi=150, bbox_inches="tight")
plt.close()
print("    EDA-10 Power by day of week saved")

daily.drop(columns=["Month", "DayType", "DayOfWeek"],
           errors="ignore", inplace=True)
print(f"\n  All 10 EDA plots saved to eda_plots/")
print()
print("STEP 7 — 7-day sliding windows + Robust Feature Engineering")
print()

daily_values = daily.values
daily_dates  = daily.index
n_days       = len(daily)
n_day_feats  = daily.shape[1]

windows   = []
win_dates = []

for start in range(n_days - WINDOW_DAYS + 1):
    end   = start + WINDOW_DAYS
    block = daily_values[start:end]
    flat  = block.flatten()

    gap_mean     = daily["Global_active_power_mean"].iloc[start:end]
    rolling_avg  = gap_mean.mean()
    trend        = float(np.polyfit(np.arange(WINDOW_DAYS), gap_mean.values, 1)[0])
    deviation    = gap_mean.std()
    max_dev      = (gap_mean - rolling_avg).abs().max()
    weekly_total = daily["Global_active_power_sum"].iloc[start:end].sum()
    pk_ratio     = daily["peak_offpeak_ratio"].iloc[start:end].mean()

    win_weekend  = daily["is_weekend"].iloc[start:end].values
    win_power    = daily["Global_active_power_mean"].iloc[start:end].values
    weekend_avg  = win_power[win_weekend == 1].mean() if (win_weekend == 1).any() else 0.0
    weekday_avg  = win_power[win_weekend == 0].mean() if (win_weekend == 0).any() else 0.0
    wk_ratio     = weekend_avg / (weekday_avg + 1e-6)

    first_half     = gap_mean.iloc[:3].mean()
    second_half    = gap_mean.iloc[4:].mean()
    recovery_ratio = second_half / (first_half + 1e-6)

    low_threshold   = rolling_avg * 0.4
    consecutive_low = float((gap_mean < low_threshold).sum())
    stability_cv = deviation / (rolling_avg + 1e-6)

    engineered = np.array([
        rolling_avg, trend, deviation, max_dev,
        weekly_total, pk_ratio, wk_ratio,
        recovery_ratio, consecutive_low, stability_cv
    ])

    windows.append(np.concatenate([flat, engineered]))
    win_dates.append((daily_dates[start].date(), daily_dates[end-1].date()))

windows = np.array(windows)

day_col_names = [
    f"day{d+1}_{feat}"
    for d in range(WINDOW_DAYS)
    for feat in daily.columns
]
eng_col_names = [
    "eng_rolling_avg_power",
    "eng_trend_slope",
    "eng_std_deviation",
    "eng_max_deviation",
    "eng_weekly_total",
    "eng_peak_offpeak_ratio",
    "eng_weekend_weekday_ratio",
    "eng_recovery_ratio",
    "eng_consecutive_low",
    "eng_stability_cv",
]
all_col_names = day_col_names + eng_col_names

print(f"  Total windows    : {len(windows):,}")
print(f"  Features/window  : {windows.shape[1]}")
print(f"    = {WINDOW_DAYS} days × {n_day_feats} daily features")
print(f"    + 7 original engineered features")
print(f"    + 3 NEW robust features")
print(f"      → eng_recovery_ratio   (handles vacation return)")
print(f"      → eng_consecutive_low  (handles short trips)")
print(f"      → eng_stability_cv     (handles new appliances)")

print()
print("STEP 8 — Theft injection at window level")
print()

n_windows = len(windows)
n_theft   = int(n_windows * THEFT_FRAC)
n_A       = int(n_theft * 0.40)
n_B       = int(n_theft * 0.35)
n_C       = n_theft - n_A - n_B

all_idx = np.arange(n_windows)
labels  = np.zeros(n_windows, dtype=int)
X       = windows.copy()

def col_idx(keyword):
    return [i for i, c in enumerate(all_col_names) if keyword in c]

gap_cols   = col_idx("Global_active_power_mean")
gi_cols    = col_idx("Global_intensity_mean")
sm1_cols   = col_idx("Sub_metering_1_mean")
sm2_cols   = col_idx("Sub_metering_2_mean")
sm3_cols   = col_idx("Sub_metering_3_mean")
volt_cols  = col_idx("Voltage")

power_like = [
    c for c in (col_idx("Global_active_power") +
                col_idx("Global_intensity") +
                col_idx("Sub_metering"))
    if c not in volt_cols
]

spike_idx = np.random.choice(all_idx, size=n_A, replace=False)
for i in spike_idx:
    scale = np.random.uniform(1.5, 3.0)
    for c in gap_cols:
        X[i, c] *= scale
    for c in gi_cols:
        X[i, c] *= scale * np.random.uniform(0.8, 1.0)
labels[spike_idx] = 1
print(f"  Pattern A (spike attack)      : {n_A:>5,} windows")

remaining_B = np.setdiff1d(all_idx, spike_idx)
block_idx   = np.random.choice(remaining_B, size=n_B, replace=False)
for i in block_idx:
    for c in power_like:
        X[i, c] *= np.random.uniform(0.0, 0.15)
labels[block_idx] = 1
print(f"  Pattern B (blackout, no Volt) : {n_B:>5,} windows")

taken_C   = np.where(labels == 1)[0]
remain_C  = np.setdiff1d(all_idx, taken_C)
burst_idx = np.random.choice(remain_C, size=n_C, replace=False)
for i in burst_idx:
    scale = np.random.uniform(2.0, 4.0)
    for c in sm1_cols + sm2_cols + sm3_cols:
        X[i, c] *= scale
labels[burst_idx] = 1
print(f"  Pattern C (sub-meter burst)   : {n_C:>5,} windows")

n_theft_act = labels.sum()
theft_pct   = n_theft_act / n_windows * 100
print(f"\n  Total windows  : {n_windows:,}")
print(f"  Normal   (0)   : {n_windows - n_theft_act:,}  ({100-theft_pct:.1f}%)")
print(f"  Theft    (1)   : {n_theft_act:,}  ({theft_pct:.1f}%)")

print()
print("STEP 9 — Split → Scale → SMOTE")
print()

X_train_raw, X_test_raw, y_train, y_test, idx_train, idx_test = train_test_split(
    X, labels, np.arange(n_windows),
    test_size    = 0.20,
    random_state = RANDOM_SEED,
    stratify     = labels
)
print(f"  Train : {len(X_train_raw):>6,} windows | theft {y_train.sum():,}")
print(f"  Test  : {len(X_test_raw):>6,} windows | theft {y_test.sum():,}")

scaler  = MinMaxScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test  = scaler.transform(X_test_raw)
joblib.dump(scaler, "scaler.pkl")
print(f"\n  Scaler saved → scaler.pkl")

print(f"\n  Before SMOTE → Normal: {(y_train==0).sum():,} | Theft: {(y_train==1).sum():,}")
smote = SMOTE(sampling_strategy=SMOTE_RATIO,
              random_state=RANDOM_SEED, k_neighbors=5)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

n0 = (y_train_smote == 0).sum()
n1 = (y_train_smote == 1).sum()
print(f"  After  SMOTE → Normal: {n0:,} | Theft: {n1:,}")
print(f"  Theft fraction after SMOTE : {n1/(n0+n1)*100:.1f}%")
print(f"  Synthetic windows created  : {n1 - (y_train==1).sum():,}")

print()
print("STEP 10 — Saving output files")
print("=" * 60)

pd.DataFrame(X_train_smote, columns=all_col_names).to_csv(
    "X_train_smote.csv", index=False)
pd.Series(y_train_smote, name="Label").to_csv(
    "y_train_smote.csv", index=False)
pd.DataFrame(X_test, columns=all_col_names).to_csv(
    "X_test.csv", index=False)
pd.Series(y_test, name="Label").to_csv(
    "y_test.csv", index=False)

win_index_df = pd.DataFrame(win_dates, columns=["window_start", "window_end"])
win_index_df["label"] = labels
win_index_df["split"] = "train"
win_index_df.iloc[idx_test,
    win_index_df.columns.get_loc("split")] = "test"
win_index_df.to_csv("windows_index.csv", index=True, index_label="window_id")

print("  Saved: X_train_smote.csv")
print("  Saved: y_train_smote.csv")
print("  Saved: X_test.csv")
print("  Saved: y_test.csv")
print("  Saved: scaler.pkl")
print("  Saved: windows_index.csv")