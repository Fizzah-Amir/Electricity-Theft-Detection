import os
import numpy as np
import pandas as pd
from datetime import datetime
import joblib
from gb_model import GradientBoostingFromScratch, DecisionStump
print("=" * 50)
print("Loading saved model and test data ...")
print("=" * 50)
model = GradientBoostingFromScratch.load("xgb_scratch_results/gb_scratch_model.pkl")
X_test = pd.read_csv("X_test.csv").values
y_test = pd.read_csv("y_test.csv").values.ravel()
windows_index = pd.read_csv("windows_index.csv")
test_windows  = windows_index[windows_index["split"] == "test"].reset_index(drop=True)
print(f"Model loaded    : xgb_scratch_results/gb_scratch_model.pkl")
print(f"Test samples    : {len(X_test)}")
print(f"Actual theft    : {y_test.sum()} cases")
print(f"Actual normal   : {(y_test==0).sum()} cases")
print("Load successfull!")
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("gb_scratch_testing", exist_ok=True)
print("\n" + "=" * 50)
print("SECTION 4.6 - Predictions on Unseen Data")
print("=" * 50)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

pred_df = pd.DataFrame({
    "window_id"    : test_windows["window_id"] if "window_id" in test_windows.columns else range(len(y_pred)),
    "window_start" : test_windows["window_start"].values,
    "window_end"   : test_windows["window_end"].values,
    "actual_label" : y_test,
    "predicted"    : y_pred,
    "theft_prob_%" : (y_prob * 100).round(2),
    "result"       : ["✓ Correct" if p == a else "✗ Wrong"
                      for p, a in zip(y_pred, y_test)]
})

pred_df["prediction_meaning"] = pred_df["predicted"].map({
    0: "Normal User",
    1: "Electricity Theft Detected"
})

print("\n  Sample Predictions (first 20 rows):")
print(pred_df[["window_start", "window_end",
               "actual_label", "predicted",
               "prediction_meaning", "theft_prob_%",
               "result"]].head(20).to_string(index=False))

pred_df.to_csv("gb_scratch_testing/all_predictions.csv", index=False)
print(f"\n  All {len(pred_df)} predictions saved → gb_scratch_testing/all_predictions.csv")
print("\n" + "=" * 50)
print("SECTION 4.7 - Post Processing")
print("=" * 50)

print("\n  4.7a — Flagging Suspicious Users")
print("  " + "-" * 40)

suspicious = pred_df[pred_df["predicted"] == 1].copy()
suspicious["risk_level"] = pd.cut(
    suspicious["theft_prob_%"],
    bins   = [0, 60, 80, 100],
    labels = ["Medium Risk", "High Risk", "Critical Risk"]
)

print(f"  Total flagged as theft : {len(suspicious)}")
print(f"  Critical Risk (>80%)   : {(suspicious['risk_level'] == 'Critical Risk').sum()}")
print(f"  High Risk    (60-80%)  : {(suspicious['risk_level'] == 'High Risk').sum()}")
print(f"  Medium Risk  (<60%)    : {(suspicious['risk_level'] == 'Medium Risk').sum()}")

if len(suspicious) > 0:
    print("\n  Flagged Windows:")
    print(suspicious[["window_start", "window_end",
                       "theft_prob_%", "risk_level",
                       "actual_label"]].to_string(index=False))

suspicious.to_csv("gb_scratch_testing/suspicious_users.csv", index=False)
print("\n  Saved → gb_scratch_testing/suspicious_users.csv")
