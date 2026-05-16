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
