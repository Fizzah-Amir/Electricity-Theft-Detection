
import os
import numpy as np
import pandas as pd
from gb_model import GradientBoostingFromScratch, DecisionStump

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


os.makedirs("xgb_scratch_results", exist_ok=True)

print("Loading saved CSV files ...")
X_train = pd.read_csv("X_train_smote.csv").values
y_train = pd.read_csv("y_train_smote.csv").values.ravel()
X_test  = pd.read_csv("X_test.csv").values
y_test  = pd.read_csv("y_test.csv").values.ravel()
all_col_names = pd.read_csv("X_train_smote.csv").columns.tolist()


print(f"  X_train shape : {X_train.shape}")
print(f"  X_test shape  : {X_test.shape}")
print(f"  y_train theft : {y_train.sum()} / {len(y_train)}")
print(f"  y_test theft  : {y_test.sum()} / {len(y_test)}")
print("Files loaded!")

print("\n" + "=" * 50)
print("Training Gradient Boosting from Scratch ...")
print("=" * 50)


scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
print(f"  scale_pos_weight = {scale_pos:.2f}")

model = GradientBoostingFromScratch(
    n_estimators=100, learning_rate=0.05,
    max_depth=3, lambda_reg=1.0,
    scale_pos_weight=scale_pos
)
model.fit(X_train, y_train)
model.save("xgb_scratch_results/gb_scratch_model.pkl")

print("\n" + "=" * 50)
print("Evaluating ...")
print("=" * 50)