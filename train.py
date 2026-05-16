
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