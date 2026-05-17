
import os
import numpy as np
import pandas as pd
from gb_model import GradientBoostingFromScratch, DecisionStump

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("xgb_scratch_results", exist_ok=True)

print("Loading training CSV files ...")
X_train      = pd.read_csv("X_train_smote.csv").values
y_train      = pd.read_csv("y_train_smote.csv").values.ravel()
all_col_names = pd.read_csv("X_train_smote.csv").columns.tolist()

print(f"  X_train shape : {X_train.shape}")
print(f"  y_train theft : {y_train.sum()} / {len(y_train)}")
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
print("Training Set Performance (not final evaluation) ...")
print("=" * 50)

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
import seaborn as sns

y_train_pred = model.predict(X_train)
y_train_prob = model.predict_proba(X_train)

acc  = accuracy_score(y_train, y_train_pred)
prec = precision_score(y_train, y_train_pred, zero_division=0)
rec  = recall_score(y_train, y_train_pred, zero_division=0)
f1   = f1_score(y_train, y_train_pred, zero_division=0)
auc  = roc_auc_score(y_train, y_train_prob)
cm   = confusion_matrix(y_train, y_train_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\n========== TRAINING SET RESULTS ==========")
print(f"  Accuracy  : {acc*100:.2f}%")
print(f"  Precision : {prec*100:.2f}%")
print(f"  Recall    : {rec*100:.2f}%")
print(f"  F1 Score  : {f1*100:.2f}%")
print(f"  ROC-AUC   : {auc:.4f}")
print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
print(f"==========================================")
print("\n  NOTE: These are training metrics only.")
print("        For final evaluation, run: test_gb_scratch.py")


fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
            xticklabels=["Normal", "Theft"],
            yticklabels=["Normal", "Theft"],
            annot_kws={"size": 16, "weight": "bold"})
ax.set_title("Confusion Matrix - Gradient Boosting (Train Data)", fontsize=13)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig("xgb_scratch_results/graph1_confusion_matrix.png", dpi=150)
plt.close()
print("\nSaved: xgb_scratch_results/graph1_confusion_matrix.png")

metric_names = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
metric_vals  = [acc*100, prec*100, rec*100, f1*100, auc*100]
bar_colors   = ["#42A5F5", "#66BB6A", "#EF5350", "#FFA726", "#AB47BC"]
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(metric_names, metric_vals, color=bar_colors, width=0.5)
ax.set_ylim(0, 115)
ax.set_ylabel("Score (%)")
ax.set_title("All Metrics - Gradient Boosting (Train Data)", fontsize=13)
for bar, val in zip(bars, metric_vals):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 1.5,
            f"{val:.1f}%",
            ha="center", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig("xgb_scratch_results/graph2_metrics_bar.png", dpi=150)
plt.close()
print("Saved: xgb_scratch_results/graph2_metrics_bar.png")


print("\n" + "=" * 50)
print("Generating Feature Importance Plot ...")
print("=" * 50)

feature_counts = np.zeros(X_train.shape[1])

def count_splits(node, counts):
    if node["leaf"]:
        return
    counts[node["feature"]] += 1
    count_splits(node["left"],  counts)
    count_splits(node["right"], counts)

for tree in model.trees:
    count_splits(tree.tree, feature_counts)

top_idx      = np.argsort(feature_counts)[::-1][:15]
top_features = [all_col_names[i] for i in top_idx]
top_values   = feature_counts[top_idx]

fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(range(15), top_values[::-1], color="#FFA726", alpha=0.85)
ax.set_yticks(range(15))
ax.set_yticklabels(top_features[::-1], fontsize=9)
ax.set_xlabel("Number of Times Used in Splits")
ax.set_title("Top 15 Important Features - Gradient Boosting (Scratch)", fontsize=13)
plt.tight_layout()
plt.savefig("xgb_scratch_results/graph1_feature_importance.png", dpi=150)
plt.close()
print("Saved: xgb_scratch_results/graph1_feature_importance.png")

print("\n" + "=" * 50)
print("Training DONE!")
print("  Model saved → xgb_scratch_results/gb_scratch_model.pkl")
print("  Next step   → run test_gb_scratch.py for final evaluation")
print("=" * 50)