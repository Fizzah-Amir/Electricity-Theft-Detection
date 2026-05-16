
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