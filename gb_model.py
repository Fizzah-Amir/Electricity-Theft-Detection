
import numpy as np
import joblib


class DecisionStump:

    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.tree      = None