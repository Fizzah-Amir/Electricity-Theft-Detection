
import numpy as np
import joblib


class DecisionStump:

    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.tree      = None
        
    
    def _best_split(self, X, gradients, hessians, lambda_reg=1.0):
        best_gain      = -np.inf
        best_feature   = None
        best_threshold = None

        for feature in range(X.shape[1]):
            values     = X[:, feature]
            thresholds = np.unique(values)
            if len(thresholds) > 20:
                thresholds = np.percentile(values, np.linspace(5, 95, 20))

            for threshold in thresholds:
                left  = values <= threshold
                right = ~left
                if left.sum() < 2 or right.sum() < 2:
                    continue

                G_L = gradients[left].sum()
                G_R = gradients[right].sum()
                H_L = hessians[left].sum()
                H_R = hessians[right].sum()
                G   = gradients.sum()
                H   = hessians.sum()

                gain = (
                    (G_L**2 / (H_L + lambda_reg)) +
                    (G_R**2 / (H_R + lambda_reg)) -
                    (G**2   / (H   + lambda_reg))
                ) * 0.5

                if gain > best_gain:
                    best_gain      = gain
                    best_feature   = feature
                    best_threshold = threshold

        return best_feature, best_threshold
    
    def _build_tree(self, X, gradients, hessians, depth, lambda_reg=1.0):
        G          = gradients.sum()
        H          = hessians.sum()
        leaf_value = -G / (H + lambda_reg)

        if depth == 0 or len(X) < 4:
            return {"leaf": True, "value": leaf_value}

        feature, threshold = self._best_split(
            X, gradients, hessians, lambda_reg)

        if feature is None:
            return {"leaf": True, "value": leaf_value}

        left  = X[:, feature] <= threshold
        right = ~left

        return {
            "leaf"      : False,
            "feature"   : feature,
            "threshold" : threshold,
            "left"      : self._build_tree(X[left],  gradients[left],
                                           hessians[left],  depth-1, lambda_reg),
            "right"     : self._build_tree(X[right], gradients[right],
                                           hessians[right], depth-1, lambda_reg),
        }
    def fit(self, X, gradients, hessians, lambda_reg=1.0):
        self.tree = self._build_tree(
            X, gradients, hessians, self.max_depth, lambda_reg)
