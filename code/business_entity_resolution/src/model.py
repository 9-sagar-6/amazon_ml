import os
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report
from utils import compute_macro_f05

FEATURE_COLS = [
    "name_ratio",
    "name_partial_ratio",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_wratio",
    "name_len_diff",
    "address_ratio",
    "address_partial_ratio",
    "address_token_sort_ratio",
    "address_token_set_ratio",
    "num_token_jaccard",
    "word_1gram_jaccard",
    "char_3gram_jaccard",
    "is_source2",
]

class EntityMatchingModel:
    def __init__(self, n_estimators=250, learning_rate=0.05, max_depth=6, num_leaves=31):
        self.model = LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_leaves=num_leaves,
            random_state=42,
            n_jobs=-1
        )
        self.optimal_threshold = 0.65

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray):
        X_feats = X_train[FEATURE_COLS]
        self.model.fit(X_feats, y_train)
        print("Model training complete.")

    def predict_proba(self, X_df: pd.DataFrame) -> np.ndarray:
        X_feats = X_df[FEATURE_COLS]
        return self.model.predict_proba(X_feats)[:, 1]

    def optimize_threshold(self, val_df: pd.DataFrame, ground_truth_dict: dict) -> float:
        """
        Finds the decision threshold theta in [0.4, 0.9] that maximizes macro F_0.5.
        """
        val_df["prob"] = self.predict_proba(val_df)
        
        best_theta = 0.65
        best_f05 = -1.0

        s1_ids = list(ground_truth_dict.keys())
        
        # Grid search threshold
        for theta in np.arange(0.40, 0.92, 0.03):
            # Group predictions above theta
            matched_pairs = val_df[val_df["prob"] >= theta]
            preds_dict = matched_pairs.groupby("s1_id")["cand_id"].apply(list).to_dict()
            
            # Ensure every S1 id exists in preds_dict
            for s1 in s1_ids:
                if s1 not in preds_dict:
                    preds_dict[s1] = []

            score = compute_macro_f05(ground_truth_dict, preds_dict)
            print(f"Threshold theta={theta:.2f} -> Macro F_0.5 = {score:.4f}")
            if score > best_f05:
                best_f05 = score
                best_theta = float(theta)

        print(f"\nOptimal Threshold theta* = {best_theta:.2f} with Best Macro F_0.5 = {best_f05:.4f}")
        self.optimal_threshold = best_theta
        return best_theta

    def save(self, filepath: str):
        joblib.dump({"model": self.model, "threshold": self.optimal_threshold}, filepath)

    def load(self, filepath: str):
        data = joblib.load(filepath)
        self.model = data["model"]
        self.optimal_threshold = data.get("threshold", 0.65)
