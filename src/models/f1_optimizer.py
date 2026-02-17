"""Utilities for global and per-user F1-oriented thresholding."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score

DEFAULT_THRESHOLD = 0.2
GLOBAL_SEARCH_STEPS = 100
LOW_PERCENTILE = 1
HIGH_PERCENTILE = 99


class F1Optimizer:
    """Optimize binary predictions for F1 score."""

    def maximize_expectation(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        user_ids: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float | None, float]:
        """Run global or per-user optimization depending on `user_ids`."""
        if user_ids is not None:
            return self._optimize_per_user(y_true, y_prob, user_ids)
        return self._optimize_global(y_true, y_prob)

    def _optimize_global(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
    ) -> tuple[np.ndarray, float, float]:
        """Search thresholds on percentiles and return best global F1 configuration."""
        best_f1 = 0.0
        best_threshold = DEFAULT_THRESHOLD
        best_predictions = (y_prob >= DEFAULT_THRESHOLD).astype(int)

        thresholds = np.linspace(
            np.percentile(y_prob, LOW_PERCENTILE),
            np.percentile(y_prob, HIGH_PERCENTILE),
            GLOBAL_SEARCH_STEPS,
        )

        for thresh in thresholds:
            preds = (y_prob >= thresh).astype(int)
            f1 = f1_score(y_true, preds)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = float(thresh)
                best_predictions = preds.copy()

        return best_predictions, best_threshold, float(best_f1)

    def _optimize_per_user(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        user_ids: np.ndarray,
    ) -> tuple[np.ndarray, None, float]:
        """Optimize top-K labels per user and report overall F1."""
        predictions = np.zeros(len(y_true))

        unique_users = np.unique(user_ids)
        total = len(unique_users)
        progress_every = max(total // 10, 1)

        print(f"  Optimizing for {total:,} users...")

        for i, user in enumerate(unique_users):
            if i % progress_every == 0:
                print(f"  Progress: {i/total:.0%}")

            user_mask = user_ids == user
            user_true = y_true[user_mask]
            user_prob = y_prob[user_mask]
            n = len(user_true)

            if n == 0:
                continue

            sorted_idx = np.argsort(user_prob)[::-1]
            user_true_sorted = user_true[sorted_idx]
            total_positive = user_true.sum()

            best_f1 = 0.0
            best_k = 1
            tp = 0

            for k in range(1, n + 1):
                tp += user_true_sorted[k - 1]
                fp = k - tp
                fn = total_positive - tp

                if tp == 0:
                    continue

                precision = tp / (tp + fp)
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = (
                    2 * precision * recall / (precision + recall)
                    if precision + recall > 0
                    else 0.0
                )

                if f1 > best_f1:
                    best_f1 = f1
                    best_k = k

            user_pred = np.zeros(n)
            user_pred[sorted_idx[:best_k]] = 1
            predictions[user_mask] = user_pred

        overall_f1 = f1_score(y_true, predictions)
        return predictions, None, float(overall_f1)

    def predict(self, y_prob: np.ndarray, user_ids: np.ndarray | None = None) -> np.ndarray:
        """Predict labels using the default static threshold."""
        _ = user_ids
        return (y_prob >= DEFAULT_THRESHOLD).astype(int)
