"""Baseline models and evaluation helpers for reorder prediction."""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

DEFAULT_TOP_PRODUCTS = 1000
DEFAULT_NUM_BOOST_ROUND = 500
DEFAULT_EARLY_STOPPING_ROUNDS = 50


class PopularityBaseline:
    """Recommend globally popular products as a non-personalized baseline."""

    def __init__(self, top_products_count: int = DEFAULT_TOP_PRODUCTS) -> None:
        self.top_products_count = top_products_count
        self.top_products: set[int] | None = None

    def fit(self, train_df: pd.DataFrame) -> "PopularityBaseline":
        """Select top products by reorder volume."""
        print("Training Popularity Baseline...")

        product_stats = train_df.groupby("product_id").agg(
            {"reordered": ["sum", "count", "mean"]}
        ).reset_index()
        product_stats.columns = ["product_id", "reorders", "total", "reorder_rate"]
        product_stats = product_stats.sort_values("reorders", ascending=False)

        self.top_products = set(
            product_stats.head(self.top_products_count)["product_id"].values
        )

        print(f"Selected top {len(self.top_products)} products")
        return self

    def predict(self, test_df: pd.DataFrame) -> pd.Series:
        """Predict a reorder label based on whether product is globally popular."""
        if self.top_products is None:
            raise ValueError("Call fit() before predict().")
        return test_df["product_id"].isin(self.top_products).astype(int)


class UserHeuristicBaseline:
    """Simple heuristic baseline based on frequency and recency cutoffs."""

    def __init__(self, times_threshold: int = 3, recency_threshold: int = 5) -> None:
        self.times_threshold = times_threshold
        self.recency_threshold = recency_threshold

    def fit(self, train_df: pd.DataFrame) -> "UserHeuristicBaseline":
        """Keep threshold parameters; no statistical training is performed."""
        _ = train_df
        print("Using heuristic thresholds:")
        print(f"  - Times bought >= {self.times_threshold}")
        print(f"  - Orders since last <= {self.recency_threshold}")
        return self

    def predict(self, test_df: pd.DataFrame) -> pd.Series:
        """Predict reorder when both frequency and recency rules are satisfied."""
        return (
            (test_df["up_times_bought"] >= self.times_threshold)
            & (test_df["up_orders_since_last"] <= self.recency_threshold)
        ).astype(int)


class LGBMBaseline:
    """LightGBM baseline model used in production and experiments."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = params or {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "min_child_samples": 20,
            "max_depth": 8,
        }
        self.model: lgb.Booster | None = None
        self.feature_cols: list[str] | None = None

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame | None = None,
        num_boost_round: int = DEFAULT_NUM_BOOST_ROUND,
        early_stopping_rounds: int = DEFAULT_EARLY_STOPPING_ROUNDS,
    ) -> "LGBMBaseline":
        """Train a LightGBM model on train_df with optional validation set."""
        print("Training LightGBM model...")

        self.feature_cols = [
            col for col in train_df.columns if col not in ["user_id", "product_id", "reordered"]
        ]
        print(f"  Using {len(self.feature_cols)} features")

        x_train = train_df[self.feature_cols]
        y_train = train_df["reordered"]
        train_data = lgb.Dataset(x_train, label=y_train)

        valid_sets = [train_data]
        valid_names = ["train"]

        if val_df is not None:
            x_val = val_df[self.feature_cols]
            y_val = val_df["reordered"]
            val_data = lgb.Dataset(x_val, label=y_val, reference=train_data)
            valid_sets.append(val_data)
            valid_names.append("valid")

        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stopping_rounds),
                lgb.log_evaluation(period=50),
            ],
        )

        print(f"Training completed. Best iteration: {self.model.best_iteration}")
        return self

    def predict(self, test_df: pd.DataFrame) -> pd.Series:
        """Predict reorder probabilities for each row in test_df."""
        if self.model is None or self.feature_cols is None:
            raise ValueError("Call fit() before predict().")
        x_test = test_df[self.feature_cols]
        return self.model.predict(x_test, num_iteration=self.model.best_iteration)

    def get_feature_importance(self, importance_type: str = "gain") -> pd.DataFrame:
        """Return feature importances sorted descending."""
        if self.model is None or self.feature_cols is None:
            raise ValueError("Call fit() before get_feature_importance().")

        importance = self.model.feature_importance(importance_type=importance_type)
        return (
            pd.DataFrame({"feature": self.feature_cols, "importance": importance})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )


def evaluate_model(
    y_true: pd.Series,
    y_pred: pd.Series,
    y_prob: pd.Series | None = None,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute precision/recall/F1 and optional AUC metrics."""
    if y_prob is not None:
        y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }

    if y_prob is not None:
        metrics["auc"] = roc_auc_score(y_true, y_prob)

    return metrics


def print_metrics(metrics: dict[str, float], model_name: str = "Model") -> None:
    """Pretty-print metrics for console-based experiments."""
    print(f"\n{model_name} Performance:")
    print("=" * 50)
    for metric, value in metrics.items():
        print(f"  {metric.upper():12s}: {value:.4f}")
    print("=" * 50)
