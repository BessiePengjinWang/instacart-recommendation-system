"""Cold-start handling via popularity fallback and score blending."""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_MIN_ORDERS_THRESHOLD = 5
DEFAULT_RECOMMENDATION_COUNT = 10
COLD_ALPHA_NO_HISTORY = 0.0
COLD_ALPHA_FEW_ORDERS = 0.3
COLD_ALPHA_SOME_ORDERS = 0.5
WARM_ALPHA_DEFAULT = 0.7


class ColdStartHandler:
    """Handle recommendation behavior for users with sparse or no history."""

    def __init__(self, min_orders_threshold: int = DEFAULT_MIN_ORDERS_THRESHOLD) -> None:
        self.min_orders_threshold = min_orders_threshold
        self.popular_products: pd.DataFrame | None = None
        self.dept_popular_products: pd.DataFrame | None = None
        self.aisle_popular_products: pd.DataFrame | None = None
        self.product_metadata: pd.DataFrame | None = None
        self.user_order_counts: pd.DataFrame | None = None
        self.is_fitted = False

    def fit(self, prior_data: pd.DataFrame, products_df: pd.DataFrame) -> "ColdStartHandler":
        """Learn popularity statistics from prior interactions."""
        print("Fitting ColdStartHandler...")

        prior_with_meta = prior_data.merge(
            products_df[["product_id", "department_id", "aisle_id", "product_name"]],
            on="product_id",
            how="left",
        )

        self.popular_products = (
            prior_with_meta.groupby(["product_id", "product_name"])
            .agg(
                total_orders=("order_id", "count"),
                reorder_rate=("reordered", "mean"),
                unique_users=("user_id", "nunique"),
            )
            .reset_index()
            .sort_values("total_orders", ascending=False)
        )
        self.popular_products["popularity_score"] = (
            self.popular_products["total_orders"]
            * (1 + self.popular_products["reorder_rate"])
        )
        self.popular_products = self.popular_products.sort_values(
            "popularity_score", ascending=False
        ).reset_index(drop=True)

        self.dept_popular_products = (
            prior_with_meta.groupby(["department_id", "product_id"])
            .agg(total_orders=("order_id", "count"), reorder_rate=("reordered", "mean"))
            .reset_index()
            .sort_values(["department_id", "total_orders"], ascending=[True, False])
        )

        self.aisle_popular_products = (
            prior_with_meta.groupby(["aisle_id", "product_id"])
            .agg(total_orders=("order_id", "count"), reorder_rate=("reordered", "mean"))
            .reset_index()
            .sort_values(["aisle_id", "total_orders"], ascending=[True, False])
        )

        self.product_metadata = products_df[
            ["product_id", "department_id", "aisle_id", "product_name"]
        ].copy()

        self.user_order_counts = (
            prior_data.groupby("user_id")["order_id"].nunique().reset_index()
        )
        self.user_order_counts.columns = ["user_id", "n_orders"]

        self.is_fitted = True

        print("ColdStartHandler fitted")
        print(f"  Total products tracked: {len(self.popular_products):,}")
        print(f"  Departments: {self.dept_popular_products['department_id'].nunique()}")
        print(f"  Aisles: {self.aisle_popular_products['aisle_id'].nunique()}")

        return self

    def get_user_type(self, user_id: int, user_order_count: int | None = None) -> str:
        """Classify user as `new`, `cold`, or `warm`."""
        if user_order_count is None and self.user_order_counts is not None:
            user_row = self.user_order_counts[self.user_order_counts["user_id"] == user_id]
            if len(user_row) == 0:
                return "new"
            user_order_count = int(user_row["n_orders"].values[0])

        if user_order_count is None:
            return "new"
        if user_order_count < self.min_orders_threshold:
            return "cold"
        return "warm"

    def recommend_for_new_user(
        self,
        n: int = DEFAULT_RECOMMENDATION_COUNT,
        department_id: int | None = None,
        aisle_id: int | None = None,
    ) -> pd.DataFrame:
        """Return popularity-based recommendations for users without history."""
        if not self.is_fitted:
            raise ValueError("Must call fit() first")

        if aisle_id is not None:
            candidates = self.aisle_popular_products[
                self.aisle_popular_products["aisle_id"] == aisle_id
            ].head(n * 2)
            if len(candidates) < n:
                candidates = self.popular_products
            reason = f"Popular in aisle {aisle_id}"

        elif department_id is not None:
            candidates = self.dept_popular_products[
                self.dept_popular_products["department_id"] == department_id
            ].head(n * 2)
            if len(candidates) < n:
                candidates = self.popular_products
            reason = f"Popular in department {department_id}"

        else:
            candidates = self.popular_products
            reason = "Overall popular"

        top_n = candidates.head(n)[["product_id"]].copy()
        top_n = top_n.merge(
            self.product_metadata[["product_id", "product_name"]],
            on="product_id",
            how="left",
        )
        top_n["score"] = 1.0 / (np.arange(len(top_n)) + 1)
        top_n["reason"] = reason
        top_n["fallback_type"] = "new_user_popularity"

        return top_n

    def blend_with_popularity(
        self,
        user_id: int,
        model_scores: np.ndarray,
        product_ids: np.ndarray,
        alpha: float = WARM_ALPHA_DEFAULT,
    ) -> tuple[np.ndarray, float]:
        """Blend model scores with popularity scores based on user history depth."""
        if self.popular_products is None or self.user_order_counts is None:
            raise ValueError("Must call fit() before blend_with_popularity().")

        user_row = self.user_order_counts[self.user_order_counts["user_id"] == user_id]
        n_orders = int(user_row["n_orders"].values[0]) if len(user_row) else 0

        if n_orders == 0:
            alpha = COLD_ALPHA_NO_HISTORY
        elif n_orders < 3:
            alpha = COLD_ALPHA_FEW_ORDERS
        elif n_orders < self.min_orders_threshold:
            alpha = COLD_ALPHA_SOME_ORDERS

        pop_scores = []
        max_pop = self.popular_products["popularity_score"].max()

        for product_id in product_ids:
            prod_row = self.popular_products[
                self.popular_products["product_id"] == product_id
            ]
            score = (
                prod_row["popularity_score"].values[0] / max_pop
                if len(prod_row) > 0
                else 0.0
            )
            pop_scores.append(score)

        pop_scores_array = np.array(pop_scores)
        blended = alpha * model_scores + (1 - alpha) * pop_scores_array
        return blended, alpha

    def get_cold_start_stats(self, user_ids: np.ndarray) -> dict[str, int]:
        """Print and return cold-start category counts for a user list."""
        stats = {"new": 0, "cold": 0, "warm": 0}

        for uid in user_ids:
            stats[self.get_user_type(int(uid))] += 1

        total = len(user_ids)
        print("\nCold Start Analysis:")
        print(f"  New users:  {stats['new']:,} ({stats['new']/total:.1%})")
        print(f"  Cold users: {stats['cold']:,} ({stats['cold']/total:.1%})")
        print(f"  Warm users: {stats['warm']:,} ({stats['warm']/total:.1%})")

        return stats
