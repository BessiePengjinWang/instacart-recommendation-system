"""Data preprocessing utilities for the Instacart recommendation pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Self

import pandas as pd


class InstacartPreprocessor:
    """Load raw Instacart files and produce processed training artifacts."""

    def __init__(self, data_dir: str = "data/raw") -> None:
        self.data_dir = Path(data_dir)
        self.orders: pd.DataFrame | None = None
        self.order_products_prior: pd.DataFrame | None = None
        self.order_products_train: pd.DataFrame | None = None
        self.products: pd.DataFrame | None = None

    def load_data(self) -> Self:
        """Load all raw data files into memory."""
        print("Loading data...")
        self.orders = pd.read_csv(self.data_dir / "orders.csv")
        self.order_products_prior = pd.read_csv(
            self.data_dir / "order_products__prior.csv"
        )
        self.order_products_train = pd.read_csv(
            self.data_dir / "order_products__train.csv"
        )
        self.products = pd.read_csv(self.data_dir / "products.csv")

        print(f"Loaded {len(self.orders)} orders")
        print(f"Loaded {len(self.order_products_prior)} prior products")
        print(f"Loaded {len(self.order_products_train)} train products")

        return self

    def create_prior_data(self) -> pd.DataFrame:
        """Combine prior order products with order-level metadata."""
        if self.orders is None or self.order_products_prior is None:
            raise ValueError("Call load_data() before create_prior_data().")

        prior_full = self.order_products_prior.merge(
            self.orders[
                [
                    "order_id",
                    "user_id",
                    "order_number",
                    "order_dow",
                    "order_hour_of_day",
                    "days_since_prior_order",
                ]
            ],
            on="order_id",
            how="left",
        )

        print(f"Created prior dataset: {prior_full.shape}")
        return prior_full

    def filter_cold_start_users(self, min_orders: int = 5) -> pd.Index:
        """Return users with at least `min_orders` prior orders."""
        if self.orders is None:
            raise ValueError("Call load_data() before filter_cold_start_users().")

        user_order_counts = self.orders[self.orders["eval_set"] == "prior"].groupby(
            "user_id"
        )["order_id"].nunique()

        valid_users = user_order_counts[user_order_counts >= min_orders].index

        print(f"Users before filter: {len(user_order_counts)}")
        print(f"Users after filter (>={min_orders} orders): {len(valid_users)}")
        print(f"Removed {len(user_order_counts) - len(valid_users)} cold start users")

        return valid_users

    def create_train_val_split(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> tuple[pd.Series, pd.Series]:
        """Split users in train eval_set into train/validation user groups."""
        if self.orders is None:
            raise ValueError("Call load_data() before create_train_val_split().")

        from sklearn.model_selection import train_test_split

        train_orders = self.orders[self.orders["eval_set"] == "train"]
        train_users = train_orders["user_id"].unique()

        train_users_split, val_users_split = train_test_split(
            train_users,
            test_size=test_size,
            random_state=random_state,
        )

        print("Train/Val split created:")
        print(f"  Train users: {len(train_users_split)}")
        print(f"  Val users: {len(val_users_split)}")

        return train_users_split, val_users_split

    def save_processed_data(self, output_dir: str = "data/processed") -> Self:
        """Save merged prior data and train/validation user lists."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        prior_full = self.create_prior_data()
        prior_full.to_parquet(output_path / "prior_orders.parquet", index=False)

        train_users, val_users = self.create_train_val_split()
        pd.DataFrame({"user_id": train_users}).to_csv(
            output_path / "train_users.csv", index=False
        )
        pd.DataFrame({"user_id": val_users}).to_csv(
            output_path / "val_users.csv", index=False
        )

        print(f"Saved processed data to {output_path}")
        return self


if __name__ == "__main__":
    preprocessor = InstacartPreprocessor(data_dir="../data/raw")
    preprocessor.load_data()
    preprocessor.save_processed_data()
