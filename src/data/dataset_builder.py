"""Build train/validation datasets by joining labels with engineered features."""

from __future__ import annotations

from pathlib import Path
from typing import Self

import pandas as pd


class DatasetBuilder:
    """Construct train and validation modeling tables."""

    def __init__(self) -> None:
        self.train_df: pd.DataFrame | None = None
        self.val_df: pd.DataFrame | None = None
        self.user_features: pd.DataFrame | None = None
        self.product_features: pd.DataFrame | None = None
        self.up_features: pd.DataFrame | None = None
        self.dept_features: pd.DataFrame | None = None
        self.aisle_features: pd.DataFrame | None = None
        self.user_dept_features: pd.DataFrame | None = None

    def load_features(self, feature_dir: str = "data/features") -> Self:
        """Load feature tables from disk."""
        feature_path = Path(feature_dir)

        self.user_features = pd.read_csv(feature_path / "user_features.csv")
        self.product_features = pd.read_csv(feature_path / "product_features.csv")
        self.up_features = pd.read_csv(feature_path / "user_product_features.csv")

        if (feature_path / "department_features.csv").exists():
            self.dept_features = pd.read_csv(feature_path / "department_features.csv")
            self.aisle_features = pd.read_csv(feature_path / "aisle_features.csv")
            self.user_dept_features = pd.read_csv(
                feature_path / "user_department_features.csv"
            )
            print("Loaded features (with dept/aisle)")
        else:
            self.dept_features = None
            self.aisle_features = None
            self.user_dept_features = None
            print("Loaded features (basic)")

        print(f"  User features: {self.user_features.shape}")
        print(f"  Product features: {self.product_features.shape}")
        print(f"  User-Product features: {self.up_features.shape}")

        if self.dept_features is not None:
            print(f"  Department features: {self.dept_features.shape}")
            print(f"  Aisle features: {self.aisle_features.shape}")
            print(f"  User-Department features: {self.user_dept_features.shape}")

        return self

    def create_labels(
        self,
        data_dir: str = "data/raw",
        processed_dir: str = "data/processed",
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Create training labels and return train/validation user ids."""
        print("Creating labels...")

        orders = pd.read_csv(Path(data_dir) / "orders.csv")
        order_products_train = pd.read_csv(Path(data_dir) / "order_products__train.csv")

        train_users = pd.read_csv(Path(processed_dir) / "train_users.csv")["user_id"].values
        val_users = pd.read_csv(Path(processed_dir) / "val_users.csv")["user_id"].values

        train_orders = orders[orders["eval_set"] == "train"]
        train_purchases = order_products_train.merge(
            train_orders[["order_id", "user_id"]],
            on="order_id",
        )[["user_id", "product_id", "reordered"]]

        print(f"Train purchases: {train_purchases.shape}")
        print(f"  Reordered items: {train_purchases['reordered'].sum()}")

        return train_purchases, train_users, val_users

    def build_dataset(
        self,
        train_purchases: pd.DataFrame,
        users_subset: pd.Series,
        data_dir: str = "data/raw",
    ) -> pd.DataFrame:
        """Build a modeling dataset for the provided user subset."""
        if self.up_features is None or self.user_features is None or self.product_features is None:
            raise ValueError("Call load_features() before build_dataset().")

        print(f"Building dataset for {len(users_subset)} users...")
        user_product_pairs = self.up_features[self.up_features["user_id"].isin(users_subset)][
            ["user_id", "product_id"]
        ].copy()

        print(f"  Total user-product pairs: {len(user_product_pairs):,}")

        subset_purchases = train_purchases[train_purchases["user_id"].isin(users_subset)][
            ["user_id", "product_id"]
        ].drop_duplicates()
        subset_purchases["reordered"] = 1

        user_product_pairs = user_product_pairs.merge(
            subset_purchases,
            on=["user_id", "product_id"],
            how="left",
        )
        user_product_pairs["reordered"] = (
            user_product_pairs["reordered"].fillna(0).astype(int)
        )

        print(f"  Positive samples (reordered): {user_product_pairs['reordered'].sum():,}")
        print(f"  Negative samples: {(user_product_pairs['reordered'] == 0).sum():,}")
        print(f"  Positive rate: {user_product_pairs['reordered'].mean():.2%}")

        print("  Merging features...")
        df = user_product_pairs.merge(self.up_features, on=["user_id", "product_id"], how="left")
        df = df.merge(self.user_features, on="user_id", how="left")
        df = df.merge(self.product_features, on="product_id", how="left")

        if self.dept_features is not None:
            print("  Merging department/aisle features...")
            products_df = pd.read_csv(Path(data_dir) / "products.csv")

            df = df.merge(
                products_df[["product_id", "department_id", "aisle_id"]],
                on="product_id",
                how="left",
            )
            df = df.merge(self.dept_features, on="department_id", how="left")
            df = df.merge(self.aisle_features, on="aisle_id", how="left")
            df = df.merge(
                self.user_dept_features,
                on=["user_id", "department_id"],
                how="left",
            )

            print("    Added dept/aisle features")

        print(f"Final dataset shape: {df.shape}")
        print(f"  Total features: {df.shape[1] - 3}")

        return df

    def build_train_val(
        self,
        data_dir: str = "data/raw",
        processed_dir: str = "data/processed",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Build both train and validation datasets."""
        train_purchases, train_users, val_users = self.create_labels(data_dir, processed_dir)

        print("\n" + "=" * 60)
        print("BUILDING TRAIN DATASET")
        print("=" * 60)
        self.train_df = self.build_dataset(train_purchases, train_users, data_dir=data_dir)

        print("\n" + "=" * 60)
        print("BUILDING VAL DATASET")
        print("=" * 60)
        self.val_df = self.build_dataset(train_purchases, val_users, data_dir=data_dir)

        return self.train_df, self.val_df

    def save_datasets(self, output_dir: str = "data/processed") -> None:
        """Persist train and validation datasets to parquet files."""
        if self.train_df is None or self.val_df is None:
            raise ValueError("Call build_train_val() before save_datasets().")

        output_path = Path(output_dir)
        print("\nSaving datasets...")
        self.train_df.to_parquet(output_path / "train_dataset.parquet", index=False)
        self.val_df.to_parquet(output_path / "val_dataset.parquet", index=False)

        print(f"Saved to {output_path}")
        print(f"  Train: {self.train_df.shape}")
        print(f"  Val: {self.val_df.shape}")


if __name__ == "__main__":
    builder = DatasetBuilder()
    builder.load_features()
    builder.build_train_val()
    builder.save_datasets()
