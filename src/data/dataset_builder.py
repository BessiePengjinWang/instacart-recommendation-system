import pandas as pd
import numpy as np
from pathlib import Path

class DatasetBuilder:
    """
    Build training dataset from features and labels
    """
    
    def __init__(self):
        self.train_df = None
        self.val_df = None
        
    def load_features(self, feature_dir='data/features'):
        """Load all feature files"""
        feature_dir = Path(feature_dir)
        
        self.user_features = pd.read_csv(feature_dir / 'user_features.csv')
        self.product_features = pd.read_csv(feature_dir / 'product_features.csv')
        self.up_features = pd.read_csv(feature_dir / 'user_product_features.csv')
        
        # NEW: Load department/aisle features if exist
        if (feature_dir / 'department_features.csv').exists():
            self.dept_features = pd.read_csv(feature_dir / 'department_features.csv')
            self.aisle_features = pd.read_csv(feature_dir / 'aisle_features.csv')
            self.user_dept_features = pd.read_csv(feature_dir / 'user_department_features.csv')
            print(f"✅ Loaded features (with dept/aisle):")
        else:
            self.dept_features = None
            self.aisle_features = None
            self.user_dept_features = None
            print(f"✅ Loaded features (basic):")
        
        print(f"   User features: {self.user_features.shape}")
        print(f"   Product features: {self.product_features.shape}")
        print(f"   User-Product features: {self.up_features.shape}")
        
        if self.dept_features is not None:
            print(f"   Department features: {self.dept_features.shape}")
            print(f"   Aisle features: {self.aisle_features.shape}")
            print(f"   User-Department features: {self.user_dept_features.shape}")
        
        return self
    
    def create_labels(self, data_dir='data/raw', processed_dir='data/processed'):
        """
        Create labels from train set
        
        For each user in train:
          - Get all products they bought in prior orders
          - Label = 1 if reordered in train order
          - Label = 0 if not reordered
        """
        print("Creating labels...")
        
        # Load orders
        orders = pd.read_csv(Path(data_dir) / 'orders.csv')
        order_products_train = pd.read_csv(Path(data_dir) / 'order_products__train.csv')
        
        # Load train/val split
        train_users = pd.read_csv(Path(processed_dir) / 'train_users.csv')['user_id'].values
        val_users = pd.read_csv(Path(processed_dir) / 'val_users.csv')['user_id'].values
        
        # Get train orders
        train_orders = orders[orders['eval_set'] == 'train']
        
        # Get what users actually bought in train order
        train_purchases = order_products_train.merge(
            train_orders[['order_id', 'user_id']], 
            on='order_id'
        )[['user_id', 'product_id', 'reordered']]
        
        print(f"✅ Train purchases: {train_purchases.shape}")
        print(f"   Reordered items: {train_purchases['reordered'].sum()}")
        
        return train_purchases, train_users, val_users
    
    def build_dataset(self, train_purchases, users_subset, data_dir='data/raw'):
        """
        Build dataset for a subset of users
        
        Updated to include department/aisle features
        """
        print(f"Building dataset for {len(users_subset)} users...")
        
        # Get user-product pairs
        user_product_pairs = self.up_features[
            self.up_features['user_id'].isin(users_subset)
        ][['user_id', 'product_id']].copy()
        
        print(f"  Total user-product pairs: {len(user_product_pairs):,}")
        
        # Create labels using merge instead of slow apply
        subset_purchases = train_purchases[
            train_purchases['user_id'].isin(users_subset)
        ][['user_id', 'product_id']].drop_duplicates()
        subset_purchases['reordered'] = 1

        user_product_pairs = user_product_pairs.merge(
            subset_purchases, on=['user_id', 'product_id'], how='left'
        )
        user_product_pairs['reordered'] = user_product_pairs['reordered'].fillna(0).astype(int)
        
        print(f"  Positive samples (reordered): {user_product_pairs['reordered'].sum():,}")
        print(f"  Negative samples: {(user_product_pairs['reordered'] == 0).sum():,}")
        print(f"  Positive rate: {user_product_pairs['reordered'].mean():.2%}")
        
        # Merge features
        print("  Merging features...")
        
        # User-product features
        df = user_product_pairs.merge(
            self.up_features,
            on=['user_id', 'product_id'],
            how='left'
        )
        
        # User features
        df = df.merge(
            self.user_features,
            on='user_id',
            how='left'
        )
        
        # Product features
        df = df.merge(
            self.product_features,
            on='product_id',
            how='left'
        )
        
        # Merge department/aisle features
        if self.dept_features is not None:
            print("  Merging department/aisle features...")
            
            products_df = pd.read_csv(Path(data_dir) / 'products.csv')
            
            df = df.merge(
                products_df[['product_id', 'department_id', 'aisle_id']],
                on='product_id',
                how='left'
            )
            
            df = df.merge(self.dept_features, on='department_id', how='left')
            df = df.merge(self.aisle_features, on='aisle_id', how='left')
            df = df.merge(self.user_dept_features, on=['user_id', 'department_id'], how='left')
            
            print(f"    ✅ Added dept/aisle features")
        
        print(f"✅ Final dataset shape: {df.shape}")
        print(f"   Total features: {df.shape[1] - 3}")  # -3 for user_id, product_id, label
        
        return df
    
    def build_train_val(self, data_dir='data/raw', processed_dir='data/processed'):
        """
        Build both train and val datasets
        """
        # Create labels
        train_purchases, train_users, val_users = self.create_labels(
            data_dir, processed_dir
        )
        
        # Build train dataset
        print("\n" + "="*60)
        print("BUILDING TRAIN DATASET")
        print("="*60)
        self.train_df = self.build_dataset(train_purchases, train_users, data_dir=data_dir)
        
        # Build val dataset
        print("\n" + "="*60)
        print("BUILDING VAL DATASET")
        print("="*60)
        self.val_df = self.build_dataset(train_purchases, val_users, data_dir=data_dir)
        
        return self.train_df, self.val_df
    
    def save_datasets(self, output_dir='data/processed'):
        """Save train/val datasets"""
        output_dir = Path(output_dir)
        
        print("\nSaving datasets...")
        self.train_df.to_parquet(output_dir / 'train_dataset.parquet', index=False)
        self.val_df.to_parquet(output_dir / 'val_dataset.parquet', index=False)
        
        print(f"✅ Saved to {output_dir}")
        print(f"   Train: {self.train_df.shape}")
        print(f"   Val: {self.val_df.shape}")


# Usage
if __name__ == "__main__":
    builder = DatasetBuilder()
    builder.load_features()
    train_df, val_df = builder.build_train_val()
    builder.save_datasets()