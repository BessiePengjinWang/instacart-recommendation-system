import pandas as pd
import numpy as np
from pathlib import Path

class InstacartPreprocessor:
    """
    Preprocessor for Instacart dataset.
    Handles data loading, cleaning, and train/val split.
    """
    
    def __init__(self, data_dir='data/raw'):
        self.data_dir = Path(data_dir)
        self.orders = None
        self.order_products_prior = None
        self.order_products_train = None
        self.products = None
        
    def load_data(self):
        """Load all raw data files"""
        print("Loading data...")
        self.orders = pd.read_csv(self.data_dir / 'orders.csv')
        self.order_products_prior = pd.read_csv(
            self.data_dir / 'order_products__prior.csv'
        )
        self.order_products_train = pd.read_csv(
            self.data_dir / 'order_products__train.csv'
        )
        self.products = pd.read_csv(self.data_dir / 'products.csv')
        
        print(f"✅ Loaded {len(self.orders)} orders")
        print(f"✅ Loaded {len(self.order_products_prior)} prior products")
        print(f"✅ Loaded {len(self.order_products_train)} train products")
        
        return self
    
    def create_prior_data(self):
        """
        Combine prior orders with order metadata
        """
        # Merge prior products with orders
        prior_full = self.order_products_prior.merge(
            self.orders[['order_id', 'user_id', 'order_number', 
                        'order_dow', 'order_hour_of_day', 
                        'days_since_prior_order']],
            on='order_id',
            how='left'
        )
        
        print(f"✅ Created prior dataset: {prior_full.shape}")
        return prior_full
    
    def filter_cold_start_users(self, min_orders=5):
        """
        Remove users with too few orders
        """
        # Count orders per user in prior set
        user_order_counts = self.orders[
            self.orders['eval_set'] == 'prior'
        ].groupby('user_id')['order_id'].nunique()
        
        # Keep users with >= min_orders
        valid_users = user_order_counts[
            user_order_counts >= min_orders
        ].index
        
        print(f"Users before filter: {len(user_order_counts)}")
        print(f"Users after filter (>={min_orders} orders): {len(valid_users)}")
        print(f"Removed {len(user_order_counts) - len(valid_users)} cold start users")
        
        return valid_users
    
    def create_train_val_split(self, test_size=0.2, random_state=42):
        """
        Split train users into train/val
        """
        from sklearn.model_selection import train_test_split
        
        # Get all users in train set
        train_orders = self.orders[self.orders['eval_set'] == 'train']
        train_users = train_orders['user_id'].unique()
        
        # Split
        train_users_split, val_users_split = train_test_split(
            train_users, 
            test_size=test_size, 
            random_state=random_state
        )
        
        print(f"✅ Train/Val split created:")
        print(f"  Train users: {len(train_users_split)}")
        print(f"  Val users: {len(val_users_split)}")
        
        return train_users_split, val_users_split
    
    def save_processed_data(self, output_dir='data/processed'):
        """
        Save processed data for later use
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save prior data
        prior_full = self.create_prior_data()
        prior_full.to_parquet(output_dir / 'prior_orders.parquet', index=False)
        
        # Save train/val split
        train_users, val_users = self.create_train_val_split()
        pd.DataFrame({'user_id': train_users}).to_csv(
            output_dir / 'train_users.csv', index=False
        )
        pd.DataFrame({'user_id': val_users}).to_csv(
            output_dir / 'val_users.csv', index=False
        )
        
        print(f"✅ Saved processed data to {output_dir}")
        
        return self


# Usage
if __name__ == "__main__":
    preprocessor = InstacartPreprocessor(data_dir='../data/raw')
    preprocessor.load_data()
    preprocessor.save_processed_data()