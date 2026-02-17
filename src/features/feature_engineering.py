import pandas as pd
import numpy as np
from pathlib import Path

class FeatureEngineer:
    """
    Create features for user-product pairs
    """
    
    def __init__(self):
        self.user_features = None
        self.product_features = None
        self.dept_features = None
        self.aisle_features = None
        
    def create_user_features(self, prior_data):
        """
        Create user-level features from prior orders
        """
        print("Creating user features...")
        
        # Basic stats
        user_stats = prior_data.groupby('user_id').agg({
            'order_id': 'nunique',
            'product_id': 'count',
            'reordered': 'mean',
            'days_since_prior_order': 'mean',
            'order_dow': lambda x: x.mode()[0] if len(x) > 0 else 0,
            'order_hour_of_day': lambda x: x.mode()[0] if len(x) > 0 else 0,
        }).reset_index()
        
        user_stats.columns = [
            'user_id', 'user_total_orders', 'user_total_products',
            'user_reorder_ratio', 'user_avg_days_between_orders',
            'user_favorite_dow', 'user_favorite_hour'
        ]
        
        # Average basket size
        user_stats['user_avg_basket_size'] = (
            user_stats['user_total_products'] / user_stats['user_total_orders']
        )
        
        # NEW: Total unique products
        user_unique_prods = prior_data.groupby('user_id')['product_id'].nunique().reset_index()
        user_unique_prods.columns = ['user_id', 'user_unique_products']
        user_stats = user_stats.merge(user_unique_prods, on='user_id')
        
        # NEW: Product diversity (unique / total)
        user_stats['user_product_diversity'] = (
            user_stats['user_unique_products'] / user_stats['user_total_products']
        )
        
        print(f"✅ Created {len(user_stats)} user feature rows")
        print(f"   Features: {len(user_stats.columns) - 1}")
        
        self.user_features = user_stats
        return user_stats
    
    def create_product_features(self, prior_data):
        """
        Create product-level features
        """
        print("Creating product features...")
        
        product_stats = prior_data.groupby('product_id').agg({
            'order_id': 'count',
            'reordered': 'mean',
            'add_to_cart_order': 'mean',
            'user_id': 'nunique',
        }).reset_index()
        
        product_stats.columns = [
            'product_id', 'product_orders', 'product_reorder_ratio',
            'product_avg_cart_position', 'product_unique_users'
        ]
        
        # NEW: Product popularity rank
        product_stats['product_rank'] = product_stats['product_orders'].rank(
            ascending=False, method='dense'
        )
        
        # NEW: Is top 100 product
        product_stats['product_in_top_100'] = (
            product_stats['product_rank'] <= 100
        ).astype(int)
        
        # NEW: Product orders ratio (relative popularity)
        total_orders = product_stats['product_orders'].sum()
        product_stats['product_orders_ratio'] = (
            product_stats['product_orders'] / total_orders
        )
        
        print(f"✅ Created {len(product_stats)} product feature rows")
        print(f"   Features: {len(product_stats.columns) - 1}")
        
        self.product_features = product_stats
        return product_stats
    
    def create_department_aisle_features(self, prior_data, products_df):
        """
        Create department and aisle level features
        
        NEW FEATURES!
        """
        print("Creating department and aisle features...")
        
        # Merge to get department and aisle info
        prior_with_dept = prior_data.merge(
            products_df[['product_id', 'department_id', 'aisle_id']], 
            on='product_id', 
            how='left'
        )
        
        # Department features
        dept_stats = prior_with_dept.groupby('department_id').agg({
            'order_id': 'count',
            'reordered': 'mean',
            'product_id': 'nunique'
        }).reset_index()
        
        dept_stats.columns = [
            'department_id', 'dept_orders', 'dept_reorder_ratio',
            'dept_unique_products'
        ]
        
        # Aisle features
        aisle_stats = prior_with_dept.groupby('aisle_id').agg({
            'order_id': 'count',
            'reordered': 'mean',
            'product_id': 'nunique'
        }).reset_index()
        
        aisle_stats.columns = [
            'aisle_id', 'aisle_orders', 'aisle_reorder_ratio',
            'aisle_unique_products'
        ]
        
        print(f"✅ Created department features: {len(dept_stats)} rows")
        print(f"✅ Created aisle features: {len(aisle_stats)} rows")
        
        self.dept_features = dept_stats
        self.aisle_features = aisle_stats
        
        return dept_stats, aisle_stats
    
    def create_user_product_features(self, prior_data):
        """
        Create user-product interaction features
        
        EXPANDED with new temporal and behavioral features
        """
        print("Creating user-product interaction features...")
        
        # Get max order number per user (for recency calculation)
        user_max_order = prior_data.groupby('user_id')['order_number'].max().reset_index()
        user_max_order.columns = ['user_id', 'user_max_order_number']
        
        # Basic aggregations
        up_stats = prior_data.groupby(['user_id', 'product_id']).agg({
            'order_id': 'count',
            'order_number': ['max', 'min'],
            'add_to_cart_order': ['mean', 'std'],
            'reordered': 'mean',
        }).reset_index()
        
        up_stats.columns = [
            'user_id', 'product_id', 'up_times_bought',
            'up_last_order_number', 'up_first_order_number',
            'up_avg_cart_position', 'up_cart_position_std',
            'up_reorder_ratio'
        ]
        
        # Fill NaN in std (happens when bought only once)
        up_stats['up_cart_position_std'] = up_stats['up_cart_position_std'].fillna(0)
        
        # Merge user max order
        up_stats = up_stats.merge(user_max_order, on='user_id')
        
        # EXISTING: Orders since last purchase
        up_stats['up_orders_since_last'] = (
            up_stats['user_max_order_number'] - up_stats['up_last_order_number']
        )
        
        # EXISTING: Order range
        up_stats['up_order_range'] = (
            up_stats['up_last_order_number'] - up_stats['up_first_order_number']
        )
        
        # EXISTING: Purchase frequency
        up_stats['up_purchase_frequency'] = np.where(
            up_stats['up_order_range'] > 0,
            up_stats['up_times_bought'] / up_stats['up_order_range'],
            up_stats['up_times_bought']
        )
        
        # NEW: Order rate since first purchase
        up_stats['up_order_rate_since_first'] = np.where(
            up_stats['user_max_order_number'] - up_stats['up_first_order_number'] > 0,
            up_stats['up_times_bought'] / (up_stats['user_max_order_number'] - up_stats['up_first_order_number'] + 1),
            up_stats['up_times_bought']
        )
        
        # NEW: Recency score (higher = more recent)
        up_stats['up_recency_score'] = 1.0 / (up_stats['up_orders_since_last'] + 1)
        
        # NEW: Frequency score (normalized by user's total orders)
        up_stats['up_frequency_score'] = (
            up_stats['up_times_bought'] / up_stats['user_max_order_number']
        )
        
        # NEW: RFM combined score
        up_stats['up_rfm_score'] = (
            up_stats['up_recency_score'] * up_stats['up_frequency_score']
        )
        
        # Drop intermediate column
        up_stats = up_stats.drop(['user_max_order_number'], axis=1)
        
        print(f"✅ Created {len(up_stats)} user-product pairs")
        print(f"   Features: {len(up_stats.columns) - 2}")  # -2 for user_id, product_id
        
        return up_stats
    
    def create_user_department_features(self, prior_data, products_df):
        """
        Create user-department interaction features
        
        NEW FEATURE SET!
        """
        print("Creating user-department interaction features...")
        
        # Merge to get department
        prior_with_dept = prior_data.merge(
            products_df[['product_id', 'department_id']], 
            on='product_id', 
            how='left'
        )
        
        # User-department stats
        user_dept_stats = prior_with_dept.groupby(['user_id', 'department_id']).agg({
            'order_id': 'count',
            'reordered': 'mean',
            'product_id': 'nunique'
        }).reset_index()
        
        user_dept_stats.columns = [
            'user_id', 'department_id',
            'user_dept_orders', 'user_dept_reorder_ratio',
            'user_dept_unique_products'
        ]
        
        print(f"✅ Created {len(user_dept_stats)} user-department pairs")
        
        return user_dept_stats
    
    def create_order_streak_features(self, prior_data):
        """
        Create order streak features (consecutive purchases)
        
        NEW FEATURE - Very important according to Kaggle winners!
        """
        print("Creating order streak features...")
        
        # Sort by user and order number
        prior_sorted = prior_data.sort_values(['user_id', 'product_id', 'order_number'])
        
        # Calculate streaks
        streaks = []
        
        for (user_id, product_id), group in prior_sorted.groupby(['user_id', 'product_id']):
            order_numbers = group['order_number'].values
            
            # Calculate max consecutive streak
            if len(order_numbers) == 1:
                max_streak = 1
                current_streak = 1
            else:
                max_streak = 1
                current_streak = 1
                
                for i in range(1, len(order_numbers)):
                    if order_numbers[i] == order_numbers[i-1] + 1:
                        current_streak += 1
                        max_streak = max(max_streak, current_streak)
                    else:
                        current_streak = 1
                
                # Check if streak continues to the end (most recent)
                if order_numbers[-1] - order_numbers[-2] == 1:
                    final_streak = current_streak
                else:
                    final_streak = 1
            
            streaks.append({
                'user_id': user_id,
                'product_id': product_id,
                'up_order_streak_max': max_streak,
                'up_order_streak_current': current_streak if len(order_numbers) > 1 else 1
            })
        
        streak_df = pd.DataFrame(streaks)
        
        print(f"✅ Created streak features for {len(streak_df)} user-product pairs")
        
        return streak_df
    
    def create_all_features(self, prior_data, products_df=None):
        """
        Create all feature sets
        
        Args:
            prior_data: Prior orders data
            products_df: Products metadata (optional, for dept/aisle features)
        """
        print("\n" + "="*60)
        print("FEATURE ENGINEERING - EXPANDED VERSION")
        print("="*60 + "\n")
        
        # Basic features
        user_feat = self.create_user_features(prior_data)
        product_feat = self.create_product_features(prior_data)
        up_feat = self.create_user_product_features(prior_data)
        
        # NEW: Streak features
        streak_feat = self.create_order_streak_features(prior_data)
        
        # Merge streak into user-product features
        up_feat = up_feat.merge(
            streak_feat, 
            on=['user_id', 'product_id'], 
            how='left'
        )
        
        # Fill missing streaks with 0 (shouldn't happen but safety)
        up_feat['up_order_streak_max'] = up_feat['up_order_streak_max'].fillna(0)
        up_feat['up_order_streak_current'] = up_feat['up_order_streak_current'].fillna(0)
        
        # If products metadata provided, create dept/aisle features
        if products_df is not None:
            dept_feat, aisle_feat = self.create_department_aisle_features(
                prior_data, products_df
            )
            user_dept_feat = self.create_user_department_features(
                prior_data, products_df
            )
            
            print("\n" + "="*60)
            print("FEATURE SUMMARY")
            print("="*60)
            print(f"User features: {len(user_feat.columns) - 1}")
            print(f"Product features: {len(product_feat.columns) - 1}")
            print(f"User-Product features: {len(up_feat.columns) - 2}")
            print(f"Department features: {len(dept_feat.columns) - 1}")
            print(f"Aisle features: {len(aisle_feat.columns) - 1}")
            print(f"User-Department features: {len(user_dept_feat.columns) - 2}")
            print(f"\nTotal features: {len(user_feat.columns) + len(product_feat.columns) + len(up_feat.columns) + len(dept_feat.columns) + len(aisle_feat.columns) + len(user_dept_feat.columns) - 7}")
            
            return (user_feat, product_feat, up_feat, 
                    dept_feat, aisle_feat, user_dept_feat)
        else:
            print("\n" + "="*60)
            print("FEATURE SUMMARY")
            print("="*60)
            print(f"User features: {len(user_feat.columns) - 1}")
            print(f"Product features: {len(product_feat.columns) - 1}")
            print(f"User-Product features: {len(up_feat.columns) - 2}")
            print(f"\nTotal features: {len(user_feat.columns) + len(product_feat.columns) + len(up_feat.columns) - 4}")
            
            return user_feat, product_feat, up_feat


# Usage
if __name__ == "__main__":
    # Load prior data
    prior_data = pd.read_parquet('../data/processed/prior_orders.parquet')
    products = pd.read_csv('../data/raw/products.csv')
    
    # Create features
    fe = FeatureEngineer()
    results = fe.create_all_features(prior_data, products)
    
    if len(results) == 6:
        user_feat, product_feat, up_feat, dept_feat, aisle_feat, user_dept_feat = results
        
        # Save all features
        import os
        os.makedirs('../data/features', exist_ok=True)
        
        user_feat.to_csv('../data/features/user_features.csv', index=False)
        product_feat.to_csv('../data/features/product_features.csv', index=False)
        up_feat.to_csv('../data/features/user_product_features.csv', index=False)
        dept_feat.to_csv('../data/features/department_features.csv', index=False)
        aisle_feat.to_csv('../data/features/aisle_features.csv', index=False)
        user_dept_feat.to_csv('../data/features/user_department_features.csv', index=False)
        
        print("\n✅ All features saved to data/features/")
    else:
        user_feat, product_feat, up_feat = results
        
        # Save features
        import os
        os.makedirs('../data/features', exist_ok=True)
        
        user_feat.to_csv('../data/features/user_features.csv', index=False)
        product_feat.to_csv('../data/features/product_features.csv', index=False)
        up_feat.to_csv('../data/features/user_product_features.csv', index=False)
        
        print("\n✅ Features saved to data/features/")