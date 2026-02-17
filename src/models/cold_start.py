import pandas as pd
import numpy as np
from pathlib import Path


class ColdStartHandler:
    """
    Handle cold start scenarios for new users and new products.
    
    Strategy:
    - New users: recommend popular products in their stated categories
    - Sparse users (<5 orders): blend model prediction with popularity
    - New products: recommend based on department/aisle popularity
    """
    
    def __init__(self, min_orders_threshold=5):
        """
        Args:
            min_orders_threshold: Users with fewer orders are "cold"
        """
        self.min_orders_threshold = min_orders_threshold
        self.popular_products = None
        self.dept_popular_products = None
        self.aisle_popular_products = None
        self.product_metadata = None
        self.is_fitted = False
    
    def fit(self, prior_data, products_df):
        """
        Learn popularity statistics from prior data
        
        Args:
            prior_data: DataFrame with user_id, product_id, order_id, reordered
            products_df: DataFrame with product_id, department_id, aisle_id
        """
        print("Fitting ColdStartHandler...")
        
        # Merge product metadata
        prior_with_meta = prior_data.merge(
            products_df[['product_id', 'department_id', 'aisle_id', 'product_name']],
            on='product_id',
            how='left'
        )
        
        # 1. Overall popular products
        self.popular_products = (
            prior_with_meta.groupby(['product_id', 'product_name'])
            .agg(
                total_orders=('order_id', 'count'),
                reorder_rate=('reordered', 'mean'),
                unique_users=('user_id', 'nunique')
            )
            .reset_index()
            .sort_values('total_orders', ascending=False)
        )
        
        # Popularity score (combines volume and reorder rate)
        self.popular_products['popularity_score'] = (
            self.popular_products['total_orders'] * 
            (1 + self.popular_products['reorder_rate'])
        )
        self.popular_products = self.popular_products.sort_values(
            'popularity_score', ascending=False
        ).reset_index(drop=True)
        
        # 2. Popular products by department
        self.dept_popular_products = (
            prior_with_meta.groupby(['department_id', 'product_id'])
            .agg(
                total_orders=('order_id', 'count'),
                reorder_rate=('reordered', 'mean')
            )
            .reset_index()
            .sort_values(['department_id', 'total_orders'], ascending=[True, False])
        )
        
        # 3. Popular products by aisle
        self.aisle_popular_products = (
            prior_with_meta.groupby(['aisle_id', 'product_id'])
            .agg(
                total_orders=('order_id', 'count'),
                reorder_rate=('reordered', 'mean')
            )
            .reset_index()
            .sort_values(['aisle_id', 'total_orders'], ascending=[True, False])
        )
        
        # 4. Save product metadata
        self.product_metadata = products_df[
            ['product_id', 'department_id', 'aisle_id', 'product_name']
        ].copy()
        
        # 5. User order counts (to identify cold users)
        self.user_order_counts = (
            prior_data.groupby('user_id')['order_id']
            .nunique()
            .reset_index()
        )
        self.user_order_counts.columns = ['user_id', 'n_orders']
        
        self.is_fitted = True
        
        print(f"✅ ColdStartHandler fitted:")
        print(f"   Total products tracked: {len(self.popular_products):,}")
        print(f"   Departments: {self.dept_popular_products['department_id'].nunique()}")
        print(f"   Aisles: {self.aisle_popular_products['aisle_id'].nunique()}")
        
        return self
    
    def get_user_type(self, user_id, user_order_count=None):
        """
        Classify user into cold/warm
        
        Returns:
            'new': Never seen before
            'cold': Too few orders (<5)
            'warm': Enough history
        """
        if user_order_count is None and self.user_order_counts is not None:
            user_row = self.user_order_counts[
                self.user_order_counts['user_id'] == user_id
            ]
            if len(user_row) == 0:
                return 'new'
            user_order_count = user_row['n_orders'].values[0]
        
        if user_order_count is None:
            return 'new'
        elif user_order_count < self.min_orders_threshold:
            return 'cold'
        else:
            return 'warm'
    
    def recommend_for_new_user(self, n=10, department_id=None, aisle_id=None):
        """
        Recommend products for a brand new user
        
        Args:
            n: Number of recommendations
            department_id: If known, filter to this department
            aisle_id: If known, filter to this aisle
        
        Returns:
            DataFrame with product_id, product_name, score, reason
        """
        assert self.is_fitted, "Must call fit() first"
        
        if aisle_id is not None:
            # Most specific: filter by aisle
            candidates = self.aisle_popular_products[
                self.aisle_popular_products['aisle_id'] == aisle_id
            ].head(n * 2)
            
            if len(candidates) < n:
                # Fallback to overall popular
                candidates = self.popular_products
            
            reason = f"Popular in aisle {aisle_id}"
            
        elif department_id is not None:
            # Filter by department
            candidates = self.dept_popular_products[
                self.dept_popular_products['department_id'] == department_id
            ].head(n * 2)
            
            if len(candidates) < n:
                candidates = self.popular_products
            
            reason = f"Popular in department {department_id}"
            
        else:
            # Overall popular
            candidates = self.popular_products
            reason = "Overall popular"
        
        top_n = candidates.head(n)[['product_id']].copy()
        top_n = top_n.merge(
            self.product_metadata[['product_id', 'product_name']],
            on='product_id',
            how='left'
        )
        top_n['score'] = 1.0 / (np.arange(len(top_n)) + 1)  # Rank-based score
        top_n['reason'] = reason
        top_n['fallback_type'] = 'new_user_popularity'
        
        return top_n
    
    def blend_with_popularity(self, user_id, model_scores, product_ids, 
                               alpha=0.7):
        """
        Blend model scores with popularity for cold users
        
        Args:
            user_id: User ID
            model_scores: Model predicted probabilities
            product_ids: Corresponding product IDs
            alpha: Weight for model (1-alpha for popularity)
                   Higher alpha = trust model more
        
        Returns:
            Blended scores array
        """
        # Get user type
        user_row = self.user_order_counts[
            self.user_order_counts['user_id'] == user_id
        ]
        
        if len(user_row) == 0:
            n_orders = 0
        else:
            n_orders = user_row['n_orders'].values[0]
        
        # Adjust alpha based on how cold the user is
        # Fewer orders → trust popularity more
        if n_orders == 0:
            alpha = 0.0   # Pure popularity
        elif n_orders < 3:
            alpha = 0.3   # Mostly popularity
        elif n_orders < 5:
            alpha = 0.5   # Equal blend
        else:
            alpha = alpha # Use provided alpha (default 0.7)
        
        # Get popularity scores for these products
        pop_scores = []
        max_pop = self.popular_products['popularity_score'].max()
        
        for pid in product_ids:
            prod_row = self.popular_products[
                self.popular_products['product_id'] == pid
            ]
            if len(prod_row) > 0:
                score = prod_row['popularity_score'].values[0] / max_pop
            else:
                score = 0.0
            pop_scores.append(score)
        
        pop_scores = np.array(pop_scores)
        
        # Blend
        blended = alpha * model_scores + (1 - alpha) * pop_scores
        
        return blended, alpha
    
    def get_cold_start_stats(self, user_ids):
        """
        Analyze cold start distribution in a user set
        """
        stats = {'new': 0, 'cold': 0, 'warm': 0}
        
        for uid in user_ids:
            user_type = self.get_user_type(uid)
            stats[user_type] += 1
        
        total = len(user_ids)
        print(f"\nCold Start Analysis:")
        print(f"  New users:  {stats['new']:,} ({stats['new']/total:.1%})")
        print(f"  Cold users: {stats['cold']:,} ({stats['cold']/total:.1%})")
        print(f"  Warm users: {stats['warm']:,} ({stats['warm']/total:.1%})")
        
        return stats