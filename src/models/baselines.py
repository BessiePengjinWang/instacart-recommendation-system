import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import lightgbm as lgb

class PopularityBaseline:
    """
    Simple baseline: recommend most popular products
    """
    
    def __init__(self):
        self.top_products = None
        
    def fit(self, train_df):
        """
        Find top products by reorder rate
        """
        print("Training Popularity Baseline...")
        
        # Calculate product popularity
        product_stats = train_df.groupby('product_id').agg({
            'reordered': ['sum', 'count', 'mean']
        }).reset_index()
        
        product_stats.columns = ['product_id', 'reorders', 'total', 'reorder_rate']
        
        # Sort by number of reorders
        product_stats = product_stats.sort_values('reorders', ascending=False)
        
        self.top_products = set(product_stats.head(1000)['product_id'].values)
        
        print(f"✅ Selected top {len(self.top_products)} products")
        
        return self
    
    def predict(self, test_df):
        """
        Predict: return 1 if product is in top products
        """
        predictions = test_df['product_id'].isin(self.top_products).astype(int)
        return predictions


class UserHeuristicBaseline:
    """
    Heuristic: predict reorder if:
    - User bought it many times before
    - User bought it recently
    """
    
    def __init__(self, times_threshold=3, recency_threshold=5):
        self.times_threshold = times_threshold
        self.recency_threshold = recency_threshold
        
    def fit(self, train_df):
        """No training needed for heuristic"""
        print(f"Using heuristic thresholds:")
        print(f"  - Times bought >= {self.times_threshold}")
        print(f"  - Orders since last <= {self.recency_threshold}")
        return self
    
    def predict(self, test_df):
        """
        Predict: return 1 if item was bought frequently and recently
        """
        predictions = (
            (test_df['up_times_bought'] >= self.times_threshold) &
            (test_df['up_orders_since_last'] <= self.recency_threshold)
        ).astype(int)
        
        return predictions


class LGBMBaseline:
    """
    LightGBM model - our main baseline
    """
    
    def __init__(self, params=None):
        if params is None:
            self.params = {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'min_child_samples': 20,
                'max_depth': 8,
            }
        else:
            self.params = params
            
        self.model = None
        self.feature_cols = None
        
    def fit(self, train_df, val_df=None, num_boost_round=500, early_stopping_rounds=50):
        """
        Train LightGBM model
        """
        print("Training LightGBM model...")
        
        # Define feature columns
        self.feature_cols = [col for col in train_df.columns 
                            if col not in ['user_id', 'product_id', 'reordered']]
        
        print(f"  Using {len(self.feature_cols)} features")
        
        # Prepare data
        X_train = train_df[self.feature_cols]
        y_train = train_df['reordered']
        
        train_data = lgb.Dataset(X_train, label=y_train)
        
        valid_sets = [train_data]
        valid_names = ['train']
        
        if val_df is not None:
            X_val = val_df[self.feature_cols]
            y_val = val_df['reordered']
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(val_data)
            valid_names.append('valid')
        
        # Train
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stopping_rounds),
                lgb.log_evaluation(period=50)
            ]
        )
        
        print(f"✅ Training completed. Best iteration: {self.model.best_iteration}")
        
        return self
    
    def predict(self, test_df):
        """
        Predict probabilities
        """
        X_test = test_df[self.feature_cols]
        predictions = self.model.predict(X_test, num_iteration=self.model.best_iteration)
        return predictions
    
    def get_feature_importance(self, importance_type='gain'):
        """
        Get feature importance
        """
        importance = self.model.feature_importance(importance_type=importance_type)
        feature_importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return feature_importance


def evaluate_model(y_true, y_pred, y_prob=None, threshold=0.5):
    """
    Evaluate model performance
    
    Args:
        y_true: true labels
        y_pred: predicted labels (0/1)
        y_prob: predicted probabilities (optional, for AUC)
        threshold: threshold for converting probabilities to labels
    """
    # Convert probabilities to labels if needed
    if y_prob is not None:
        y_pred = (y_prob >= threshold).astype(int)
    
    # Calculate metrics
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }
    
    # Add AUC if probabilities available
    if y_prob is not None:
        auc = roc_auc_score(y_true, y_prob)
        metrics['auc'] = auc
    
    return metrics


def print_metrics(metrics, model_name="Model"):
    """
    Pretty print metrics
    """
    print(f"\n{model_name} Performance:")
    print("=" * 50)
    for metric, value in metrics.items():
        print(f"  {metric.upper():12s}: {value:.4f}")
    print("=" * 50)