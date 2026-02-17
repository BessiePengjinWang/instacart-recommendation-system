import numpy as np
from sklearn.metrics import f1_score

class F1Optimizer:
    """
    Fast F1 optimization using incremental calculation
    """
    
    def maximize_expectation(self, y_true, y_prob, user_ids=None):
        if user_ids is not None:
            return self._optimize_per_user(y_true, y_prob, user_ids)
        else:
            return self._optimize_global(y_true, y_prob)
    
    def _optimize_global(self, y_true, y_prob):
        """
        Fast global threshold search - O(n) not O(n²)
        Just tries 100 thresholds instead of n thresholds
        """
        best_f1 = 0
        best_threshold = 0.2
        best_predictions = (y_prob >= 0.2).astype(int)
        
        # Try 100 thresholds between min and max prob
        thresholds = np.linspace(
            np.percentile(y_prob, 1),
            np.percentile(y_prob, 99),
            100
        )
        
        for thresh in thresholds:
            preds = (y_prob >= thresh).astype(int)
            f1 = f1_score(y_true, preds)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = thresh
                best_predictions = preds.copy()
        
        return best_predictions, best_threshold, best_f1
    
    def _optimize_per_user(self, y_true, y_prob, user_ids):
        """
        Per-user F1 optimization - fast version
        
        For each user: try top-K predictions where K = 1 to N
        Uses incremental F1 calculation (no repeated full scan)
        """
        predictions = np.zeros(len(y_true))
        
        unique_users = np.unique(user_ids)
        total = len(unique_users)
        
        print(f"  Optimizing for {total:,} users...")
        
        for i, user in enumerate(unique_users):
            # Progress every 10%
            if i % (total // 10) == 0:
                print(f"  Progress: {i/total:.0%}")
            
            user_mask = user_ids == user
            user_true = y_true[user_mask]
            user_prob = y_prob[user_mask]
            n = len(user_true)
            
            if n == 0:
                continue
            
            # Sort by probability descending
            sorted_idx = np.argsort(user_prob)[::-1]
            user_true_sorted = user_true[sorted_idx]
            
            # Incremental F1 calculation - O(n) not O(n²)
            total_positive = user_true.sum()
            
            best_f1 = 0
            best_k = 1
            
            tp = 0
            for k in range(1, n + 1):
                # Update tp incrementally
                tp += user_true_sorted[k - 1]
                fp = k - tp
                fn = total_positive - tp
                
                if tp == 0:
                    continue
                
                precision = tp / (tp + fp)
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                
                if precision + recall > 0:
                    f1 = 2 * precision * recall / (precision + recall)
                else:
                    f1 = 0
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_k = k
            
            # Apply best K
            user_pred = np.zeros(n)
            user_pred[sorted_idx[:best_k]] = 1
            predictions[user_mask] = user_pred
        
        overall_f1 = f1_score(y_true, predictions)
        return predictions, None, overall_f1
    
    def predict(self, y_prob, user_ids=None):
        """Simple predict using global threshold"""
        return (y_prob >= 0.2).astype(int)