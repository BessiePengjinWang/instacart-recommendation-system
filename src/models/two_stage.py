"""DEPRECATED: Two-stage recommender kept only for historical comparison experiments."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix

DEFAULT_N_CANDIDATES = 500
DEFAULT_ALS_FACTORS = 100


class TwoStageRecommender:
    """Deprecated two-stage recommender (ALS retrieval + LightGBM ranking)."""

    def __init__(
        self,
        ranker_model: Any,
        n_candidates: int = DEFAULT_N_CANDIDATES,
        als_factors: int = DEFAULT_ALS_FACTORS,
    ) -> None:
        self.ranker_model = ranker_model
        self.n_candidates = n_candidates
        self.als_factors = als_factors
        self.als_model: AlternatingLeastSquares | None = None
        self.user_id_map: dict[int, int] | None = None
        self.product_id_map: dict[int, int] | None = None
        self.reverse_product_map: dict[int, int] | None = None
        self.user_item_matrix = None

    def fit_als(
        self,
        prior_data: pd.DataFrame,
        all_products: np.ndarray | None = None,
    ) -> "TwoStageRecommender":
        """Fit ALS model for candidate generation."""
        print("Training ALS for candidate generation...")

        interactions = (
            prior_data.groupby(["user_id", "product_id"]).size().reset_index(name="count")
        )

        unique_users = interactions["user_id"].unique()
        unique_products = interactions["product_id"].unique()

        if all_products is not None:
            missing_products = set(all_products) - set(unique_products)
            if len(missing_products) > 0:
                print(
                    f"  Adding {len(missing_products)} products to vocabulary (rare items)"
                )
                unique_products = np.concatenate([unique_products, list(missing_products)])

        self.user_id_map = {uid: idx for idx, uid in enumerate(unique_users)}
        self.product_id_map = {pid: idx for idx, pid in enumerate(unique_products)}
        self.reverse_product_map = {idx: pid for pid, idx in self.product_id_map.items()}

        print(f"  Users: {len(self.user_id_map):,}")
        print(f"  Products: {len(self.product_id_map):,}")

        interactions["user_idx"] = interactions["user_id"].map(self.user_id_map)
        interactions["product_idx"] = interactions["product_id"].map(self.product_id_map)

        self.user_item_matrix = csr_matrix(
            (interactions["count"], (interactions["user_idx"], interactions["product_idx"])),
            shape=(len(self.user_id_map), len(self.product_id_map)),
        )

        print(f"  Matrix shape: {self.user_item_matrix.shape}")
        print(
            "  Sparsity: "
            f"{1 - self.user_item_matrix.nnz / np.prod(self.user_item_matrix.shape):.6f}"
        )

        self.als_model = AlternatingLeastSquares(
            factors=self.als_factors,
            regularization=0.01,
            iterations=15,
            random_state=42,
        )
        self.als_model.fit(self.user_item_matrix)

        print("ALS training completed")
        return self

    def generate_candidates(
        self,
        user_id: int,
        n: int | None = None,
    ) -> list[tuple[int, float]]:
        """Generate top-N ALS candidates for one user."""
        if n is None:
            n = self.n_candidates

        if user_id not in self.user_id_map:
            return []

        user_idx = self.user_id_map[user_id]
        item_indices, scores = self.als_model.recommend(
            user_idx,
            user_items=self.user_item_matrix[user_idx],
            N=n,
            filter_already_liked_items=False,
        )

        return [
            (self.reverse_product_map[idx], score)
            for idx, score in zip(item_indices, scores)
        ]

    def predict(
        self,
        test_df: pd.DataFrame,
        return_candidates: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, float]]:
        """Run two-stage prediction on a test dataframe."""
        print("Running two-stage prediction...")
        start_time = time.time()

        print("  Stage 1: Generating candidates...")
        stage1_start = time.time()

        unique_users = test_df["user_id"].unique()
        candidate_set = self._batch_generate_candidates(unique_users)

        candidate_pairs = []
        for uid, pids in candidate_set.items():
            for pid in pids:
                candidate_pairs.append((uid, pid))

        if candidate_pairs:
            cand_df = pd.DataFrame(candidate_pairs, columns=["user_id", "product_id"])
            cand_df["is_candidate"] = True
            test_df = test_df.copy()
            test_df = test_df.merge(cand_df, on=["user_id", "product_id"], how="left")
            test_df["is_candidate"] = test_df["is_candidate"].fillna(False)
        else:
            test_df = test_df.copy()
            test_df["is_candidate"] = False

        candidates_mask = test_df["is_candidate"].values
        candidates_df = test_df[candidates_mask]

        stage1_time = time.time() - stage1_start
        print(f"    Generated {len(candidates_df)} candidates from {len(test_df)} pairs")
        print(f"    Candidate coverage: {len(candidates_df) / len(test_df):.1%}")
        print(f"    Stage 1 time: {stage1_time:.2f}s")

        print("  Stage 2: Ranking candidates...")
        stage2_start = time.time()

        predictions = np.zeros(len(test_df))
        if len(candidates_df) > 0:
            candidate_scores = self.ranker_model.predict(candidates_df)
            predictions[candidates_mask] = candidate_scores

        stage2_time = time.time() - stage2_start
        total_time = time.time() - start_time

        print(f"    Stage 2 time: {stage2_time:.2f}s")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Avg time per user: {total_time / len(unique_users) * 1000:.1f}ms")

        if return_candidates:
            candidates_info = {
                "n_candidates": len(candidates_df),
                "n_total": len(test_df),
                "coverage": len(candidates_df) / len(test_df),
                "stage1_time": stage1_time,
                "stage2_time": stage2_time,
                "total_time": total_time,
            }
            return predictions, candidates_info

        return predictions

    def _batch_generate_candidates(self, user_ids: np.ndarray) -> dict[int, set[int]]:
        """Batch-generate ALS candidates for many users."""
        user_candidates: dict[int, set[int]] = {}
        valid_user_ids = [uid for uid in user_ids if uid in self.user_id_map]

        if len(valid_user_ids) == 0:
            return user_candidates

        user_indices = np.array([self.user_id_map[uid] for uid in valid_user_ids])
        all_item_indices, _ = self.als_model.recommend(
            user_indices,
            user_items=self.user_item_matrix[user_indices],
            N=self.n_candidates,
            filter_already_liked_items=False,
        )

        for i, user_id in enumerate(valid_user_ids):
            candidate_pids = set()
            for idx in all_item_indices[i]:
                if idx in self.reverse_product_map:
                    candidate_pids.add(self.reverse_product_map[idx])
            user_candidates[user_id] = candidate_pids

        return user_candidates

    def evaluate_stage1_recall(self, test_df: pd.DataFrame, y_true: np.ndarray) -> float:
        """Measure stage-1 candidate recall against positive labels."""
        print("Evaluating Stage 1 recall...")

        unique_users = test_df["user_id"].unique()
        candidate_set = self._batch_generate_candidates(unique_users)

        candidate_pairs = []
        for uid, pids in candidate_set.items():
            for pid in pids:
                candidate_pairs.append((uid, pid))

        cand_df = pd.DataFrame(candidate_pairs, columns=["user_id", "product_id"])
        cand_df["is_candidate"] = True

        merged = test_df[["user_id", "product_id"]].merge(
            cand_df,
            on=["user_id", "product_id"],
            how="left",
        )
        is_candidate = merged["is_candidate"].fillna(False).values

        true_positives_mask = y_true == 1
        recall = is_candidate[true_positives_mask].mean() if true_positives_mask.sum() > 0 else 0.0

        print(f"  Stage 1 Recall: {recall:.4f}")
        print(f"  (i.e., {recall:.1%} of true reorders are in candidates)")
        return float(recall)


def compare_single_stage_vs_two_stage(
    single_stage_model: Any,
    two_stage_model: TwoStageRecommender,
    test_df: pd.DataFrame,
    y_true: np.ndarray,
) -> pd.DataFrame:
    """Compare single-stage and deprecated two-stage model performance."""
    from sklearn.metrics import f1_score, roc_auc_score

    print("=" * 60)
    print("SINGLE-STAGE vs TWO-STAGE COMPARISON")
    print("=" * 60)

    print("\n1. Single-Stage (LightGBM on all products):")
    start = time.time()
    y_pred_single = single_stage_model.predict(test_df)
    single_time = time.time() - start

    print(f"  Time: {single_time:.2f}s")
    print(f"  Avg per user: {single_time / test_df['user_id'].nunique() * 1000:.1f}ms")

    print("\n2. Two-Stage (ALS + LightGBM):")
    y_pred_two, info = two_stage_model.predict(test_df, return_candidates=True)
    print(f"  Candidate coverage: {info['coverage']:.1%}")

    best_f1_single, best_thresh_single = 0.0, 0.5
    for thresh in np.arange(0.1, 0.6, 0.05):
        f1 = f1_score(y_true, (y_pred_single >= thresh).astype(int))
        if f1 > best_f1_single:
            best_f1_single, best_thresh_single = f1, thresh

    best_f1_two, best_thresh_two = 0.0, 0.5
    for thresh in np.arange(0.1, 0.6, 0.05):
        f1 = f1_score(y_true, (y_pred_two >= thresh).astype(int))
        if f1 > best_f1_two:
            best_f1_two, best_thresh_two = f1, thresh

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    results = pd.DataFrame(
        {
            "Metric": ["AUC", "Best F1", "Best Threshold", "Latency (s)", "Speedup"],
            "Single-Stage": [
                f"{roc_auc_score(y_true, y_pred_single):.4f}",
                f"{best_f1_single:.4f}",
                f"{best_thresh_single:.2f}",
                f"{single_time:.2f}",
                "1.0x",
            ],
            "Two-Stage": [
                f"{roc_auc_score(y_true, y_pred_two):.4f}",
                f"{best_f1_two:.4f}",
                f"{best_thresh_two:.2f}",
                f"{info['total_time']:.2f}",
                f"{single_time / info['total_time']:.1f}x",
            ],
        }
    )

    print(results.to_string(index=False))
    return results
