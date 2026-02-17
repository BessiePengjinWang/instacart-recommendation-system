"""FastAPI service for Instacart product recommendations."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.models.f1_optimizer import F1Optimizer

from .models import (
    HealthResponse,
    ProductRecommendation,
    RecommendRequest,
    RecommendResponse,
)

MODEL_SCORE_THRESHOLD = 0.2

app = FastAPI(
    title="Instacart Recommendation API",
    description="Production-ready grocery recommendation system with cold start handling",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AppState:
    """Container for app-wide model and feature objects loaded at startup."""

    model = None
    cold_start_handler = None
    f1_optimizer = None
    user_features = None
    product_features = None
    up_features = None
    dept_features = None
    aisle_features = None
    user_dept_features = None
    products_metadata = None
    feature_cols = None
    feature_dir = None


state = AppState()


@app.on_event("startup")
async def load_models() -> None:
    """Load trained artifacts and small feature tables into app state."""
    print("Loading models and features...")

    try:
        print("Step 1: Loading LightGBM model...")
        model_path = Path("models/saved/lgbm_baseline.pkl")
        state.model = joblib.load(model_path)
        print(f"Loaded LightGBM model ({model_path.stat().st_size / 1024 / 1024:.1f} MB)")

        state.feature_dir = Path("data/features")

        print("Step 2: Loading user features...")
        state.user_features = pd.read_csv(state.feature_dir / "user_features.csv")
        print(f"  User features: {state.user_features.shape}")

        print("Step 3: Loading product features...")
        state.product_features = pd.read_csv(state.feature_dir / "product_features.csv")
        print(f"  Product features: {state.product_features.shape}")

        print("Step 4: Loading department features...")
        state.dept_features = pd.read_csv(state.feature_dir / "department_features.csv")
        print(f"  Dept features: {state.dept_features.shape}")

        print("Step 5: Loading aisle features...")
        state.aisle_features = pd.read_csv(state.feature_dir / "aisle_features.csv")
        print(f"  Aisle features: {state.aisle_features.shape}")

        print("Step 6: Loading product metadata...")
        state.products_metadata = pd.read_csv("data/raw/products.csv")
        print(f"  Products metadata: {state.products_metadata.shape}")

        print("Step 7: Loading cold start handler...")
        cold_start_cache = Path("models/saved/cold_start_cache.pkl")
        if not cold_start_cache.exists():
            raise FileNotFoundError(f"Cold start cache not found: {cold_start_cache}")

        state.cold_start_handler = joblib.load(cold_start_cache)
        print(f"  Cold start cache size: {cold_start_cache.stat().st_size / 1024 / 1024:.1f} MB")

        print("Step 8: Initializing F1 optimizer...")
        state.f1_optimizer = F1Optimizer()
        state.feature_cols = state.model.feature_cols

        print(f"Model uses {len(state.feature_cols)} features")
        print("API ready to serve")

    except Exception as exc:
        print(f"Error loading models: {exc}")
        raise


def get_user_candidates(user_id: int) -> pd.DataFrame:
    """Load candidate user-product rows and join all required feature tables."""
    up_file = state.feature_dir / "user_product_features.parquet"
    user_up_features = pd.read_parquet(up_file, filters=[("user_id", "==", user_id)])

    if len(user_up_features) == 0:
        return pd.DataFrame()

    user_pairs = user_up_features[["user_id", "product_id"]].copy()
    df = user_pairs.merge(user_up_features, on=["user_id", "product_id"], how="left")
    df = df.merge(state.user_features, on="user_id", how="left")
    df = df.merge(state.product_features, on="product_id", how="left")

    df = df.merge(
        state.products_metadata[["product_id", "department_id", "aisle_id"]],
        on="product_id",
        how="left",
    )
    df = df.merge(state.dept_features, on="department_id", how="left")
    df = df.merge(state.aisle_features, on="aisle_id", how="left")

    user_dept_file = state.feature_dir / "user_department_features.parquet"
    user_dept = pd.read_parquet(user_dept_file, filters=[("user_id", "==", user_id)])
    df = df.merge(user_dept, on=["user_id", "department_id"], how="left")

    return df


@app.get("/", response_model=Dict[str, str])
async def root() -> Dict[str, str]:
    """Return API metadata."""
    return {
        "message": "Instacart Recommendation API",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Report service health and artifact readiness."""
    return HealthResponse(
        status="healthy" if state.model is not None else "unhealthy",
        model_loaded=state.model is not None,
        model_version="lgbm_baseline_v1",
        features_count=len(state.feature_cols) if state.feature_cols else 0,
    )


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(request: RecommendRequest) -> RecommendResponse:
    """Generate recommendations for a given user."""
    start_time = time.time()

    try:
        user_id = request.user_id
        k = request.k
        strategy = request.strategy

        if state.model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        user_type = state.cold_start_handler.get_user_type(user_id)

        if user_type == "new" or strategy == "popularity":
            recs_df = state.cold_start_handler.recommend_for_new_user(n=k)
            strategy_used = "popularity_fallback"

        elif user_type == "cold" and strategy == "auto":
            candidates = get_user_candidates(user_id)
            if len(candidates) == 0:
                recs_df = state.cold_start_handler.recommend_for_new_user(n=k)
                strategy_used = "popularity_fallback"
            else:
                model_scores = state.model.predict(candidates)
                product_ids = candidates["product_id"].values
                blended_scores, alpha = state.cold_start_handler.blend_with_popularity(
                    user_id,
                    model_scores,
                    product_ids,
                )
                top_k_idx = np.argsort(blended_scores)[::-1][:k]
                recs_df = pd.DataFrame(
                    {
                        "product_id": product_ids[top_k_idx],
                        "score": blended_scores[top_k_idx],
                    }
                )
                strategy_used = f"blended_alpha_{alpha:.2f}"

        else:
            candidates = get_user_candidates(user_id)
            if len(candidates) == 0:
                recs_df = state.cold_start_handler.recommend_for_new_user(n=k)
                strategy_used = "popularity_fallback"
            else:
                model_scores = state.model.predict(candidates)
                sorted_idx = np.argsort(model_scores)[::-1]
                high_conf_idx = sorted_idx[model_scores[sorted_idx] > MODEL_SCORE_THRESHOLD]
                top_k_idx = high_conf_idx[:k] if len(high_conf_idx) > 0 else sorted_idx[:k]

                recs_df = pd.DataFrame(
                    {
                        "product_id": candidates.iloc[top_k_idx]["product_id"].values,
                        "score": model_scores[top_k_idx],
                    }
                )
                strategy_used = "lightgbm_per_user_f1"

        recs_df = recs_df.merge(
            state.products_metadata[["product_id", "product_name"]],
            on="product_id",
            how="left",
        )

        recommendations = [
            ProductRecommendation(
                product_id=int(row["product_id"]),
                product_name=row["product_name"],
                score=float(row["score"]),
                reason=_get_reason(strategy_used),
            )
            for _, row in recs_df.iterrows()
        ]

        latency_ms = (time.time() - start_time) * 1000
        return RecommendResponse(
            user_id=user_id,
            recommendations=recommendations,
            strategy_used=strategy_used,
            user_type=user_type,
            latency_ms=latency_ms,
        )

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating recommendations: {exc}",
        )


def _get_reason(strategy: str) -> str:
    """Return a reason string for the response payload."""
    if "popularity" in strategy:
        return "Popular choice"
    if "blended" in strategy:
        return "Based on your history and popular items"
    return "Frequently purchased by you"


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
