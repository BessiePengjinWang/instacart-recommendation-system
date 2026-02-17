"""Pydantic request/response schemas for the recommendation API."""

from pydantic import BaseModel, Field
from typing import List, Optional

class RecommendRequest(BaseModel):
    """Request body for recommendation endpoint"""
    user_id: int = Field(..., description="User ID to generate recommendations for")
    k: int = Field(10, ge=1, le=50, description="Number of recommendations to return")
    strategy: str = Field(
        "auto",
        description="Recommendation strategy: 'auto', 'model', 'popularity'"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": 12345,
                "k": 10,
                "strategy": "auto"
            }
        }


class ProductRecommendation(BaseModel):
    """Single product recommendation"""
    product_id: int
    product_name: str
    score: float = Field(..., ge=0.0, le=1.0)
    reason: Optional[str] = None


class RecommendResponse(BaseModel):
    """Response from recommendation endpoint"""
    user_id: int
    recommendations: List[ProductRecommendation]
    strategy_used: str
    user_type: str  # "warm", "cold", "new"
    latency_ms: float
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": 12345,
                "recommendations": [
                    {
                        "product_id": 24852,
                        "product_name": "Banana",
                        "score": 0.85,
                        "reason": "Frequently purchased"
                    }
                ],
                "strategy_used": "lightgbm_per_user_f1",
                "user_type": "warm",
                "latency_ms": 45.2
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    model_version: str
    features_count: int
