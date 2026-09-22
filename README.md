# Instacart Recommendation System

A machine learning system for predicting grocery reorder behavior, trained on 13M+ user-product interactions.

## Problem

Given a user's purchase history, predict which previously purchased products will appear in their next order. This is useful for pre-filling carts and surfacing relevant items, and it addresses a long-tail problem where the top 20% of products account for 91% of sales.

Dataset: 206,209 users, 49,677 products, 32M+ prior order interactions, reduced to 13M user-product pairs for modeling.

## Results

| Metric | Popularity baseline | Model |
|---|---|---|
| F1 | 0.13 | 0.507 |
| AUC | - | 0.845 |
| Precision | - | 0.38 |
| Recall | - | 0.53 |

## Approach

The model is a LightGBM classifier over 45 engineered features, grouped into four categories:

- **User-product interaction (17 features)**: recency, frequency, and consistency of a given user buying a given item. The strongest predictors were orders elapsed since last purchase, total times bought, and consecutive-order streaks.
- **User features (9)**: order frequency, basket size, overall reorder ratio.
- **Product features (7)**: popularity, reorder rate, typical cart position.
- **Department/aisle features (12)**: category-level reorder rates and user preferences within categories.

LightGBM was chosen over deep learning approaches because the features are sparse and tabular, training is fast, and feature importances are directly interpretable.

### Per-user threshold optimization

Rather than applying one global probability threshold, the model selects the threshold that maximizes F1 separately for each user:

```python
for K in range(1, n_candidates):
    f1 = calculate_f1(top_K_predictions, true_labels)
best_K = argmax(f1)
```

This improved F1 from 0.438 to 0.507.

### Cold start

21.1% of users have fewer than 5 orders. These are handled with a blend of model output and popularity-based recommendations, weighted by order count:

| User type | Orders | Strategy |
|---|---|---|
| New | 0 | Popularity only |
| Cold | 1-4 | Blend: model × α + popularity × (1-α), α from 0.3 to 0.5 |
| Warm | 5+ | Model prediction, α = 0.7 |

## What didn't work

- **Two-stage ranking (ALS candidate generation + LightGBM ranking)**: coverage dropped to 55% and AUC fell from 0.83 to 0.72. The dataset already restricts candidates to each user's history (~65 items), so a separate candidate-generation stage added complexity without benefit.
- **A single global F1 threshold**: gave F1 = 0.438, well below the per-user optimum, since reorder rates vary substantially across users.
- **The initial feature set of 19 features**: gave AUC = 0.831. Expanding to 45 features, based on published approaches from top Kaggle solutions, improved this meaningfully — feature engineering mattered more than model choice here.

## Setup

```bash
git clone https://github.com/BessiePengjinWang/instacart-recommendation-system
cd instacart-recommendation-system
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Data (requires Kaggle API credentials)
kaggle competitions download -c instacart-market-basket-analysis
unzip instacart-market-basket-analysis.zip -d data/raw/

# Pipeline
python src/data/preprocessing.py
python src/features/feature_engineering.py
python src/data/dataset_builder.py

# Training
jupyter notebook notebooks/05_baseline_models.ipynb

# API
uvicorn src.api.main:app --reload
```

## API

### `GET /health`

Returns service and model status.

### `POST /recommend`

Request:
```json
{
  "user_id": 12345,
  "k": 10,
  "strategy": "auto"
}
```

`strategy` is one of `auto` (default; chooses based on user history), `model` (force LightGBM), or `popularity`.

Response:
```json
{
  "user_id": 12345,
  "recommendations": [
    {"product_id": 24852, "product_name": "Banana", "score": 0.85, "reason": "Frequently purchased by you"}
  ],
  "strategy_used": "lightgbm_per_user_f1",
  "user_type": "warm",
  "latency_ms": 45.2
}
```

Interactive docs available at `/docs` (Swagger) and `/redoc` once the service is running.

## Deployment

Local development (no Docker) uses roughly 3-4GB of memory:
```bash
uvicorn src.api.main:app --reload --port 8000
```

Docker deployment requires more memory (12GB recommended) because the app loads all user, product, and user-product interaction features into memory at once, along with the trained model:
```bash
docker build -t instacart-recommender:v1 .
docker-compose up -d
```

In a production deployment this memory footprint would be reduced (to roughly 2GB) by moving feature storage to Postgres/Redis or a feature store (Feast, Tecton) and serving the model through a dedicated serving framework (TensorFlow Serving, BentoML) instead of loading everything into process memory.

## Project structure

```
instacart-recommendation-system/
├── data/               # raw, processed, and feature data
├── notebooks/          # EDA through model training and cold-start analysis
├── src/
│   ├── data/           # preprocessing and dataset assembly
│   ├── features/       # feature engineering
│   ├── models/         # baselines, cold start, F1 optimization
│   └── api/             # FastAPI service
├── models/saved/       # trained model artifacts
└── docs/diagrams/       # architecture diagrams
```

## EDA highlights

- Peak ordering time is Monday 10am, consistent with weekly restocking.
- Overall reorder rate is 59.2%.
- 21.1% of users have fewer than 5 orders (cold start).

## References

- [Instacart Kaggle competition](https://www.kaggle.com/c/instacart-market-basket-analysis)
- [Top 2% solution writeup](https://jacquespeeters.github.io/2017/08/28/kaggle-instacart-solution-overview/)
- [2nd place solution writeup](https://medium.com/kaggle-blog/instacart-market-basket-analysis-feda2700cded)
- [LightGBM documentation](https://lightgbm.readthedocs.io)