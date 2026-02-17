# scripts/prepare_cold_start_cache.py

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import joblib
from src.models.cold_start import ColdStartHandler

print("Preparing cold start cache...")

# Load data
print("Loading prior orders...")
prior_data = pd.read_parquet('data/processed/prior_orders.parquet')

print("Loading products...")
products = pd.read_csv('data/raw/products.csv')

# Fit cold start handler
print("Fitting ColdStartHandler...")
handler = ColdStartHandler()
handler.fit(prior_data, products)

# Save
output_path = Path('models/saved/cold_start_cache.pkl')
output_path.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(handler, output_path)

print(f"✅ Saved cold start cache to {output_path}")
print(f"   File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")