# scripts/convert_features_to_parquet.py

import pandas as pd
from pathlib import Path

print("Converting large CSV files to Parquet...")

feature_dir = Path("data/features")

# Convert user_product_features (1.4GB)
print("Converting user_product_features.csv...")
df = pd.read_csv(feature_dir / "user_product_features.csv")
df.to_parquet(feature_dir / "user_product_features.parquet", index=False)
print(f"  ✅ Saved as parquet ({(feature_dir / 'user_product_features.parquet').stat().st_size / 1024 / 1024:.1f} MB)")

# Convert user_department_features (50MB)
print("Converting user_department_features.csv...")
df = pd.read_csv(feature_dir / "user_department_features.csv")
df.to_parquet(feature_dir / "user_department_features.parquet", index=False)
print(f"  ✅ Saved as parquet ({(feature_dir / 'user_department_features.parquet').stat().st_size / 1024 / 1024:.1f} MB)")

print("\n✅ Conversion complete!")