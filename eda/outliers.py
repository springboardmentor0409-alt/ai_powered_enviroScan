import os
import numpy as np
import pandas as pd
from scipy.stats import zscore

pollutants = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def detect_outliers(df, output_dir):
    ensure_dir(output_dir)

    print("\n=== OUTLIER ANALYSIS (Z-score & ZERO values) ===")

    numeric_df = df[pollutants]

    # ---- Z-score Outliers ----
    z_scores = np.abs(zscore(numeric_df))
    z_outliers = (z_scores > 3).sum()

    print("\n📌 Z-score Outliers (|z| > 3):")
    print(z_outliers)

    # ---- Zero-value Outliers ----
    zero_outliers = (numeric_df == 0).sum()

    print("\n📌 Zero-value Outliers (unrealistic pollutant readings = 0):")
    print(zero_outliers)

    # ---- Combine both ----
    combined_outliers = z_outliers + zero_outliers

    print("\n📌 Combined Outliers (Z-score + Zero values):")
    print(combined_outliers)

    # ---- Save to CSV ----
    out_df = pd.DataFrame({
        "zscore_outliers": z_outliers,
        "zero_outliers": zero_outliers,
        "combined_outliers": combined_outliers
    })

    out_df.to_csv(f"{output_dir}/outliers.csv")

    print(f"\nOutlier report saved to: {output_dir}/outliers.csv\n")

    return out_df
