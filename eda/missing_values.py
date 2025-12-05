import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def check_missing(df, output_dir):
    ensure_dir(output_dir)

    print("\n=== MISSING VALUES REPORT (NaN + Zero) ===")

    # --- True missing ---
    missing_nan = df.isnull().sum()

    # --- Zero-value missing (only numeric) ---
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    missing_zero = (df[numeric_cols] == 0).sum()
    missing_zero = missing_zero.reindex(df.columns, fill_value=0)

    # --- Combined ---
    combined = missing_nan + missing_zero

    print("\nMissing (NaN):")
    print(missing_nan)

    print("\nMissing (Zero values):")
    print(missing_zero)

    print("\nCombined Missing:")
    print(combined)

    # Save CSVs
    missing_nan.to_csv(f"{output_dir}/missing_nan.csv")
    missing_zero.to_csv(f"{output_dir}/missing_zero.csv")
    combined.to_csv(f"{output_dir}/missing_combined.csv")

    # =====================================================
    # FIXED: Boolean mask for heatmap
    # =====================================================
    mask_nan = df.isnull()
    mask_zero = (df[numeric_cols] == 0).reindex(columns=df.columns, fill_value=False)

    # Combine correctly
    mask = (mask_nan | mask_zero).astype(bool)

    # Plot heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(mask, cbar=False)
    plt.title("Missing Values Heatmap (NaN + Zero-values)")

    plt.savefig(f"{output_dir}/missing_heatmap.png")
    plt.close()

    print(f"\nSaved missing value reports and heatmap to {output_dir}\n")

    return combined
