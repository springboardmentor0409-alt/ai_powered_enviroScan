import numpy as np
from scipy.stats import zscore

pollutants = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]

def detect_outliers(df):
    numeric_df = df[pollutants]
    z_scores = np.abs(zscore(numeric_df))
    outliers = (z_scores > 3).sum()

    print("\n=== Z-score Outliers Count (Threshold = 3) ===")
    print(outliers)
