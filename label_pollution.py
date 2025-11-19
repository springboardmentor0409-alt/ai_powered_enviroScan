"""
label_pollution.py

Reads data from CSV, adds 'season', handles NaNs, normalizes pollutant columns,
computes proximity features, assigns pollution-source labels based on rules,
computes a confidence score, and writes a processed CSV and a timestamped backup.

Usage:
    python scripts/label_pollution.py --input data/EnviroFinal_final_unlabled.csv --outdir .

Outputs:
    backups/EnviroFinal_final_unlabled_YYYYmmdd_HHMMSS.csv
    processed/EnviroFinal_final_unlabled_processed.csv
"""

import os
import argparse
from datetime import datetime
import math
from functools import partial

import numpy as np
import pandas as pd
from dateutil import parser

# -------------------------
# Configuration / thresholds
# -------------------------
POLLUTANT_COLS = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]

# Proximity decay scale factors (km)
K_ROAD = 0.3
K_INDUSTRY = 1.5
K_FARM = 4.0

# Rule thresholds (from your methodology)
TH_PM25_VEH = 40
TH_PM10_VEH = 90
TH_NO2_VEH = 20
TH_ROAD_DIST_VEH = 0.2

TH_SO2_IND = 8
TH_NO2_IND = 35
TH_IND_DIST = 1.5

TH_PM10_AGRI = 80
TH_FARM_DIST = 4.0

TH_PM25_BURN = 45  # for burning label if fire evidence present

# Photochemical rule thresholds (reasonable defaults)
# O3 will be compared to 75th percentile (calculated from data)
TH_TEMP_HIGH = 30.0      # Celsius
TH_HUMIDITY_LOW = 30.0   # percent
TH_NO2_LOW = 10.0        # µg/m3

# Confidence score weights
W_POLLUTANT = 0.65
W_PROXIMITY = 0.30
W_FIRE = 0.05

# -------------------------
# Utility functions
# -------------------------
def ensure_dirs(dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def backup_file(filepath, backup_dir="backups"):
    ensure_dirs([backup_dir])
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.basename(filepath)
    name, ext = os.path.splitext(base)
    dest = os.path.join(backup_dir, f"{name}_{ts}{ext}")
    pd.read_csv(filepath).to_csv(dest, index=False)  # copy preserving CSV encoding
    return dest

# -------------------------
# Robust date parsing + season
# -------------------------
def add_season_col(df, date_col="date"):
    """
    Adds: date_raw (original string), date_parsed (pd.Timestamp or NaT),
    date_parse_status ('OK' or 'FAILED'), and season ('Winter','Summer','Monsoon','Post-Monsoon','Unknown').
    This function is non-destructive: rows are NOT dropped if date fails.
    """
    print("Fixing inconsistent date formats...")

    # Keep original raw column
    df = df.copy()
    df["date_raw"] = df[date_col].astype(str)

    s = df["date_raw"].str.strip().replace("/", "-", regex=False)

    # Try vectorized parsing with a few common strict formats first — fast for large data
    parsed = pd.to_datetime(s, format="%d-%m-%Y", errors="coerce", dayfirst=True)
    parsed = parsed.fillna(pd.to_datetime(s, format="%m-%d-%Y", errors="coerce", dayfirst=False))
    parsed = parsed.fillna(pd.to_datetime(s, format="%Y-%m-%d", errors="coerce"))
    parsed = parsed.fillna(pd.to_datetime(s, format="%Y-%d-%m", errors="coerce"))  # rare but included

    # For remaining unparsed entries, fall back to dateutil parser (slower, but more flexible)
    mask_unparsed = parsed.isna()
    if mask_unparsed.any():
        # apply parser only to the remaining subset
        def try_parse(x):
            try:
                # try dayfirst first -> matches DD-MM or D-M
                return parser.parse(x, dayfirst=True)
            except Exception:
                try:
                    return parser.parse(x, dayfirst=False)
                except Exception:
                    return pd.NaT

        parsed_remaining = s[mask_unparsed].apply(try_parse)
        parsed.loc[mask_unparsed] = parsed_remaining

    # Assign parsed column
    df["date_parsed"] = parsed

    # Status column
    df["date_parse_status"] = np.where(df["date_parsed"].isna(), "FAILED", "OK")

    # Season calculation: if date_parsed is NaT -> 'Unknown'
    def season_from_ts(ts):
        if pd.isna(ts):
            return "Unknown"
        m = ts.month
        if m in (12, 1, 2):
            return "Winter"
        if m in (3, 4, 5):
            return "Summer"
        if m in (6, 7, 8, 9):
            return "Monsoon"
        if m in (10, 11):
            return "Post-Monsoon"
        return "Unknown"

    df["season"] = df["date_parsed"].apply(season_from_ts)

    return df

# -------------------------
# NaN handling
# -------------------------
def handle_nans(df):
    out = df.copy()

    # fill fire_count with 0 if missing (reasonable assumption)
    if "fire_count" in out.columns:
        out["fire_count"] = out["fire_count"].fillna(0)

    # fire_min_dist_km -> median if exists
    if "fire_min_dist_km" in out.columns:
        out["fire_min_dist_km"] = out["fire_min_dist_km"].fillna(out["fire_min_dist_km"].median())

    # numeric columns -> median (do not overwrite season or date_parsed)
    numeric_cols = out.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ("season",)]
    if numeric_cols:
        medians = out[numeric_cols].median()
        out[numeric_cols] = out[numeric_cols].fillna(medians)

    # categorical columns -> mode, excluding season and date_raw/date_parse_status
    obj_cols = out.select_dtypes(include=["object"]).columns.tolist()
    obj_cols = [c for c in obj_cols if c not in ("season", "date_raw", "date_parse_status")]
    for c in obj_cols:
        modes = out[c].mode(dropna=True)
        if not modes.empty:
            out[c] = out[c].fillna(modes.iloc[0])

    return out

# -------------------------
# Normalization / proximity / labeling (unchanged logic)
# -------------------------
def normalize_col(series):
    """Min-max normalize a pandas Series to [0,1].
       If min == max, fall back to z-score then logistic mapping to 0-1.
    """
    s = series.dropna()
    if s.empty:
        return series * 0.0  # all NaN -> zeros
    pmin = s.min()
    pmax = s.max()
    if pmax > pmin:
        return (series - pmin) / (pmax - pmin)
    # fallback: z-score to logistic mapping
    mean = s.mean()
    std = s.std()
    if std == 0 or np.isnan(std):
        return pd.Series(0.5, index=series.index)  # constant value
    z = (series - mean) / std
    # logistic to map to 0..1
    return 1 / (1 + np.exp(-z))

def compute_proximity(dist_series, k):
    # prox = exp(-d / k)
    d = dist_series.fillna(np.inf).astype(float)
    prox = np.exp(-d / float(k))
    prox = prox.fillna(0.0)
    return prox

def assign_label_and_confidence(row, o3_75th):
    pollutant_score = max([row.get(f"{c}_norm", 0.0) for c in POLLUTANT_COLS])

    prox_vals = [row.get("prox_road", 0.0), row.get("prox_industry", 0.0), row.get("prox_farm", 0.0)]
    proximity_score = max(prox_vals)

    fire_flag = 1 if (row.get("fire_nearby", 0) == 1 or row.get("fire_count", 0) >= 1) else 0

    label = "Natural"

    cond_veh_poll = (row.get("PM2.5", 0) >= TH_PM25_VEH) or (row.get("PM10", 0) >= TH_PM10_VEH) or (row.get("NO2", 0) >= TH_NO2_VEH)
    cond_veh_dist = (row.get("dist_to_road", np.inf) <= TH_ROAD_DIST_VEH)
    is_vehicular = cond_veh_poll and cond_veh_dist

    cond_ind_poll = (row.get("SO2", 0) >= TH_SO2_IND) or (row.get("NO2", 0) >= TH_NO2_IND)
    cond_ind_dist = (row.get("dist_to_industry", np.inf) <= TH_IND_DIST)
    is_industrial = cond_ind_poll and cond_ind_dist

    is_agricultural = (row.get("PM10", 0) >= TH_PM10_AGRI) and (row.get("dist_to_farm", np.inf) <= TH_FARM_DIST)

    fire_evidence = (row.get("fire_nearby", 0) == 1) or (row.get("fire_count", 0) >= 1)
    is_burning = fire_evidence and (row.get("PM2.5", 0) >= TH_PM25_BURN)

    o3_val = row.get("O3", np.nan)
    temp_val = row.get("temperature", np.nan)
    hum_val = row.get("humidity", np.nan)
    no2_val = row.get("NO2", np.nan)

    photochem_conditions = (
        (not pd.isna(o3_val) and o3_val >= o3_75th)
        and (
            (not pd.isna(temp_val) and temp_val >= TH_TEMP_HIGH)
            or (not pd.isna(hum_val) and hum_val <= TH_HUMIDITY_LOW)
            or (not pd.isna(no2_val) and no2_val <= TH_NO2_LOW)
        )
    )

    if is_burning:
        label = "Burning"
    elif is_industrial:
        label = "Industrial"
    elif is_vehicular:
        label = "Vehicular"
    elif is_agricultural:
        label = "Agricultural"
    elif photochem_conditions:
        label = "Photochemical"
    else:
        label = "Natural"

    C = (W_POLLUTANT * pollutant_score) + (W_PROXIMITY * proximity_score) + (W_FIRE * fire_flag)
    C = float(max(0.0, min(1.0, C)))

    return label, C

# -------------------------
# Main processing function
# -------------------------
def process(input_path, outdir):
    ensure_dirs([outdir, os.path.join(outdir, "backups"), os.path.join(outdir, "processed")])

    print("Creating backup...")
    backup_path = backup_file(input_path, backup_dir=os.path.join(outdir, "backups"))
    print("Backup saved to:", backup_path)

    print("Loading CSV...")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns")

    print("Adding season column...")
    df = add_season_col(df, date_col="date")

    print("Handling NaNs...")
    df = handle_nans(df)

    # Compute O3 75th percentile (for photochemical rule)
    if "O3" in df.columns:
        o3_75th = df["O3"].dropna().quantile(0.75)
    else:
        o3_75th = np.nan

    # Normalize pollutant columns and store as e.g. "PM2.5_norm"
    for col in POLLUTANT_COLS:
        if col in df.columns:
            df[f"{col}_norm"] = normalize_col(df[col])
        else:
            df[f"{col}_norm"] = 0.0

    # Proximity features
    if "dist_to_road" in df.columns:
        df["prox_road"] = compute_proximity(df["dist_to_road"], K_ROAD)
    else:
        df["prox_road"] = 0.0

    if "dist_to_industry" in df.columns:
        df["prox_industry"] = compute_proximity(df["dist_to_industry"], K_INDUSTRY)
    else:
        df["prox_industry"] = 0.0

    if "dist_to_farm" in df.columns:
        df["prox_farm"] = compute_proximity(df["dist_to_farm"], K_FARM)
    else:
        df["prox_farm"] = 0.0

    # Apply labeling and confidence computation row-wise
    labels = []
    confidences = []
    print("Assigning labels and computing confidence (this may take a moment)...")
    for _, row in df.iterrows():
        label, conf = assign_label_and_confidence(row, o3_75th=o3_75th)
        labels.append(label)
        confidences.append(round(conf, 4))

    df["pollution_label"] = labels
    df["label_confidence"] = confidences

    # Save processed file
    input_base = os.path.basename(input_path)
    out_path = os.path.join(outdir, "processed", os.path.splitext(input_base)[0] + "_processed.csv")
    df.to_csv(out_path, index=False)
    print("Processed file saved to:", out_path)

    # Print a small summary
    print("\nLabel distribution:")
    print(df["pollution_label"].value_counts(dropna=False))
    print("\nExample rows with labels and confidence:")
    # Show original raw date + parsed + labels to help debug rows with FAILED status
    show_cols = ["date_raw", "date_parsed", "date_parse_status", "city", "pollution_label", "label_confidence"]
    # if some columns missing, fall back to minimal view
    show_cols = [c for c in show_cols if c in df.columns]
    print(df[show_cols].head(10).to_string(index=False))

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Label pollution source and compute confidence")
    parser.add_argument("--input", "-i", type=str, default="data/EnviroFinal_final_unlabled.csv", help="Path to input CSV")
    parser.add_argument("--outdir", "-o", type=str, default=".", help="Base output directory (creates backups/ and processed/ within it)")
    args = parser.parse_args()

    process(args.input, args.outdir)
