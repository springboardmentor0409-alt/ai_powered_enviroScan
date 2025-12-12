# utilities.py
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np
from pathlib import Path
from model_loader import get_artifacts
import math

class PollutionInput(BaseModel):
    date: str  # DD-MM-YYYY
    city: str
    location_id: str
    latitude: float
    longitude: float
    PM2_5: Optional[float] = Field(None, alias="PM2_5")
    PM10: Optional[float]
    NO2: Optional[float]
    SO2: Optional[float]
    CO: Optional[float]
    O3: Optional[float]
    temp: Optional[float]
    humidity: Optional[float]
    wind_speed: Optional[float]
    wind_dir: Optional[float]
    dist_to_road: Optional[float]
    dist_to_industry: Optional[float]
    dist_to_farm: Optional[float]
    traffic_index: Optional[float]
    fire_nearby: Optional[int]
    fire_min_dist_km: Optional[float]

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "date": "15-01-2023",
                "city": "Delhi",
                "location_id": "LOC_001",
                "latitude": 28.61,
                "longitude": 77.23,
                "PM2_5": 120.5,
                "PM10": 200.1,
                "NO2": 45.2,
                "SO2": 12.5,
                "CO": 1.2,
                "O3": 30.5,
                "temp": 15.2,
                "humidity": 60.5,
                "wind_speed": 5.5,
                "wind_dir": 120.0,
                "dist_to_road": 500.0,
                "dist_to_industry": 2000.0,
                "dist_to_farm": 5000.0,
                "traffic_index": 80.5,
                "fire_nearby": 0,
                "fire_min_dist_km": 50.0
            }
        }

def preprocess_input(data: PollutionInput) -> pd.DataFrame:
    """
    Convert a PollutionInput to a preprocessed pandas DataFrame ready for model input.
    Uses artifacts loaded into model_loader.get_artifacts()
    """
    artifacts = get_artifacts() or {}
    input_dict = data.dict(by_alias=True)
    # Map Pydantic PM2_5 to DF column PM2.5
    mapping = {"PM2_5": "PM2.5"}
    input_dict = {mapping.get(k, k): v for k, v in input_dict.items()}

    df = pd.DataFrame([input_dict])

    # Date handling
    df['date'] = pd.to_datetime(df['date'], format="%d-%m-%Y", errors='coerce')
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['dayofyear'] = df['date'].dt.dayofyear
    df['season'] = df['month'].map({
        12:"winter",1:"winter",2:"winter",
        3:"summer",4:"summer",5:"summer",
        6:"monsoon",7:"monsoon",8:"monsoon",
        9:"post",10:"post",11:"post"
    })

    pollutant_cols = ["PM2.5","PM10","NO2","SO2","CO","O3"]

    # 1. Clip Outliers
    clip_limits = artifacts.get("clip_limits", {})
    for col in pollutant_cols:
        if col in clip_limits and col in df.columns:
            low, high = clip_limits[col]
            df[col] = df[col].astype(float).clip(lower=low, upper=high)

    # 2. Imputation (KNN) - Spatial
    knn_cols = ["dist_to_road","dist_to_industry","dist_to_farm"]
    if "imputer" in artifacts:
        try:
            imputer = artifacts["imputer"]
            # ensure all knn_cols present
            for c in knn_cols:
                if c not in df.columns:
                    df[c] = np.nan
            df[knn_cols] = imputer.transform(df[knn_cols])
        except Exception:
            # don't crash on missing artifact or transform failure
            pass

    # 3. Fill Missing with Median
    median_values = artifacts.get("median_values", {})
    for col in pollutant_cols:
        if col in df.columns and pd.isna(df[col].iloc[0]):
            df[col] = df[col].fillna(median_values.get(col, 0))

    # 4. Wind direction -> components
    df["wind_dir"] = df.get("wind_dir", np.nan)
    df["wind_dir_rad"] = np.deg2rad(df["wind_dir"].fillna(0))
    df["wind_u"] = np.cos(df["wind_dir_rad"])
    df["wind_v"] = np.sin(df["wind_dir_rad"])

    # 5. Scaling (if scalers present)
    if "scaler_pollutants" in artifacts and "scaler_meta" in artifacts:
        try:
            scaler_pollutants = artifacts["scaler_pollutants"]
            scaler_meta = artifacts["scaler_meta"]

            meta_cols = ["temp","humidity","wind_speed","traffic_index",
                        "dist_to_road","dist_to_industry","dist_to_farm","fire_min_dist_km"]

            # ensure columns exist
            for c in pollutant_cols:
                if c not in df.columns:
                    df[c] = np.nan
            for c in meta_cols:
                if c not in df.columns:
                    df[c] = np.nan

            p_scaled = scaler_pollutants.transform(df[pollutant_cols])
            for i, c in enumerate(pollutant_cols):
                df[f"{c}_s"] = p_scaled[:, i]

            m_scaled = scaler_meta.transform(df[meta_cols])
            for i, c in enumerate(meta_cols):
                df[f"{c}_s"] = m_scaled[:, i]
        except Exception:
            # if transformers fail, continue without scaled columns
            pass

    # 6. Wind alignment synthetic bearings
    def align(w_u, w_v, bearing_deg):
        b_rad = np.deg2rad(bearing_deg)
        return (w_u*np.cos(b_rad) + w_v*np.sin(b_rad) + 1) / 2

    lat = df["latitude"].iloc[0]
    lon = df["longitude"].iloc[0]
    # synthetic bearings (keep same formula as original)
    df["road_bearing"] = (lat * 31 + lon * 17) % 360
    df["industry_bearing"] = (lat * 41 + lon * 23) % 360
    df["farm_bearing"] = (lat * 53 + lon * 29) % 360
    df["fire_bearing"] = (lat * 67 + lon * 11) % 360

    df["align_r"] = align(df["wind_u"], df["wind_v"], df["road_bearing"])
    df["align_i"] = align(df["wind_u"], df["wind_v"], df["industry_bearing"])
    df["align_f"] = align(df["wind_u"], df["wind_v"], df["farm_bearing"])
    df["align_fire"] = align(df["wind_u"], df["wind_v"], df["fire_bearing"])

    # Final: return dataframe (model expects specific column order—ensure your training pipeline columns align)
    return df
