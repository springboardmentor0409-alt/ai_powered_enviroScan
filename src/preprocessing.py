import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler

def preprocess_pollution_data(input_path: str, output_path: str):

    print("Loading dataset...")
    df = pd.read_csv(input_path)

    # DATE HANDLING
    df['date'] = pd.to_datetime(df['date'], format="%d-%m-%Y", errors='coerce')
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['dayofyear'] = df['date'].dt.dayofyear

    # Seasonal categories
    df['season'] = df['month'].map({
        12:"winter",1:"winter",2:"winter",
        3:"summer",4:"summer",5:"summer",
        6:"monsoon",7:"monsoon",8:"monsoon",
        9:"post",10:"post",11:"post"
    })

    # TREAT ZERO VALUES AS MISSING (from EDA)
    pollutant_cols = ["PM2.5","PM10","NO2","SO2","CO","O3"]
    for col in pollutant_cols:
        df[col] = df[col].replace(0, np.nan)

    # OUTLIER CLIPPING (1st–99th percentile)
    print("Clipping extreme outliers...")
    for col in pollutant_cols:
        low = df[col].quantile(0.01)
        high = df[col].quantile(0.99)
        df[col] = df[col].clip(lower=low, upper=high)

    # IMPUTATION
    print("Imputing spatial and pollution columns...")
    
    # Spatial columns using KNN
    knn_cols = ["dist_to_road","dist_to_industry","dist_to_farm"]
    imputer = KNNImputer(n_neighbors=5)
    df[knn_cols] = imputer.fit_transform(df[knn_cols])

    # lat/lon should never change → ffill/bfill
    df["latitude"] = df["latitude"].ffill().bfill()
    df["longitude"] = df["longitude"].ffill().bfill()

    # Fire distance → fill missing with minimum
    df["fire_min_dist_km"] = df["fire_min_dist_km"].fillna(df["fire_min_dist_km"].min())

    # Remaining missing pollutant values → median
    for col in pollutant_cols:
        df[col] = df[col].fillna(df[col].median())

    # WIND DIRECTION → SIN & COS
    print("Encoding wind direction...")
    df["wind_dir_rad"] = np.deg2rad(df["wind_dir"])
    df["wind_u"] = np.cos(df["wind_dir_rad"])
    df["wind_v"] = np.sin(df["wind_dir_rad"])

    # SCALING (RobustScaler)
    print("Scaling features (RobustScaler)...")

    meta_cols = ["temp","humidity","wind_speed","traffic_index",
                 "dist_to_road","dist_to_industry","dist_to_farm","fire_min_dist_km"]

    scaler = RobustScaler()
    df[[c+"_s" for c in pollutant_cols]] = scaler.fit_transform(df[pollutant_cols])
    df[[c+"_s" for c in meta_cols]] = scaler.fit_transform(df[meta_cols])

    # WIND ALIGNMENT VECTOR
    print("Computing wind alignment...")

    def align(w_u, w_v, bearing_deg):
        b_rad = np.deg2rad(bearing_deg)
        return (w_u*np.cos(b_rad) + w_v*np.sin(b_rad) + 1) / 2

    # Synthetic bearings (consistent & deterministic)
    df["road_bearing"] = (df["latitude"]*31 + df["longitude"]*17) % 360
    df["industry_bearing"] = (df["latitude"]*41 + df["longitude"]*23) % 360
    df["farm_bearing"] = (df["latitude"]*53 + df["longitude"]*29) % 360
    df["fire_bearing"] = (df["latitude"]*67 + df["longitude"]*11) % 360

    df["align_r"] = align(df["wind_u"], df["wind_v"], df["road_bearing"])
    df["align_i"] = align(df["wind_u"], df["wind_v"], df["industry_bearing"])
    df["align_f"] = align(df["wind_u"], df["wind_v"], df["farm_bearing"])
    df["align_fire"] = align(df["wind_u"], df["wind_v"], df["fire_bearing"])

    # SOURCE SCORE MODEL (Corrected)
    print("Calculating source label scores...")

    fire_flag = (df["fire_nearby"] == 1).astype(float)

    veh = (0.45*df["NO2_s"] + 0.35*df["CO_s"] + 0.25*df["PM2.5_s"]
           + 0.20*df["traffic_index_s"] + 0.15*(1-df["dist_to_road_s"])) * (0.8 + 0.2*df["align_r"])

    ind = (0.45*df["SO2_s"] + 0.35*df["PM10_s"] + 0.20*df["PM2.5_s"]
           + 0.15*(1-df["dist_to_industry_s"])) * (0.8 + 0.2*df["align_i"])

    ag = (0.40*df["PM10_s"] + 0.30*df["humidity_s"]
          + 0.20*(1-df["dist_to_farm_s"]) + 0.10*fire_flag) * (0.8 + 0.2*df["align_f"])

    burn = (0.50*fire_flag + 0.30*(1-df["fire_min_dist_km_s"])
            + 0.25*df["PM2.5_s"] + 0.25*df["PM10_s"]) * (0.8 + 0.2*df["align_fire"])

    score_matrix = np.vstack([veh, ind, ag, burn]).T
    labels = np.array(["Vehicular","Industrial","Agricultural","Burning"])
    df["source_label"] = labels[np.argmax(score_matrix, axis=1)]

    # SEASON-AWARE SOFT BALANCING
    print("Soft balancing using seasonal proportions...")

    final_parts = []
    for cls in labels:
        class_df = df[df["source_label"] == cls]

        seasonal = class_df.groupby("season").apply(
            lambda x: x.sample(
                max(5000, len(x)), 
                replace=len(x)<5000,
                random_state=42
            )
        )
        final_parts.append(seasonal.reset_index(drop=True))

    final_df = pd.concat(final_parts).sample(len(df), replace=True, random_state=42)

    # SAVE OUTPUT
    print(f"Saving processed output → {output_path}")
    final_df.to_csv(output_path, index=False)
    print("\n Preprocessing completed successfully!\n")


if __name__ == "__main__":
    preprocess_pollution_data(
        "data/unlabeled_pollution_data.csv",
        "data/labeled_pollution_data.csv"
    )
