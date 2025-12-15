import joblib
import pandas as pd
from datetime import datetime
import math

# ----------- MODEL PATHS -------------
MODEL_PATHS = {
    1: "../models/random_forest/random_forest.joblib",
    2: "../models/logistic_regression/logistic_regression.joblib",
    3: "../models/xgboost_model/xgboost.joblib",
    4: "../models/decision_tree/decision_tree.joblib",
}

ENCODER_PATHS = {
    1: "../models/random_forest/label_encoder.joblib",
    2: "../models/logistic_regression/label_encoder.joblib",
    3: "../models/xgboost_model/label_encoder.joblib",
    4: "../models/decision_tree/label_encoder.joblib",
}

# ----------- DEFAULT VALUES (REQUIRED BY PIPELINE) -------------
default_values = {
    "PM2.5_s": 0, "PM10_s": 0, "NO2_s": 0, "SO2_s": 0, "CO_s": 0, "O3_s": 0,
    "temp_s": 0, "humidity_s": 0, "wind_speed_s": 0,
    "traffic_index_s": 0,
    "dist_to_road_s": 0, "dist_to_industry_s": 0, "dist_to_farm_s": 0,
    "fire_min_dist_km_s": 0,
    "road_bearing": 0, "industry_bearing": 0,
    "farm_bearing": 0, "fire_bearing": 0,
    "align_r": 0, "align_i": 0, "align_f": 0, "align_fire": 0,
    "dist_to_farm": 1.0
}

# ----------- SEASON INFERENCE -------------
def infer_season(month):
    if month in [12, 1, 2]:
        return "winter"
    elif month in [3, 4, 5]:
        return "summer"
    elif month in [6, 7, 8]:
        return "monsoon"
    else:
        return "autumn"

# ---------------------------------------------------
#         USER INPUT FUNCTION (FINAL)
# ---------------------------------------------------
def get_user_input():
    print("\nENTER REQUIRED PARAMETERS")
    print("-----------------------------------")

    date_str = input("Date (YYYY-MM-DD): ")
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")

    city = input("City: ")
    location_id = input("Location ID: ")
    latitude = float(input("Latitude: "))
    longitude = float(input("Longitude: "))

    PM25 = float(input("PM2.5: "))
    PM10 = float(input("PM10: "))
    NO2 = float(input("NO2: "))
    SO2 = float(input("SO2: "))
    CO = float(input("CO: "))
    O3 = float(input("O3: "))

    temp = float(input("Temperature (°C): "))
    humidity = float(input("Humidity (%): "))
    wind_speed = float(input("Wind Speed (m/s): "))
    wind_dir = float(input("Wind Direction (0–360): "))

    dist_to_road = float(input("Distance to nearest road (km): "))
    dist_to_industry = float(input("Distance to nearest industry (km): "))
    fire_nearby = int(input("Fire nearby? (1=Yes, 0=No): "))
    fire_min_dist_km = float(input("Minimum distance to fire (km): "))
    traffic_index = float(input("Traffic Index: "))

    wind_dir_rad = math.radians(wind_dir)

    return {
        "date": date_str,
        "city": city,
        "location_id": location_id,
        "latitude": latitude,
        "longitude": longitude,

        "year": date_obj.year,
        "month": date_obj.month,
        "dayofyear": date_obj.timetuple().tm_yday,
        "season": infer_season(date_obj.month),

        "PM2.5": PM25,
        "PM10": PM10,
        "NO2": NO2,
        "SO2": SO2,
        "CO": CO,
        "O3": O3,

        "temp": temp,
        "humidity": humidity,
        "wind_speed": wind_speed,
        "wind_dir": wind_dir,
        "wind_dir_rad": wind_dir_rad,
        "wind_u": wind_speed * math.cos(wind_dir_rad),
        "wind_v": wind_speed * math.sin(wind_dir_rad),

        "dist_to_road": dist_to_road,
        "dist_to_industry": dist_to_industry,
        "fire_nearby": fire_nearby,
        "fire_min_dist_km": fire_min_dist_km,
        "traffic_index": traffic_index,
    }

# ---------------------------------------------------
#        MODEL LOADER + PREDICTOR
# ---------------------------------------------------
def predict_with_model(model_choice, user_inputs):
    print(f"\nLoading model {model_choice} ...")

    model = joblib.load(MODEL_PATHS[model_choice])
    encoder = joblib.load(ENCODER_PATHS[model_choice])

    full_input = {**default_values, **user_inputs}
    df = pd.DataFrame([full_input])

    pred = model.predict(df)[0]
    label = encoder.inverse_transform([pred])[0]

    return label

# ---------------------------------------------------
#                     MAIN
# ---------------------------------------------------
if __name__ == "__main__":

    print("\n===== SELECT MODEL FOR PREDICTION =====")
    print("1. Random Forest (default)")
    print("2. Logistic Regression")
    print("3. XGBoost")
    print("4. Decision Tree")

    choice = input("\nEnter model number (Press Enter for default): ")
    model_choice = int(choice) if choice.strip() else 1

    user_inputs = get_user_input()
    result = predict_with_model(model_choice, user_inputs)

    print("\n==============================")
    print(" Predicted Pollution Source →", result)
    print("==============================\n")
