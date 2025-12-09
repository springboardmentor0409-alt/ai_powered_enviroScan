import joblib
import pandas as pd

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

# ----------- DEFAULT MISSING VALUE FILLER -------------
default_values = {
    # Required columns that preprocessor expects
    "city": "UnknownCity",
    "location_id": "LOC_0",

    # engineered / unused fields (keep zero)
    "wind_dir_rad": 3.14,
    "wind_u": 0,
    "wind_v": 0,
    "PM2.5_s": 0, "PM10_s": 0, "NO2_s": 0, "SO2_s": 0, "CO_s": 0, "O3_s": 0,
    "temp_s": 0, "humidity_s": 0, "wind_speed_s": 0,
    "traffic_index_s": 0,
    "dist_to_road_s": 0, "dist_to_industry_s": 0, "dist_to_farm_s": 0,
    "fire_min_dist_km_s": 0,
    "road_bearing": 0, "industry_bearing": 0, "farm_bearing": 0, "fire_bearing": 0,
    "align_r": 0, "align_i": 0, "align_f": 0, "align_fire": 0,

    # unused but required structure
    "year": 2020,
    "month": 1,
    "dayofyear": 1,
    "dist_to_farm": 1.0,
}

# ---------------------------------------------------
#         USER INPUT PREDICTION FUNCTION
# ---------------------------------------------------
def get_user_input():
    print("\nENTER IMPORTANT PARAMETERS ONLY")
    print("-----------------------------------")

    season = input("Season (summer/winter/autumn/monsoon): ")

    PM25 = float(input("PM2.5: "))
    PM10 = float(input("PM10: "))
    NO2 = float(input("NO2: "))
    SO2 = float(input("SO2: "))
    CO = float(input("CO: "))
    O3 = float(input("O3: "))

    temp = float(input("Temperature (°C): "))
    humidity = float(input("Humidity (%): "))
    wind_speed = float(input("Wind Speed: "))

    # ------- NEW REQUIRED INPUTS -------
    wind_dir = float(input("Wind Direction (0–360 degrees): "))
    dist_to_road = float(input("Distance to nearest road (km): "))
    dist_to_industry = float(input("Distance to nearest industry (km): "))
    fire_nearby = int(input("Is there a fire nearby? (1=Yes, 0=No): "))
    fire_min_dist_km = float(input("Minimum distance to fire (km): "))

    traffic_index = float(input("Traffic Index: "))

    return {
        "season": season,
        "PM2.5": PM25,
        "PM10": PM10,
        "NO2": NO2,
        "SO2": SO2,
        "CO": CO,
        "O3": O3,
        "temp": temp,
        "humidity": humidity,
        "wind_speed": wind_speed,

        # new keys
        "wind_dir": wind_dir,
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
    print(f"\nLoading Model {model_choice} ...")
    model = joblib.load(MODEL_PATHS[model_choice])
    encoder = joblib.load(ENCODER_PATHS[model_choice])

    # Merge user data + default values
    full_input = {**default_values, **user_inputs}
    df = pd.DataFrame([full_input])

    # Predict
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

    if choice.strip() == "":
        model_choice = 1
    else:
        model_choice = int(choice)

    user_inputs = get_user_input()
    result = predict_with_model(model_choice, user_inputs)

    print("\n==============================")
    print(" Predicted Pollution Source →", result)
    print("==============================\n")
