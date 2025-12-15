import streamlit as st
import pandas as pd
import joblib
import math
from pathlib import Path
from datetime import date

# -----------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------
st.set_page_config(
    page_title="EnviroScan – AI Pollution Source Identifier",
    layout="wide"
)

# -----------------------------------------------------------
# UI STYLE
# -----------------------------------------------------------
st.markdown("""
<style>
body { background-color: #f4f6f8; }
h1, h2, h3 { color: #2E7D32 !important; font-weight: 700; }
.card {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 14px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
    margin-bottom: 25px;
}
.stButton>button {
    background-color: #2E7D32;
    color: white;
    font-size: 18px;
    padding: 10px 25px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# MODEL LOADER
# -----------------------------------------------------------
def load_model(model_name):
    model_paths = {
        "Random Forest": "../models/random_forest/random_forest.joblib",
        "Logistic Regression": "../models/logistic_regression/logistic_regression.joblib",
        "XGBoost": "../models/xgboost_model/xgboost.joblib",
        "Decision Tree": "../models/decision_tree/decision_tree.joblib",
    }
    encoder_paths = {
        "Random Forest": "../models/random_forest/label_encoder.joblib",
        "Logistic Regression": "../models/logistic_regression/label_encoder.joblib",
        "XGBoost": "../models/xgboost_model/label_encoder.joblib",
        "Decision Tree": "../models/decision_tree/label_encoder.joblib",
    }
    return joblib.load(model_paths[model_name]), joblib.load(encoder_paths[model_name])

# -----------------------------------------------------------
# HELPER: SEASON FROM DATE (INTERNAL)
# -----------------------------------------------------------
def infer_season(month):
    if month in [12, 1, 2]:
        return "winter"
    elif month in [3, 4, 5]:
        return "summer"
    elif month in [6, 7, 8]:
        return "monsoon"
    else:
        return "autumn"

# -----------------------------------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------------------------------
st.sidebar.title("🌿 Navigation")
menu = st.sidebar.radio(
    "Select Page",
    ["Home", "Predict Source", "Model Insights", "Data Visualization", "About Project"],
    label_visibility="collapsed"
)

# -----------------------------------------------------------
# HOME
# -----------------------------------------------------------
if menu == "Home":
    st.title("🌍 EnviroScan: AI-Powered Pollution Source Identifier")
    st.subheader("Machine Learning + Geospatial Analytics")

    st.markdown("""
EnviroScan identifies the **most probable source of air pollution**
using environmental measurements and spatial context.

### Pollution Sources Identified
- Vehicular Emissions  
- Industrial Pollution  
- Agricultural Burning  
- Natural Causes  
""")

# -----------------------------------------------------------
# PREDICT SOURCE
# -----------------------------------------------------------
elif menu == "Predict Source":

    st.title("🔍 Predict Pollution Source")

    model_choice = st.selectbox(
        "Select Model",
        ["Random Forest", "Logistic Regression", "XGBoost", "Decision Tree"]
    )

    model, encoder = load_model(model_choice)

    with st.form("prediction_form"):
        st.subheader("📍 Location & Time")
        colA, colB = st.columns(2)

        with colA:
            date_val = st.date_input("Date", value=date.today())
            city = st.text_input("City", value="Hyderabad")
            location_id = st.text_input("Location ID", value="LOC_001")

        with colB:
            latitude = st.number_input("Latitude", value=17.3850, format="%.4f")
            longitude = st.number_input("Longitude", value=78.4867, format="%.4f")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🌫 Pollution Levels")
            PM25 = st.number_input("PM2.5", value=80.0)
            PM10 = st.number_input("PM10", value=120.0)
            NO2 = st.number_input("NO2", value=30.0)
            SO2 = st.number_input("SO2", value=8.0)
            CO = st.number_input("CO", value=1.2)
            O3 = st.number_input("O3", value=25.0)

            st.subheader("🌦 Weather")
            temp = st.number_input("Temperature (°C)", value=28.0)
            humidity = st.number_input("Humidity (%)", value=60.0)
            wind_speed = st.number_input("Wind Speed (m/s)", value=2.0)

        with col2:
            st.subheader("🌍 Spatial Factors")
            wind_dir = st.number_input("Wind Direction (°)", value=180.0)
            dist_to_road = st.number_input("Distance to Road (km)", value=0.2)
            dist_to_industry = st.number_input("Distance to Industry (km)", value=5.0)
            dist_to_farm = st.number_input("Distance to Farm (km)", value=1.0)
            traffic = st.number_input("Traffic Index", value=40.0)

            st.subheader("🔥 Fire Data")
            fire_nearby = st.selectbox("Fire Nearby?", [0, 1])
            fire_min_dist_km = st.number_input("Fire Distance (km)", value=15.0)

        submit = st.form_submit_button("Predict Pollution Source")

    if submit:
        wind_rad = math.radians(wind_dir)

        df = pd.DataFrame([{
            "date": date_val.strftime("%Y-%m-%d"),
            "city": city,
            "location_id": location_id,
            "latitude": latitude,
            "longitude": longitude,
            "year": date_val.year,
            "month": date_val.month,
            "dayofyear": date_val.timetuple().tm_yday,
            "season": infer_season(date_val.month),
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
            "wind_dir_rad": wind_rad,
            "wind_u": wind_speed * math.cos(wind_rad),
            "wind_v": wind_speed * math.sin(wind_rad),
            "dist_to_road": dist_to_road,
            "dist_to_industry": dist_to_industry,
            "dist_to_farm": dist_to_farm,
            "fire_nearby": fire_nearby,
            "fire_min_dist_km": fire_min_dist_km,
            "traffic_index": traffic,

            # placeholders
            "PM2.5_s": 0, "PM10_s": 0, "NO2_s": 0, "SO2_s": 0,
            "CO_s": 0, "O3_s": 0, "temp_s": 0, "humidity_s": 0,
            "wind_speed_s": 0, "traffic_index_s": 0,
            "dist_to_road_s": 0, "dist_to_industry_s": 0,
            "dist_to_farm_s": 0, "fire_min_dist_km_s": 0,
            "road_bearing": 0, "industry_bearing": 0,
            "farm_bearing": 0, "fire_bearing": 0,
            "align_r": 0, "align_i": 0,
            "align_f": 0, "align_fire": 0
        }])

        pred = model.predict(df)[0]
        label = encoder.inverse_transform([pred])[0]
        st.success(f"🌱 Predicted Pollution Source: **{label}**")

# -----------------------------------------------------------
# MODEL INSIGHTS
# -----------------------------------------------------------
elif menu == "Model Insights":

    st.title("📈 Model Insights")

    model_select = st.selectbox(
        "Choose Model",
        ["Random Forest", "Logistic Regression", "XGBoost", "Decision Tree"]
    )

    folder_map = {
        "Random Forest": "random_forest",
        "Logistic Regression": "logistic_regression",
        "XGBoost": "xgboost_model",
        "Decision Tree": "decision_tree",
    }

    base_folder = Path("../results") / folder_map[model_select]

    explanations = {
        "confusion_matrix": "Confusion matrix showing correct and incorrect predictions.",
        "classification_report": "Precision, recall, and F1-score for each class.",
        "feature_importance": "Most influential features for model decisions.",
        "cv_f1": "Cross-validation F1 score stability."
    }

    if not base_folder.exists():
        st.error("No results found for this model.")
    else:
        for img in sorted(base_folder.glob("*.png")):
            st.subheader(img.stem.replace("_", " ").title())
            st.image(str(img), width="stretch")
            for key in explanations:
                if key in img.name:
                    st.caption(explanations[key])
            st.markdown("---")

# -----------------------------------------------------------
# DATA VISUALIZATION (EDA)
# -----------------------------------------------------------
elif menu == "Data Visualization":

    st.title("📊 Data Visualization (EDA)")

    eda_root = Path("../results/eda")

    if not eda_root.exists():
        st.error("EDA folder not found.")
    else:
        for section in ["", "boxplots", "distributions"]:
            folder = eda_root if section == "" else eda_root / section
            if folder.exists():
                st.subheader(section.capitalize() if section else "Overall EDA")
                cols = st.columns(2)
                for i, img in enumerate(sorted(folder.glob("*.png"))):
                    with cols[i % 2]:
                        st.image(
                            str(img),
                            caption=img.stem.replace("_", " ").title(),
                            width="stretch"
                        )

# -----------------------------------------------------------
# ABOUT
# -----------------------------------------------------------
elif menu == "About Project":
    st.title("ℹ️ About EnviroScan")
    st.markdown("""
EnviroScan uses Machine Learning and geospatial indicators
to **identify pollution sources**, not just pollution levels.
""")
