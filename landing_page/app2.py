import streamlit as st
import pandas as pd
import joblib
import math
import os
from pathlib import Path

# -----------------------------------------------------------
# PAGE CONFIGURATION + UI THEME
# -----------------------------------------------------------
st.set_page_config(
    page_title="EnviroScan – AI Pollution Source Identifier",
    layout="wide"
)

st.markdown(
    """
    <style>
        body {
            background-color: #f4f7fb;
        }

        h1, h2, h3 {
            color: #1565C0 !important;   /* Blue Theme */
            font-weight: 700 !important;
        }

        .card {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 14px;
            box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
            margin-bottom: 25px;
            border-left: 5px solid #1565C0; /* Blue Accent */
        }

        .stButton>button {
            background-color: #1565C0; /* Button Blue */
            color: white;
            font-size: 18px;
            padding: 10px 25px;
            border-radius: 10px;
            border: none;
        }

        .stButton>button:hover {
            background-color: #0D47A1; /* Dark Blue hover */
            color: white;
        }

        .sidebar .sidebar-content {
            background-color: #E3F2FD !important; /* Soft blue sidebar */
        }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------------
# MODEL LOADER
# -----------------------------------------------------------
def load_model(model_name):
    model_paths = {
        "Random Forest": "models/random_forest/random_forest.joblib",
        "Logistic Regression": "models/logistic_regression/logistic_regression.joblib",
        "XGBoost": "models/xgboost_model/xgboost.joblib",
        "Decision Tree": "models/decision_tree/decision_tree.joblib",
    }
    encoder_paths = {
        "Random Forest": "models/random_forest/label_encoder.joblib",
        "Logistic Regression": "models/logistic_regression/label_encoder.joblib",
        "XGBoost": "models/xgboost_model/label_encoder.joblib",
        "Decision Tree": "models/decision_tree/label_encoder.joblib",
    }

    return joblib.load(model_paths[model_name]), joblib.load(encoder_paths[model_name])

# -----------------------------------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------------------------------
st.sidebar.title("🌿 Navigation")

menu = st.sidebar.radio(
    "",
    ["Home", "Predict Source", "Model Insights", "Historical Data", "About Project"]
)

# -----------------------------------------------------------
# HOME PAGE
# -----------------------------------------------------------
if menu == "Home":
    st.markdown("<h1> EnviroScan: AI-Powered Pollution Source Identifier</h1>", unsafe_allow_html=True)
    st.subheader("Using Machine Learning & Geospatial Analytics")

    st.markdown("""
EnviroScan is an AI-driven system designed to **identify the most probable source of pollution** by combining  
air quality data, weather parameters, and geospatial features. Traditional monitoring systems only indicate  
pollution levels — EnviroScan reveals **where the pollution is coming from**, enabling smarter environmental decisions.

### 🔍 What EnviroScan Analyzes
- Pollutant concentrations (PM2.5, PM10, NO₂, SO₂, CO, O₃)  
- Weather factors (temperature, humidity, wind speed & direction)  
- Geospatial indicators (distance to roads, industries, farms, and fire events)  
- Seasonal and temporal patterns  

Based on these features, EnviroScan predicts whether pollution originates from:
- **Vehicular emissions**  
- **Industrial zones**  
- **Agricultural burning**  
- **Natural environmental causes**

This makes the system highly valuable for environmental agencies, researchers, and policymakers.
""")

    img_path = Path("assets/enviro_dashboard.png")
    if img_path.exists():
        st.image(str(img_path), width=450)

# -----------------------------------------------------------
# PREDICT SOURCE PAGE
# -----------------------------------------------------------
elif menu == "Predict Source":
    st.markdown("<h1>🔍 Predict Pollution Source</h1>", unsafe_allow_html=True)

    model_choice = st.selectbox(
        "Select Model",
        ["Random Forest", "Logistic Regression", "XGBoost", "Decision Tree"]
    )

    # load model (will raise if model files missing)
    model, encoder = load_model(model_choice)

    # create two equal columns and put content inside them (no empty placeholders)
    col1, col2 = st.columns([1, 1])

    # ---------------- COLUMN 1 ----------------
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🌫 Pollution Levels")
        PM25 = st.slider("PM2.5", 0, 500, 80)
        PM10 = st.slider("PM10", 0, 600, 120)
        NO2 = st.slider("NO2", 0, 200, 30)
        SO2 = st.slider("SO2", 0, 100, 8)
        CO = st.slider("CO", 0.0, 10.0, 1.2)
        O3 = st.slider("O3", 0, 200, 25)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🌦 Weather Conditions")
        temp = st.slider("Temperature (°C)", -10, 50, 28)
        humidity = st.slider("Humidity (%)", 0, 100, 60)
        wind_speed = st.slider("Wind Speed (m/s)", 0, 20, 2)
        season = st.selectbox("Season", ["summer", "winter", "autumn", "monsoon"])
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- COLUMN 2 ----------------
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🌬 Wind & Distance Factors")
        wind_dir = st.slider("Wind Direction (°)", 0, 360, 180)
        dist_to_road = st.slider("Distance to Road (km)", 0.0, 5.0, 0.2)
        dist_to_industry = st.slider("Distance to Industry (km)", 0.0, 20.0, 5.0)
        fire_nearby = st.selectbox("Fire Nearby?", [0, 1])
        fire_min_dist_km = st.slider("Nearest Fire Distance (km)", 0, 50, 15)
        traffic = st.slider("Traffic Index", 0, 100, 40)
        st.markdown("</div>", unsafe_allow_html=True)

    # -----------------------------------------------------------
    # PREDICT BUTTON
    # -----------------------------------------------------------
    if st.button("Predict Pollution Source"):

        wind_dir_rad = math.radians(wind_dir)
        wind_u = wind_speed * math.cos(wind_dir_rad)
        wind_v = wind_speed * math.sin(wind_dir_rad)

        data = {
            "season": season,
            "PM2.5": PM25, "PM10": PM10,
            "NO2": NO2, "SO2": SO2,
            "CO": CO, "O3": O3,
            "temp": temp, "humidity": humidity,
            "wind_speed": wind_speed,
            "wind_dir": wind_dir,
            "wind_dir_rad": wind_dir_rad,
            "wind_u": wind_u,
            "wind_v": wind_v,
            "dist_to_road": dist_to_road,
            "dist_to_industry": dist_to_industry,
            "dist_to_farm": 1.0,
            "fire_nearby": fire_nearby,
            "fire_min_dist_km": fire_min_dist_km,
            "traffic_index": traffic,
            "city": "UnknownCity",
            "location_id": "LOC_0",
            "year": 2020, "month": 1, "dayofyear": 1,

            # Required placeholder scaled features
            "PM2.5_s": 0, "PM10_s": 0, "NO2_s": 0, "SO2_s": 0,
            "CO_s": 0, "O3_s": 0, "temp_s": 0, "humidity_s": 0,
            "wind_speed_s": 0, "traffic_index_s": 0,
            "dist_to_road_s": 0, "dist_to_industry_s": 0,
            "dist_to_farm_s": 0, "fire_min_dist_km_s": 0,
            "road_bearing": 0, "industry_bearing": 0,
            "farm_bearing": 0, "fire_bearing": 0,
            "align_r": 0, "align_i": 0,
            "align_f": 0, "align_fire": 0,
        }

        df = pd.DataFrame([data])
        pred = model.predict(df)[0]
        label = encoder.inverse_transform([pred])[0]

        st.success(f"🌱 Predicted Pollution Source: **{label}**")

# -----------------------------------------------------------
# MODEL INSIGHTS PAGE
# -----------------------------------------------------------
elif menu == "Model Insights":
    st.markdown("<h1>📈 Model Insights</h1>", unsafe_allow_html=True)

    model_select = st.selectbox(
        "Choose Model",
        ["Random Forest", "Logistic Regression", "XGBoost", "Decision Tree"]
    )

    result_folders = {
        "Random Forest": "results/random_forest",
        "Logistic Regression": "results/logistic_regression",
        "XGBoost": "results/xgboost_model",
        "Decision Tree": "results/decision_tree",
    }

    img_folder = Path(result_folders[model_select])

    descriptions = {
        "confusion_matrix": """
### 🟦 Confusion Matrix  
Shows how accurately the model predicts each pollution source.  
- Diagonal = correct predictions  
- Off-diagonal = misclassifications  
""",

        "classification_report": """
### 📘 Classification Report  
Shows Precision, Recall, and F1-score for each pollution class.  
Higher values = better performance.  
""",

        "feature_importance": """
### ⭐ Feature Importance  
Displays which features influence the model the most.  
Helps understand how the model makes decisions.  
""",

        "cv_f1_scores": """
### 🔁 Cross-Validation F1 Scores  
Indicates how stable the model is across multiple folds.  
Stable scores mean the model generalizes well.  
"""
    }

    if not img_folder.exists():
        st.error("⚠️ No insight results found for this model.")
    else:
        for img_file in sorted(os.listdir(img_folder)):
            if img_file.endswith((".png", ".jpg")):
                img_name = img_file.replace(".png", "")
                clean_title = img_name.replace("_", " ").title()

                st.subheader(clean_title)
                st.image(str(img_folder / img_file))

                # Show matching description
                for key in descriptions:
                    if key in img_name:
                        st.markdown(descriptions[key])
                        break

                st.markdown("---")

# -----------------------------------------------------------
# HISTORICAL DATA PAGE
# -----------------------------------------------------------
elif menu == "Historical Data":
    st.markdown("<h1>📜 Historical Data Explorer</h1>", unsafe_allow_html=True)
    st.info("Upload or visualize historical pollution datasets here (coming soon).")

# -----------------------------------------------------------
# ABOUT PAGE
# -----------------------------------------------------------
elif menu == "About Project":
    st.markdown("<h1>ℹ️ About EnviroScan</h1>", unsafe_allow_html=True)
    st.markdown("""
### 🌟 Purpose of the Project  
Most existing pollution monitoring systems only measure how bad the air quality is,  
but they do not identify **why** pollution is increasing. EnviroScan solves this using AI + geospatial analytics.

### 🎯 Key Outcomes  
- Predicts pollution source categories  
- Creates pollution heatmaps  
- Triggers alerts  
- Offers interactive visual analytics  

### 🧠 Modules Implemented  
- Data Collection (OpenAQ, OpenWeather, OSMnx)  
- Feature Engineering  
- Source Labeling  
- ML Models (RF, XGBoost, LR, DT)  
- Geospatial Mapping  
- Interactive Dashboard  

### 🏗 System Architecture  
API Data → Cleaning → Feature Engineering → Labeling → Model → Dashboard  

### 📝 Deliverables  
- ML models  
- Dashboard  
- Heatmaps  
- Documentation  
""")
