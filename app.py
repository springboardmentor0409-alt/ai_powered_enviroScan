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
    layout="wide",
    page_icon="🌍"
)

# -----------------------------------------------------------
# MODERN UI STYLE
# -----------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: linear-gradient(135deg, #f3f8fb, #eef6f7);
}

/* Headings */
h1 {
    font-size: 2.6rem !important;
    font-weight: 800 !important;
    background: linear-gradient(90deg, #005f73, #0bbcd6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

h2, h3 {
    color: #005f73 !important;
    font-weight: 700;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #023047, #005f73);
}

section[data-testid="stSidebar"] * {
    color: #ffffff !important;
}

/* Cards */
.card {
    background: rgba(255, 255, 255, 0.88);
    backdrop-filter: blur(14px);
    border-radius: 18px;
    padding: 26px;
    margin-bottom: 24px;
    box-shadow: 0 12px 32px rgba(0,0,0,0.08);
}

/* Inputs */
input, select, textarea {
    border-radius: 10px !important;
    border: 1px solid #bde0fe !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #005f73, #0bbcd6);
    color: white;
    border-radius: 14px;
    font-size: 18px;
    font-weight: 600;
    padding: 12px 30px;
    border: none;
    transition: 0.3s;
    box-shadow: 0 6px 18px rgba(11,188,214,0.45);
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 26px rgba(11,188,214,0.6);
}

/* Alerts */
div[data-testid="stAlert"] {
    border-radius: 14px;
    font-size: 17px;
}

/* Images */
img {
    border-radius: 16px;
}
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------
# MODEL LOADER
# -----------------------------------------------------------
@st.cache_resource
def load_model(model_name):
    try:
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
        model = joblib.load(model_paths[model_name])
        encoder = joblib.load(encoder_paths[model_name])
        return model, encoder
    except FileNotFoundError as e:
        st.error(f"Model or encoder file not found for {model_name}: {e}")
        return None, None
    except Exception as e:
        st.error(f"Error loading model {model_name}: {e}")
        return None, None

# -----------------------------------------------------------
# HELPER
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
# SIDEBAR
# -----------------------------------------------------------
st.sidebar.title("🌿 EnviroScan")
menu = st.sidebar.radio(
    "Navigation",
    ["Home", "Predict Source", "Model Insights", "Data Visualization", "About Project"]
)

# -----------------------------------------------------------
# HOME
# -----------------------------------------------------------
if menu == "Home":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.title("AI-EnviroScan")
    st.subheader("AI-Powered Pollution Source Identifier using Geospatial Analytics")

    st.markdown("""
AI-EnviroScan is an **intelligent AI system** designed to identify the  
**probable source of air pollution**, not just pollutant concentration levels.

By combining **machine learning, environmental data, weather parameters,  
and geospatial analytics**, the system determines whether pollution is caused by:

- 🚗 Vehicular Emissions  
- 🏭 Industrial Activities  
- 🌾 Agricultural Burning  
- 🔥 Waste / Open Burning  
- 🌲 Natural Factors  

This enables **targeted mitigation strategies** instead of generic pollution control.
""")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🔍 Key Capabilities")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
- ML-based pollution source prediction  
- Automated feature engineering  
- Weather & seasonal impact analysis  
- Real-time alerts
""")

    with col2:
        st.markdown("""
- Geospatial heatmaps & hotspots  
- Pollution trend visualization  
- Decision-support dashboard  
- Smart-city ready architecture
""")
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------
# PREDICTION
# -----------------------------------------------------------
elif menu == "Predict Source":

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.title("🔍 Predict Pollution Source")
    st.markdown('</div>', unsafe_allow_html=True)

    model_choice = st.selectbox(
        "Select Model",
        ["Random Forest", "Logistic Regression", "XGBoost", "Decision Tree"]
    )

    model, encoder = load_model(model_choice)

    if model is None or encoder is None:
        st.error("Failed to load the selected model. Please check the model files.")
    else:
        # Initialize session state for predictions
        if 'predictions' not in st.session_state:
            st.session_state.predictions = []

        # Input validation function
        def validate_inputs(date_val, city, location_id, latitude, longitude, PM25, PM10, NO2, SO2, CO, O3, temp, humidity, wind_speed, wind_dir, dist_to_road, dist_to_industry, dist_to_farm, traffic, fire_nearby, fire_min_dist_km):
            errors = []
            if not city.strip():
                errors.append("City cannot be empty.")
            if not location_id.strip():
                errors.append("Location ID cannot be empty.")
            if not (-90 <= latitude <= 90):
                errors.append("Latitude must be between -90 and 90.")
            if not (-180 <= longitude <= 180):
                errors.append("Longitude must be between -180 and 180.")
            if PM25 < 0 or PM10 < 0 or NO2 < 0 or SO2 < 0 or CO < 0 or O3 < 0:
                errors.append("Pollutant values must be non-negative.")
            if not (0 <= humidity <= 100):
                errors.append("Humidity must be between 0 and 100.")
            if wind_speed < 0:
                errors.append("Wind speed must be non-negative.")
            if not (0 <= wind_dir <= 360):
                errors.append("Wind direction must be between 0 and 360.")
            if dist_to_road < 0 or dist_to_industry < 0 or dist_to_farm < 0 or fire_min_dist_km < 0:
                errors.append("Distances must be non-negative.")
            if traffic < 0:
                errors.append("Traffic index must be non-negative.")
            return errors

        with st.form("prediction_form"):
            tabs = st.tabs(["📍 Location & Date", "🌫️ Pollutants", "🌤️ Weather", "📏 Distances & Other"])

            with tabs[0]:
                colA, colB = st.columns(2)
                with colA:
                    date_val = st.date_input("Date", value=date.today(), help="Select the date for prediction.")
                    city = st.text_input("City", value="Hyderabad", help="Enter the city name.")
                    location_id = st.text_input("Location ID", value="LOC_001", help="Unique identifier for the location.")
                with colB:
                    latitude = st.number_input("Latitude", value=17.3850, format="%.4f", min_value=-90.0, max_value=90.0, help="Latitude in degrees.")
                    longitude = st.number_input("Longitude", value=78.4867, format="%.4f", min_value=-180.0, max_value=180.0, help="Longitude in degrees.")

            with tabs[1]:
                col1, col2 = st.columns(2)
                with col1:
                    PM25 = st.number_input("PM2.5 (µg/m³)", value=80.0, min_value=0.0, help="Particulate matter 2.5 concentration.")
                    PM10 = st.number_input("PM10 (µg/m³)", value=120.0, min_value=0.0, help="Particulate matter 10 concentration.")
                    NO2 = st.number_input("NO2 (µg/m³)", value=30.0, min_value=0.0, help="Nitrogen dioxide concentration.")
                    SO2 = st.number_input("SO2 (µg/m³)", value=8.0, min_value=0.0, help="Sulfur dioxide concentration.")
                    CO = st.number_input("CO (mg/m³)", value=1.2, min_value=0.0, help="Carbon monoxide concentration.")
                    O3 = st.number_input("O3 (µg/m³)", value=25.0, min_value=0.0, help="Ozone concentration.")

            with tabs[2]:
                col1, col2 = st.columns(2)
                with col1:
                    temp = st.number_input("Temperature (°C)", value=28.0, help="Ambient temperature.")
                    humidity = st.number_input("Humidity (%)", value=60.0, min_value=0.0, max_value=100.0, help="Relative humidity percentage.")
                    wind_speed = st.number_input("Wind Speed (m/s)", value=2.0, min_value=0.0, help="Wind speed in meters per second.")
                with col2:
                    wind_dir = st.number_input("Wind Direction (°)", value=180.0, min_value=0.0, max_value=360.0, help="Wind direction in degrees (0-360).")

            with tabs[3]:
                col1, col2 = st.columns(2)
                with col1:
                    dist_to_road = st.number_input("Distance to Road (km)", value=0.2, min_value=0.0, help="Distance to nearest road.")
                    dist_to_industry = st.number_input("Distance to Industry (km)", value=5.0, min_value=0.0, help="Distance to nearest industry.")
                    dist_to_farm = st.number_input("Distance to Farm (km)", value=1.0, min_value=0.0, help="Distance to nearest farm.")
                    traffic = st.number_input("Traffic Index", value=40.0, min_value=0.0, help="Traffic density index.")
                with col2:
                    fire_nearby = st.selectbox("Fire Nearby?", [0, 1], help="Is there a fire nearby? (0=No, 1=Yes)")
                    fire_min_dist_km = st.number_input("Fire Distance (km)", value=15.0, min_value=0.0, help="Distance to nearest fire.")

            submit = st.form_submit_button("Predict")

        if submit:
            errors = validate_inputs(date_val, city, location_id, latitude, longitude, PM25, PM10, NO2, SO2, CO, O3, temp, humidity, wind_speed, wind_dir, dist_to_road, dist_to_industry, dist_to_farm, traffic, fire_nearby, fire_min_dist_km)
            if errors:
                for error in errors:
                    st.error(error)
            else:
                with st.spinner("Predicting pollution source..."):
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
                        **{k: 0 for k in [
                            "PM2.5_s","PM10_s","NO2_s","SO2_s","CO_s","O3_s","temp_s",
                            "humidity_s","wind_speed_s","traffic_index_s",
                            "dist_to_road_s","dist_to_industry_s","dist_to_farm_s",
                            "fire_min_dist_km_s","road_bearing","industry_bearing",
                            "farm_bearing","fire_bearing","align_r","align_i",
                            "align_f","align_fire"
                        ]}
                    }])

                    try:
                        pred = model.predict(df)[0]
                        label = encoder.inverse_transform([pred])[0]

                        # Store prediction in session state
                        prediction_record = {
                            'timestamp': pd.Timestamp.now(),
                            'model': model_choice,
                            'date': date_val.strftime("%Y-%m-%d"),
                            'city': city,
                            'latitude': latitude,
                            'longitude': longitude,
                            'predicted_source': label,
                            'PM2.5': PM25,
                            'PM10': PM10,
                            'NO2': NO2,
                            'SO2': SO2,
                            'CO': CO,
                            'O3': O3,
                            'temp': temp,
                            'humidity': humidity,
                            'wind_speed': wind_speed,
                            'wind_dir': wind_dir
                        }
                        st.session_state.predictions.append(prediction_record)

                        # Display prediction result
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        st.success(f"🌱 Predicted Pollution Source: **{label}**")
                        if hasattr(model, 'predict_proba'):
                            probas = model.predict_proba(df)[0]
                            classes = encoder.classes_
                            prob_dict = {cls: round(prob*100, 2) for cls, prob in zip(classes, probas)}
                            st.write("**Prediction Confidence:**")
                            for cls, prob in prob_dict.items():
                                st.write(f"- {cls}: {prob}%")
                        st.markdown('</div>', unsafe_allow_html=True)

                    except Exception as e:
                        st.error(f"Prediction failed: {str(e)}")

        # Prediction History
        if st.session_state.predictions:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("📜 Prediction History")
            pred_df = pd.DataFrame(st.session_state.predictions)
            st.dataframe(pred_df, width='stretch')

            csv = pred_df.to_csv(index=False)
            st.download_button(
                label="Download Prediction History as CSV",
                data=csv,
                file_name="pollution_predictions.csv",
                mime="text/csv"
            )

            if st.button("Clear History"):
                st.session_state.predictions = []
                st.success("Prediction history cleared.")
            st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------
# MODEL INSIGHTS
# -----------------------------------------------------------
elif menu == "Model Insights":

    st.title("📈 Model Insights")

    model_select = st.selectbox(
        "Choose Model",
        ["Random Forest", "Logistic Regression", "XGBoost", "Decision Tree"]
    )

    base_folder = Path("results") / {
        "Random Forest": "random_forest",
        "Logistic Regression": "logistic_regression",
        "XGBoost": "xgboost_model",
        "Decision Tree": "decision_tree",
    }[model_select]

    if not base_folder.exists():
        st.error("No results found.")
    else:
        for img in sorted(base_folder.glob("*.png")):
            st.subheader(img.stem.replace("_", " ").title())
            st.image(str(img), width='stretch')
            st.markdown("---")

# -----------------------------------------------------------
# DATA VISUALIZATION
# -----------------------------------------------------------
elif menu == "Data Visualization":

    st.title("📊 Data Visualization")

    eda_root = Path("results/eda")

    if not eda_root.exists():
        st.error("EDA folder not found.")
    else:
        cols = st.columns(2)
        for i, img in enumerate(sorted(eda_root.glob("**/*.png"))):
            with cols[i % 2]:
                st.image(
                    str(img),
                    caption=img.stem.replace("_", " ").title(),
                    width='stretch'
                )

# -----------------------------------------------------------
# ABOUT
# -----------------------------------------------------------
elif menu == "About Project":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.title("ℹ️ About AI-EnviroScan")

    st.markdown("""
AI-EnviroScan is a **machine-learning driven pollution source identification system**
that integrates **environmental data, weather parameters, and geospatial intelligence**
to determine the **root cause of air pollution events**.

### 🏗️ System Architecture
APIs (OpenAQ, OpenWeather, OSMnx)
→ Data Cleaning & Feature Engineering
→ ML Models (Random Forest, XGBoost, Logistic Regression, Decision Tree)
→ Pollution Source Prediction
→ Geospatial Mapping
→ Streamlit Dashboard & Alerts

### 🎯 Use Cases
- Smart city air-quality monitoring
- Environmental research & analysis
- Government decision support
- Academic & student projects

### 📊 Supported Pollution Sources
- 🚗 Vehicular Emissions
- 🏭 Industrial Activities
- 🌾 Agricultural Burning
- 🔥 Waste / Open Burning
- 🌲 Natural Factors
""")
    st.markdown('</div>', unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown("**@AI-EnviroScan** made by Ustela Sukesh Reddy")
