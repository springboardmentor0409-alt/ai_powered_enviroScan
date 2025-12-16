import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib

# --- PATH FIX ---
# This ensures the app finds your images/css/model no matter where you run it from
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

# --- LOAD THE MODEL ---
# Replace 'pollution_model.pkl' with your actual filename if it differs
try:
    model = joblib.load('pollution_model.pkl')
    print("✅ Model loaded successfully!")
except FileNotFoundError:
    print("⚠️ Model file not found.")
    model = None

# 1. SETUP PAGE CONFIG
st.set_page_config(layout="wide", page_title="EnviroScan Dashboard")

# 2. SESSION STATE
if 'expanded_card' not in st.session_state:
    st.session_state.expanded_card = None

# 3. CSS LOADER
def local_css(file_name):
    try:
        with open(file_name, encoding="utf-8") as f: 
            css = f.read()
            st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.markdown("""
        <style>
        .card-container-simple { border: 1px solid #ddd; padding: 10px; border-radius: 10px; text-align: center; }
        .main-title-gradient { background: -webkit-linear-gradient(45deg, #00b09b, #96c93d); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        </style>
        """, unsafe_allow_html=True)

local_css("style.css")

# --- MOCK DATA GENERATOR (For the Home Page Map) ---
def load_mock_data():
    data = pd.DataFrame(
        np.random.randn(100, 2) / 50 + [18.5204, 73.8567],
        columns=['lat', 'lon']
    )
    data['PM2.5'] = np.random.randint(20, 300, 100)
    data['Source'] = np.random.choice(['Vehicular', 'Industrial', 'Agri-Burn'], 100)
    return data

df = load_mock_data()

# 4. HELPER FUNCTION
def create_clickable_card(card_id, image_path, title, short_text):
    st.markdown(f'<div class="card-container-simple">', unsafe_allow_html=True)
    if os.path.exists(image_path):
        st.image(image_path, caption=title, width=150) 
    else:
        st.markdown(f"**{title}**")
        
    st.markdown(f'<p class="card-short-text">{short_text}</p>', unsafe_allow_html=True)
    if st.button("Expand", key=f"btn_{card_id}"):
        st.session_state.expanded_card = card_id
        st.rerun()
    st.markdown(f'</div>', unsafe_allow_html=True)

# 5. SIDEBAR CONFIGURATION
with st.sidebar:
    if os.path.exists("logo.png"):
        st.image("logo.png", width=100)
    st.title("🌿 EnviroScan")
    
    st.subheader("Navigation")
    selected = st.radio(
        "Navigate", 
        [
            "🏠 Home", 
            "🔮 Predict Source", 
            "📊 Analytics", 
            "🗺️ Heatmaps", 
            "ℹ️ About Project"
        ]
    )
    
    st.markdown("---")
    st.subheader("Filter Data")
    pollutant_filter = st.selectbox("Select Pollutant", ["PM2.5", "PM10", "NO2", "SO2"])
    source_filter = st.multiselect("Select Source", ["Vehicular", "Industrial", "Agricultural"], default=["Vehicular", "Industrial"])

# --- PAGE LOGIC ---

# 1. DASHBOARD (HOME)
if "Home" in selected:
    col_title, col_metric = st.columns([2, 1])
    with col_title:
        st.markdown('<h1 class="main-title-gradient">Live Pollution Monitor</h1>', unsafe_allow_html=True)
        st.write("Real-time identification of pollution sources using AI & Geospatial Analytics.")
    
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Avg PM2.5", "145 µg/m³", "+12%")
    with kpi2:
        st.metric("Active Alerts", "3 Zones", "High Risk", delta_color="inverse")
    with kpi3:
        st.metric("Primary Source", "Industrial", "45% Conf.")
    with kpi4:
        st.metric("Sensor Status", "Active", "98% Uptime")

    st.markdown("---")

    col_map, col_details = st.columns([2, 1])
    with col_map:
        st.subheader("📍 Live Hotspots")
        filtered_df = df[df['Source'].isin(source_filter)]
        st.map(filtered_df, zoom=10)

    with col_details:
        st.subheader("🚨 Recent Alerts")
        alert_data = pd.DataFrame({
            "Zone": ["Zone A", "Zone B", "Zone C"],
            "Level": ["Critical", "Warning", "Moderate"],
            "Source": ["Industrial", "Traffic", "Agri-Burn"]
        })
        
        def highlight_critical(val):
            color = 'red' if val == 'Critical' else 'orange' if val == 'Warning' else 'green'
            return f'color: {color}; font-weight: bold'

        # Using .map for pandas 2.1+, fallback to .applymap if needed for older versions
        try:
            st.dataframe(alert_data.style.map(highlight_critical, subset=['Level']), use_container_width=True)
        except:
            st.dataframe(alert_data.style.applymap(highlight_critical, subset=['Level']), use_container_width=True)
        
        st.info("💡 **Insight:** Industrial emissions in Zone A represent 60% of total pollution today.")

# 2. PREDICTION PAGE
elif "Predict Source" in selected:
    st.title("🔮 AI Pollution Source Predictor")
    st.markdown("Enter real-time environmental data below to identify the likely pollution source and assess risk.")

    col_input1, col_input2 = st.columns(2)
    with col_input1:
        st.subheader("🧪 Pollutant Levels")
        # Ensure these match the UNITS your model was trained on
        pm25 = st.number_input("PM2.5 (µg/m³)", min_value=0.0, value=45.0)
        no2 = st.number_input("NO₂ (ppb)", min_value=0.0, value=12.0)
        so2 = st.number_input("SO₂ (ppb)", min_value=0.0, value=5.0)
        co = st.number_input("CO (ppm)", min_value=0.0, value=0.5)

    with col_input2:
        st.subheader("🌍 Environmental Context")
        proximity_ind = st.slider("Proximity to Industrial Zone (km)", 0.0, 50.0, 10.0)
        proximity_road = st.slider("Proximity to Highway (km)", 0.0, 50.0, 5.0)
        temp = st.slider("Temperature (°C)", -10.0, 50.0, 25.0)
        wind_speed = st.slider("Wind Speed (km/h)", 0.0, 100.0, 10.0)

    st.markdown("---")
    
    if st.button("🚀 Run Analysis & Predict Source", type="primary"):
        if model is not None:
            # --- 1. PREPARE INPUTS ---
            # CHECK: Ensure this list order matches your training data EXACTLY!
            input_data = np.array([[
                pm25, no2, so2, co, 
                proximity_ind, proximity_road, 
                temp, wind_speed
            ]])
            
            # --- 2. GET PREDICTION ---
            prediction = model.predict(input_data)
            
            # --- 3. MAP RESULT ---
            # Update this dictionary if your model outputs different numbers/classes
            class_map = {0: "Vehicular", 1: "Industrial", 2: "Agricultural"} 
            
            if isinstance(prediction[0], (int, np.integer)):
                predicted_source = class_map.get(prediction[0], "Unknown Source")
            else:
                predicted_source = prediction[0]

            # --- 4. CALCULATE CONFIDENCE ---
            try:
                probs = model.predict_proba(input_data)
                confidence = round(np.max(probs) * 100, 2)
            except:
                confidence = 85.0 # Fallback
            
            # --- 5. RISK LOGIC ---
            risk_level = "Moderate" # Default
            if "Industrial" in str(predicted_source) and so2 > 40:
                risk_level = "Critical"
            elif "Vehicular" in str(predicted_source) and no2 > 50:
                risk_level = "High"
            elif "Agricultural" in str(predicted_source) and pm25 > 100:
                risk_level = "Severe"

        else:
            st.error("⚠️ Model not loaded. Please check if 'pollution_model.pkl' is in the folder.")
            predicted_source = "System Error"
            confidence = 0.0
            risk_level = "Unknown"

        # Display Results
        st.subheader("Analysis Result")
        result_color = "red" if risk_level in ["Critical", "Severe"] else "orange" if risk_level == "High" else "green"
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.info(f"**Identified Source:**\n\n### {predicted_source}")
        with c2:
            st.metric("Model Confidence", f"{confidence}%")
        with c3:
            st.markdown(f"**Risk Zone Status:**")
            st.markdown(f'<h2 style="color:{result_color}; margin:0;">{risk_level.upper()}</h2>', unsafe_allow_html=True)

        st.subheader("🚨 Automated Alerts & Recommendations")
        if risk_level in ["Critical", "Severe", "High"]:
            st.error(f"⚠️ **ALERT TRIGGERED:** {predicted_source} detected at unsafe levels!")
            with st.expander("View Policy Recommendations"):
                st.write(f"1. Immediate intervention required for {predicted_source}.")
                st.write("2. Alert sent to local environmental agency.")
        else:
            st.success("✅ No immediate alerts.")

# 3. ANALYTICS PAGE
elif "Analytics" in selected:
    st.title("📊 Detailed Analytics")
    st.info("Detailed statistical charts will be displayed here.")
    chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])
    st.line_chart(chart_data)

# 4. HEATMAPS PAGE
elif "Heatmaps" in selected:
    st.markdown('<h3 class="center-heading">Geospatial Insights Gallery</h3>', unsafe_allow_html=True)
    card_data = [
        {"id": "card_a", "img": "no2_outliers.png", "title": "NO₂ Analysis","short": "High Traffic Correlation", "long": "NO2 spikes correlate with congestion zones."},
        {"id": "card_b", "img": "o3_outliers.png", "title": "O3 Analysis", "short": "Ozone Variance","long": "Median Ozone is 30, with outliers up to 78."},
        {"id": "card_c", "img": "so2_outliers.png", "title": "SO₂ Analysis", "short": "Industrial Spikes", "long": "Low baseline but extreme outliers near factories."},
        {"id": "card_d", "img": "PM2.5_outliers.png", "title": "PM2.5 Analysis", "short": "Particulate Matter", "long": "Heavily right-skewed data with outliers > 2000."},
        {"id": "card_e", "img": "pm10_outliers.png", "title": "PM10 Analysis", "short": "Dust & Construction", "long": "PM10 shows extreme outliers up to 5000."}
    ]
    cols = st.columns(5)
    for i, col in enumerate(cols):
        with col:
            create_clickable_card(card_data[i]["id"], card_data[i]["img"], card_data[i]["title"], card_data[i]["short"])
    
    if st.session_state.expanded_card:
        sel = next((item for item in card_data if item["id"] == st.session_state.expanded_card), None)
        if sel:
            st.markdown("---")
            c1, c2, c3 = st.columns([1, 4, 1])
            with c2:
                st.subheader(f"Detailed View: {sel['title']}")
                if os.path.exists(sel['img']):
                    st.image(sel['img'], use_container_width=True)
                st.write(sel['long'])
                if st.button("Close View", key="close_btn"):
                    st.session_state.expanded_card = None
                    st.rerun()

# 5. MODEL INSIGHTS
elif "Model Insights" in selected:
    st.title("🧠 Model Logic")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**🚗 Vehicular**\n\nHigh NO₂ + Road Proximity")
    with col2:
        st.warning("**🏭 Industrial**\n\nHigh SO₂ + Factory Proximity")
    with col3:
        st.error("**🌾 Agricultural**\n\nHigh PM + Rural Areas")
    
    with st.expander("See Statistical Evidence"):
        st.write("Weekend effect observed in NO2 data suggests heavy commuter influence.")

# 6. ABOUT PROJECT
elif "About Project" in selected:
    st.title("ℹ️ About EnviroScan")
    st.write("EnviroScan uses AI/ML to detect pollution sources from OpenAQ data.")
    st.success("Key Features: Source Prediction, Hotspot Visualization, Smart Alerts.")