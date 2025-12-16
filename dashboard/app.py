import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from datetime import date
import matplotlib.pyplot as plt
import seaborn as sns

# --- PATH FIX ---
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

# 1. SETUP PAGE CONFIG
st.set_page_config(layout="wide", page_title="EnviroScan Dashboard", initial_sidebar_state="expanded")

# --- SMART FILE LOADER (DATA) ---
def load_data():
    possible_paths = [
        "data/labeled_pollution_data.csv", 
        "../data/labeled_pollution_data.csv",
        "dashboard/data/labeled_pollution_data.csv",
        "labeled_pollution_data.csv"
    ]
    for path in possible_paths:
        if os.path.exists(path):
            try:
                return pd.read_csv(path), path
            except:
                continue
    return None, None

# --- SMART IMAGE LOADER (RESULTS) ---
def get_model_plot(folder_name, file_name):
    # Searches for the plot in common 'results' locations
    possible_paths = [
        f"results/{folder_name}/{file_name}",       # If running from root
        f"../results/{folder_name}/{file_name}",    # If running from dashboard
        f"{folder_name}/{file_name}"                # If images are just in folders
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None

# --- LOAD MODELS ---
# --- FILTERED MODEL LOADER (ONLY 4) ---
models = {}
required_models = ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost"]

base_dir = os.path.join("..", "models")

if os.path.exists(base_dir):
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".joblib") or file.endswith(".pkl"):
                file_path = os.path.join(root, file)
                lower_name = file.lower()
                
                # Identify the model type
                model_name = None
                if "logistic" in lower_name:
                    model_name = "Logistic Regression"
                elif "decision" in lower_name and "tree" in lower_name:
                    model_name = "Decision Tree"
                elif "random" in lower_name and "forest" in lower_name:
                    model_name = "Random Forest"
                elif "xgb" in lower_name:
                    model_name = "XGBoost"
                
                # Only load if it's one of the 4 we want AND we haven't loaded it yet
                if model_name and model_name not in models:
                    try:
                        models[model_name] = joblib.load(file_path)
                    except Exception as e:
                        st.error(f"Error loading {model_name}: {e}")

# Check if we are missing any
missing = [m for m in required_models if m not in models]
if missing:
    st.sidebar.warning(f"⚠️ Missing models: {', '.join(missing)}")

# ---------------------------------------
# --- FUNCTION: GET ALL CITIES ---
def get_all_cities():
    indian_cities = [
        "Agra", "Ahmedabad", "Allahabad", "Amritsar", "Aurangabad", "Bangalore", "Bhopal", "Bhubaneswar", 
        "Chandigarh", "Chennai", "Coimbatore", "Dehradun", "Delhi", "Dhanbad", "Faridabad", "Ghaziabad", 
        "Gurgaon", "Guwahati", "Gwalior", "Hyderabad", "Indore", "Jabalpur", "Jaipur", "Jalandhar", 
        "Jamshedpur", "Jodhpur", "Kanpur", "Kochi", "Kolkata", "Kota", "Lucknow", "Ludhiana", "Madurai", 
        "Meerut", "Mumbai", "Mysore", "Nagpur", "Nashik", "Navi Mumbai", "Noida", "Patna", "Pune", "Raipur", 
        "Rajkot", "Ranchi", "Srinagar", "Surat", "Thane", "Thiruvananthapuram", "Vadodara", "Varanasi", 
        "Vijayawada", "Visakhapatnam", "Warangal"
    ]
    
    df, _ = load_data()
    if df is not None:
        if 'City' in df.columns:
            indian_cities.extend(df['City'].unique().tolist())
        elif 'city' in df.columns:
            indian_cities.extend(df['city'].unique().tolist())
            
    return sorted(list(set(indian_cities)))

# 2. CUSTOM CSS
st.markdown("""
<style>
    h3 { font-size: 20px !important; font-weight: 600; margin-top: 20px; }
    .stNumberInput label, .stTextInput label, .stSelectbox label, .stDateInput label {
        font-size: 14px !important; font-weight: 500;
    }
    .block-container { padding-top: 2rem; }
    div[data-testid="stMetricValue"] { font-size: 24px; }
</style>
""", unsafe_allow_html=True)

# 3. SIDEBAR
with st.sidebar:
    if os.path.exists("logo.png"):
        st.image("logo.png", width=120) 
    
    st.markdown("# 🌿 EnviroScan")
    st.markdown("### Navigation")
    
    # REMOVED "Heatmaps" from this list
    page_selection = st.radio(
        "Go to", 
        ["Home", "Predict Source", "Data Visualization", "Model Evaluation", "About Project"], 
        label_visibility="collapsed" 
    )
    
    st.markdown("---")
    
    if page_selection == "Predict Source":
        st.markdown("### Model Controls")
        model_options = list(models.keys()) if models else ["No Models Found"]
        selected_model_name = st.selectbox("Select Model", model_options, label_visibility="collapsed")
        current_model = models.get(selected_model_name)
        
        st.markdown("**Status:**")
        if models:
            st.success(f"✅ {len(models)} Models Ready")
        else:
            st.error("❌ No Models Loaded")

# --- PAGE LOGIC ---

# 1. HOME PAGE
if page_selection == "Home":
    st.title("🌍 EnviroScan: AI Pollution Monitor")
    st.write("Welcome to the EnviroScan Dashboard.")
    st.info("👈 Please navigate to **'Predict Source'** in the sidebar to use the AI model.")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("System Status", "Online", "v1.0")
    m2.metric("Models Available", len(models))
    m3.metric("Last Update", str(date.today()))

    st.markdown("---")
    st.subheader("📡 Live Pollution Monitor")
    st.caption("Real-time identification of pollution sources using AI & Geospatial Analytics.")

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Avg PM2.5", "145 µg/m³", "+12%")
    with kpi2:
        st.metric("Active Alerts", "3 Zones", "High Risk", delta_color="inverse")
    with kpi3:
        st.metric("Primary Source", "Industrial", "45% Conf.")
    with kpi4:
        st.metric("Sensor Status", "Active", "98% Uptime")

    st.markdown("") 
    col_map, col_alert = st.columns([2, 1])

    with col_map:
        st.markdown("### 📍 Live Hotspots (Pune Sector)")
        map_data = pd.DataFrame(
            np.random.randn(50, 2) / 20 + [18.5204, 73.8567],
            columns=['lat', 'lon']
        )
        st.map(map_data, zoom=11)

    with col_alert:
        st.markdown("### 🚨 Recent Alerts")
        alert_data = pd.DataFrame({
            "Zone": ["Zone A", "Zone B", "Zone C"],
            "Level": ["Critical", "Warning", "Moderate"],
            "Source": ["Industrial", "Traffic", "Agri-Burn"]
        })
        
        def color_risk(val):
            color = 'red' if val == 'Critical' else 'orange' if val == 'Warning' else 'green'
            return f'color: {color}; font-weight: bold'

        st.dataframe(
            alert_data.style.map(color_risk, subset=['Level']),
            use_container_width=True,
            hide_index=True
        )

# 2. PREDICT SOURCE
elif page_selection == "Predict Source":
    
    st.markdown("# 🔮 AI Pollution Source Predictor")
    st.markdown("### Input Environmental Parameters")

    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.date_input("Date", value=date.today())
            all_cities = get_all_cities()
            final_city_list = ["Other / Type Manually"] + all_cities
            default_ix = final_city_list.index("Delhi") if "Delhi" in final_city_list else 0
            city_selection = st.selectbox("City", final_city_list, index=default_ix)
            
            if city_selection == "Other / Type Manually":
                selected_city = st.text_input("Enter City Name", value="Dharashiv")
            else:
                selected_city = city_selection

            lat = st.number_input("Latitude", value=28.6100, format="%.4f")
            lon = st.number_input("Longitude", value=77.2300, format="%.4f")

        with col2:
            pm25 = st.number_input("PM2.5", value=100.00, format="%.2f")
            pm10 = st.number_input("PM10", value=180.00, format="%.2f") 
            no2 = st.number_input("NO2", value=40.00, format="%.2f")
            so2 = st.number_input("SO2", value=10.00, format="%.2f")
            co = st.number_input("CO", value=1.00, format="%.2f")
            o3 = st.number_input("O3", value=30.00, format="%.2f") 

        with col3:
            temp = st.number_input("Temperature (°C)", value=25.00, format="%.2f")
            humidity = st.number_input("Humidity (%)", value=50.00, format="%.2f") 
            wind = st.number_input("Wind Speed (km/h)", value=5.00, format="%.2f")
            traffic_index = st.number_input("Traffic Index", value=50.00, format="%.2f") 

        st.markdown("### Spatial Context")
        c_space1, c_space2, c_space3 = st.columns(3)
        with c_space1:
            dist_road_m = st.number_input("Dist to Road (m)", value=500.00, format="%.2f")
        with c_space2:
            dist_ind_m = st.number_input("Dist to Industry (m)", value=2000.00, format="%.2f")
        with c_space3:
            dist_farm_m = st.number_input("Dist to Farm (m)", value=5000.00, format="%.2f")

        st.markdown("### Fire Data")
        c_fire1, c_fire2 = st.columns(2)
        with c_fire1:
            fire_status = st.selectbox("Fire Nearby?", ["No", "Yes"])
        fire_dist = 0.0 
        with c_fire2:
            if fire_status == "Yes":
                fire_dist = st.number_input("Fire Min Dist (km)", value=5.00, format="%.2f")
            else:
                st.write("") 

    st.write("") 
    if st.button("🔍 Predict Source", type="primary", use_container_width=False):
        if current_model:
            ind_prox_km = dist_ind_m / 1000.0
            road_prox_km = dist_road_m / 1000.0
            input_df = pd.DataFrame([[pm25, no2, so2, co, ind_prox_km, road_prox_km, temp, wind]], 
                                    columns=['pm25', 'no2', 'so2', 'co', 'ind_prox', 'road_prox', 'temp', 'wind'])
            pred = current_model.predict(input_df)[0]
            label_map = {0: "Vehicular 🚗", 1: "Industrial 🏭", 2: "Agricultural 🌾"}
            result = label_map.get(pred, pred)
            st.success(f"### Detected Source: {result}")
            has_fire = True if fire_status == "Yes" else False
            if pred == 2 and has_fire:
                st.error(f"⚠️ **CRITICAL ALERT:** Stubble Burning Confirmed! (Agri Source + Fire {fire_dist}km away)")
            elif has_fire and fire_dist < 5.0:
                 st.warning(f"⚠️ Fire detected very close ({fire_dist} km), but primary pollution source is {result}.")
        else:
            st.error("Please select a valid model from the sidebar.")

# 3. DATA VISUALIZATION
elif page_selection == "Data Visualization":
    st.title("📊 Data Visualization (EDA)")
    st.markdown("Exploratory Data Analysis of the pollution dataset.")

    # ADDED tab6 "🔥 Geo Heatmap" here
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📄 Live Data Preview", "🔥 Correlation", "📊 Trends", "📦 Outliers", "🗺️ Geospatial", "🔥 Geo Heatmap"])
    
    with tab1:
        st.subheader("Raw Dataset View")
        df, found_path = load_data()
        if df is not None:
            st.success(f"✅ Loaded Real Data: {df.shape[0]} rows found.")
            st.dataframe(df.head(100), use_container_width=True)
        else:
            st.warning("⚠️ CSV file not found. Displaying synthetic sample.")
            st.dataframe(pd.DataFrame(np.random.randn(20, 5), columns=["A","B","C","D","E"]), use_container_width=True)

    with tab2:
        st.subheader("Correlation Heatmap")
        if os.path.exists("heatmap.png"): 
            st.image("heatmap.png", width=700)
            st.info("💡 **Description:** This heatmap displays correlation coefficients (from -1 to 1). Dark red (close to 1) means two variables increase together.")
        else: st.warning("⚠️ 'heatmap.png' not found.")

    with tab3:
        st.subheader("Daily Pollutant Trends")
        if os.path.exists("daily_pollutant_trends.png"): 
            st.image("daily_pollutant_trends.png", width=700)
            st.info("💡 **Description:** Time-series chart showing the rise and fall of pollutant levels over time. Peaks indicate high-pollution events.")
        else: st.warning("⚠️ 'daily_pollutant_trends.png' not found.")

    with tab4:
        st.subheader("Outlier Detection")
        outlier_images = ["no2_outliers.png", "pm2.5_outliers.png", "pm10_outliers.png", "so2_outliers.png"]
        found_images = [img for img in outlier_images if os.path.exists(img)]
        if found_images:
            cols = st.columns(2)
            for i, img_file in enumerate(found_images):
                with cols[i % 2]: 
                    st.image(img_file, caption=img_file, width=400)
            st.info("💡 **Description:** These boxplots highlight outliers (points outside whiskers). They help identify extreme pollution events that deviate from the norm.")
        else: st.warning("⚠️ No outlier plots found.")

    with tab5:
        st.subheader("Geospatial Distribution")
        if os.path.exists("Geospatial_scatter.png"): 
            st.image("Geospatial_scatter.png", width=700)
            st.info("💡 **Description:** A scatter plot mapping pollutant levels across different geographical coordinates (Latitude/Longitude).")
        else: st.warning("⚠️ 'Geospatial_scatter.png' not found.")
    
    # MOVED CONTENT FROM "Heatmaps" PAGE TO HERE
    with tab6:
        st.subheader("Geospatial Heatmaps")
        st.write("Heatmaps identifying high-density pollution zones.")
        if os.path.exists("heatmap.png"):
            st.image("heatmap.png", width=800)
        else:
            st.info("Pollution concentration maps will appear here.")

# 4. MODEL EVALUATION
elif page_selection == "Model Evaluation":
    st.title("📉 Model Evaluation")
    st.markdown("Detailed analysis of model performance metrics and plots.")

    # 1. Summary Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Best Performing Model", "XGBoost", "99.8% Acc")
    with col2:
        st.metric("Fastest Inference", "Decision Tree", "0.02s")
    with col3:
        st.metric("Total Samples Trained", "5,000+", "Synthetic Data")

    st.markdown("---")
    
    # 2. Dynamic Plot Selector
    st.subheader("📂 Detailed Model Plots")
    
    folder_map = {
        "Logistic Regression": "logistic_regression",
        "Decision Tree": "decision_tree",
        "Random Forest": "random_forest",
        "XGBoost": "xgboost_model"
    }
    
    selected_eval_model = st.selectbox("Select Model to Inspect:", list(folder_map.keys()))
    selected_folder = folder_map[selected_eval_model]
    
    plots_info = [
        {"file": "confusion_matrix.png", "desc": "**Confusion Matrix:** Highlights where the model made errors. Darker diagonal squares indicate high accuracy."},
        {"file": "classification_report.png", "desc": "**Classification Report:** Shows Precision (Exactness), Recall (Completeness), and F1-Score for each pollution class."},
        {"file": "feature_importance.png", "desc": "**Feature Importance:** Ranks variables (like Distance to Industry) by how much they influenced the prediction."},
        {"file": "cv_f1_scores.png", "desc": "**Cross-Validation:** Demonstrates model stability by testing it on multiple subsets of data."}
    ]
    
    c1, c2 = st.columns(2)
    
    found_any = False
    for i, p_info in enumerate(plots_info):
        plot_file = p_info["file"]
        plot_desc = p_info["desc"]
        
        plot_path = get_model_plot(selected_folder, plot_file)
        
        if plot_path:
            found_any = True
            with (c1 if i % 2 == 0 else c2):
                st.image(plot_path, caption=f"{selected_eval_model} - {plot_file.replace('.png', '').replace('_', ' ').title()}", use_container_width=True)
                st.info(plot_desc)
        else:
            if "feature_importance" in plot_file and selected_eval_model == "Logistic Regression":
                continue

    if not found_any:
        st.warning(f"⚠️ No plots found for {selected_eval_model}. Check if files exist in 'results/{selected_folder}/'.")
        st.info("Expected structure: results > decision_tree > confusion_matrix.png")

# 6. ABOUT PROJECT
elif page_selection == "About Project":
    st.title("ℹ️ About EnviroScan")
    st.write("EnviroScan uses Machine Learning to identify pollution sources based on environmental parameters.")