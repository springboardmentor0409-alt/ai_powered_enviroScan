import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import plotly.express as px
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# -------------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------------
st.set_page_config(
    page_title="EnviroScan – AI Pollution Dashboard",
    page_icon="🌿",
    layout="wide"
)

# -------------------------------------------------------------
# GLOBAL STYLING
# -------------------------------------------------------------
st.markdown("""
<style>
    [data-testid="stSidebar"] {
        background-color: #e8f5e9 !important;
        padding: 25px;
    }

    [data-testid="stSidebar"] * {
        color: #1b5e20 !important;
        font-weight: 600;
    }

    .stApp {
        background-color: #f1f8e9 !important;
        color: #1b5e20 !important;
    }

    h1, h2, h3, h4 { color: #1b5e20 !important; }

    .predict-card {
        background: #c8e6c9;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        border-left: 8px solid #2e7d32;
        margin-bottom: 20px;
        font-size: 22px;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# DYNAMIC FOOTER FUNCTION
# -------------------------------------------------------------
def eco_footer(tab_name):
    messages = {
        "🏠 Home": "🌱 Building a sustainable future with AI-driven environmental intelligence.",
        "📊 EDA Overview": "📘 Data speaks — understanding pollution starts with good EDA.",
        "📈 Model Comparison": "🤖 Smarter models, clearer insights — choose what works best.",
        "🔍 Pollution Predictor": "🌿 Your air quality insights are just one prediction away!",
        "📽 Advanced Visual Dashboard": "🌍 Visualizing environmental change — insights that matter."
    }

    msg = messages.get(tab_name, "🌍 EnviroScan — Cleaner Air Through Smarter Analytics")

    st.markdown(f"""
    <div style="
        margin-top:40px;
        padding:15px;
        text-align:center;
        background:#e8f5e9;
        border-radius:10px;
        color:#1b5e20;
        font-size:15px;
        font-weight:600;
        border-top:3px solid #2e7d32;
    ">
        {msg}
    </div>
    """, unsafe_allow_html=True)

# -------------------------------------------------------------
# LOAD MAIN DATA FOR ORIGINAL TABS
# -------------------------------------------------------------
@st.cache_data
def load_main_data():
    return pd.read_csv("data/labeled_pollution_data.csv")

df = load_main_data()

# -------------------------------------------------------------
# LOAD MODEL
# -------------------------------------------------------------
MODEL_PATH = "models/light_model.joblib"
ENCODER_PATH = "models/light_label_encoder.joblib"

model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)

FEATURES = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "traffic_index"]

# -------------------------------------------------------------
# SIDEBAR NAVIGATION
# -------------------------------------------------------------
st.sidebar.title("🌿 EnviroScan Navigation")

page = st.sidebar.radio(
    "Go To:",
    [
        "🏠 Home",
        "📊 EDA Overview",
        "📈 Model Comparison",
        "🔍 Pollution Predictor",
        "📽 Advanced Visual Dashboard"   # NEW TAB ADDED HERE
    ]
)

# -------------------------------------------------------------
#  HOME PAGE 
# -------------------------------------------------------------
if page == "🏠 Home":
    st.markdown("""
    <div style="
        animation: fadeIn 2s ease-in-out;
        background: linear-gradient(to right, #a5d6a7, #c8e6c9);
        padding: 35px;
        border-radius: 12px;
        text-align: center;
        color: #1b5e20;
        margin-bottom: 25px;">
        <div style="font-size: 35px;">🌿 🌍 🌱</div>
        <h1>EnviroScan – AI-Powered Pollution Intelligence System</h1>
        <h3>Machine Learning • Environmental Analytics • Real-Time Pollution Source Detection</h3>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### 🌱 Introduction  
    EnviroScan predicts the **likely source of air pollution** using intelligent pollutant-level analysis.

    ### Why It Matters
    - 🚗 Vehicular emissions  
    - 🏭 Industrial activity  
    - 🔥 Biomass burning  
    - 🌾 Agriculture  
    """)

    eco_footer("🏠 Home")


# -------------------------------------------------------------
#  EDA OVERVIEW
# -------------------------------------------------------------
elif page == "📊 EDA Overview":

    st.title("📊 Exploratory Data Analysis")

    st.subheader("📌 Dataset Preview")
    st.dataframe(df.head())

    st.subheader("📌 Summary Statistics")
    st.write(df.describe())

    st.subheader("📌 Missing Values")
    st.write(df.isnull().sum())

    st.subheader("📌 Correlation Matrix")
    st.write(df[FEATURES].corr())

    eco_footer("📊 EDA Overview")


# -------------------------------------------------------------
# MODEL COMPARISON
# -------------------------------------------------------------
elif page == "📈 Model Comparison":

    st.title("📈 Model Comparison Dashboard")

    st.write("Model plots and evaluation metrics will be shown here.")

    eco_footer("📈 Model Comparison")


# -------------------------------------------------------------
#  POLLUTION PREDICTOR TAB
# -------------------------------------------------------------
elif page == "🔍 Pollution Predictor":

    st.markdown("<h1 style='text-align:center;'>🔍 AI Pollution Source Predictor</h1>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        pm25 = st.number_input("PM2.5", value=45.0)
        no2 = st.number_input("NO₂", value=20.0)
        co = st.number_input("CO", value=1.0)

    with col2:
        pm10 = st.number_input("PM10", value=70.0)
        so2 = st.number_input("SO₂", value=5.0)
        o3 = st.number_input("O₃", value=12.0)
        traffic = st.number_input("Traffic Index", value=30.0)

    if st.button("🌿 Predict Pollution Source"):

        input_data = pd.DataFrame([{
            "PM2.5": pm25, "PM10": pm10,
            "NO2": no2, "SO2": so2,
            "CO": co, "O3": o3,
            "traffic_index": traffic
        }])

        pred_encoded = model.predict(input_data)[0]
        pred_label = label_encoder.inverse_transform([pred_encoded])[0]
        confidence = model.predict_proba(input_data).max() * 100

        st.success(f"🌿 Predicted Source: {pred_label} ({confidence:.2f}% confidence)")

    eco_footer("🔍 Pollution Predictor")


# -------------------------------------------------------------
# 📽 NEW TAB: ADVANCED VISUAL DASHBOARD
# -------------------------------------------------------------
elif page == "📽 Advanced Visual Dashboard":

    st.markdown("""
        <div style="text-align:center; padding: 10px;">
            <h1>🌍 AI-Powered <span style="color:#1E8449">EnviroScan Dashboard</span></h1>
            <h3>Smart • Sustainable • Environmental Intelligence</h3>
        </div>
    """, unsafe_allow_html=True)

    # ---------------- LOAD SEPARATE DATASET SAFELY ----------------
    @st.cache_data
    def load_dashboard_data():
        df = pd.read_csv("pollution_labeled_output.csv")
        df["date"] = pd.to_datetime(df["date"], errors="ignore")
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df["dayofyear"] = df["date"].dt.dayofyear
        df.fillna(df.median(numeric_only=True), inplace=True)
        df.fillna("Unknown", inplace=True)
        return df

    df_dash = load_dashboard_data()

    st.header("🖼 Visual EDA Insights — Eco Perspective")

    image_folder = "dashboard"
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]

    cols = st.columns(2)
    for i, img in enumerate(images):
        with cols[i % 2]:
            st.image(f"{image_folder}/{img}", caption=img, use_container_width=True)

    st.header("🌿 Environmental Visualizations")

    pollutants = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
    df_sorted = df_dash.sort_values("date")

    st.subheader("📈 Time-Based Pollutant Trend")
    fig_line = px.line(
        df_sorted,
        x="date",
        y=pollutants,
        animation_frame=df_sorted["year"].astype(str),
        title="Pollutant Levels Over Time"
    )
    st.plotly_chart(fig_line, use_container_width=True)

    st.subheader("🍃 Seasonal Pollution Behavior")
    df_season = df_dash.groupby(["Season", "year"])[pollutants].mean().reset_index()

    fig_bar = px.bar(
        df_season,
        x="Season",
        y="PM2.5",
        color="Season",
        animation_frame="year",
        title="Season-wise PM2.5 Levels"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("📍 Pollution Movement Across Regions")
    fig_geo = px.scatter(
        df_sorted,
        x="longitude",
        y="latitude",
        color="PM2.5",
        animation_frame=df_sorted["year"].astype(str),
        title="Animated Pollution Spread",
        color_continuous_scale="greens"
    )
    st.plotly_chart(fig_geo, use_container_width=True)

    # ------------------- AI PREDICTOR (CITY-BASED) -------------------
    st.header("🤖 AI Pollution Source Predictor")

    @st.cache_resource
    def train_city_model(df_dash):
        target = "Pollution_Source"
        y = LabelEncoder().fit_transform(df_dash[target])
        X = df_dash.drop(columns=[target, "date"])

        num_cols = X.select_dtypes(include=["int64", "float64"]).columns
        cat_cols = X.select_dtypes(include=["object"]).columns

        processor = ColumnTransformer([
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
        ])

        model = Pipeline([
            ("prep", processor),
            ("clf", RandomForestClassifier(n_estimators=180))
        ])

        model.fit(X, y)
        return model, num_cols, cat_cols

    model2, numeric_cols, cat_cols = train_city_model(df_dash)

    city = st.text_input("City Name")
    season = st.selectbox("Season", df_dash["Season"].unique())

    city_means = df_dash.groupby("city").mean(numeric_only=True)

    if st.button(" Predict Pollution Source"):

        input_data = {
            "city": city,
            "Season": season
        }

        if city in city_means.index:
            for col in numeric_cols:
                input_data[col] = city_means.loc[city, col]
        else:
            for col in numeric_cols:
                input_data[col] = df_dash[col].median()

        for col in cat_cols:
            if col not in input_data:
                input_data[col] = df_dash[col].mode()[0]

        pred_df = pd.DataFrame([input_data])
        pred = model2.predict(pred_df)[0]

        labels = df_dash["Pollution_Source"].unique()

        st.success(f"🌱 Predicted Source: {labels[pred]}")

    eco_footer("📽 Advanced Visual Dashboard")

