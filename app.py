# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import folium
from folium.plugins import HeatMap
import plotly.express as px
from streamlit.components.v1 import html
from datetime import datetime
import urllib.request

# -------------------------------------------------
# Streamlit config
# -------------------------------------------------
st.set_page_config(layout="wide", page_title="EnviroScan — Pollution Source Dashboard")

# -------------------------------------------------
# Paths
# -------------------------------------------------
DATA_PATH = os.path.join("data", "labeled_pollution_data.csv")
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")

# ⚠️ IMPORTANT: replace if your repo/branch differs
MODEL_URL = (
    "https://github.com/springboardmentor0409-alt/"
    "ai_powered_enviroScan/raw/main/models/best_model.pkl"
)

# -------------------------------------------------
# Dataset loader
# -------------------------------------------------
@st.cache_data(ttl=3600)
def load_dataset(path=DATA_PATH):
    if not os.path.exists(path):
        st.error(f"Dataset not found at: {path}")
        return None
    df = pd.read_csv(path, parse_dates=["date"])
    return df

# -------------------------------------------------
# Lazy model loader (Streamlit Cloud SAFE)
# -------------------------------------------------
@st.cache_resource
def load_model():
    os.makedirs(MODEL_DIR, exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 Downloading ML model (one-time)..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle["label_encoder"], bundle["feature_columns"]

# -------------------------------------------------
# Feature alignment for prediction
# -------------------------------------------------
def preprocess_input_for_model(input_dict, feature_columns):
    df = pd.DataFrame([input_dict])
    df = pd.get_dummies(df, drop_first=True)

    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    return df[feature_columns]

def predict_single(input_dict):
    model, le, feat_cols = load_model()
    X = preprocess_input_for_model(input_dict, feat_cols)
    pred = model.predict(X)[0]
    conf = float(np.max(model.predict_proba(X)))
    return le.inverse_transform([pred])[0], conf

# -------------------------------------------------
# KPI helper
# -------------------------------------------------
def top_kpis(df):
    latest_date = df["date"].max()
    latest_df = df[df["date"] == latest_date]
    return (
        latest_date,
        float(latest_df["PM2.5"].mean()),
        float(latest_df["PM2.5"].max()),
        df["Source"].mode().iat[0],
    )

# -------------------------------------------------
# Folium helpers
# -------------------------------------------------
def make_folium_map(df, lat="latitude", lon="longitude", value="PM2.5"):
    m = folium.Map(
        location=[df[lat].mean(), df[lon].mean()],
        zoom_start=6,
        tiles="CartoDB positron",
    )

    heat_data = df[[lat, lon, value]].dropna().values.tolist()
    HeatMap(heat_data, radius=10, blur=15).add_to(m)
    return m

def folium_static(m, width=700, height=500):
    html(m._repr_html_(), height=height, width=width)

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
st.sidebar.title("EnviroScan — Controls")
df = load_dataset()

if df is None:
    st.stop()

cities = sorted(df["city"].unique().tolist())
selected_city = st.sidebar.selectbox("City", ["All"] + cities)

date_min, date_max = df["date"].min(), df["date"].max()
date_range = st.sidebar.date_input(
    "Date range",
    value=(date_min.date(), date_max.date()),
    min_value=date_min.date(),
    max_value=date_max.date(),
)

pollutant = st.sidebar.selectbox(
    "Pollutant", ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
)

filtered = df.copy()
if selected_city != "All":
    filtered = filtered[filtered["city"] == selected_city]

filtered = filtered[
    (filtered["date"] >= pd.to_datetime(date_range[0]))
    & (filtered["date"] <= pd.to_datetime(date_range[1]))
]

# -------------------------------------------------
# Main Dashboard
# -------------------------------------------------
st.title("EnviroScan — AI-Powered Pollution Source Dashboard")
st.markdown("Visualize pollution, predict sources, and explore historical trends.")

latest_date, avg_pm25, max_pm25, common_source = top_kpis(df)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Latest Date", latest_date.strftime("%Y-%m-%d"))
c2.metric("Avg PM2.5", f"{avg_pm25:.2f}")
c3.metric("Max PM2.5", f"{max_pm25:.2f}")
c4.metric("Most Common Source", common_source)

# Charts + Map
colA, colB = st.columns((1.2, 1))

with colA:
    st.subheader("Source Distribution")
    fig1 = px.pie(
        filtered["Source"].value_counts().reset_index(),
        names="index",
        values="Source",
    )
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader(f"{pollutant} Trend")
    ts = (
        filtered.set_index("date")
        .resample("D")[pollutant]
        .mean()
        .reset_index()
    )
    st.plotly_chart(px.line(ts, x="date", y=pollutant), use_container_width=True)

with colB:
    st.subheader("Geospatial Heatmap")
    map_df = filtered.sample(min(3000, len(filtered)))
    folium_static(make_folium_map(map_df, value=pollutant), height=520)

# -------------------------------------------------
# Prediction Section
# -------------------------------------------------
st.markdown("---")
st.header("Predict Pollution Source")

col1, col2, col3 = st.columns(3)

with col1:
    latitude = st.number_input("Latitude", value=float(df["latitude"].median()))
    longitude = st.number_input("Longitude", value=float(df["longitude"].median()))
    date_input = st.date_input("Date", value=datetime.today())

with col2:
    PM25 = st.number_input("PM2.5", value=float(df["PM2.5"].median()))
    PM10 = st.number_input("PM10", value=float(df["PM10"].median()))
    NO2 = st.number_input("NO2", value=float(df["NO2"].median()))

with col3:
    SO2 = st.number_input("SO2", value=float(df["SO2"].median()))
    CO = st.number_input("CO", value=float(df["CO"].median()))
    O3 = st.number_input("O3", value=float(df["O3"].median()))

if st.button("Predict Source"):
    sample = {
        "latitude": latitude,
        "longitude": longitude,
        "PM2.5": PM25,
        "PM10": PM10,
        "NO2": NO2,
        "SO2": SO2,
        "CO": CO,
        "O3": O3,
        "temperature": float(df["temperature"].median()),
        "humidity": float(df["humidity"].median()),
        "wind_speed": float(df["wind_speed"].median()),
        "wind_dir": float(df["wind_dir"].median()),
        "dist_to_road": float(df["dist_to_road"].median()),
        "dist_to_industry": float(df["dist_to_industry"].median()),
        "dist_to_farm": float(df["dist_to_farm"].median()),
        "fire_nearby": 0,
        "fire_count": 0,
        "fire_min_dist_km": 50,
        "dayofyear": date_input.timetuple().tm_yday,
        "month": date_input.month,
        "year": date_input.year,
        "Season": date_input.strftime("%B"),
    }

    pred, conf = predict_single(sample)
    st.success(f"**Predicted Source:** {pred}")
    st.metric("Confidence", f"{conf:.2%}")

st.caption("EnviroScan — Streamlit Deployment | Lazy-loaded ML model for cloud scalability")
