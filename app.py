# app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import folium
from folium.plugins import HeatMap
import plotly.express as px
from streamlit.components.v1 import html
from datetime import datetime

# -------------------------------------------------
# Streamlit config
# -------------------------------------------------
st.set_page_config(layout="wide", page_title="EnviroScan — Pollution Source Dashboard")

st.write("🚀 App booted successfully")  # DEBUG MARKER (IMPORTANT)

# -------------------------------------------------
# Paths
# -------------------------------------------------
DATA_PATH = "data/labeled_pollution_data.csv"
MODEL_PATH = "models/best_model.pkl"

# -------------------------------------------------
# Load dataset
# -------------------------------------------------
@st.cache_data
def load_dataset():
    if not os.path.exists(DATA_PATH):
        st.error(f"Dataset not found: {DATA_PATH}")
        return None
    return pd.read_csv(DATA_PATH, parse_dates=["date"])

# -------------------------------------------------
# SAFE model loader (NO DOWNLOAD, NO CRASH)
# -------------------------------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, None, None
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle["label_encoder"], bundle["feature_columns"]

# -------------------------------------------------
# Prediction helpers
# -------------------------------------------------
def preprocess_input(input_dict, feature_columns):
    df = pd.DataFrame([input_dict])
    df = pd.get_dummies(df)
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    return df[feature_columns]

def predict_single(input_dict):
    model, le, feats = load_model()
    if model is None:
        return "Model not loaded", 0.0
    X = preprocess_input(input_dict, feats)
    pred = model.predict(X)[0]
    conf = np.max(model.predict_proba(X))
    return le.inverse_transform([pred])[0], conf

# -------------------------------------------------
# Load data
# -------------------------------------------------
df = load_dataset()
if df is None:
    st.stop()

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
st.sidebar.title("EnviroScan Controls")

cities = sorted(df["city"].unique())
city = st.sidebar.selectbox("City", ["All"] + cities)

pollutant = st.sidebar.selectbox(
    "Pollutant", ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
)

start, end = st.sidebar.date_input(
    "Date Range",
    value=(df["date"].min().date(), df["date"].max().date()),
)

filtered = df.copy()
if city != "All":
    filtered = filtered[filtered["city"] == city]

filtered = filtered[
    (filtered["date"] >= pd.to_datetime(start))
    & (filtered["date"] <= pd.to_datetime(end))
]

# -------------------------------------------------
# Dashboard
# -------------------------------------------------
st.title("EnviroScan — AI-Powered Pollution Dashboard")

k1, k2, k3 = st.columns(3)
k1.metric("Average PM2.5", f"{filtered['PM2.5'].mean():.2f}")
k2.metric("Max PM2.5", f"{filtered['PM2.5'].max():.2f}")
k3.metric("Records", filtered.shape[0])

col1, col2 = st.columns((1.2, 1))

with col1:
    st.subheader("Pollution Trend")
    ts = filtered.set_index("date").resample("D")[pollutant].mean().reset_index()
    st.plotly_chart(px.line(ts, x="date", y=pollutant), use_container_width=True)

    st.subheader("Source Distribution")
    st.plotly_chart(
        px.pie(filtered, names="Source"),
        use_container_width=True,
    )

with col2:
    st.subheader("Geospatial Heatmap")
    sample = filtered.sample(min(3000, len(filtered)))
    m = folium.Map(
        location=[sample["latitude"].mean(), sample["longitude"].mean()],
        zoom_start=5,
        tiles="CartoDB positron",
    )
    HeatMap(sample[["latitude", "longitude", pollutant]].values.tolist()).add_to(m)
    html(m._repr_html_(), height=520)

# -------------------------------------------------
# Prediction Section
# -------------------------------------------------
st.markdown("---")
st.header("Predict Pollution Source")

lat = st.number_input("Latitude", value=float(df["latitude"].median()))
lon = st.number_input("Longitude", value=float(df["longitude"].median()))
pm25 = st.number_input("PM2.5", value=float(df["PM2.5"].median()))
pm10 = st.number_input("PM10", value=float(df["PM10"].median()))
date = st.date_input("Date", value=datetime.today())

if st.button("Predict"):
    sample = {
        "latitude": lat,
        "longitude": lon,
        "PM2.5": pm25,
        "PM10": pm10,
        "NO2": float(df["NO2"].median()),
        "SO2": float(df["SO2"].median()),
        "CO": float(df["CO"].median()),
        "O3": float(df["O3"].median()),
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
        "dayofyear": date.timetuple().tm_yday,
        "month": date.month,
        "year": date.year,
        "Season": date.strftime("%B"),
    }

    pred, conf = predict_single(sample)
    st.success(f"Prediction: {pred}")
    st.metric("Confidence", f"{conf:.2%}")

st.caption("EnviroScan | Streamlit Cloud Stable Build")
