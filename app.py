# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
from datetime import datetime

st.set_page_config(
    layout="wide",
    page_title="EnviroScan — Pollution Source Dashboard"
)

# ----------------------------
# Paths
# ----------------------------
DATA_PATH = "data/labeled_pollution_data.csv"
MODEL_PATH = "models/best_model.pkl"

# ----------------------------
# Loaders
# ----------------------------
@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH, parse_dates=["date"])

@st.cache_resource
def load_model():
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle["label_encoder"], bundle["feature_columns"]

# ----------------------------
# Prediction helper
# ----------------------------
def preprocess_input(data, feature_cols):
    df = pd.DataFrame([data])
    df = pd.get_dummies(df)
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    return df[feature_cols]

def predict_single(data):
    model, le, feature_cols = load_model()
    X = preprocess_input(data, feature_cols)
    pred = model.predict(X)[0]
    prob = model.predict_proba(X).max()
    return le.inverse_transform([pred])[0], prob

# ----------------------------
# Load data
# ----------------------------
df = load_dataset()

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.title("EnviroScan Filters")

city = st.sidebar.selectbox(
    "City",
    ["All"] + sorted(df["city"].unique())
)

pollutant = st.sidebar.selectbox(
    "Pollutant",
    ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
)

date_range = st.sidebar.date_input(
    "Date Range",
    (df["date"].min().date(), df["date"].max().date())
)

filtered = df.copy()

if city != "All":
    filtered = filtered[filtered["city"] == city]

filtered = filtered[
    (filtered["date"] >= pd.to_datetime(date_range[0])) &
    (filtered["date"] <= pd.to_datetime(date_range[1]))
]

# ----------------------------
# Title
# ----------------------------
st.title("🌍 EnviroScan — AI Powered Pollution Source Dashboard")

# ----------------------------
# KPI Row
# ----------------------------
k1, k2, k3, k4 = st.columns(4)

k1.metric("Records", len(filtered))
k2.metric(f"Avg {pollutant}", f"{filtered[pollutant].mean():.2f}")
k3.metric(f"Max {pollutant}", f"{filtered[pollutant].max():.2f}")
k4.metric("Top Source", filtered["Source"].mode()[0])

# ----------------------------
# Charts
# ----------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Pollution Source Distribution")
    fig1 = px.pie(
        filtered,
        names="Source",
        title="Source Contribution"
    )
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader(f"{pollutant} Trend")
    ts = filtered.groupby("date")[pollutant].mean().reset_index()
    fig2 = px.line(ts, x="date", y=pollutant)
    st.plotly_chart(fig2, use_container_width=True)

with col2:
    st.subheader("🗺 Geospatial Pollution Map")

    map_df = filtered.sample(
        min(3000, len(filtered)),
        random_state=42
    )

    fig_map = px.scatter_geo(
        map_df,
        lat="latitude",
        lon="longitude",
        color="Source",
        size=pollutant,
        hover_name="city",
        title=f"{pollutant} Distribution"
    )

    st.plotly_chart(fig_map, use_container_width=True)

# ----------------------------
# Prediction Section
# ----------------------------
st.markdown("---")
st.header("🔮 Predict Pollution Source")

c1, c2, c3 = st.columns(3)

with c1:
    lat = st.number_input("Latitude", value=float(df["latitude"].median()))
    lon = st.number_input("Longitude", value=float(df["longitude"].median()))
    date_val = st.date_input("Date", datetime.today())

with c2:
    pm25 = st.number_input("PM2.5", value=float(df["PM2.5"].median()))
    pm10 = st.number_input("PM10", value=float(df["PM10"].median()))
    no2 = st.number_input("NO2", value=float(df["NO2"].median()))

with c3:
    so2 = st.number_input("SO2", value=float(df["SO2"].median()))
    co = st.number_input("CO", value=float(df["CO"].median()))
    o3 = st.number_input("O3", value=float(df["O3"].median()))

if st.button("Predict Source"):
    input_data = {
        "latitude": lat,
        "longitude": lon,
        "PM2.5": pm25,
        "PM10": pm10,
        "NO2": no2,
        "SO2": so2,
        "CO": co,
        "O3": o3,
        "temperature": df["temperature"].median(),
        "humidity": df["humidity"].median(),
        "wind_speed": df["wind_speed"].median(),
        "wind_dir": df["wind_dir"].median(),
        "dist_to_road": df["dist_to_road"].median(),
        "dist_to_industry": df["dist_to_industry"].median(),
        "dist_to_farm": df["dist_to_farm"].median(),
        "fire_nearby": 0,
        "fire_count": 0,
        "fire_min_dist_km": 50,
        "dayofyear": pd.to_datetime(date_val).dayofyear,
        "month": pd.to_datetime(date_val).month,
        "year": pd.to_datetime(date_val).year,
        "Season": pd.to_datetime(date_val).strftime("%B")
    }

    label, confidence = predict_single(input_data)
    st.success(f"Predicted Source: **{label}**")
    st.metric("Confidence", f"{confidence:.2%}")

st.caption("EnviroScan — Streamlit Dashboard (Cloud-safe version)")
