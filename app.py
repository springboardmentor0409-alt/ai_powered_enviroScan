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
DATA_PATH = os.path.join("data", "labeled_pollution_data.csv")
MODEL_PATH = os.path.join("models", "best_model.pkl")

# ----------------------------
# Load dataset & model
# ----------------------------
@st.cache_data(ttl=3600)
def load_dataset():
    if not os.path.exists(DATA_PATH):
        st.error("Dataset not found.")
        return None
    return pd.read_csv(DATA_PATH, parse_dates=["date"])

@st.cache_resource(ttl=3600)
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("Model not found.")
        return None
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle["label_encoder"], bundle["feature_columns"]

# ----------------------------
# Prediction helper
# ----------------------------
def preprocess_input(input_dict, feature_columns):
    df = pd.DataFrame([input_dict])
    df = pd.get_dummies(df, drop_first=True)

    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    return df[feature_columns]

def predict_single(input_dict):
    model_data = load_model()
    if model_data is None:
        return None, None

    model, le, feat_cols = model_data
    X = preprocess_input(input_dict, feat_cols)
    pred = model.predict(X)[0]
    prob = model.predict_proba(X).max()
    return le.inverse_transform([pred])[0], prob

# ----------------------------
# Load data
# ----------------------------
df = load_dataset()
if df is None:
    st.stop()

# ----------------------------
# Sidebar filters
# ----------------------------
st.sidebar.title("EnviroScan Controls")

cities = sorted(df["city"].unique())
selected_city = st.sidebar.selectbox("City", ["All"] + cities)

pollutant = st.sidebar.selectbox(
    "Pollutant",
    ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
)

date_min, date_max = df["date"].min(), df["date"].max()
date_range = st.sidebar.date_input(
    "Date range",
    value=(date_min.date(), date_max.date())
)

filtered = df.copy()
if selected_city != "All":
    filtered = filtered[filtered["city"] == selected_city]

filtered = filtered[
    (filtered["date"] >= pd.to_datetime(date_range[0])) &
    (filtered["date"] <= pd.to_datetime(date_range[1]))
]

# ----------------------------
# KPIs
# ----------------------------
st.title("🌍 EnviroScan — AI-Powered Pollution Dashboard")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Records", len(filtered))
k2.metric(f"Avg {pollutant}", f"{filtered[pollutant].mean():.2f}")
k3.metric(f"Max {pollutant}", f"{filtered[pollutant].max():.2f}")
k4.metric("Top Source", filtered["Source"].mode()[0])

# ----------------------------
# Charts
# ----------------------------
col1, col2 = st.columns((1.2, 1))

with col1:
    st.subheader("Pollution Source Distribution")
    fig1 = px.pie(
        filtered,
        names="Source",
        title="Predicted Pollution Sources"
    )
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader(f"{pollutant} Trend Over Time")
    ts = filtered.groupby("date")[pollutant].mean().reset_index()
    fig2 = px.line(ts, x="date", y=pollutant)
    st.plotly_chart(fig2, use_container_width=True)

with col2:
    st.subheader("Geospatial Pollution Heatmap")

    map_df = filtered.sample(min(3000, len(filtered)))

    fig_map = px.density_mapbox(
        map_df,
        lat="latitude",
        lon="longitude",
        z=pollutant,
        radius=15,
        center=dict(
            lat=map_df["latitude"].mean(),
            lon=map_df["longitude"].mean()
        ),
        zoom=4,
        mapbox_style="carto-positron",
        title=f"{pollutant} Density Map"
    )

    st.plotly_chart(fig_map, use_container_width=True)

# ----------------------------
# Prediction section
# ----------------------------
st.markdown("---")
st.header("🔮 Predict Pollution Source")

colA, colB, colC = st.columns(3)

with colA:
    city = st.selectbox("City", cities)
    lat = st.number_input("Latitude", value=float(df["latitude"].median()))
    lon = st.number_input("Longitude", value=float(df["longitude"].median()))
    date_val = st.date_input("Date", value=datetime.today())

with colB:
    pm25 = st.number_input("PM2.5", value=float(df["PM2.5"].median()))
    pm10 = st.number_input("PM10", value=float(df["PM10"].median()))
    no2 = st.number_input("NO2", value=float(df["NO2"].median()))
    so2 = st.number_input("SO2", value=float(df["SO2"].median()))

with colC:
    co = st.number_input("CO", value=float(df["CO"].median()))
    o3 = st.number_input("O3", value=float(df["O3"].median()))
    temp = st.number_input("Temperature", value=float(df["temperature"].median()))
    humidity = st.number_input("Humidity", value=float(df["humidity"].median()))

if st.button("Predict"):
    input_data = {
        "city": city,
        "latitude": lat,
        "longitude": lon,
        "PM2.5": pm25,
        "PM10": pm10,
        "NO2": no2,
        "SO2": so2,
        "CO": co,
        "O3": o3,
        "temperature": temp,
        "humidity": humidity,
        "wind_speed": df["wind_speed"].median(),
        "wind_dir": df["wind_dir"].median(),
        "dist_to_road": df["dist_to_road"].median(),
        "dist_to_industry": df["dist_to_industry"].median(),
        "dist_to_farm": df["dist_to_farm"].median(),
        "fire_nearby": 0,
        "fire_count": 0,
        "fire_min_dist_km": df["fire_min_dist_km"].median(),
        "dayofyear": date_val.timetuple().tm_yday,
        "month": date_val.month,
        "year": date_val.year,
        "Season": date_val.strftime("%B")
    }

    label, confidence = predict_single(input_data)
    st.success(f"Predicted Source: **{label}**")
    st.metric("Confidence", f"{confidence:.2%}")

# ----------------------------
st.caption("EnviroScan • Streamlit Deployment • No Folium • Stable Build")
