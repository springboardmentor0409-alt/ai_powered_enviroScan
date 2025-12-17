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

st.set_page_config(layout="wide", page_title="EnviroScan — Pollution Source Dashboard")

# ----------------------------
# Paths (edit if needed)
# ----------------------------
DATA_PATH = os.path.join("data", "labeled_pollution_data.csv")
MODEL_PATH = os.path.join("models", "best_model.pkl")

# ----------------------------
# Utility: load data & model
# ----------------------------
@st.cache_data(ttl=3600)
def load_dataset(path=DATA_PATH):
    if not os.path.exists(path):
        st.error(f"Dataset not found at: {path}")
        return None
    df = pd.read_csv(path, parse_dates=["date"], infer_datetime_format=True)
    return df

@st.cache_resource(ttl=3600)
def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        st.error(f"Model not found at: {path}")
        return None
    bundle = joblib.load(path)
    model = bundle["model"]
    label_encoder = bundle["label_encoder"]
    feature_columns = bundle.get("feature_columns", [])
    return model, label_encoder, feature_columns

# Preprocess single input dict -> aligned DF
def preprocess_input_for_model(input_dict, feature_columns):
    df = pd.DataFrame([input_dict])
    df = pd.get_dummies(df, drop_first=True)
    # add missing features
    for c in feature_columns:
        if c not in df.columns:
            df[c] = 0
    # drop extra columns
    df = df[feature_columns]
    return df

def predict_single(input_dict):
    model_bundle = load_model()
    if model_bundle is None:
        return None, None
    model, le, feat_cols = model_bundle
    X = preprocess_input_for_model(input_dict, feat_cols)
    preds = model.predict(X)
    probs = model.predict_proba(X)
    pred_label = le.inverse_transform(preds)[0]
    conf = float(np.max(probs))
    return pred_label, conf

# ----------------------------
# Small helpers
# ----------------------------
def top_kpis(df):
    latest_date = df['date'].max()
    latest_df = df[df['date'] == latest_date]
    avg_pm25 = float(latest_df['PM2.5'].mean())
    max_pm25 = float(latest_df['PM2.5'].max())
    most_common_source = df['Source'].mode().iat[0] if 'Source' in df.columns else "N/A"
    return latest_date, avg_pm25, max_pm25, most_common_source

def make_folium_map(df, lat_col='latitude', lon_col='longitude', value_col='PM2.5', radius=10):
    # Center map
    center = [df[lat_col].mean(), df[lon_col].mean()]
    m = folium.Map(location=center, tiles="CartoDB positron", zoom_start=6)
    # HeatMap expects list of [lat, lon, weight]
    heat_data = df[[lat_col, lon_col, value_col]].dropna().values.tolist()
    if len(heat_data) > 0:
        HeatMap(heat_data, radius=radius, blur=15, max_zoom=10).add_to(m)
    # add some markers for top N extremes (optional)
    topn = df.nlargest(30, value_col)
    for _, r in topn.iterrows():
        popup = folium.Popup(f"{r.get('city','')}, {value_col}: {r.get(value_col):.1f}<br>Source: {r.get('Source','')}", max_width=250)
        folium.CircleMarker(location=[r[lat_col], r[lon_col]], radius=4, color='red', fill=True, popup=popup).add_to(m)
    return m

def folium_static(m, width=700, height=500):
    """Render Folium map in Streamlit via HTML iframe"""
    return html(m._repr_html_(), height=height, width=width)

# ----------------------------
# UI Layout: Sidebar
# ----------------------------
st.sidebar.title("EnviroScan — Controls")
df = load_dataset()
model_bundle = load_model()

if df is None:
    st.stop()

# Sidebar filters
cities = sorted(df['city'].unique().tolist())
selected_city = st.sidebar.selectbox("City", ["All"] + cities)
date_min = df['date'].min()
date_max = df['date'].max()
date_range = st.sidebar.date_input("Date range", value=(date_min.date(), date_max.date()), min_value=date_min.date(), max_value=date_max.date())

pollutant = st.sidebar.selectbox("Pollutant for plots & heatmap", ["PM2.5","PM10","NO2","SO2","CO","O3"])

# Quick filter
filtered = df.copy()
if selected_city != "All":
    filtered = filtered[filtered['city'] == selected_city]

start_date = pd.to_datetime(date_range[0])
end_date = pd.to_datetime(date_range[1])
filtered = filtered[(filtered['date'] >= start_date) & (filtered['date'] <= end_date)]

# ----------------------------
# Main layout
# ----------------------------
st.title("EnviroScan — AI-Powered Pollution Source Dashboard")
st.markdown("Predict pollution sources, visualize hotspots, explore historical trends, and export results.")

# KPI row
latest_date, avg_pm25, max_pm25, most_common_source = top_kpis(df)
k1, k2, k3, k4 = st.columns(4)
k1.metric("Latest Date in Dataset", latest_date.strftime("%Y-%m-%d"))
k2.metric(f"Avg {pollutant} (filtered)", f"{filtered[pollutant].mean():.2f}")
k3.metric(f"Max {pollutant} (filtered)", f"{filtered[pollutant].max():.2f}")
k4.metric("Most common source (all data)", most_common_source)

# Two-column: charts + map
c1, c2 = st.columns((1.2, 1))

with c1:
    st.subheader("Source Distribution")
    src_counts = filtered['Source'].value_counts().reset_index()
    src_counts.columns = ['Source', 'count']
    fig1 = px.pie(src_counts, names='Source', values='count', title="Predicted Source Distribution")
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader(f"{pollutant} over time")
    # Aggregate daily average
    ts = filtered.set_index('date').resample('D')[pollutant].mean().reset_index()
    fig2 = px.line(ts, x='date', y=pollutant, title=f"{pollutant} — daily average")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Pollutant levels by city")
    city_avg = filtered.groupby('city')[pollutant].mean().reset_index().sort_values(pollutant, ascending=False).head(20)
    fig3 = px.bar(city_avg, x=pollutant, y='city', orientation='h', title=f"Top cities by {pollutant}")
    st.plotly_chart(fig3, use_container_width=True)

with c2:
    st.subheader("Geospatial Heatmap")
    # create a smaller sample for the map to keep interactive smooth
    map_df = filtered[[ 'latitude','longitude', pollutant, 'Source']].dropna()
    if map_df.shape[0] > 3000:
        map_df = map_df.sample(3000, random_state=42)
    folium_map = make_folium_map(map_df.rename(columns={pollutant: pollutant}), value_col=pollutant)
    folium_static(folium_map, height=550, width=700)

st.markdown("---")

# ----------------------------
# Prediction page (form)
# ----------------------------
st.header("Predict Pollution Source for a Location & Time")

col1, col2, col3 = st.columns(3)

with col1:
    in_city = st.selectbox("City (optional)", [""] + cities)
    latitude = st.number_input("Latitude", value=float(df['latitude'].median()))
    longitude = st.number_input("Longitude", value=float(df['longitude'].median()))
    date_input = st.date_input("Date", value=datetime.today().date())

with col2:
    PM25 = st.number_input("PM2.5", value=float(df['PM2.5'].median()))
    PM10 = st.number_input("PM10", value=float(df['PM10'].median()))
    NO2 = st.number_input("NO2", value=float(df['NO2'].median()))
    SO2 = st.number_input("SO2", value=float(df['SO2'].median()))

with col3:
    CO = st.number_input("CO", value=float(df['CO'].median()))
    O3 = st.number_input("O3", value=float(df['O3'].median()))
    temperature = st.number_input("Temperature (°C)", value=float(df['temperature'].median()))
    humidity = st.number_input("Humidity (%)", value=float(df['humidity'].median()))

# extra features
dist_to_road = st.number_input("Distance to road (km)", value=float(df['dist_to_road'].median()))
dist_to_industry = st.number_input("Distance to industry (km)", value=float(df['dist_to_industry'].median()))
dist_to_farm = st.number_input("Distance to farm (km)", value=float(df['dist_to_farm'].median()))
fire_nearby = st.selectbox("Fire nearby?", [0,1], index=0)
fire_count = st.number_input("Fire count nearby", value=int(df['fire_count'].median()))
fire_min_dist_km = st.number_input("Closest fire (km)", value=float(df['fire_min_dist_km'].median()))

# derived features
dayofyear = pd.to_datetime(date_input).dayofyear
month = pd.to_datetime(date_input).month
year = pd.to_datetime(date_input).year

predict_btn = st.button("Predict Source")

if predict_btn:
    input_dict = {
        "city": in_city or None,
        "latitude": latitude,
        "longitude": longitude,
        "PM2.5": PM25,
        "PM10": PM10,
        "NO2": NO2,
        "SO2": SO2,
        "CO": CO,
        "O3": O3,
        "temperature": temperature,
        "humidity": humidity,
        "wind_speed": float(df['wind_speed'].median()) if 'wind_speed' in df.columns else 1.0,
        "wind_dir": float(df['wind_dir'].median()) if 'wind_dir' in df.columns else 180,
        "dist_to_road": dist_to_road,
        "dist_to_industry": dist_to_industry,
        "dist_to_farm": dist_to_farm,
        "fire_nearby": int(fire_nearby),
        "fire_count": int(fire_count),
        "fire_min_dist_km": float(fire_min_dist_km),
        "dayofyear": int(dayofyear),
        "month": int(month),
        "year": int(year),
        # Season as string (model expects Season dummies if present in feature_columns)
        "Season": pd.to_datetime(date_input).strftime("%B")
    }

    pred_label, conf = predict_single(input_dict)
    if pred_label is None:
        st.error("Model not loaded — cannot predict.")
    else:
        st.markdown(f"### Prediction: **{pred_label}**")
        st.metric("Confidence", f"{conf:.2%}")

st.markdown("---")

# ----------------------------
# Historical explorer + upload/download
# ----------------------------
st.header("Historical Data Explorer")
with st.expander("Filters & Export"):
    colA, colB, colC = st.columns(3)
    with colA:
        city_sel = st.multiselect("Filter cities", options=df['city'].unique().tolist(), default=[df['city'].unique().tolist()[0]])
    with colB:
        source_sel = st.multiselect("Filter source", options=df['Source'].unique().tolist(), default=df['Source'].unique().tolist())
    with colC:
        download_btn = st.button("Download filtered CSV")

    filtered2 = df[df['city'].isin(city_sel) & df['Source'].isin(source_sel)]
    st.write(f"Filtered rows: {filtered2.shape[0]}")

    if download_btn:
        csv = filtered2.to_csv(index=False).encode('utf-8')
        st.download_button("Click to download CSV", data=csv, file_name="enviro_filtered.csv", mime="text/csv")

st.dataframe(filtered2.head(200), use_container_width=True)

st.markdown("### Upload new dataset (optional)")
uploaded = st.file_uploader("Upload CSV to append and preview", type=["csv"])
if uploaded is not None:
    newdf = pd.read_csv(uploaded, parse_dates=["date"], infer_datetime_format=True)
    st.write("Preview of uploaded file")
    st.dataframe(newdf.head(200))

st.markdown("---")
st.caption("EnviroScan — Streamlit Dashboard (prototype). Model trained on the provided dataset. Use prediction form for single-row predictions.")
