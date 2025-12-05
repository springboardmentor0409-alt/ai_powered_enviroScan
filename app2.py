import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier



st.set_page_config(
    page_title="AI-Powered EnviroScan",
    layout="wide",
)


st.markdown("""
    <style>
        .main {
            background: linear-gradient(135deg, #e9f7ef, #d4efdf);
        }
        h1, h2, h3 {
            color: #145A32 !important;
            font-weight: 700;
        }
        .glass-card {
            padding: 25px;
            background: rgba(255,255,255,0.45);
            border-radius: 15px;
            border: 1px solid rgba(255,255,255,0.3);
            box-shadow: 0px 8px 20px rgba(0,0,0,0.08);
            margin-bottom: 20px;
        }
        .stButton>button {
            background-color: #28B463 !important;
            color: white !important;
            border-radius: 10px !important;
        }
        .stButton>button:hover {
            background-color: #1D8348 !important;
        }
    </style>
""", unsafe_allow_html=True)



st.markdown("""
    <div style="text-align:center; padding: 10px;">
        <h1>🌍 AI-Powered <span style="color:#1E8449">EnviroScan Dashboard</span></h1>
        <h3>Smart • Sustainable • Environmental Intelligence</h3>
        <p style="font-size:18px; color:#145A32;">
            EnviroScan is an AI-driven environmental analytics platform designed to 
            visualize pollution patterns, discover trends, and predict the most likely 
            pollution sources using machine learning.
            <br><br>
            Powered by real-world environmental datasets, advanced geospatial visualization, 
            and AI predictions — this dashboard helps cities move toward a cleaner, healthier future.
        </p>
    </div>
""", unsafe_allow_html=True)



st.sidebar.markdown("""
## 🌿 Eco Facts
- Trees can lower urban temperature by **up to 8°C**  
- Air pollution kills **7 million people each year**  
- Clean air boosts productivity and public health  
""")

@st.cache_data
def load_data():
    df = pd.read_csv("pollution_labeled_output.csv")

    df["date"] = pd.to_datetime(df["date"], errors="ignore")
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["dayofyear"] = df["date"].dt.dayofyear

    df.fillna(df.median(numeric_only=True), inplace=True)
    df.fillna("Unknown", inplace=True)

    return df

df = load_data()

st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
st.header("🖼 Visual EDA Insights — Eco Perspective")

image_folder = "dashboard"
images = [img for img in os.listdir(image_folder) if img.endswith(".png")]

cols = st.columns(2)
for i, img in enumerate(images):
    with cols[i % 2]:
        st.image(f"{image_folder}/{img}", caption=img,
                 use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
st.header(" Environmental Visualizations")

pollutants = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
df_anim = df.sort_values("date")


st.subheader(" Time-Based Pollutant Trend ")
fig_line = px.line(
    df_anim,
    x="date",
    y=pollutants,
    animation_frame=df_anim["year"].astype(str),
    title="Pollutant Levels Over Time"
)
st.plotly_chart(fig_line, use_container_width=True)
st.subheader("Seasonal Pollution Behavior")
df_season = df.groupby(["Season", "year"])[pollutants].mean().reset_index()

fig_bar = px.bar(
    df_season,
    x="Season",
    y="PM2.5",
    color="Season",
    animation_frame="year",
    title="Season-wise PM2.5 Levels"
)
st.plotly_chart(fig_bar, use_container_width=True)
st.subheader(" Pollution Movement Across Regions")
fig_geo = px.scatter(
    df_anim,
    x="longitude",
    y="latitude",
    color="PM2.5",
    animation_frame=df_anim["year"].astype(str),
    title="Animated Pollution Spread",
    color_continuous_scale="greens"
)
st.plotly_chart(fig_geo, use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)
st.header(" AI Pollution Source Predictor")

@st.cache_resource
def train_model(df):
    target = "Pollution_Source"
    y = LabelEncoder().fit_transform(df[target])

    X = df.drop(columns=[target, "date"])
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

st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
st.subheader("🔍 Predict Pollution Source (AI)")

city = st.text_input("City Name")
season = st.selectbox("Season", df["Season"].unique())

city_means = df.groupby("city").mean(numeric_only=True)

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
            input_data[col] = df[col].median()

    for col in cat_cols:
        if col not in input_data:
            input_data[col] = df[col].mode()[0]

    pred_df = pd.DataFrame([input_data])
    pred = model.predict(pred_df)[0]

    labels = df["Pollution_Source"].unique()
    st.success(f"🌱 **Predicted Source:** {labels[pred]}")

st.markdown("</div>", unsafe_allow_html=True)



st.markdown("---")
st.caption("Made  for a cleaner planet — AI-Powered EnviroScan Dashboard")
