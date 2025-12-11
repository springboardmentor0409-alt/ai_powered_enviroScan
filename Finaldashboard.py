import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os
import plotly.express as px
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
# LOAD DATA
# -------------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/labeled_pollution_data.csv")

df = load_data()

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
    ["🏠 Home", "📊 EDA Overview", "📈 Model Comparison", "🔍 Pollution Predictor", "📽 Advanced Visual Dashboard"]
)

# -------------------------------------------------------------
#  HOME PAGE 
# -------------------------------------------------------------
if page == "🏠 Home":

    # Header Animation
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
    EnviroScan is an advanced environmental AI dashboard capable of predicting the **likely source of air pollution**
    using real-time pollutant levels.  

    It is designed for:
    - Environmental researchers  
    - Pollution control authorities  
    - Smart city management  
    - Students & ML engineers  
    - Sustainability & climate analytics  

    ---

    ### 🔍 Why This Project Matters
    Pollution arises from:
    - 🚗 Vehicular emissions  
    - 🏭 Industrial activities  
    - 🔥 Biomass & waste burning  
    - 🌾 Agricultural residue burning  

    EnviroScan helps identify the **root cause**, not just measure pollutant values.

    ---

    ### 🧠 How the AI Model Works  
    - Optimized Random Forest classifier  
    - Uses 7 environmental indicators  
    - Fast + lightweight prediction  
    - Feature importance & explanation  
    - Fully interactive through Streamlit  

    ---

    ### 🌟 Project Overview  
    - 📌 Pollution Data + Weather Indicators  
    - 📌 Clean UI with eco-friendly theme  
    - 📌 Prediction + Explanation Engine  
    - 📌 Multi-model comparison support  
    - 📌 Real-time analytics ready  
    """)

    # 3 Overview Cards
    col1, col2, col3 = st.columns(3)
    col1.markdown("""<div style="background:#e8f5e9; padding:15px; border-radius:12px;">
        <h3>📘 Dataset</h3>
        <ul><li>Pollutant Indicators</li><li>Weather Features</li><li>Traffic Index</li><li>Geo Attributes</li></ul></div>""",
        unsafe_allow_html=True
    )
    col2.markdown("""<div style="background:#e3f2fd; padding:15px; border-radius:12px;">
        <h3>🤖 Machine Learning</h3>
        <ul><li>Random Forest</li><li>Feature Importance</li><li>Confidence Scoring</li></ul></div>""",
        unsafe_allow_html=True
    )
    col3.markdown("""<div style="background:#fff3e0; padding:15px; border-radius:12px;">
        <h3>📊 Dashboard</h3>
        <ul><li>EDA Tools</li><li>Model Comparison</li><li>AI Prediction</li></ul></div>""",
        unsafe_allow_html=True
    )

    # Timeline
    st.markdown("""
    ---
    ### 🛠 Project Workflow Timeline
    <div style="display:flex; justify-content:space-between;">
        <div style="width:22%; background:#e8f5e9; padding:15px; border-radius:10px; text-align:center;">
            <h3>📥 Data Collection</h3><p>Pollutant, weather & traffic data gathered.</p></div>
        <div style="width:22%; background:#e3f2fd; padding:15px; border-radius:10px; text-align:center;">
            <h3>⚙️ Processing</h3><p>Cleaning, filtering, engineering.</p></div>
        <div style="width:22%; background:#fff9c4; padding:15px; border-radius:10px; text-align:center;">
            <h3>🤖 Model Training</h3><p>AI learns pollutant signatures.</p></div>
        <div style="width:22%; background:#ffe0b2; padding:15px; border-radius:10px; text-align:center;">
            <h3>📊 Dashboard</h3><p>Visualization & realtime inference.</p></div>
    </div>
    """, unsafe_allow_html=True)

    # Key Highlights
    st.markdown("""
    ---
    ### ⭐ Key Highlights
    <div style="display:flex; justify-content:space-around; padding:20px; background:#e8f5e9; border-radius:12px;">
        <div style='text-align:center;'><h2>⚡</h2><p><b>Fast Predictions</b><br><small>Under 100ms</small></p></div>
        <div style='text-align:center;'><h2>📊</h2><p><b>Clear Analytics</b><br><small>User-friendly visuals</small></p></div>
        <div style='text-align:center;'><h2>🌍</h2><p><b>Eco-Focused</b><br><small>Built for sustainability</small></p></div>
        <div style='text-align:center;'><h2>🤖</h2><p><b>AI Powered</b><br><small>Smart inference engine</small></p></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <hr>
    <div style='text-align:center; padding:10px; color:#1b5e20;'>
        🌿 <b>EnviroScan</b> — Together we build a cleaner, greener tomorrow.
    </div>
    """, unsafe_allow_html=True)


# -------------------------------------------------------------
#  EDA OVERVIEW
# -------------------------------------------------------------
elif page == "📊 EDA Overview":

    st.title("📊 Exploratory Data Analysis")

    st.markdown("""
    ### 📘 EDA Insights  
    This section provides:
    - Dataset preview  
    - Summary statistics  
    - Missing value overview  
    - Correlation understanding  
    """)

    st.subheader("📌 Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    st.subheader("📌 Summary Statistics")
    st.write(df.describe())

    st.subheader("📌 Missing Values")
    st.write(df.isnull().sum())

    st.subheader("📌 Correlation Matrix")
    st.write(df[FEATURES].corr())

    st.markdown("""
    <hr>
    <div style='text-align:center; padding:10px; color:#1b5e20;'>
        📊 Exploration fuels understanding — Nature reveals patterns if we observe deeply.
    </div>
    """, unsafe_allow_html=True)


# -------------------------------------------------------------
# MODEL COMPARISON (SMART DISPLAY WITH DESCRIPTIONS)
# -------------------------------------------------------------
elif page == "📈 Model Comparison":

    st.title("📈 Model Comparison Dashboard")

    model_choice = st.selectbox(
        "Choose a model to inspect:",
        ["Decision Tree", "Logistic Regression", "Random Forest", "XGBoost"]
    )

    import os

    # Model folder paths
    model_paths = {
        "Decision Tree": "results/decision_tree/",
        "Logistic Regression": "results/logistic_regression/",
        "Random Forest": "results/random_forest/",
        "XGBoost": "results/xgboost_model/"
    }

    base = model_paths[model_choice]

    # Correct file names inside folders
    feature_img = base + "feature_importance.png"
    class_img = base + "classification_report.png"
    cm_img = base + "confusion_matrix.png"
    cv_img = base + "cv_f1_scores.png"   

    st.markdown(f"## 📌 Results for **{model_choice}**")

    # -------------------------------------------------
    #  FEATURE IMPORTANCE 
    # -------------------------------------------------
    st.subheader("🌿 Feature Importance")

    if model_choice == "Logistic Regression":
        st.info("""
        ℹ **Logistic Regression does not support feature importance plots** 
        because it uses coefficients instead of tree-based importance.
        """)
    else:
        st.markdown("""
        **Meaning of this plot:**  
        - Shows how strongly each pollutant influences predictions  
        - Higher bar → stronger importance  
        - Helps explain model behavior  
        """)

        if os.path.exists(feature_img):
            st.image(feature_img, use_container_width=True)
        else:
            st.warning("⚠ Feature importance image missing.")

    st.markdown("---")

    # -------------------------------------------------
    #  CLASSIFICATION REPORT
    # -------------------------------------------------
    st.subheader("📊 Classification Report (Precision • Recall • F1 Score)")

    st.markdown("""
    **Interpretation:**  
    - **Precision:** Accuracy of predicted positives  
    - **Recall:** Ability to detect real positives  
    - **F1-score:** Balance between precision & recall  
    """)

    if os.path.exists(class_img):
        st.image(class_img, use_container_width=True)
    else:
        st.error("❌ Classification report image missing!")

    st.markdown("---")

    # -------------------------------------------------
    #  CONFUSION MATRIX
    # -------------------------------------------------
    st.subheader("🔷 Confusion Matrix")

    st.markdown("""
    **How to read this:**  
    - Rows = actual labels  
    - Columns = predicted labels  
    - Diagonal = correct predictions  
    - Off-diagonal = misclassifications  
    """)

    if os.path.exists(cm_img):
        st.image(cm_img, use_container_width=True)
    else:
        st.error("❌ Confusion matrix image missing!")

    st.markdown("---")

    # -------------------------------------------------
    # CROSS VALIDATION F1 MACRO SCORES
    # -------------------------------------------------
    st.subheader("📈 Cross-Validation F1-Macro Scores")

    st.markdown("""
    **Why this matters:**  
    - Evaluates model stability across multiple folds  
    - Higher & consistent F1-macro → better generalization  
    """)

    if os.path.exists(cv_img):
        st.image(cv_img, use_container_width=True)
    else:
        st.error("❌ CV F1-score plot missing! Expected file: " + cv_img)

        st.markdown("""
    <hr>
    <div style='text-align:center; padding:10px; color:#1b5e20;'>
        🤖 Smarter Models. Cleaner Air. A Sustainable Future Awaits.
    </div>
    """, unsafe_allow_html=True)




# -------------------------------------------------------------
#  POLLUTION PREDICTOR
# -------------------------------------------------------------
elif page == "🔍 Pollution Predictor":

    
    st.markdown("""
    <style>
    /* Fix labels in number_input (text before input box) */
    .css-10trblm, .css-1fcdlhc, label, .stNumberInput label {
        color: #1b5e20 !important;
        font-weight: 600 !important;
    }

    /* Fix placeholder / value text color */
    .stNumberInput input {
        color: #1b5e20 !important;
        font-weight: 600;
    }

    /* Fix minus/plus button text color */
    .stNumberInput button {
        color: #1b5e20 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # ------------------------- TITLE -------------------------
    st.markdown("""
    <h1 style="color:#1b5e20; text-align:center;">🔍 AI Pollution Source Predictor</h1>
    <p style="text-align:center; font-size:18px; color:#2e7d32;">
        Enter pollutant levels below to generate an instant AI-driven pollution source prediction.
    </p>
    <hr style="border:1px solid #c8e6c9;">
    """, unsafe_allow_html=True)

    st.markdown("<h3 style='color:#1b5e20;'>🧪 Enter Pollution Indicators</h3>", unsafe_allow_html=True)

    # ------------------------- INPUT FIELDS -------------------------
    col1, col2 = st.columns(2)

    with col1:
        pm25 = st.number_input("PM2.5 (µg/m³)", value=45.0)
        no2 = st.number_input("NO₂ (µg/m³)", value=20.0)
        co = st.number_input("CO (mg/m³)", value=1.0)

    with col2:
        pm10 = st.number_input("PM10 (µg/m³)", value=70.0)
        so2 = st.number_input("SO₂ (µg/m³)", value=5.0)
        o3 = st.number_input("O₃ (µg/m³)", value=12.0)
        traffic = st.number_input("Traffic Index", value=30.0)

    st.markdown("<br>", unsafe_allow_html=True)

    # ------------------------- BUTTON -------------------------
    predict = st.button("🌿 Predict Pollution Source", use_container_width=True)

    if predict:

        # Create input dataframe
        input_data = pd.DataFrame([{
            "PM2.5": pm25, "PM10": pm10,
            "NO2": no2, "SO2": so2,
            "CO": co, "O3": o3,
            "traffic_index": traffic
        }])

        # Predict
        pred_encoded = model.predict(input_data)[0]
        pred_label = label_encoder.inverse_transform([pred_encoded])[0]
        confidence = model.predict_proba(input_data).max() * 100

        # ---------------- CONFIDENCE LEVEL ----------------
        if confidence >= 80:
            conf_color = "#2e7d32"
            conf_text = "High Confidence"
        elif confidence >= 60:
            conf_color = "#f9a825"
            conf_text = "Moderate Confidence"
        else:
            conf_color = "#c62828"
            conf_text = "Low Confidence"

        # ---------------- RESULT CARD ----------------
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #a5d6a7, #81c784);
            padding: 25px;
            border-radius: 15px;
            text-align: center;
            color: #1b5e20;
            font-size: 24px;
            font-weight: bold;
            border-left: 10px solid #2e7d32;">
            🌿 Predicted Source: <span style="font-size:28px;">{pred_label}</span><br>
            <span style="background:{conf_color}; padding:8px 15px; border-radius:10px; color:white; font-size:18px;">
                {conf_text} — {confidence:.2f}%
            </span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ======================================================================
        # HEALTH RISK ASSESSMENT
        # ======================================================================
        avg_pollution = np.mean([pm25, pm10, no2, so2, co, o3])

        if avg_pollution < 50:
            risk = "🟢 Good Air Quality"
            risk_desc = "Minimal health risk."
        elif avg_pollution < 100:
            risk = "🟡 Moderate Pollution"
            risk_desc = "Sensitive groups may be affected."
        elif avg_pollution < 150:
            risk = "🟠 Unhealthy"
            risk_desc = "Respiratory issues possible."
        else:
            risk = "🔴 Hazardous"
            risk_desc = "Serious health effects likely."

        st.markdown(f"""
        <div style="background:#e8f5e9; padding:15px; border-radius:12px; border-left:6px solid #2e7d32;">
            <h3 style="color:#1b5e20;">🏥 Health Risk Indicator</h3>
            <p style="font-size:20px;"><b>{risk}</b></p>
            <p>{risk_desc}</p>
        </div>
        """, unsafe_allow_html=True)

        # ======================================================================
        # 📏 DISTANCE MATRIX (NEW)
        # ======================================================================
        st.markdown("<h3 style='color:#1b5e20;'>📏 Distance Matrix (Similarity to Known Sources)</h3>", unsafe_allow_html=True)

        CLASS_PROFILES = {
            "Vehicular":     [80, 100, 60, 20, 1.5, 15, 70],
            "Industrial":    [70, 90, 50, 30, 2.0, 20, 40],
            "Agricultural":  [40, 60, 20, 10, 0.9, 10, 15],
            "Natural":       [20, 30, 10, 5, 0.5, 8, 5],
            "Photochemical": [50, 70, 40, 15, 1.0, 30, 25],
            "Burning":       [100, 130, 80, 40, 3.0, 25, 20]
        }

        user_vec = np.array([pm25, pm10, no2, so2, co, o3, traffic])

        distances = {cls: np.linalg.norm(user_vec - np.array(profile)) for cls, profile in CLASS_PROFILES.items()}

        dist_df = pd.DataFrame(list(distances.items()), columns=["Source Type", "Distance"]).sort_values("Distance")
        st.dataframe(dist_df, use_container_width=True)

        closest = dist_df.iloc[0]["Source Type"]

        st.markdown(f"""
        <div style="background:#fff3e0; padding:15px; border-radius:12px; border-left:6px solid #fb8c00;">
            <h4 style="color:#e65100;">📌 Closest Pollution Profile:</h4>
            <p style="font-size:18px;"><b>{closest}</b> — lowest similarity distance.</p>
        </div>
        """, unsafe_allow_html=True)

        # ======================================================================
        # FEATURE IMPORTANCE
        # ======================================================================
        st.markdown("<h3 style='color:#1b5e20;'>📘 Feature Importance</h3>", unsafe_allow_html=True)

        feature_importance = model.feature_importances_

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(FEATURES, feature_importance, color="#2e7d32")
        ax.set_xlabel("Importance Score")
        ax.set_title("Feature Influence on Prediction")
        st.pyplot(fig)

        # ======================================================================
        # RADAR CHART
        # ======================================================================
        st.markdown("<h3 style='color:#1b5e20;'>🕸 Pollution Fingerprint (Radar Chart)</h3>", unsafe_allow_html=True)

        labels = FEATURES
        values = [pm25, pm10, no2, so2, co, o3, traffic]

        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
        values = np.concatenate((values, [values[0]]))
        angles = np.concatenate((angles, [angles[0]]))

        fig2 = plt.figure(figsize=(6, 6))
        ax2 = fig2.add_subplot(111, polar=True)
        ax2.plot(angles, values, linewidth=2, linestyle="solid", color="#2e7d32")
        ax2.fill(angles, values, alpha=0.3, color="#66bb6a")
        ax2.set_thetagrids(angles[:-1] * 180/np.pi, labels)
        ax2.set_title("Pollution Radar Profile", color="#1b5e20")

        st.pyplot(fig2)

        # ======================================================================
        # TEXTUAL EXPLANATION
        # ======================================================================
        st.markdown("<h3 style='color:#1b5e20;'>📝 Explanation Summary</h3>", unsafe_allow_html=True)

        top_features = sorted(
            list(zip(FEATURES, feature_importance)),
            key=lambda x: x[1],
            reverse=True
        )[:3]

        explanation_text = f"""
        The prediction is primarily influenced by:
        - **{top_features[0][0]}** (highest impact)  
        - **{top_features[1][0]}**  
        - **{top_features[2][0]}**  

        These pollutant levels closely match typical patterns of **{pred_label}** emissions.
        """

        st.markdown(f"""
        <div style="background:#e3f2fd; padding:18px; border-radius:12px; border-left:6px solid #1e88e5;">
            {explanation_text}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
    <hr>
    <div style='text-align:center; padding:10px; color:#1b5e20;'>
        🌿 AI-powered insights guiding every breath — Choose a healthier path.
    </div>
    """, unsafe_allow_html=True)

        
# -------------------------------------------------------------
# 📽 NEW TAB: ADVANCED VISUAL DASHBOARD
# -------------------------------------------------------------
elif page == " Advanced Visual Dashboard":

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

    st.markdown("""
    <hr>
    <div style='text-align:center; padding:10px; color:#1b5e20;'>
        📽 Visualizing Earth’s heartbeat — Data for a greener generation.
    </div>
    """, unsafe_allow_html=True)


