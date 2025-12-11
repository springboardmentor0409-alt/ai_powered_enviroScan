import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

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
    h1, h2, h3, h4 {
        color: #1b5e20 !important;
    }
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
# LOAD BASE MODEL (Predictor uses lightweight model)
# -------------------------------------------------------------
model = joblib.load("models/light_model.joblib")
label_encoder = joblib.load("models/light_label_encoder.joblib")

FEATURES = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "traffic_index"]

# -------------------------------------------------------------
# SIDEBAR NAVIGATION
# -------------------------------------------------------------
st.sidebar.title("🌿 EnviroScan Navigation")

page = st.sidebar.radio(
    "Go To:",
    ["🏠 Home", "📊 EDA Overview", "📈 Model Comparison", "🔍 Pollution Predictor"]
)

# -------------------------------------------------------------
# 🏠 HOME PAGE 
# -------------------------------------------------------------
if page == "🏠 Home":

    st.markdown("""
    <div style="
        background: linear-gradient(to right, #a5d6a7, #c8e6c9);
        padding: 35px;
        border-radius: 12px;
        text-align: center;
        color: #1b5e20;
        margin-bottom: 30px;">
        <h1>🌍 EnviroScan – AI-Powered Pollution Intelligence System</h1>
        <h3>Machine Learning • Environmental Analytics • Real-Time Pollution Source Detection</h3>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### 🌱 Introduction  
    EnviroScan predicts the **most likely pollution source** using machine learning.

    It supports:
    - Pollution authorities  
    - Smart city teams  
    - Researchers  
    - Students & ML engineers  
    - Sustainability groups  

    ---

    ### 🔍 Why This Project Matters
    Major pollution comes from:
    - 🚗 Vehicles  
    - 🏭 Industries  
    - 🔥 Biomass & waste burning  
    - 🌾 Agricultural residue  

    EnviroScan helps reveal the **cause**, not just the values.

    ---

    ### 🧠 How AI Works  
    - Random Forest ML Model  
    - Uses 7 pollutant indicators  
    - Lightweight & fast  
    - Provides feature importance  

    ---

    ### 🌟 Project Overview  
    - 📌 Pollution Data + Weather Indicators  
    - 📌 Clean eco-friendly UI  
    - 📌 Prediction + Explanation Engine  
    - 📌 Multi-model comparison support  
    - 📌 Real-time analytics ready  
    """)

    # CARDS
    col1, col2, col3 = st.columns(3)

    col1.markdown("""
    <div style="background:#e8f5e9; padding:15px; border-radius:12px;">
        <h3>📘 Dataset</h3>
        <ul><li>Pollutant Indicators</li><li>Weather Features</li><li>Traffic Index</li><li>Geo Attributes</li></ul>
    </div>""", unsafe_allow_html=True)

    col2.markdown("""
    <div style="background:#e3f2fd; padding:15px; border-radius:12px;">
        <h3>🤖 Machine Learning</h3>
        <ul><li>Random Forest</li><li>Feature Importance</li><li>Confidence Scores</li></ul>
    </div>""", unsafe_allow_html=True)

    col3.markdown("""
    <div style="background:#fff3e0; padding:15px; border-radius:12px;">
        <h3>📊 Dashboard</h3>
        <ul><li>EDA Tools</li><li>Model Comparison</li><li>Prediction Engine</li></ul>
    </div>""", unsafe_allow_html=True)

    # TIMELINE
    st.markdown("""
    ---  
    ### 🛠 Project Workflow Timeline  

    <div style="display:flex; justify-content:space-between;">

        <div style="width:22%; background:#e8f5e9; padding:15px; border-radius:12px; text-align:center;">
            <h3>📥 Data Collection</h3>
            <p>Pollutant, weather & traffic data gathered.</p>
        </div>

        <div style="width:22%; background:#e3f2fd; padding:15px; border-radius:12px; text-align:center;">
            <h3>⚙️ Processing</h3>
            <p>Cleaning, filtering, engineering.</p>
        </div>

        <div style="width:22%; background:#fff9c4; padding:15px; border-radius:12px; text-align:center;">
            <h3>🤖 Model Training</h3>
            <p>AI learns pollutant signatures.</p>
        </div>

        <div style="width:22%; background:#ffe0b2; padding:15px; border-radius:12px; text-align:center;">
            <h3>📊 Dashboard</h3>
            <p>Visualization & real-time inference.</p>
        </div>

    </div>
    """, unsafe_allow_html=True)

# -------------------------------------------------------------
# 📊 EDA OVERVIEW
# -------------------------------------------------------------
elif page == "📊 EDA Overview":

    st.title("📊 Exploratory Data Analysis")

    st.markdown("""
    ### 📘 EDA Insights  
    This section provides:  
    - Dataset preview  
    - Summary statistics  
    - Missing value overview  
    - Correlation insights  
    """)

    st.subheader("📌 Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    st.subheader("📌 Summary Statistics")
    st.write(df.describe())

    st.subheader("📌 Missing Values")
    st.write(df.isnull().sum())

    st.subheader("📌 Correlation Matrix")
    st.write(df.corr())

# -------------------------------------------------------------
# 📈 MODEL COMPARISON  (YOUR UPDATED SMART VERSION)
# -------------------------------------------------------------
# -------------------------------------------------------------
# 📈 MODEL COMPARISON (SMART DISPLAY WITH DESCRIPTIONS)
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
    cv_img = base + "cv_f1_scores.png"   # 👈 FIXED NAME

    st.markdown(f"## 📌 Results for **{model_choice}**")

    # -------------------------------------------------
    # 🌿 FEATURE IMPORTANCE (Not available for Logistic Regression)
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
    # 📊 CLASSIFICATION REPORT
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
    # 🔷 CONFUSION MATRIX
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
    # 📈 CROSS VALIDATION F1 MACRO SCORES
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


# -------------------------------------------------------------
# 🔍 POLLUTION PREDICTOR
# -------------------------------------------------------------
elif page == "🔍 Pollution Predictor":

    st.title("🔍 AI Pollution Source Predictor")

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

        st.markdown(
            f"<div class='predict-card'>🌿 <b>Predicted Source:</b> {pred_label}<br>"
            f"🔎 <b>Confidence:</b> {confidence:.2f}%</div>",
            unsafe_allow_html=True
        )

        st.subheader("📘 Feature Importance")

        feature_importance = model.feature_importances_
        fig, ax = plt.subplots()
        ax.barh(FEATURES, feature_importance, color="#2e7d32")
        st.pyplot(fig)






















1)
# -------------------------------------------------------------
# 🔍 POLLUTION PREDICTOR
# -------------------------------------------------------------
elif page == "🔍 Pollution Predictor":

    st.title("🔍 AI Pollution Source Predictor")

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

        st.markdown(
            f"<div class='predict-card'>"
            f"🌿 <b>Predicted Source:</b> {pred_label}<br>"
            f"🔎 <b>Confidence:</b> {confidence:.2f}%"
            f"</div>",
            unsafe_allow_html=True
        )

        st.subheader("📘 Feature Importance")
        feature_importance = model.feature_importances_

        fig, ax = plt.subplots()
        ax.barh(FEATURES, feature_importance, color="#2e7d32")
        ax.set_xlabel("Importance Score")
        st.pyplot(fig)



2)
# -------------------------------------------------------------
# 🔍 POLLUTION PREDICTOR (UPGRADED UI)
# -------------------------------------------------------------
elif page == "🔍 Pollution Predictor":

    # Title
    st.markdown("""
    <h1 style="color:#1b5e20; text-align:center;">🔍 AI Pollution Source Predictor</h1>
    <p style="text-align:center; font-size:18px; color:#2e7d32;">
        Enter pollutant levels below to generate an instant AI-driven source prediction.
    </p>
    <hr style="border:1px solid #c8e6c9;">
    """, unsafe_allow_html=True)

    # Input columns
    st.markdown("<h3 style='color:#1b5e20;'>🧪 Enter Pollutant Levels</h3>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        pm25 = st.number_input("PM2.5 (µg/m³)", value=45.0, help="Fine particulate matter")
        no2 = st.number_input("NO₂ (µg/m³)", value=20.0, help="Nitrogen Dioxide")
        co = st.number_input("CO (mg/m³)", value=1.0, help="Carbon Monoxide")

    with col2:
        pm10 = st.number_input("PM10 (µg/m³)", value=70.0)
        so2 = st.number_input("SO₂ (µg/m³)", value=5.0)
        o3 = st.number_input("O₃ (µg/m³)", value=12.0)
        traffic = st.number_input("Traffic Index", value=30.0, help="Relative traffic density score")

    st.markdown("<br>", unsafe_allow_html=True)

    # Predict button
    predict = st.button("🌿 Predict Pollution Source", use_container_width=True)

    if predict:
        # Prepare input
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

        # Result Card (Beautiful Green Gradient)
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
            🌿 Predicted Pollution Source: <span style="font-size:28px;">{pred_label}</span><br>
            🔎 Confidence: <span style="font-size:26px;">{confidence:.2f}%</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br><h3 style='color:#1b5e20;'>📘 Feature Importance</h3>", unsafe_allow_html=True)

        # Feature Importance Chart Container
        st.markdown("""
        <div style="
            background:#e8f5e9;
            padding:20px;
            border-radius:12px;
            border: 2px solid #a5d6a7;
            margin-bottom:15px;">
        """, unsafe_allow_html=True)

        # Plot feature importance
        feature_importance = model.feature_importances_
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(FEATURES, feature_importance, color="#2e7d32")
        ax.set_xlabel("Importance Score", color="#1b5e20")
        ax.set_ylabel("Features", color="#1b5e20")
        ax.set_title("Feature Influence on Prediction", color="#1b5e20")

        st.pyplot(fig)

        # Close container
        st.markdown("</div>", unsafe_allow_html=True)



3)
# -------------------------------------------------------------
# 🔍 POLLUTION PREDICTOR (PREMIUM UPGRADED UI)
# -------------------------------------------------------------
elif page == "🔍 Pollution Predictor":

    # Title
    st.markdown("""
    <h1 style="color:#1b5e20; text-align:center;">🔍 AI Pollution Source Predictor</h1>
    <p style="text-align:center; font-size:18px; color:#2e7d32;">
        Enter pollutant levels below to generate an instant AI-driven pollution source prediction.
    </p>
    <hr style="border:1px solid #c8e6c9;">
    """, unsafe_allow_html=True)

    st.markdown("<h3 style='color:#1b5e20;'>🧪 Enter Pollution Indicators</h3>", unsafe_allow_html=True)

    # -------------- INPUT FIELDS --------------
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

    # -------------- PREDICT BUTTON --------------
    predict = st.button("🌿 Predict Pollution Source", use_container_width=True)

    # ============================================================================
    # RUN PREDICTION
    # ============================================================================
    if predict:

        # Build input DataFrame
        input_data = pd.DataFrame([{
            "PM2.5": pm25, "PM10": pm10,
            "NO2": no2, "SO2": so2,
            "CO": co, "O3": o3,
            "traffic_index": traffic
        }])

        pred_encoded = model.predict(input_data)[0]
        pred_label = label_encoder.inverse_transform([pred_encoded])[0]
        confidence = model.predict_proba(input_data).max() * 100

        # -------------- CONFIDENCE BADGE --------------
        if confidence >= 80:
            conf_color = "#2e7d32"   # Green
            conf_text = "High Confidence"
        elif confidence >= 60:
            conf_color = "#f9a825"   # Yellow
            conf_text = "Moderate Confidence"
        else:
            conf_color = "#c62828"   # Red
            conf_text = "Low Confidence"

        # Beautiful Result Card
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

        # ============================================================================
        # HEALTH RISK ASSESSMENT CARD
        # ============================================================================
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

        # ============================================================================
        # FEATURE IMPORTANCE
        # ============================================================================
        st.markdown("<h3 style='color:#1b5e20;'>📘 Feature Importance</h3>", unsafe_allow_html=True)

        feature_importance = model.feature_importances_

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(FEATURES, feature_importance, color="#2e7d32")
        ax.set_xlabel("Importance Score")
        ax.set_title("Feature Influence on Prediction")
        st.pyplot(fig)

        # ============================================================================
        # RADAR CHART (Pollution Fingerprint)
        # ============================================================================
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

        # ============================================================================
        # EXPLANATION SUMMARY
        # ============================================================================
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
