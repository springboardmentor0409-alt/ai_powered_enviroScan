import streamlit as st
import requests
import pandas as pd
import datetime
from pathlib import Path
from PIL import Image
import os

# --- CONFIGURATION ---
API_URL = "http://localhost:8000"
st.set_page_config(page_title="EnviroScan Pollution Source", layout="wide")

# --- Helper functions for browsing local result folders ---
ROOT = Path(".").resolve()
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"

def list_files(folder: Path, exts=None):
    if not folder.exists() or not folder.is_dir():
        return []
    if exts:
        return sorted([p for p in folder.iterdir() if p.suffix.lower() in exts])
    return sorted([p for p in folder.iterdir() if p.is_file()])

def safe_image_display(path: Path, caption=None, width=600):
    try:
        img = Image.open(path)
        st.image(img, caption=caption or path.name, use_container_width=True)
    except Exception as e:
        st.write(f"Could not open image {path.name}: {e}")

def safe_csv_display(path: Path, nrows=10):
    try:
        df = pd.read_csv(path)
        st.dataframe(df.head(nrows))
        if df.shape[0] > nrows:
            st.write(f"Showing first {nrows} rows of {df.shape[0]} total.")
    except Exception as e:
        st.write(f"Could not read CSV {path.name}: {e}")

# --- SIDEBAR: Sections ---
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["Live Demo", "About", "Data Visualization (EDA)", "Model Evaluation"])

# --- ABOUT ---
if section == "About":
    st.title("🏭 EnviroScan — About")
    st.markdown(
        """
        **EnviroScan Pollution Source Prediction** is a lightweight tool that predicts the likely source of
        air pollution (e.g., Vehicular, Industrial, Agricultural, Burning) using environmental measurements
        and spatial context.

        **Highlights**
        - Preprocessing pipeline with clipping, imputation, scaling, and wind alignment.
        - Multiple model backends (Random Forest, Decision Tree, XGBoost, Logistic Regression).
        - Visual EDA and model evaluation results are available under `results/`.
        - Live demo allows trying models served by the backend API.
        """
    )
    st.markdown("**Project structure (expected):**")
    st.code(
        """
    .
    ├─ backend/
    ├─ models/
    ├─ results/
    │  ├─ eda/
    │  ├─ decision_tree/
    │  ├─ logistic_regression/
    │  ├─ random_forest/
    │  └─ xgboost_model/
    └─ frontend_streamlit.py
        """,
        language="bash",
    )
    st.write("---")
    st.markdown("If anything is missing, place the corresponding files (images/CSVs/models) in the folders above and reload the app.")

# --- DATA VISUALIZATION (EDA) ---
elif section == "Data Visualization (EDA)":
    st.title("📊 Data Visualization — EDA Results")
    eda_folder = RESULTS_DIR / "eda"
    if not eda_folder.exists():
        st.warning(f"No EDA results found at `{eda_folder}`. Place EDA images and CSVs there.")
    else:
        st.markdown(f"Browsing `{eda_folder}`")
        image_files = list_files(eda_folder, exts={".png", ".jpg", ".jpeg"})
        csv_files = list_files(eda_folder, exts={".csv"})

        if image_files:
            img_choice = st.selectbox("Select an image to view", [p.name for p in image_files])
            chosen = eda_folder / img_choice
            safe_image_display(chosen)
        else:
            st.info("No images found in EDA folder.")

        st.write("---")
        if csv_files:
            csv_choice = st.selectbox("Select a CSV to preview", [p.name for p in csv_files], key="eda_csv")
            safe_csv_display(eda_folder / csv_choice, nrows=15)
            # Option to download CSV
            with open(eda_folder / csv_choice, "rb") as f:
                st.download_button("Download CSV", f, file_name=csv_choice)
        else:
            st.info("No CSV files found in EDA folder.")

# --- MODEL EVALUATION ---
elif section == "Model Evaluation":
    st.title("📈 Model Evaluation Results")
    model_result_dirs = {
        "Decision Tree": RESULTS_DIR / "decision_tree",
        "Logistic Regression": RESULTS_DIR / "logistic_regression",
        "Random Forest": RESULTS_DIR / "random_forest",
        "XGBoost": RESULTS_DIR / "xgboost_model",
    }

    chosen_model_eval = st.selectbox("Choose model folder", list(model_result_dirs.keys()))
    folder = model_result_dirs[chosen_model_eval]

    if not folder.exists():
        st.warning(f"No results found for `{chosen_model_eval}` at `{folder}`.")
    else:
        st.markdown(f"Browsing `{folder}`")
        image_files = list_files(folder, exts={".png", ".jpg", ".jpeg"})
        csv_files = list_files(folder, exts={".csv"})

        if image_files:
            # show multiple thumbnails
            st.subheader("Images / Plots")
            cols = st.columns(3)
            for i, img_path in enumerate(image_files):
                with cols[i % 3]:
                    try:
                        img = Image.open(img_path)
                        st.image(img, caption=img_path.name, use_container_width=True)
                    except Exception as e:
                        st.write(f"Unable to open {img_path.name}: {e}")
        else:
            st.info("No evaluation images found for this model.")

        st.write("---")
        if csv_files:
            st.subheader("Tabular Results / Metrics")
            csv_choice = st.selectbox("Choose CSV to preview", [p.name for p in csv_files], key=f"eval_{chosen_model_eval}")
            safe_csv_display(folder / csv_choice, nrows=25)
            with open(folder / csv_choice, "rb") as f:
                st.download_button("Download CSV", f, file_name=csv_choice)
        else:
            st.info("No CSV metric files found for this model.")

# --- LIVE DEMO (Default) ---
elif section == "Live Demo":
    st.title("🔬 Live Demo — Predict Source")

    # Discover models available in models/ directory (folders)
    available_model_dirs = []
    if MODELS_DIR.exists():
        for p in MODELS_DIR.iterdir():
            if p.is_dir():
                available_model_dirs.append(p.name)
    # Fallback to a minimal list if nothing there
    if not available_model_dirs:
        available_model_dirs = ["random_forest", "decision_tree", "logistic_regression", "xgboost_model"]

    st.sidebar.markdown("### Live Demo Controls")
    # Primary model selection (keeps old selected_model UI behavior)
    selected_model_name = st.sidebar.selectbox("Select Model (to call API)", available_model_dirs, index=available_model_dirs.index("random_forest") if "random_forest" in available_model_dirs else 0)
    # Optionally let user choose a model file inside that folder (for reference)
    model_files = list_files(MODELS_DIR / selected_model_name, exts={".joblib", ".pkl"}) if (MODELS_DIR / selected_model_name).exists() else []
    if model_files:
        st.sidebar.write("Model files detected:")
        st.sidebar.selectbox("Model artifact (for info only)", [p.name for p in model_files])
    else:
        st.sidebar.info("No local model artifact found — backend must host the model to respond.")

    # --- MAIN FORM (same as original) ---
    st.subheader("Input Environmental Parameters")
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            date_val = st.date_input("Date", datetime.date(2023, 1, 15))
            city = st.selectbox("City", ["Delhi", "Mumbai", "Kolkata", "Chennai", "Bangalore", "Hyderabad"])
            location_id = st.text_input("Location ID", "LOC_001")
            latitude = st.number_input("Latitude", value=28.61, format="%.4f")
            longitude = st.number_input("Longitude", value=77.23, format="%.4f")

        with col2:
            pm25 = st.number_input("PM2.5", value=100.0)
            pm10 = st.number_input("PM10", value=180.0)
            no2 = st.number_input("NO2", value=40.0)
            so2 = st.number_input("SO2", value=10.0)
            co = st.number_input("CO", value=1.0)
            o3 = st.number_input("O3", value=30.0)

        with col3:
            temp = st.number_input("Temperature (°C)", value=25.0)
            humidity = st.number_input("Humidity (%)", value=50.0)
            wind_speed = st.number_input("Wind Speed (km/h)", value=5.0)
            wind_dir = st.number_input("Wind Direction (deg)", value=120.0, min_value=0.0, max_value=360.0)
            traffic = st.number_input("Traffic Index", value=50.0)

        st.subheader("Spatial Context")
        c1, c2, c3 = st.columns(3)
        with c1:
            dist_road = st.number_input("Dist to Road (m)", value=500.0)
        with c2:
            dist_ind = st.number_input("Dist to Industry (m)", value=2000.0)
        with c3:
            dist_farm = st.number_input("Dist to Farm (m)", value=5000.0)

        st.subheader("Fire Data")
        cd1, cd2 = st.columns(2)
        with cd1:
            fire_nearby = st.selectbox("Fire Nearby?", ["No", "Yes"])
            fire_val = 1 if fire_nearby == "Yes" else 0
        with cd2:
            fire_dist = st.number_input("Fire Min Dist (km)", value=50.0)

        submit_button = st.form_submit_button("🔍 Predict Source")

    if submit_button:
        # Prepare payload
        payload = {
            "date": date_val.strftime("%d-%m-%Y"),
            "city": city,
            "location_id": location_id,
            "latitude": latitude,
            "longitude": longitude,
            "PM2_5": pm25,
            "PM10": pm10,
            "NO2": no2,
            "SO2": so2,
            "CO": co,
            "O3": o3,
            "temp": temp,
            "humidity": humidity,
            "wind_speed": wind_speed,
            "wind_dir": wind_dir,
            "dist_to_road": dist_road,
            "dist_to_industry": dist_ind,
            "dist_to_farm": dist_farm,
            "traffic_index": traffic,
            "fire_nearby": fire_val,
            "fire_min_dist_km": fire_dist
        }

        with st.spinner(f"Querying model `{selected_model_name}` on backend..."):
            try:
                model_slug = selected_model_name.replace(" ", "_").lower()
                response = requests.post(f"{API_URL}/predict/{model_slug}", json=payload, timeout=8)

                if response.status_code == 200:
                    result = response.json()
                    pred_class = result.get("prediction", "Unknown")
                    conf = result.get("confidence", "N/A")

                    st.success(f"**Predicted Source:** {pred_class}")
                    st.write(f"**Model:** {result.get('model', selected_model_name)}")
                    st.write(f"**Confidence:** {conf}")

                    # Visual enhancement based on result
                    if pred_class == "Vehicular":
                        st.info("High contribution from traffic emissions detected. (NO2, CO, Proximity to Roads)")
                    elif pred_class == "Industrial":
                        st.warning("Industrial emissions detected. (SO2, PM10, Proximity to Industry)")
                    elif pred_class == "Agricultural":
                        st.success("Agricultural source detected. (Ammonia, PM10, Farm proximity)")
                    elif pred_class == "Burning":
                        st.error("Biomass/Waste burning detected. (PM2.5, Fire detection)")
                else:
                    st.error(f"Error {response.status_code}: {response.text}")

            except requests.exceptions.ConnectionError:
                st.error("Could not connect to Backend API. Is it running?")
            except requests.exceptions.ReadTimeout:
                st.error("Request timed out. Backend might be busy.")
            except Exception as e:
                st.error(f"An error occurred: {e}")
