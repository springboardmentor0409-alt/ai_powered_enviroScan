# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster, HeatMap
import matplotlib.pyplot as plt
import numpy as np
import datetime

# ------------------- STRONG CSS OVERRIDES (force light widgets everywhere) -------------------
st.markdown("""
<style>
/* Generic force-light fallback for Streamlit widgets */
:root { color-scheme: light !important; }

/* Basic page bg / title as before */
.stApp { background: linear-gradient(135deg, #b3e5fc 0%, #f8bbd0 100%) !important; }

/* Force main text dark */
* { color: #111 !important; }

/* Sidebar panel */
section[data-testid="stSidebar"] {
  background: #ffffff !important;
  color: #111 !important;
}

/* Most widget clickable areas */
div[role="combobox"],
div[data-baseweb="select"],
div[data-baseweb="select"] > div[role="button"],
div[role="listbox"],
div[role="option"],
input[type="text"], input[type="date"], textarea,
button, .stButton > button {
  background: #ffffff !important;
  color: #111 !important;
  border: 1px solid #d1d5db !important;
  border-radius: 10px !important;
  box-shadow: none !important;
}

/* Select internal list panel */
div[role="listbox"] {
  background: #ffffff !important;
  color: #111 !important;
}

/* Placeholder color */
::placeholder { color: #666 !important; opacity: 1 !important; }

/* Override some Streamlit auto-generated classes */
.css-1n76uvr, .css-1vencpc, .css-ocqkz7, .css-1aumxhk {
  background: transparent !important;
  color: #111 !important;
}

/* Caret / arrow */
div[data-baseweb="select"] svg { fill: #111 !important; }

/* Buttons hover */
.stButton > button:hover { background-color: #f3f4f6 !important; }

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* Page background */
.stApp {
    background: linear-gradient(135deg, #b3e5fc 0%, #f8bbd0 100%) !important;
}

/* Title styling */
.title-container {
    display: flex;
    align-items: center;
    background: linear-gradient(90deg,#4facfe 0%,#00f2fe 100%) !important;
    border-radius: 14px;
    padding: 15px 25px;
    margin-bottom: 1.5rem;
    color: white;
    box-shadow: 0 8px 30px rgba(0,0,0,0.3);
}
.title-container h1 { font-size: 2.7rem; font-weight: 900; margin: 0; text-shadow: 2px 2px 7px rgba(0,0,0,0.7); }

/* Make default text dark everywhere */
* { color: #111 !important; }

/* ---------- Sidebar: white panel ---------- */
section[data-testid="stSidebar"] {
    background: #ffffff !important;
    color: #111 !important;
    padding: 1.2rem !important;
    width: 340px !important;
    box-shadow: 6px 0 18px rgba(0,0,0,0.12);
    border-left: 1px solid rgba(0,0,0,0.04);
}

/* ---------- FORCE LIGHT STYLING FOR ALL WIDGETS (main + sidebar) ---------- */
/* Selectbox / Dropdown button */
div[data-baseweb="select"] > div[role="button"],
.stSelectbox > div[role="button"],
div[role="listbox"],
div[role="option"],
/* Text inputs & textareas */
.stTextInput input,
textarea,
input[type="text"],
/* Date input */
.stDateInput input,
input[type="date"],
/* Multiselect */
.stMultiSelect > div[role="button"],
/* Buttons */
.stButton > button,
button {
    background-color: #ffffff !important;
    color: #111 !important;
    border: 1px solid #d1d5db !important;
    border-radius: 10px !important;
    box-shadow: none !important;
}

/* Dropdown list panel (when expanded) */
div[data-testid="stHorizontalBlock"] div[role="listbox"],
div[role="listbox"],
div[role="presentation"] .rc-virtual-list {
    background-color: #ffffff !important;
    color: #111 !important;
}

/* Options inside dropdown */
div[role="option"],
div[role="option"] * {
    background-color: #ffffff !important;
    color: #111 !important;
}

/* Make placeholder and selected text dark */
::placeholder { color: #666 !important; opacity: 1 !important; }
.stSelectbox, .stTextInput, .stDateInput, .stMultiSelect {
    color: #111 !important;
}

/* Remove any dark theme backgrounds applied deeper */
.css-1n76uvr, .css-1vencpc, .css-1aumxhk, .css-ocqkz7 {
    background: transparent !important;
}

/* Tweak select caret color */
div[data-baseweb="select"] svg { fill: #111 !important; }

/* Buttons hover */
.stButton > button:hover {
    background-color: #f3f4f6 !important;
}

/* Ensure charts and headings are readable */
h1, h2, h3, h4, h5, h6 { color: #111 !important; text-shadow: none !important; }

/* Sidebar scrollbar */
section[data-testid="stSidebar"] ::-webkit-scrollbar { width: 10px; }
section[data-testid="stSidebar"] ::-webkit-scrollbar-thumb { background: rgba(0,0,0,0.08); border-radius: 6px; }
</style>
""", unsafe_allow_html=True)

# ------------------- Robust data loading -------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/labeled_pollution_data.csv")
    df.columns = [c.strip() for c in df.columns]

    # detect datetime-like column
    def find_dt_col(cols):
        cands = ["timestamp", "date", "time", "datetime"]
        lower = {c.lower(): c for c in cols}
        for c in cands:
            if c in lower:
                return lower[c]
        for c in cols:
            if "date" in c.lower() or "time" in c.lower():
                return c
        return None

    dt_col = find_dt_col(df.columns)

    if dt_col:
        df[dt_col] = df[dt_col].astype(str).str.strip()
        df[dt_col] = df[dt_col].replace({"": pd.NA, "NA": pd.NA, "N/A": pd.NA})
        df["Timestamp"] = pd.to_datetime(df[dt_col], errors="coerce", dayfirst=True, infer_datetime_format=True)
    else:
        df["Timestamp"] = pd.to_datetime("now")

    if "city" in df.columns:
        df["city"] = df["city"].astype(str).str.strip()

    for col in ["latitude", "longitude"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "source_label" in df.columns and "source" not in df.columns:
        df["source"] = df["source_label"]

    return df

df = load_data()

# pollutant mapping
POLLUTANT_MAP = {
    "PM2.5": "PM2.5",
    "PM10": "PM10",
    "NO2": "NO2",
    "SO2": "SO2",
    "CO": "CO",
    "O3": "O3",
}

# title
st.markdown("""
<div class="title-container">
    <h1>EnviroScan: AI-Powered Pollution Dashboard</h1>
</div>
""", unsafe_allow_html=True)

# session defaults
if "section" not in st.session_state:
    st.session_state["section"] = "Pollution Trends"
if "filters_applied" not in st.session_state:
    st.session_state["filters_applied"] = False

# ------------------- SIDEBAR -------------------
with st.sidebar:
    st.markdown("### Navigation")
    section_choice = st.radio(
        "Choose View:",
        ("Pollution Trends", "Source Distribution", "Map & Alerts", "Future Prediction"),
        index=0
    )
    st.session_state["section"] = section_choice

    st.markdown("---")
    st.header("Filters")

    if "city" in df.columns:
        cities = df["city"].dropna().unique().tolist()
        selected_city = st.selectbox("Select a city:", cities)
    else:
        selected_city = st.text_input("Enter city name:")

    if df["Timestamp"].notna().any():
        min_dt = df["Timestamp"].min().date()
        max_dt = df["Timestamp"].max().date()
    else:
        min_dt = datetime.date.today()
        max_dt = datetime.date.today()

    start_date = st.date_input("Start date", value=min_dt, min_value=min_dt, max_value=max_dt)
    end_date = st.date_input("End date", value=max_dt, min_value=min_dt, max_value=max_dt)

    if st.button("Apply City Filter"):
        st.session_state["filters_applied"] = True

# ------------------- Filtering -------------------
try:
    if st.session_state.get("filters_applied", False):
        if "city" in df.columns and selected_city:
            sel_norm = str(selected_city).strip().lower()
            df_city = df[df["city"].astype(str).str.strip().str.lower() == sel_norm]
        else:
            df_city = df.copy()

        df_filtered = df_city[
            (df_city["Timestamp"].dt.date >= start_date) &
            (df_city["Timestamp"].dt.date <= end_date)
        ]
    else:
        # default behaviour: filter by provided dates (change to df.head(0) if you want empty until apply)
        df_filtered = df[
            (df["Timestamp"].dt.date >= start_date) &
            (df["Timestamp"].dt.date <= end_date)
        ]
except Exception:
    df_filtered = df.copy()

# read section
section = st.session_state["section"]

# -------- Pollutant Trends --------
if section == "Pollution Trends":
    pollutants = list(POLLUTANT_MAP.keys())
    pollutant = st.selectbox("Select pollutant:", pollutants)
    col = POLLUTANT_MAP[pollutant]

    if col in df_filtered.columns:
        if "city" in df_filtered.columns and selected_city:
            sel_norm = str(selected_city).strip().lower()
            trend_df = df_filtered[df_filtered["city"].astype(str).str.strip().str.lower() == sel_norm]
        else:
            trend_df = df_filtered.copy()

        if not trend_df.empty:
            trend_data = trend_df.groupby("Timestamp")[col].mean().reset_index()
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(trend_data["Timestamp"], trend_data[col], marker="o", linestyle="-")
            ax.set_title(f"{pollutant} Trend")
            ax.set_xlabel("Date and Time")
            ax.set_ylabel(pollutant)
            ax.grid(True, linestyle="--", alpha=0.6)
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("No data available for the selected filters.")
    else:
        st.warning(f"Selected pollutant '{pollutant}' not available in the filtered data.")

# -------- Source Distribution --------
elif section == "Source Distribution":
    if "source" not in df_filtered.columns:
        st.info("No 'source' column available in the data to show distribution.")
    else:
        if "city" in df_filtered.columns and selected_city:
            sel_norm = str(selected_city).strip().lower()
            src_df = df_filtered[df_filtered["city"].astype(str).str.strip().str.lower() == sel_norm]
        else:
            src_df = df_filtered.copy()

        src = src_df["source"].value_counts()
        if src.empty:
            st.info("No source data available for this city/selection.")
        else:
            labels = src.index.tolist()
            sizes = src.values.tolist()
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
            ax.axis("equal")
            plt.title(f"Source Distribution")
            st.pyplot(fig)

# -------- Map & Alerts --------
elif section == "Map & Alerts":
    st.subheader(f"Map & Alerts")
    if "city" in df_filtered.columns and selected_city:
        sel_norm = str(selected_city).strip().lower()
        selected_city_data = df_filtered[df_filtered["city"].astype(str).str.strip().str.lower() == sel_norm].copy()
    else:
        selected_city_data = df_filtered.copy()

    if selected_city_data.empty:
        st.info("No data available for selected city/selection to show map.")
    else:
        pollutant_cols = [col for col in POLLUTANT_MAP.values() if col in selected_city_data.columns]
        thresholds = {"PM2.5": 60, "PM10": 100, "NO2": 40, "SO2": 20, "CO": 10, "O3": 70}
        alerts = []
        for disp, col in POLLUTANT_MAP.items():
            if col in selected_city_data.columns:
                max_val = selected_city_data[col].max()
                thr = thresholds.get(disp)
                if pd.notnull(max_val) and thr is not None and max_val > thr:
                    alerts.append(f"{disp}: Max {max_val:.1f} (> {thr})")

        if "latitude" in selected_city_data.columns and "longitude" in selected_city_data.columns:
            mean_lat = selected_city_data["latitude"].dropna().mean()
            mean_lon = selected_city_data["longitude"].dropna().mean()
            if pd.isna(mean_lat) or pd.isna(mean_lon):
                m = folium.Map(location=[15.9129, 79.7400], zoom_start=7)
            else:
                m = folium.Map(location=[mean_lat, mean_lon], zoom_start=11)

            mc = MarkerCluster().add_to(m)
            for _, row in selected_city_data.iterrows():
                if pd.notnull(row.get("latitude")) and pd.notnull(row.get("longitude")):
                    popup_parts = []
                    for disp, col in POLLUTANT_MAP.items():
                        if col in row.index and pd.notnull(row[col]):
                            try:
                                popup_parts.append(f"{disp}: {row[col]:.1f}")
                            except Exception:
                                popup_parts.append(f"{disp}: {row[col]}")
                    src_text = row.get("source", "Unknown")
                    popup = "<br>".join(popup_parts + [f"Source: {src_text}"])
                    try:
                        folium.CircleMarker(
                            location=[float(row["latitude"]), float(row["longitude"])],
                            radius=7,
                            color="#009688",
                            fill=True,
                            fill_opacity=0.75,
                            popup=f"{row.get('city', 'City')}<br>{popup}"
                        ).add_to(mc)
                    except Exception:
                        continue

            # heatmap
            heatcols = [c for c in pollutant_cols]
            avg_pollutant = selected_city_data[["latitude", "longitude"] + heatcols].dropna()
            heat_data = [
                [r["latitude"], r["longitude"], np.mean([r[c] for c in heatcols])]
                for _, r in avg_pollutant.iterrows()
            ]
            if heat_data:
                max_val = max([h[2] for h in heat_data])
                HeatMap(heat_data, radius=12, blur=16, max_val=max_val).add_to(m)

            st_folium(m, width=800, height=500)

            if alerts:
                st.warning(";  ".join(alerts))
            else:
                st.success("All pollutant levels within safe limits in selected city.")
        else:
            st.info("No spatial (latitude/longitude) data available for selected city/selection to show map.")

# -------- Future Prediction --------
elif section == "Future Prediction":
    st.subheader("Future AQI Prediction")
    pollutants = list(POLLUTANT_MAP.keys())
    pred_pollutant = st.selectbox("Select pollutant:", pollutants, key="prediction_pollutant")
    pred_col = POLLUTANT_MAP.get(pred_pollutant)

    if df["Timestamp"].notna().any():
        min_pred_date = max(df["Timestamp"].dt.date.max(), datetime.date.today())
    else:
        min_pred_date = datetime.date.today()
    max_pred_date = min_pred_date + datetime.timedelta(days=90)
    future_date = st.date_input("Select future date for prediction", value=min_pred_date, min_value=min_pred_date, max_value=max_pred_date, key="future_date_predict")
    predict_button = st.button("Predict AQI")
    if predict_button:
        if "city" in df.columns and selected_city:
            sel_norm = str(selected_city).strip().lower()
            pred_df = df[df["city"].astype(str).str.strip().str.lower() == sel_norm]
        else:
            pred_df = df.copy()

        if pred_col not in pred_df.columns:
            st.warning(f"No historical data for {pred_pollutant} to make predictions.")
        else:
            if not pred_df.empty:
                pred_trend = pred_df.groupby("Timestamp")[pred_col].mean().reset_index()
                last_values = pred_trend[pred_col].tail(5)
                predicted_value = last_values.mean() if not last_values.empty else None

                def get_aqi_category(val, pollutant):
                    if val is None or pd.isna(val):
                        return "Unknown"
                    if pollutant in ["PM2.5", "PM10"]:
                        if val <= 50: return "Good"
                        elif val <= 100: return "Satisfactory"
                        elif val <= 250: return "Moderate"
                        elif val <= 350: return "Poor"
                        else: return "Very Poor"
                    elif pollutant == "NO2":
                        if val <= 40: return "Good"
                        elif val <= 80: return "Satisfactory"
                        elif val <= 180: return "Moderate"
                        elif val <= 280: return "Poor"
                        else: return "Very Poor"
                    elif pollutant == "SO2":
                        if val <= 20: return "Good"
                        elif val <= 40: return "Satisfactory"
                        elif val <= 80: return "Moderate"
                        elif val <= 380: return "Poor"
                        else: return "Very Poor"
                    elif pollutant == "CO":
                        if val <= 1: return "Good"
                        elif val <= 2: return "Satisfactory"
                        elif val <= 10: return "Moderate"
                        elif val <= 17: return "Poor"
                        else: return "Very Poor"
                    elif pollutant == "O3":
                        if val <= 50: return "Good"
                        elif val <= 100: return "Satisfactory"
                        elif val <= 168: return "Moderate"
                        elif val <= 208: return "Poor"
                        else: return "Very Poor"
                    return "Unknown"

                aqi_category = get_aqi_category(predicted_value, pred_pollutant) if predicted_value is not None else "Unknown"
                if predicted_value is not None and not pd.isna(predicted_value):
                    st.success(f"⚡ Forecasted AQI: {predicted_value:.2f} ({aqi_category})")
                    fig, ax = plt.subplots(figsize=(7,4))
                    ax.scatter([future_date], [predicted_value], s=200)
                    ax.set_title(f'Future Prediction for {pred_pollutant} in {selected_city}')
                    ax.set_xlabel('Date')
                    ax.set_ylabel(pred_pollutant)
                    ax.grid(True, linestyle='--', alpha=0.5)
                    st.pyplot(fig)
                else:
                    st.warning("Insufficient historical values to produce a prediction.")
            else:
                st.warning("No data for predictions.")

# ------------------- Download -------------------
csv_bytes = df_filtered.to_csv(index=False).encode("utf-8")
st.download_button("Download filtered data", csv_bytes, "pollution_report.csv", "text/csv")
