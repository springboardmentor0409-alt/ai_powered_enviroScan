import streamlit as st
import pandas as pd
import numpy as np
from streamlit_option_menu import option_menu
from pathlib import Path

# Base folder: the directory where this file (app2.py) lives
HERE = Path(__file__).parent

# Helper to safely load local CSS
def local_css(filename):
    css_path = HERE / filename
    try:
        with open(css_path, encoding="utf-8") as f:
            css = f.read()
            st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"CSS file not found: {css_path}. Continuing without custom styles.")

# Helper to display images safely (falls back to an external placeholder)
def safe_image(filename_or_url, **kwargs):
    """
    If filename_or_url is a local file name (exists inside landing_page/), load it.
    Otherwise, if it's a full URL, Streamlit will load it directly.
    If local file missing, use an external placeholder so the app doesn't crash.
    """
    candidate = HERE / filename_or_url
    # If it's a path-like and exists locally, show it
    if candidate.exists():
        st.image(str(candidate), **kwargs)
    else:
        # If input already looks like a URL (http...), try to show it directly
        if isinstance(filename_or_url, str) and filename_or_url.startswith(("http://", "https://")):
            st.image(filename_or_url, **kwargs)
        else:
            # fallback placeholder
            st.image("https://via.placeholder.com/400x200.png?text=Missing+Image", **kwargs)

# Ensure session state for expanded card
if 'expanded_card' not in st.session_state:
    st.session_state.expanded_card = None

st.set_page_config(layout="wide")

# Load CSS from landing_page/style.css (works even if you run Streamlit from repo root)
local_css("style.css")


def create_clickable_card(card_id, image_path, title, short_text):
    """
    Creates a clickable small card and an Expand button that sets session_state.expanded_card.
    Uses safe_image to prevent crashes if the image is missing.
    """
    st.markdown(f'<div class="card-container-simple">', unsafe_allow_html=True)

    # Display the small card elements (thumbnail)
    safe_image(image_path, caption=title, width=150)
    st.markdown(f'<p class="card-short-text">{short_text}</p>', unsafe_allow_html=True)

    # Button that expands the card when clicked
    if st.button("Expand", key=f"btn_{card_id}"):
        st.session_state.expanded_card = card_id

    st.markdown(f'</div>', unsafe_allow_html=True)


# Sidebar menu
with st.sidebar:
    selected = option_menu(
        menu_title=None,
        options=["Home", "Visualization", "Contact"],
        icons=["house", "book", "envelope"],
        default_index=0
    )


if selected == "Home":
    col_logo, col_title = st.columns([1.2, 7.6])

    with col_logo:
        # Display the image in the smaller column (safe)
        safe_image("logo.png", width=170)

    with col_title:
        st.markdown(
            ''' 
            <h1 class="main-title-gradient" style="margin-top: 0; padding-top: 0;">
                EnviroScan: AI-Powered Pollution Source Identifier using Geospatial Analytics
            </h1>
            ''',
            unsafe_allow_html=True
        )

    st.markdown('<h3>🎯<span class="gradient-text-only">Objective of the Project</span></h3>', unsafe_allow_html=True)
    st.write(
        "The primary objective of the EnviroScan project is to develop an AI-powered system that uses machine learning "
        "and geospatial analytics to not only monitor pollutant levels but, more critically, to identify and predict the "
        "specific likely sources of pollution (e.g., industrial, vehicular, agricultural). This system aims to transform raw "
        "sensor data into actionable insights for environmental authorities and urban planners."
    )

    st.markdown('<h3>❓ <span class="gradient-text-only">Why We Need This Project</span></h3>', unsafe_allow_html=True)
    st.markdown("""
        - Source Attribution Gap: Current systems measure pollution levels but fail to identify the specific sources (the "Why"), hindering effective action.
        - Targeted Action Requirement: Authorities need to move beyond generic measures to targeted interventions (e.g., regulating a specific factory vs. general traffic control).
        - Complex Urban Pollution: Urban pollution is a mix of sources (vehicular, industrial, natural); an intelligent system is needed to disentangle and prioritize these factors.
        - Data-Driven Policy: The project provides the quantifiable evidence and insights necessary for effective environmental policy-making and urban planning.    
        - Proactive Management: Enables real-time, source-specific alerts, allowing for a more proactive and timely response to high-risk pollution events.
    """)
    st.markdown('<h3>🚀 <span class="gradient-text-only">Future Scope</span></h3>', unsafe_allow_html=True)
    st.markdown("""
    - Advanced Forecasting: Implement models (like LSTM) to predict future pollution levels and source contributions hours or days in advance.
    - Satellite and External Data Integration: Incorporate satellite imagery and social media/citizen reports to enhance detection, especially for large-scale events like agricultural burning.
    - Mobile Application & Enhanced Alerts: Create a mobile app for field use and integrate advanced alerting via SMS/Voice for critical conditions.
    - Conversational AI for Data Exploration: Integrate a Conversational Analytics Chatbot into the dashboard, allowing users (planners, inspectors) to ask complex, multi-step questions about the data in plain language.
    - Deep Learning Migration for Source Attribution: Transition from traditional machine learning (Random Forest, XGBoost) to Deep Learning (e.g., CNN-LSTM hybrid models) to better capture complex, non-linear spatio-temporal patterns.
    """, unsafe_allow_html=True)


if selected == "Visualization":
    st.markdown('<h3 class="center-heading">Geospatial Insights Gallery</h3>', unsafe_allow_html=True)

    # --- 1. HORIZONTAL LAYOUT OF CARDS (5 boxes) ---
    col1, col2, col3, col4, col5 = st.columns(5)

    # card metadata
    card_data = [
        {"id": "card_a", "img": "no2_outliers.png", "title": "NO₂ Outliers", "short": "NO2 Boxplot Outliers",
         "long": "This analysis highlights high-risk nitrogen dioxide outliers. The concentration spikes are correlated with specific industrial zones, suggesting primary source contributions."},
        {"id": "card_b", "img": "o3_outliers.png", "title": "O3 Outliers", "short": "O3 Boxplot Outliers",
         "long": "The O3 box plot shows that the median Ozone concentration is around 30, with the middle 50% clustered between 25 and 35. A significant set of data points ranging from 60 to 78 are identified as outliers, indicating periods of unusually high Ozone levels."},
        {"id": "card_c", "img": "so2_outliers.png", "title": "SO₂ Outliers", "short": "SO2 Boxplot Outliers",
         "long": "This SO2 box plot shows that Sulphur Dioxide concentrations are extremely low, with the middle 50% of the data close to zero. The plot is severely right-skewed and features a large number of extreme outliers extending up to around 400."},
        {"id": "card_d", "img": "PM2.5_outliers.png", "title": "PM2.5 Outliers", "short": "PM2.5 Boxplot Outliers",
         "long": "This PM2.5 box plot shows that Particulate Matter concentrations are very low (median near zero), with the core data being highly concentrated. The plot is heavily right-skewed, displaying numerous extreme outliers that reach up to approximately 2100."},
        {"id": "card_e", "img": "pm10_outliers.png", "title": "PM10 Outliers", "short": "PM10 Boxplot Outliers",
         "long": "The PM10 box plot indicates that the median Particulate Matter concentration is very low (near zero), showing most data points are clustered at the bottom end. The distribution is extremely right-skewed with many outliers, some reaching concentrations as high as 5000."}
    ]

    # create the small cards
    with col1:
        create_clickable_card(
            card_id=card_data[0]["id"],
            image_path=card_data[0]["img"],
            title=card_data[0]["title"],
            short_text=card_data[0]["short"]
        )
    with col2:
        create_clickable_card(
            card_id=card_data[1]["id"],
            image_path=card_data[1]["img"],
            title=card_data[1]["title"],
            short_text=card_data[1]["short"]
        )
    with col3:
        create_clickable_card(
            card_id=card_data[2]["id"],
            image_path=card_data[2]["img"],
            title=card_data[2]["title"],
            short_text=card_data[2]["short"]
        )
    with col4:
        create_clickable_card(
            card_id=card_data[3]["id"],
            image_path=card_data[3]["img"],
            title=card_data[3]["title"],
            short_text=card_data[3]["short"]
        )
    with col5:
        create_clickable_card(
            card_id=card_data[4]["id"],
            image_path=card_data[4]["img"],
            title=card_data[4]["title"],
            short_text=card_data[4]["short"]
        )

    st.markdown("---")  # Separator line

    # --- 2. THE EXPANDED CENTRAL VIEW (MODAL SIMULATION) ---
    if st.session_state.expanded_card:
        selected_data = next((item for item in card_data if item["id"] == st.session_state.expanded_card), None)

        if selected_data:
            center_col, content_col, end_col = st.columns([1, 4, 1])
            with content_col:
                if st.button("Close ❌", key="close_modal_btn"):
                    st.session_state.expanded_card = None

                st.markdown('<div class="expanded-modal-box">', unsafe_allow_html=True)
                st.subheader(selected_data['title'])
                st.markdown("---")
                # show large image (safe)
                safe_image(selected_data['img'], use_container_width=True)
                st.markdown(f'<p class="modal-text">{selected_data["long"]}</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

    # show summary images safely
    safe_image("daily_pollutant_trends.png", use_container_width=True)
    st.markdown("---")
    st.subheader("Geospatial Distribution and Numeric Feature Relationships")

    plot_col1, plot_col2 = st.columns(2)
    with plot_col1:
        safe_image("Geospatial_scatter.png", caption="Geospatial Scatter — Data Points", width='stretch')
    with plot_col2:
        safe_image("heatmap.png", caption="Correlation Heatmap (Numeric Features Only)", width='stretch')


if selected == "Contact":
    st.title("Contact")
    st.write("For collaboration or help, contact the project owner or mentor.")
