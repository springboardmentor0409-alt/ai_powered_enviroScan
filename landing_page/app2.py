import streamlit as st
import pandas as pd
import numpy as np
from streamlit_option_menu import option_menu

if 'expanded_card' not in st.session_state:
    # Set the initial state to None, meaning no card is expanded yet.
    st.session_state.expanded_card = None

st.set_page_config(layout="wide")
def local_css(file_name):
    try:
        # Using a context manager for reliable file reading
        with open(file_name, encoding="utf-8") as f: 
            css = f.read()
            st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
           # st.success("CSS file successfully injected!") # line for confomation css working properly
    except FileNotFoundError:
        st.error(f"FATAL ERROR: Could not find the CSS file at {file_name}. Please check its path.")
local_css("style.css")

def create_clickable_card(card_id, image_path, title, short_text):
    """
    Creates a clickable card that updates the session state when pressed.
    This function should only receive the data necessary for the small card display.
    """
    
    st.markdown(f'<div class="card-container-simple">', unsafe_allow_html=True)
    
    # Display the small card elements
    st.image(image_path, caption=title, width=150) 
    st.markdown(f'<p class="card-short-text">{short_text}</p>', unsafe_allow_html=True)
    
    # The Button is the unique element that needs a key.
    # We now pass card_id as the primary argument.
    if st.button("Expand", key=f"btn_{card_id}"):
        st.session_state.expanded_card = card_id
        # We don't need st.experimental_rerun() here as the button click
        # already triggers a rerun by default when it returns True.

    st.markdown(f'</div>', unsafe_allow_html=True)
    

with st.sidebar:
    selected =option_menu(
       menu_title=None,
        options=["Home","Visualization","Contact"],
        icons=["house","book","envelope"],
        default_index=0
    )


if selected=="Home":
    col_logo, col_title = st.columns([1.2, 7.6])
    
    with col_logo:
        # Display the image in the smaller column
        # Using the file name from your uploaded content
        st.image("logo.png", width=170) 

    with col_title:
        # The title is placed right next to the logo in the second column
        # We added an inline style to ensure the H1 margin doesn't push it too far down
        st.markdown(
            ''' 
            <h1 class="main-title-gradient" style="margin-top: 0; padding-top: 0;">
                EnviroScan:AI-Powered Pollution Source Identifier using Geospatial Analytics
            </h1>
            ''',
            unsafe_allow_html=True
        )

    st.markdown(
        '<h3>🎯<span class="gradient-text-only">Objective of the Project</span></h3>', 
        unsafe_allow_html=True
        )
    st.write("The primary objective of the EnviroScan project is to develop an AI-powered system that uses machine learning and geospatial analytics to not only monitor pollutant levels but, more critically, to identify and predict the specific likely sources of pollution (e.g., industrial, vehicular, agricultural). This system aims to transform raw sensor data into actionable insights for environmental authorities and urban planners.")
    st.markdown('<h3>❓ <span class="gradient-text-only">Why We Need This Project</span></h3>', unsafe_allow_html=True)
    st.markdown("""
        - Source Attribution Gap: Current systems measure pollution levels but fail to identify the specific sources (the "Why"), hindering effective action.
        - Targeted Action Requirement: Authorities need to move beyond generic measures to targeted interventions (e.g., regulating a specific factory vs. general traffic control).
        - Complex Urban Pollution: Urban pollution is a mix of sources (vehicular, industrial, natural); an intelligent system is needed to disentangle and prioritize these factors.
        - Data-Driven Policy: The project provides the quantifiable evidence and insights necessary for effective environmental policy-making and urban planning.    
        - Proactive Management: Enables real-time, source-specific alerts, allowing for a more proactive and timely response to high-risk pollution events.
        """)
    st.markdown(
        '<h3>🚀 <span class="gradient-text-only">Future Scope</span></h3>', 
        unsafe_allow_html=True
        )
    st.markdown("""
    - Advanced Forecasting:  Implement models (like LSTM) to predict future pollution levels and source contributions hours or days in advance.
    - Satellite and External Data Integration:  Incorporate satellite imagery and social media/citizen reports to enhance detection, especially for large-scale events like agricultural burning.
    - Mobile Application & Enhanced Alerts:  Create a mobile app for field use and integrate advanced alerting via SMS/Voice for critical conditions.
    - Conversational AI for Data Exploration: Integrate a Conversational Analytics Chatbot into the dashboard, allowing users (planners, inspectors) to ask complex, multi-step questions about the data in plain language (e.g., "Show the predicted industrial contribution near the river last Tuesday").
    -  Deep Learning Migration for Source Attribution: Transition from traditional machine learning (Random Forest, XGBoost) to Deep Learning (e.g., CNN-LSTM hybrid models) to better capture complex, non-linear spatio-temporal patterns and improve source-tracing accuracy across different regions.    
    """)

if selected == "Visualization":
    #st.title(f"You have selected {selected}")
    st.markdown('<h3 class="center-heading">Geospatial Insights Gallery</h3>', unsafe_allow_html=True)
    # --- 1. HORIZONTAL LAYOUT OF CARDS (5 boxes) ---
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # Use a clearer structure for card_data
    card_data = [
        {"id": "card_a", "img": "no2_outliers.png", "title": "NO₂ Outliers","short": "NO2 Boxplot Outliers", "long": "This analysis highlights high-risk nitrogen dioxide outliers. The concentration spikes are correlated with specific industrial zones, suggesting primary source contributions."},
        {"id": "card_b", "img": "o3_outliers.png", "title": "O3 Outliers", "short": "O3 Boxplot Outliers","long": "The O3 box plot shows that the median Ozone concentration is around 30, with the middle 50% clustered between 25 and 35. A significant set of data points ranging from 60 to 78 are identified as outliers, indicating periods of unusually high Ozone levels."},
        {"id": "card_c", "img": "so2_outliers.png", "title": "SO₂ Outliers", "short": "SO2 Boxplot Outliers", "long": "This SO2 box plot shows that Sulphur Dioxide concentrations are extremely low, with the middle 50% of the data close to zero. The plot is severely right-skewed and features a large number of extreme outliers extending up to around 400."},
        {"id": "card_d", "img": "PM2.5_outliers.png", "title": "PM2.5 Outliers", "short": "PM2.5 Boxplot Outliers", "long": "This PM2.5 box plot shows that Particulate Matter concentrations are very low (median near zero), with the core data being highly concentrated. The plot is heavily right-skewed, displaying numerous extreme outliers that reach up to approximately 2100."},
        {"id": "card_e", "img": "pm10_outliers.png", "title": "PM10 Outliers", "short": "PM10 Boxplot Outliers", "long": "The PM10 box plot indicates that the median Particulate Matter concentration is very low (near zero), showing most data points are clustered at the bottom end. The distribution is extremely right-skewed with many outliers, some reaching concentrations as high as 5000."}
    ]
    # Create the cards in the columns
    # We now explicitly pass the required arguments for the *small* card display.
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
    
    st.markdown("---") # Separator line
    
    # --- 2. THE EXPANDED CENTRAL VIEW (MODAL SIMULATION) ---
    
    if st.session_state.expanded_card:
        # Retrieve the data for the currently selected card directly from card_data
        selected_data = next((item for item in card_data if item["id"] == st.session_state.expanded_card), None)
        
        if selected_data:
            # Create a large container in the middle
            center_col, content_col, end_col = st.columns([1, 4, 1])
            
            with content_col:
                # Add a button to close the modal
                if st.button("Close ❌", key="close_modal_btn"):
                    st.session_state.expanded_card = None
                    #st.experimental_rerun()
                    
                st.markdown('<div class="expanded-modal-box">', unsafe_allow_html=True)
                st.subheader(selected_data['title'])
                st.markdown("---")
                st.image(selected_data['img'], use_container_width=True) 
                st.markdown(f'<p class="modal-text">{selected_data["long"]}</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
    # -----------------------------------------------------------------    
    st.image("daily_pollutant_trends.png")
    st.markdown("---") # Separator line
    # -----------------------------------------------------------------
    st.subheader("Geospatial Distribution and Numeric Feature Relationships")

    plot_col1, plot_col2 = st.columns(2) 
    with plot_col1:
        st.image("Geospatial_scatter.png", 
             caption="Geospatial Scatter — Data Points", 
             width='stretch') 
    with plot_col2:
        st.image("heatmap.png", 
             caption="Correlation Heatmap (Numeric Features Only)", 
             width='stretch')
    # -----------------------------------------------------------------
if selected == "Contact":
    st.title(f"You have selected {selected}")

