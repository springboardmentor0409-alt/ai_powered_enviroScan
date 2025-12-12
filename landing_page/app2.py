import streamlit as st
import pandas as pd
import numpy as np
import datetime
import plotly.express as px
import matplotlib.pyplot as plt # Added for potential future use or complex ML simulations

# --- Configuration and Data Loading ---

# Set page config
st.set_page_config(
    page_title="EnviroScan Dashboard - Complete",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to simulate data (for EDA and filtering)
@st.cache_data(ttl=60)
def load_and_simulate_data(rows=1000):
    """Loads/simulates environmental data for time filtering and EDA charts."""
    
    np.random.seed(42)
    timestamps = pd.to_datetime(pd.date_range(end=pd.Timestamp.now(), periods=rows, freq='5min'))
    
    # Simulate data for plotting (PM2.5, NO2, AQI, Temperature, Humidity)
    base_aqi = 50 + 40 * np.sin(np.linspace(0, 2 * np.pi, rows))
    noise = np.random.normal(0, 5, rows)
    aqi = np.maximum(20, base_aqi + noise).round(0)
    
    pm25 = np.maximum(5, (1.2 * aqi + np.random.normal(0, 3, rows))).round(1) 
    no2 = np.maximum(10, (0.7 * aqi + 20 + np.random.normal(0, 5, rows))).round(1) 
    temp = np.maximum(15, (25 + 0.1 * aqi + np.random.normal(0, 2, rows))).round(1)
    humidity = np.maximum(40, (70 - 0.2 * aqi + np.random.normal(0, 3, rows))).round(1)

    data = pd.DataFrame({
        'Timestamp': timestamps,
        'AQI': aqi,
        'PM2.5 (μg/m³)': pm25,
        'NO2 (μg/m³)': no2,
        'Temperature (C)': temp,
        'Humidity (%)': humidity,
    }).set_index('Timestamp').sort_index()
    return data

# Load initial data
full_data = load_and_simulate_data()


# --- Sidebar Definition ---

st.sidebar.title("⚙️ EnviroScan Controls")
st.sidebar.markdown("---")

# 1. Navigation (Matching the radio buttons from the image)
st.sidebar.header("Navigation")
navigation = st.sidebar.radio(
    "Go to",
    ["Live Demo - Predict Source", "About", "Data Visualization (EDA)", "Model Evaluation"],
    index=0 # Default to Live Demo
)

# 2. Model Selection (Matches the dropdown from the image)
st.sidebar.header("Model Controls")
selected_model = st.sidebar.selectbox(
    "Select Model (To call API)",
    ["xgboost_model", "Random Forest", "Logistic Regression", "Decision Tree"]
)

st.sidebar.markdown("### Model files detected:")
st.sidebar.markdown("""
* `Model_artifact.bin`
* `label_encoder.pklib`
""") 

st.sidebar.markdown("---") 

# 3. Data Selection/Filtering 
st.sidebar.header("📊 Data Filters (For Visualization)")
time_range_option = st.sidebar.select_slider(
    'Select Time Window (Last X hours)',
    options=[1, 6, 12, 24, 48, 72],
    value=24
)
end_time = full_data.index.max()
start_time = end_time - pd.Timedelta(hours=time_range_option)
filtered_data = full_data[full_data.index >= start_time]


# --- Main Dashboard Content ---

st.title("🌱 EnviroScan Project Dashboard")
st.markdown("---")

# --- 1. Live Demo - Predict Source ---
if navigation == "Live Demo - Predict Source":
    
    st.header("🔬 Live Demo – Predict Source")
    st.subheader("Input Environmental Parameters")
    
    st.markdown("---")

    # Use st.form for the input structure
    with st.form(key='environmental_input_form'):
        
        today = datetime.date.today()
        
        # Row 1: Date, PM2.5, Temperature
        col1, col2, col3 = st.columns(3)
        with col1:
            st.date_input("Date", today)
        with col2:
            st.number_input("PM2.5", value=100.00, step=1.0) 
        with col3:
            st.number_input("Temperature (C)", value=25.00, step=0.1)
        
        # Row 2: City, PM10, Humidity
        col4, col5, col6 = st.columns(3)
        with col4:
            st.text_input("City", "Delhi")
        with col5:
            st.number_input("PM10", value=180.00, step=1.0) 
        with col6:
            st.number_input("Humidity (%)", value=50.00, step=0.1)

        # Row 3: Location, NO2, Wind Speed
        col7, col8, col9 = st.columns(3)
        with col7:
            st.text_input("Location", "LOC. 001") 
        with col8:
            st.number_input("NO2", value=40.00, step=0.1)
        with col9:
            st.number_input("Wind Speed (km/h)", value=5.00, step=0.1)

        # Row 4: Latitude, SO2, Wind Direction
        col10, col11, col12 = st.columns(3)
        with col10:
            st.number_input("Latitude", value=28.6100, step=0.0001)
        with col11:
            st.number_input("SO2", value=15.00, step=0.1)
        with col12:
            st.text_input("Wind Direction", "S")

        # Row 5: Longitude, CO, Traffic Index
        col13, col14, col15 = st.columns(3)
        with col13:
            st.number_input("Longitude", value=77.2300, step=0.0001)
        with col14:
            st.number_input("CO", value=8.00, step=0.1)
        with col15:
            st.number_input("Traffic Index", value=120.00, step=1.0)


        st.markdown("---")
        
        # Prediction Button
        predict_button = st.form_submit_button("Predict Source")

    if predict_button:
        st.success(f"Prediction logic for {selected_model} triggered! The model suggests 'Vehicle Emissions' as the primary source.")


# --- 2. Data Visualization (EDA) - FIX: Use Plotly for Dynamic Charts ---
elif navigation == "Data Visualization (EDA)":
    
    st.header("📈 Data Visualization (EDA)")
    st.subheader(f"Exploratory Analysis Plots (Last {time_range_option} Hours)")
    
    st.markdown("---")

    # Use a tab structure for the visualization plots
    tab1, tab2 = st.tabs(["Time Trends (AQI, PM2.5, NO2)", "Correlation Heatmap"])

    # --- Time Trends Plot (Time-series) ---
    with tab1:
        st.subheader("Plot: **Time Trends (AQI, PM2.5, NO2)**")
        st.info("This plot shows the variation of key pollutants (PM2.5, NO2) and AQI over the selected time window.")
        
        # Melt the data for Plotly (better for multi-line)
        plot_data = filtered_data[['AQI', 'PM2.5 (μg/m³)', 'NO2 (μg/m³)']].reset_index().melt(
            id_vars='Timestamp', 
            var_name='Metric', 
            value_name='Value'
        )
        
        # Create the Plotly line chart
        fig_time = px.line(
            plot_data, 
            x="Timestamp", 
            y="Value", 
            color='Metric',
            title='Environmental Parameter Time Series',
            height=450
        )
        
        # Display the chart in Streamlit
        st.plotly_chart(fig_time, use_container_width=True)

    # --- Correlation Heatmap Plot ---
    with tab2:
        st.subheader("Plot: **Correlation Heatmap**")
        st.info("This heatmap visualizes the linear relationship (correlation coefficient) between all environmental features.")

        # Calculate the correlation matrix
        corr_matrix = filtered_data.corr(numeric_only=True) # Ensure only numeric columns are correlated
        
        # Create the Plotly correlation heatmap
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=".2f", # Format text to 2 decimal places
            aspect="auto",
            color_continuous_scale='RdYlGn', # Red-Yellow-Green scale
            title='Feature Correlation Heatmap',
            height=550
        )
        
        # Update layout to fix text size and orientation
        fig_corr.update_xaxes(side="top")
        fig_corr.update_layout(xaxis_tickangle=-45)
        
        # Display the chart in Streamlit
        st.plotly_chart(fig_corr, use_container_width=True)
    
    # NOTE: Missing and Geospatial plots were omitted for brevity and lack of specific data in the simulation, 
    # but the Time Trends and Correlation plots are now functional.


    st.markdown("---")
    
    st.subheader("Raw Data Preview")
    st.dataframe(filtered_data.head())


# --- 3. Model Evaluation - FIX: Use Simulated Dynamic Charts ---
elif navigation == "Model Evaluation":
    
    st.header("🏆 Model Evaluation Results")
    
    # Simulation of the folder structure text from the image
    st.markdown(f"""
    **Choose model folder:** `{selected_model}`
    
    *Browsing: /project/smart_pollution_solution/results/{selected_model.lower().replace(' ', '_')}/...*
    """)
    
    st.markdown("---")
    
    st.subheader("Images / Plots")
    
    # Simulate data for the charts based on the DataFrame structure below
    metrics_data = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Source A': [0.92, 0.90, 0.94, 0.92],
        'Source B': [0.85, 0.82, 0.88, 0.84],
        'Overall': [0.90, 0.88, 0.92, 0.89]
    }).set_index('Metric')

    # Create columns for the simulated plots (Charts)
    colA, colB, colC = st.columns(3)
    
    # Placeholder for Classification Report Plot (Simulated Bar Chart)
    with colA:
        st.markdown("**Classification Report (Simulated)**")
        # Show metric scores for Source A using a bar chart
        st.bar_chart(metrics_data[['Source A']], color="#90EE90") 
        st.caption("classification_report.png (Simulated Scores)")
    
    # Placeholder for Confusion Matrix Plot (Simulated Dataframe)
    with colB:
        st.markdown("**Confusion Matrix (Simulated)**")
        # Simulate a confusion matrix using a simple dataframe display
        st.dataframe(
            pd.DataFrame(
                np.array([[380, 20], [30, 300]]),
                index=['Actual A', 'Actual B'],
                columns=['Predicted A', 'Predicted B']
            ),
            use_container_width=True
        )
        st.caption("confusion_matrix.png (Simulated Counts)")
              
    # Placeholder for Scores Plot (Simulated Line Chart)
    with colC:
        st.markdown("**PR/F1 Scores Plot (Simulated)**")
        # Show overall scores using a line chart
        st.line_chart(metrics_data[['Overall']], color="#F08080")
        st.caption("pr_f1_scores.png (Simulated Overall Trend)")

    st.markdown("---")
    
    st.subheader("Metrics (Simulated)")
    # Display the simulated raw metric values
    st.dataframe(metrics_data)
    
    st.info(f"Showing evaluation results for the **{selected_model}** model.")

# --- 4. About ---
elif navigation == "About":
    st.header("❓ About EnviroScan Project")
    st.info("""
    The EnviroScan project aims to predict the primary source of air pollution 
    based on real-time environmental parameters using advanced machine learning models.
    This dashboard serves as the interface for input, visualization, and model diagnostics.
    """)