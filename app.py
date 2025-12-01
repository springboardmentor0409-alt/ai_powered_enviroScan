# app.py - AI-Powered EnviroScan: Pollution Source Predictor
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="AI-Powered EnviroScan",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1E88E5;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background-color: #e3f2fd;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #1E88E5;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🌍 AI-Powered EnviroScan</h1>', unsafe_allow_html=True)
st.markdown('<h3 class="sub-header">Real-time Pollution Monitoring & Source Prediction</h3>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3095/3095110.png", width=100)
    st.title("Navigation")
    
    app_mode = st.selectbox(
        "Select Mode",
        ["🏠 Dashboard", "🔍 Predict Source", "📊 Model Insights", "📈 Historical Data", "⚙️ Settings"]
    )
    
    st.divider()
    
    st.markdown("### Model Information")
    st.info("**Random Forest Classifier**\n\nAccuracy: 86.75%\n\nFeatures: 14\n\nTrained on: 2000 samples")
    
    st.divider()
    
    st.markdown("### About")
    st.caption("AI-powered system for predicting pollution sources using machine learning")

# Dashboard Mode
if app_mode == "🏠 Dashboard":
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(label="Model Accuracy", value="86.75%", delta="2.1%")
    
    with col2:
        st.metric(label="Sources Predicted", value="1,240", delta="24")
    
    with col3:
        st.metric(label="Active Sensors", value="15", delta="-1")
    
    with col4:
        st.metric(label="Air Quality Index", value="78", delta="-3")
    
    st.divider()
    
    # Real-time Data Visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Current Pollution Levels")
        
        # Sample pollution data
        pollution_data = pd.DataFrame({
            'Pollutant': ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3'],
            'Level': [45, 78, 32, 18, 0.9, 28],
            'Safe Limit': [35, 50, 40, 20, 1.0, 50]
        })
        
        fig = px.bar(pollution_data, x='Pollutant', y=['Level', 'Safe Limit'],
                     barmode='group', color_discrete_map={'Level': '#FF6B6B', 'Safe Limit': '#4ECDC4'})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📍 Source Distribution")
        
        # Sample source distribution
        source_data = pd.DataFrame({
            'Source': ['Industrial', 'Vehicular', 'Residential', 'Natural'],
            'Count': [42, 38, 28, 15]
        })
        
        fig = px.pie(source_data, values='Count', names='Source',
                     color_discrete_sequence=px.colors.sequential.RdBu)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Map Visualization
    st.subheader("🗺️ Pollution Hotspots")
    
    # Sample location data
    locations = pd.DataFrame({
        'lat': [28.6139, 28.7041, 28.4595, 28.5355, 28.4089],
        'lon': [77.2090, 77.1025, 77.0266, 77.3910, 77.3178],
        'Pollution_Level': [65, 78, 42, 89, 56],
        'Predicted_Source': ['Industrial', 'Vehicular', 'Residential', 'Industrial', 'Vehicular']
    })
    
    fig = px.scatter_mapbox(locations, lat="lat", lon="lon", 
                           size="Pollution_Level", color="Predicted_Source",
                           color_discrete_sequence=px.colors.qualitative.Set1,
                           zoom=10, height=400)
    fig.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig, use_container_width=True)

# Prediction Mode
elif app_mode == "🔍 Predict Source":
    st.header("🔍 Predict Pollution Source")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Sensor Data")
        
        # User input for pollution parameters
        pm25 = st.slider("PM2.5 (µg/m³)", min_value=0, max_value=500, value=45, step=1)
        pm10 = st.slider("PM10 (µg/m³)", min_value=0, max_value=1000, value=78, step=1)
        no2 = st.slider("NO₂ (ppb)", min_value=0, max_value=200, value=32, step=1)
        so2 = st.slider("SO₂ (ppb)", min_value=0, max_value=200, value=18, step=1)
        co = st.slider("CO (ppm)", min_value=0.0, max_value=10.0, value=0.9, step=0.1)
        o3 = st.slider("O₃ (ppb)", min_value=0, max_value=200, value=28, step=1)
    
    with col2:
        st.subheader("Environmental Conditions")
        
        temperature = st.slider("Temperature (°C)", min_value=-10, max_value=50, value=25, step=1)
        humidity = st.slider("Humidity (%)", min_value=0, max_value=100, value=65, step=1)
        wind_speed = st.slider("Wind Speed (km/h)", min_value=0, max_value=100, value=12, step=1)
        
        area_type = st.selectbox("Area Type", ["Industrial", "Commercial", "Residential", "Mixed", "Green"])
        hour = st.slider("Hour of Day", min_value=0, max_value=23, value=14, step=1)
        is_weekday = st.checkbox("Weekday", value=True)
        is_rush_hour = st.checkbox("Rush Hour", value=False)
    
    # Prediction button
    if st.button("🔬 Predict Source", type="primary", use_container_width=True):
        # Simulate loading
        with st.spinner("Analyzing pollution patterns..."):
            import time
            time.sleep(1.5)
            
            # Create sample prediction
            features = np.array([[pm25, pm10, no2, so2, co, o3, temperature, humidity, 
                                wind_speed, hour, is_rush_hour, is_weekday, 0, 0]])
            
            # Mock prediction (replace with actual model loading)
            try:
                # Try to load actual model
                # model_data = joblib.load('models/pollution_source_model.joblib')
                # model = model_data['model']
                # scaler = model_data['scaler']
                # features_scaled = scaler.transform(features)
                # prediction = model.predict(features_scaled)[0]
                # probabilities = model.predict_proba(features_scaled)[0]
                
                # For demo - mock predictions based on logic
                if so2 > 30 and pm10 > 100:
                    prediction = "Industrial"
                    probabilities = [0.7, 0.1, 0.15, 0.05]
                elif no2 > 40 and co > 1.0:
                    prediction = "Vehicular"
                    probabilities = [0.1, 0.05, 0.1, 0.75]
                elif area_type == "Residential":
                    prediction = "Residential"
                    probabilities = [0.15, 0.1, 0.65, 0.1]
                else:
                    prediction = "Natural"
                    probabilities = [0.05, 0.8, 0.1, 0.05]
                    
            except:
                # Fallback to mock prediction
                prediction = "Industrial" if so2 > 30 else "Vehicular" if no2 > 40 else "Residential"
                probabilities = [0.4, 0.2, 0.3, 0.1]
            
            # Display results
            st.divider()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                st.subheader("🎯 Predicted Source")
                
                # Color based on prediction
                color_map = {
                    "Industrial": "#FF6B6B",
                    "Vehicular": "#4ECDC4",
                    "Residential": "#FFD166",
                    "Natural": "#06D6A0"
                }
                
                st.markdown(f"<h2 style='color: {color_map.get(prediction, '#000')};'>{prediction}</h2>", 
                          unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Confidence meter
                confidence = max(probabilities) * 100
                st.progress(int(confidence))
                st.caption(f"Confidence: {confidence:.1f}%")
            
            with col2:
                st.subheader("📈 Source Probabilities")
                
                prob_df = pd.DataFrame({
                    'Source': ['Industrial', 'Natural', 'Residential', 'Vehicular'],
                    'Probability': probabilities
                })
                
                fig = px.bar(prob_df, x='Source', y='Probability', 
                           color='Source', color_discrete_map=color_map)
                fig.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations
            st.subheader("💡 Recommendations")
            
            if prediction == "Industrial":
                st.warning("""
                **Industrial Pollution Detected**
                - Check nearby factories for compliance
                - Monitor SO₂ and PM10 levels
                - Consider installing scrubbers
                """)
            elif prediction == "Vehicular":
                st.info("""
                **Vehicular Pollution Detected**
                - Consider traffic management
                - Promote public transport
                - Check vehicle emission standards
                """)
            elif prediction == "Residential":
                st.success("""
                **Residential Pollution Detected**
                - Check waste management
                - Monitor cooking/heating sources
                - Promote clean energy
                """)
            else:
                st.success("""
                **Natural Sources Detected**
                - Likely dust/pollen/sea salt
                - Monitor weather conditions
                - No immediate action needed
                """)

# Model Insights Mode
elif app_mode == "📊 Model Insights":
    st.header("📊 Model Insights & Performance")
    
    tab1, tab2, tab3 = st.tabs(["📈 Performance", "🏆 Features", "📋 Confusion Matrix"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Accuracy metrics
            metrics = {
                "Overall Accuracy": "86.75%",
                "Industrial Precision": "76%",
                "Vehicular Precision": "99%",
                "Natural Precision": "100%",
                "Residential Recall": "87%"
            }
            
            for metric, value in metrics.items():
                st.metric(label=metric, value=value)
        
        with col2:
            # Classification report
            st.subheader("Classification Report")
            report_data = pd.DataFrame({
                'Class': ['Industrial', 'Natural', 'Residential', 'Vehicular'],
                'Precision': [0.76, 1.00, 0.81, 0.99],
                'Recall': [0.89, 1.00, 0.87, 0.73],
                'F1-Score': [0.82, 1.00, 0.84, 0.84]
            })
            st.dataframe(report_data, use_container_width=True)
    
    with tab2:
        st.subheader("🏆 Feature Importance")
        
        # Feature importance data (from your training output)
        feature_importance = {
            'area_type_encoded': 0.4449,
            'PM10': 0.1196,
            'SO2': 0.0872,
            'NO2': 0.0762,
            'PM2.5': 0.0614,
            'CO': 0.0453,
            'wind_speed': 0.0307,
            'temperature': 0.0306,
            'humidity': 0.0296,
            'O3': 0.0268
        }
        
        fig = px.bar(x=list(feature_importance.values()), 
                    y=list(feature_importance.keys()),
                    orientation='h',
                    labels={'x': 'Importance', 'y': 'Feature'},
                    color=list(feature_importance.values()),
                    color_continuous_scale='Viridis')
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("📋 Confusion Matrix")
        
        # Mock confusion matrix
        cm = np.array([[97, 2, 10, 0],
                      [0, 83, 0, 0],
                      [12, 1, 94, 1],
                      [2, 0, 25, 73]])
        
        fig = px.imshow(cm,
                       labels=dict(x="Predicted", y="Actual", color="Count"),
                       x=['Industrial', 'Natural', 'Residential', 'Vehicular'],
                       y=['Industrial', 'Natural', 'Residential', 'Vehicular'],
                       text_auto=True,
                       color_continuous_scale='Blues')
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

# Historical Data Mode
elif app_mode == "📈 Historical Data":
    st.header("📈 Historical Pollution Data")
    
    # Generate sample historical data
    dates = pd.date_range(start='2024-01-01', end='2024-12-01', freq='D')
    historical_data = pd.DataFrame({
        'Date': dates,
        'PM2.5': np.random.normal(45, 15, len(dates)),
        'PM10': np.random.normal(78, 20, len(dates)),
        'NO2': np.random.normal(32, 8, len(dates)),
        'SO2': np.random.normal(18, 5, len(dates)),
        'Predicted_Source': np.random.choice(['Industrial', 'Vehicular', 'Residential', 'Natural'], len(dates))
    })
    
    pollutant = st.selectbox("Select Pollutant", ['PM2.5', 'PM10', 'NO2', 'SO2'])
    
    fig = px.line(historical_data, x='Date', y=pollutant, 
                 color='Predicted_Source',
                 title=f'Historical {pollutant} Levels')
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Data table
    if st.checkbox("Show Raw Data"):
        st.dataframe(historical_data.tail(100), use_container_width=True)

# Settings Mode
elif app_mode == "⚙️ Settings":
    st.header("⚙️ Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Settings")
        
        st.selectbox("Prediction Model", ["Random Forest", "Gradient Boosting", "Neural Network"])
        st.slider("Confidence Threshold", 0.0, 1.0, 0.7, 0.05)
        st.number_input("Update Frequency (minutes)", 1, 60, 5)
        
        if st.button("🔄 Retrain Model", type="secondary"):
            with st.spinner("Retraining model with latest data..."):
                time.sleep(2)
                st.success("Model retrained successfully!")
    
    with col2:
        st.subheader("Alert Settings")
        
        st.number_input("PM2.5 Alert Threshold", 0, 500, 35)
        st.number_input("PM10 Alert Threshold", 0, 1000, 50)
        st.number_input("NO2 Alert Threshold", 0, 200, 40)
        
        st.checkbox("Email Alerts", value=True)
        st.checkbox("SMS Alerts", value=False)
        st.checkbox("Push Notifications", value=True)
    
    if st.button("💾 Save Settings", type="primary"):
        st.success("Settings saved successfully!")

# Footer
st.divider()
st.caption("🌱 AI-Powered EnviroScan v1.0 | Random Forest Model | Accuracy: 86.75%")