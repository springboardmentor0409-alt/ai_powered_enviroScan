import streamlit as st
from pathlib import Path
import base64

# ---------------------------------------------------
# 🌍 App Configuration
# ---------------------------------------------------
st.set_page_config(
    page_title="EnviroScan: AI-Powered Pollution Source Identifier",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom page style
st.markdown(
    """
    <style>
    body {
        background-color: #f9fafc;
        color: #1c1c1c;
    }
    .stSidebar {
        background-color: #e8f5e9 !important;
    }
    h1, h2, h3 {
        color: #2e7d32;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------
# 🌐 Sidebar Navigation
# ---------------------------------------------------
st.sidebar.title("🌿 Dashboard")
menu = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "📊 Analytics", "🗺️ Heatmaps", "🧠 Model Insights", "ℹ️ About Project"]
)

# ---------------------------------------------------
# 🏠 HOME PAGE
# ---------------------------------------------------
if menu == "🏠 Home":
    st.markdown("<br>", unsafe_allow_html=True)

    # ✅ Load image from assets folder
    image_path = Path("assets/enviro_dashboard.png")
    if image_path.exists():
        with open(image_path, "rb") as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode()

        # Title and Image side by side
        st.markdown(
            f"""
            <div style='display: flex; align-items: center; justify-content: space-between;'>
                <div style='flex: 1;'>
                    <h1 style='color:#2e7d32;'>🌍 EnviroScan</h1>
                    <h3 style='color:#4e4e4e;'>AI-Powered Pollution Source Identifier using Geospatial Analytics</h3>
                </div>
                <div style='flex: 0 0 220px; text-align: right;'>
                    <img src='data:image/png;base64,{img_base64}' 
                         style='width:180px; border-radius:12px; box-shadow:0 4px 8px rgba(0,0,0,0.15);'>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.error("❌ Image not found: Please place it in `assets/enviro_dashboard.png`")

    # Description
    st.markdown(
        """
        Welcome to **EnviroScan**, an AI-powered system designed to **detect, visualize, and analyze pollution sources** in real time.  
        This dashboard allows you to monitor environmental conditions, visualize hotspots, and view predictive insights powered by AI.
        """
    )

    # Features
    st.markdown("### 🚀 Key Features (Coming Soon)")
    st.markdown(
        """
        - 🌫️ Real-time pollution data tracking  
        - 🗺️ Dynamic heatmap visualization  
        - 🧠 AI-powered pollution source classification  
        - ☁️ Integration with weather datasets  
        - 📈 Interactive analytics and automated alerts  
        """
    )

    st.success("✅ Dashboard Interface Loaded Successfully!")

# ---------------------------------------------------
# 📊 ANALYTICS PAGE
# ---------------------------------------------------
elif menu == "📊 Analytics":
    st.title("📊 Pollution Analytics Overview")
    st.markdown(
        """
        This section will display **data visualizations and metrics** once pollution and weather datasets are connected.  
        Below are some placeholder stats for demonstration.
        """
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Average AQI (Placeholder)", "178", "+5 from last week")
    col2.metric("Most Polluted City", "Delhi", "PM2.5 ↑")
    col3.metric("Safest City", "Kochi", "PM2.5 ↓")

    st.info("📊 Charts & comparisons will appear here once data integration is complete.")

# ---------------------------------------------------
# 🗺️ HEATMAPS PAGE
# ---------------------------------------------------
# ---------------------------------------------------
# 🗺️ HEATMAPS PAGE
# ---------------------------------------------------
elif menu == "🗺️ Heatmaps":
    st.title("🗺️ Pollution Heatmaps & EDA Visualizations")

    st.markdown("### Below are the EDA visualizations generated from the dataset.")

    import os
    from pathlib import Path
    from PIL import Image

    # Suppress Streamlit logs in terminal
    import logging
    logging.getLogger("streamlit").setLevel(logging.CRITICAL)

    eda_folder = Path("assets/eda/")

    if not eda_folder.exists():
        st.error("❌ Folder not found: assets/eda/. Please add your EDA images there.")
    else:
        image_files = sorted([
            f for f in os.listdir(eda_folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

        if not image_files:
            st.warning("⚠️ No images found in assets/eda/")
        else:

            # 🔥 Display 2 images per row
            for i in range(0, len(image_files), 2):

                col1, col2 = st.columns(2)

                # --- First Image ---
                with col1:
                    img_name = image_files[i]
                    clean_title = img_name.split(".")[0].replace("_", " ").title()
                    st.subheader(f"📌 {clean_title}")
                    st.image(str(eda_folder / img_name), width="stretch")

                # --- Second Image ---
                if i + 1 < len(image_files):
                    with col2:
                        img_name_2 = image_files[i + 1]
                        clean_title_2 = img_name_2.split(".")[0].replace("_", " ").title()
                        st.subheader(f"📌 {clean_title_2}")
                        st.image(str(eda_folder / img_name_2), width="stretch")

                st.markdown("---")

# ---------------------------------------------------
# 🧠 MODEL INSIGHTS PAGE
# ---------------------------------------------------
elif menu == "🧠 Model Insights":
    st.title("🧠 AI Model Insights (Coming Soon)")
    st.markdown(
        """
        This section will display **machine learning predictions** such as pollution source detection, 
        feature importance, and model performance summaries.
        """
    )
    st.info("💡 You can integrate your trained ML model later here for predictions.")

# ---------------------------------------------------
# ℹ️ ABOUT PAGE
# ---------------------------------------------------
elif menu == "ℹ️ About Project":
    st.title("ℹ️ About EnviroScan Project")
    st.markdown(
        """
        ### 🌟 Project Vision
        To develop an intelligent, AI-based system that identifies and visualizes pollution sources in real time.

        ### 🧩 Technologies Used
        -  Python  
        -  Streamlit (Frontend UI)  
        -  Pandas, NumPy (Data Processing)  
        -  Scikit-learn (Machine Learning)  
        -  Folium / Plotly (Visualization)

        ### 🎯 Future Goals
        - Integrate live pollution & weather APIs  
        - Build automated pollution alerts  
        - Deploy dashboard on the web for public access  
        - Enable city-level comparison and historical trend analysis  
        """
    )

    st.success("✅ Dashboard base setup complete — more modules coming soon!")
