🌪️ AI-EnviroScan
Pinpointing the Invisible — Discovering Where Pollution Really Comes From

Most air quality systems stop at “how bad is the air?”
AI-EnviroScan goes one step further — it answers “who’s responsible?”

AI-EnviroScan is an AI-powered geospatial intelligence platform that not only measures air pollution, but tracks its likely source using a fusion of machine learning, wind-flow modeling, and OpenStreetMap-based spatial analytics.

🔥 What Makes This Project Unique?
| Traditional AQ Systems  | AI-EnviroScan                   |
| ----------------------- | ------------------------------- |
| Shows pollution level   | **Identifies pollution origin** |
| Static AQI charts       | **Interactive real-time maps**  |
| No spatial intelligence | **Wind-aware plume tracing**    |
| Data only               | **Actionable insights**         |

🎯 Capabilities

🏭 Classifies Pollution Source
Industrial • Vehicular • Agricultural Burning • Natural Dust • Biomass Fire

🗺️ Live Geospatial Hotspot Detection
Pinpoints high-risk zones with interactive Folium maps

🌬️ Wind-Aware Source Attribution
Tracks pollution plumes using wind vectors

📊 AI-Driven Analytics Dashboard
Trend analysis, alerts, and source distribution

🚨 Real-Time Alerts
Instant notification when thresholds exceed

🧠 How It Works
1️⃣ Multi-Source Data Fusion
| Source             | Purpose                                 |
| ------------------ | --------------------------------------- |
| **OpenAQ**         | Real-time PM2.5, PM10, NO₂, SO₂, CO, O₃ |
| **OpenWeatherMap** | Wind speed & direction                  |
| **OSMnx**          | Factories, highways, farmland proximity |

2️⃣ Smart Feature Engineering
| Signal                              | Interpretation                |
| ----------------------------------- | ----------------------------- |
| High **SO₂ / NO₂**                  | Industrial / Coal combustion  |
| High **NO₂ near roads**             | Vehicular traffic             |
| High **PM + farmland + dry season** | Agricultural burning          |
| Wind-upstream factory               | Industrial plume confirmation |

3️⃣ AI Models

Random Forest

XGBoost

Trained using a semi-supervised hybrid labeling pipeline — heuristic rules + manual expert validation.

OpenAQ + OpenWeather + OSMnx  
        ↓  
Data Cleaning & Feature Engineering  
        ↓  
Random Forest / XGBoost Model  
        ↓  
Pollution Source Prediction  
        ↓  
Geospatial Intelligence Engine  
        ↓  
Streamlit Dashboard + Alerts

📸 Dashboard Preview

🗺️ Pollution Hotspot Map | 📊 Source Distribution | 📈 Trend Analytics


🧰 Tech Stack
| Layer      | Tools                             |
| ---------- | --------------------------------- |
| Language   | Python 3.9+                       |
| ML         | Scikit-Learn, XGBoost             |
| Geospatial | GeoPandas, Folium, OSMnx, Shapely |
| APIs       | OpenAQ, OpenWeatherMap            |
| Interface  | Streamlit                         |
```bash
git clone https://github.com/your-username/AI-EnviroScan.git
cd AI-EnviroScan
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
