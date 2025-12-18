🌍 AI-EnviroScan

AI-Powered Pollution Source Identifier using Geospatial Analytics AI-EnviroScan is an intelligent analytical system that goes beyond measuring air quality; it identifies the probable origin of pollution. By fusing machine learning with real-time weather and geospatial data, the system distinguishes between industrial emissions, traffic congestion, agricultural burning, and natural phenomena.

🚀 Overview
Most air quality monitors tell you how much pollution there is, but not where it's coming from. AI-EnviroScan bridges this gap by correlating pollutant ratios (like the SO2 / NO2 ratio) with proximity to points of interest (factories, highways, farms) extracted via OpenStreetMap.
---

##  Key Features

- Predict pollution sources (Industrial, Vehicular, Agricultural, Burning, Natural)  
- Interactive geospatial heatmaps & hotspot visualization  
- Real-time pollution alerts  
- Dashboard with pollutant trends and source distribution  
- Integration with OpenAQ, OpenWeather, OSMnx  
- Automated data cleaning & feature engineering  
- ML-based pollution source prediction  
- Streamlit-powered real-time dashboard  

---
📸 Dashboard Preview
[Insert Screenshot of Map View] | [Insert Screenshot of Analytics Charts]

---
🛠️ Technical Workflow
1. Data Acquisition & FusionThe system pulls data from three primary sources:OpenAQ: Real-time PM2.5, PM10, $NO_2$, $SO_2$, $CO$, and $O_3$ levels.OpenWeather API: Wind speed and direction (crucial for plume dispersion modeling).OSMnx (OpenStreetMap): Spatial features such as distance to the nearest industrial zone or primary highway.

2. Feature Engineering & LogicThe model doesn't just look at raw numbers. 
It calculates:Pollutant Ratios: High $SO_2/NO_2$ often indicates coal-burning (Industrial), while high $NO_2$ alone suggests combustion engines (Vehicular).Wind Vectors: Correlates wind direction with upwind land-use types to validate sources.
3. Machine Learning StackModels: Random Forest & XGBoost (chosen for their ability to handle non-linear spatial relationships).Labeling: A semi-supervised approach using heuristic rules for initial labeling followed by manual validation.

---
##  Project Workflow

Data Collection → Preprocessing → Source Labeling → Model Training  
Geospatial Mapping ← Predictions ← Real-Time Dashboard & Alerts  

---

💻 Tech Stack:

Language: Python 3.9+
Data Science: Pandas, NumPy, Scikit-learn, XGBoost
Geospatial: GeoPandas, Folium, OSMnx, Shapely
Interface: Streamlit
APIs: OpenAQ, OpenWeatherMap

---
##  System Architecture

APIs (OpenAQ, OpenWeather, OSMnx)  
→ Data Cleaning & Feature Engineering  
→ ML Model (RF/XGBoost)  
→ Predictions  
→ Geospatial Mapping  
→ Dashboard & Alerts (Streamlit)

---

##  Modules Overview

### 1. Data Collection  
- Collect pollutant data  
- Collect weather data  
- Extract geospatial features  
- Store in CSV/JSON  

### 2. Data Cleaning  
- Remove duplicates  
- Handle missing values  
- Normalize features  
- Build unified DataFrame  

### 3. Source Labeling  
Rule examples:  
- Road + NO₂ → Vehicular  
- Factory + SO₂ → Industrial  
- Farmland + dry season + PM → Agricultural  

### 4. Model Training  
- Train Random Forest, XGBoost  
- Evaluate using accuracy, precision, recall, F1-score  

### 5. Geospatial Mapping  
- Create heatmaps  
- Create marker maps  
- Highlight high-risk zones  

### 6. Real-Time Dashboard  
- Show predictions  
- Alert when thresholds exceed  
- Trend charts  
- Source distribution pie charts  

### 7. Documentation  
- Architecture diagrams  
- Model metrics  
- Final report & presentation  

---

⚙️ Installation & Setup
1. Clone & Environment
```
git clone https://github.com/your-username/AI-EnviroScan.git
cd AI-EnviroScan
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```
2. API Configuration
Create a .env file in the root directory and add your keys:

#Code snippet
```
OPENWEATHER_API_KEY=your_key_here
OPENAQ_API_KEY=your_key_here
```
3. Run the Application
# To train the model with latest data
```
python src/train_model.py
```
# To launch the dashboard
```
streamlit run dashboard.py
```

---

## Future Enhancements

- Satellite-based pollution estimation  
- Predictive pollution forecasting  
- Multi-sensor integration  
- Cloud deployment  

---


