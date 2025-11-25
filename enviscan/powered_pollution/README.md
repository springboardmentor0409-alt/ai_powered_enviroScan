# AI-EnviroScan  
### **AI-Powered Pollution Source Identifier using Geospatial Analytics**

AI-EnviroScan is an intelligent AI system that identifies the **probable source of air pollution** using machine learning, weather data, pollutant readings, and geospatial analytics.  
Instead of only measuring pollutant levels, the system predicts whether pollution is caused by **industrial activity, vehicles, agricultural burning, dumping sites, or natural factors**, and visualizes hotspots on interactive maps.

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

##  Project Workflow

Data Collection → Preprocessing → Source Labeling → Model Training  
Geospatial Mapping ← Predictions ← Real-Time Dashboard & Alerts  

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

##  How to Run

### Clone Repository
```
git clone https://github.com/your-username/AI-EnviroScan.git
cd AI-EnviroScan
```

### Install Dependencies
```
pip install -r requirements.txt
```

### Run Pipeline
```
python data_pipeline.py
python train_model.py
```

### Launch Dashboard
```
streamlit run dashboard.py
```

---

## Tech Stack

- Python  
- Pandas, NumPy  
- Scikit-learn, XGBoost  
- GeoPandas, Folium, OSMnx  
- Streamlit  
- OpenAQ & OpenWeather APIs  

---

## Future Enhancements

- Satellite-based pollution estimation  
- Predictive pollution forecasting  
- Multi-sensor integration  
- Cloud deployment  

---


