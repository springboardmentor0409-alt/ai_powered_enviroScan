Here’s your content **cleaned, polished, and structured professionally**, without changing **any meaning or scope**.
Think of this as turning raw code into a well-formatted dashboard ✨

---

# 🌍 AI-EnviroScan

### **AI-Powered Pollution Source Identifier using Geospatial Analytics**

**AI-EnviroScan** is an intelligent AI system that identifies the **probable source of air pollution** using machine learning, weather data, pollutant readings, and geospatial analytics.

Instead of only measuring pollutant levels, the system predicts whether pollution is caused by **industrial activity, vehicles, agricultural burning, dumping sites, or natural factors**, and visualizes pollution hotspots on interactive maps.

---

## 🚀 Key Features

- Predict pollution sources
  _(Industrial, Vehicular, Agricultural, Burning, Natural)_
- Interactive geospatial heatmaps and hotspot visualization
- Real-time pollution alerts
- Dashboard with pollutant trends and source distribution
- Integration with **OpenAQ**, **OpenWeather**, **OSMnx**
- Automated data cleaning and feature engineering
- Machine learning–based pollution source prediction
- **Streamlit-powered** real-time dashboard

---

## 🔄 Project Workflow

```
Data Collection → Preprocessing → Source Labeling → Model Training
                                   ↓
                         Geospatial Mapping
                                   ↓
                   Real-Time Dashboard & Alerts
```

---

## 🏗️ System Architecture

```
APIs (OpenAQ, OpenWeather, OSMnx)
        ↓
Data Cleaning & Feature Engineering
        ↓
ML Model (Random Forest / XGBoost)
        ↓
Predictions
        ↓
Geospatial Mapping
        ↓
Dashboard & Alerts (Streamlit)
```

---

## 📦 Modules Overview

### 1️⃣ Data Collection

- Collect pollutant data
- Collect weather data
- Extract geospatial features
- Store data in CSV / JSON format

---

### 2️⃣ Data Cleaning

- Remove duplicates
- Handle missing values
- Normalize features
- Build a unified DataFrame

---

### 3️⃣ Source Labeling

Rule-based examples:

- **Road proximity + NO₂** → Vehicular
- **Factory proximity + SO₂** → Industrial
- **Farmland + dry season + PM** → Agricultural

---

### 4️⃣ Model Training

- Train **Random Forest** and **XGBoost** models
- Evaluate using:

  - Accuracy
  - Precision
  - Recall
  - F1-score

---

### 5️⃣ Geospatial Mapping

- Create pollution heatmaps
- Generate marker-based maps
- Highlight high-risk pollution zones

---

### 6️⃣ Real-Time Dashboard

- Display predicted pollution sources
- Trigger alerts when thresholds exceed limits
- Show pollutant trend charts
- Visualize source distribution using pie charts

---

### 7️⃣ Documentation

- System architecture diagrams
- Model evaluation metrics
- Final report and presentation

---

## ▶️ How to Run

### Clone the Repository

```bash
git clone https://github.com/your-username/AI-EnviroScan.git
cd AI-EnviroScan
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Launch the Dashboard

```bash
streamlit run app.py
```

---

## 🧰 Tech Stack

- **Python**
- **Pandas**, **NumPy**
- **Scikit-learn**, **XGBoost**
- **GeoPandas**, **Folium**, **OSMnx**
- **Streamlit**

---

## 🔮 Future Enhancements

- Satellite-based pollution estimation
- Predictive pollution forecasting
- Multi-sensor data integration
- Cloud deployment and scalability
