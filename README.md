# 🌿 EnviroScan

### AI-Powered Pollution Source Identifier using Geospatial Analytics

EnviroScan is an intelligent **machine learning–based system** designed to identify the **probable source of air pollution** by analyzing pollutant concentrations, weather conditions, and geospatial context.

Unlike traditional air quality monitoring systems that only report pollution levels, EnviroScan focuses on **identifying where pollution originates**—such as vehicular emissions, industrial activity, agricultural practices, or natural causes—and presents the results through an interactive **Streamlit dashboard**.

---

## 📌 Key Features

* Predicts major pollution sources:
  * 🚗 Vehicular
  * 🏭 Industrial
  * 🌾 Agricultural
  * 🔥 Burning
* Machine learning–based source classification
* Weather-aware and geospatial feature engineering
* Interactive and user-friendly Streamlit dashboard
* Model performance analysis and visual insights
---

## 🧠 Machine Learning Models

The following models are implemented and evaluated:

* Random Forest
* Logistic Regression
* Decision Tree
* XGBoost

### 📊 Evaluation Metrics

Each model is evaluated using:

* Confusion Matrix
* Classification Report (Precision, Recall, F1-score)
* Cross-validation F1 Score
* Feature Importance (where applicable)

---

## 📊 Input Data & Features

### 🌫 Pollution Parameters

* PM2.5
* PM10
* NO₂
* SO₂
* CO
* O₃

### 🌦 Weather Data

* Temperature
* Humidity
* Wind Speed
* Wind Direction

### 🌍 Geospatial & Contextual Features

* City and Location ID
* Latitude & Longitude
* Distance to roads, industries, and farmlands
* Traffic Index
* Fire proximity indicators
* Date-based features:
  * Year
  * Month
  * Day of Year
---

## 🖥 Dashboard Modules

* **Home** – Project overview and pollution source categories
* **Predict Source** – User input form with real-time prediction
* **Model Insights** – Confusion matrix, reports, feature importance
* **Data Visualization** – Exploratory data analysis and plots
* **About Project** – Purpose, scope, and future vision of EnviroScan

---

## ▶️ How to Run the Project

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/ai_powered_enviroScan.git
cd ai_powered_enviroScan
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Launch the Dashboard

```bash
cd dashboard
streamlit run app.py
```

---

## 🧰 Tech Stack

* **Programming Language:** Python
* **Libraries & Frameworks:**

  * Pandas, NumPy
  * Scikit-learn
  * XGBoost
  * Matplotlib, Seaborn
  * Streamlit
  * Joblib

---

## 🚀 Future Enhancements

* Satellite-based pollution estimation
* Multi-sensor integration
* Pollution forecasting and trend analysis
* Cloud deployment for public access

---

---

🌱 *EnviroScan aims to support smarter environmental monitoring and informed decision-making through AI-powered insights.*
