import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline




# -----------------------------------------------------------
# LOAD DATA + DATE EXTRACTION
# -----------------------------------------------------------
def load_data_and_split():
    print("Loading dataset...")

    df = pd.read_csv("pollution_labeled_output.csv")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["year"] = df["date"].dt.year.fillna(2000).astype(int)
    df["month"] = df["date"].dt.month.fillna(1).astype(int)
    df["dayofyear"] = df["date"].dt.dayofyear.fillna(1).astype(int)

    df = df.drop_duplicates()
    df = df.fillna(df.median(numeric_only=True))

    global label_encoder
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["Pollution_Source"])

    X = df.drop(columns=["Pollution_Source", "date"])

    return (*train_test_split(X, y, test_size=0.2, random_state=42, stratify=y), df)



# -----------------------------------------------------------
# PREPROCESSING
# -----------------------------------------------------------
def define_preprocessing(df):

    num_cols = [
        'latitude','longitude','PM2.5','PM10','NO2','SO2','CO','O3',
        'temperature','humidity','wind_speed','wind_dir',
        'dist_to_road','dist_to_industry','dist_to_farm',
        'fire_nearby','fire_count','fire_min_dist_km',
        'Confidence_Score',
        'year','month','dayofyear'
    ]

    num_cols = [c for c in num_cols if c in df.columns]

    cat_cols = ['city','Season']

    preprocess = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])

    return preprocess, num_cols, cat_cols



# -----------------------------------------------------------
# MODEL TRAINING
# -----------------------------------------------------------
def train_models(X_train, X_test, y_train, y_test, preprocess):

    print("\nTraining Logistic Regression...")
    log_model = Pipeline([
        ('pre', preprocess),
        ('clf', LogisticRegression(max_iter=2000))
    ])
    log_model.fit(X_train, y_train)
    pred_lr = log_model.predict(X_test)

    print("\nLogistic Regression Results:")
    print(classification_report(y_test, pred_lr))

    sns.heatmap(confusion_matrix(y_test, pred_lr).astype(int),
                annot=True, cmap="Blues", fmt="d")
    plt.title("Confusion Matrix - Logistic Regression")
    plt.show()

    print("\nTraining Random Forest + SMOTE...")
    rf_model = ImbPipeline([
        ('pre', preprocess),
        ('smote', SMOTE(random_state=42)),
        ('rf', RandomForestClassifier(
            n_estimators=250, random_state=42, n_jobs=-1,
            max_depth=15, min_samples_leaf=5, class_weight="balanced"
        ))
    ])
    rf_model.fit(X_train, y_train)
    pred_rf = rf_model.predict(X_test)

    print("\nRandom Forest Results:")
    print(classification_report(y_test, pred_rf))

    sns.heatmap(confusion_matrix(y_test, pred_rf).astype(int),
                annot=True, cmap="Greens", fmt="d")
    plt.title("Confusion Matrix - Random Forest")
    plt.show()

    return rf_model



# -----------------------------------------------------------
# INTERACTIVE PREDICTOR (LIKE YOUR SCREENSHOT)
# -----------------------------------------------------------
def interactive_predictor(model):
    print("\n====================================================")
    print("4. INTERACTIVE POLLUTION SOURCE PREDICTOR")
    print("====================================================\n")

    lat = float(input("Enter Latitude (e.g., 12.9722): "))
    lon = float(input("Enter Longitude (e.g., 77.5936): "))
    city = input("Enter City name (e.g., Bengaluru): ")
    season = input("Enter Season (e.g., Summer, Winter): ")

    # Create input row
    sample = {
        "latitude": lat,
        "longitude": lon,
        "city": city,
        "Season": season,
        "PM2.5": 0, "PM10": 0, "NO2": 0, "SO2": 0, "CO": 0,
        "O3": 0, "temperature": 0, "humidity": 0,
        "wind_speed": 0, "wind_dir": 0,
        "dist_to_road": 0, "dist_to_industry": 0, "dist_to_farm": 0,
        "fire_nearby": 0, "fire_count": 0, "fire_min_dist_km": 0,
        "Confidence_Score": 0,
        "year": 2024, "month": 1, "dayofyear": 1
    }

    df_input = pd.DataFrame([sample])

    pred_class = model.predict(df_input)[0]
    pred_label = label_encoder.inverse_transform([pred_class])[0]

    print("\n----------------------------------------------------")
    print(f"INPUT: Location ({lat}, {lon}), City: {city}, Season: {season}")
    print(f"PREDICTION: The pollution source is most likely: **{pred_label}**")
    print("----------------------------------------------------\n")



# -----------------------------------------------------------
# MAIN
# -----------------------------------------------------------
def main():
    print("\n===== Starting EnviroScan ML Pipeline =====\n")

    X_train, X_test, y_train, y_test, df = load_data_and_split()

    preprocess, _, _ = define_preprocessing(df)

    final_model = train_models(X_train, X_test, y_train, y_test, preprocess)

    interactive_predictor(final_model)

    print("\n===== TRAINING COMPLETE =====\n")




if __name__ == "__main__":
    main()
