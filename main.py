import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
import joblib
import os

warnings.filterwarnings('ignore')
def load_and_clean_data(filepath):
    print(f"Loading data from {filepath}...")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f" ERROR: File not found at {filepath}")

    # Detect CSV or Excel
    if filepath.endswith(".xlsx") or filepath.endswith(".xls"):
        df = pd.read_excel(filepath)
    else:
        df = pd.read_csv(filepath)

    print("✔ File loaded successfully")

    df = df.drop_duplicates()

    cols_with_zeros_as_missing = [
        'PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3',
        'temp', 'humidity', 'wind_speed', 'wind_dir'
    ]

    # Replace zeros/NaN with median for sensor columns
    for col in cols_with_zeros_as_missing:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"Imputed 0/NaN in {col} -> median {median_val}")

    if 'fire_min_dist_km' in df.columns:
        df['fire_min_dist_km'].fillna(0, inplace=True)

    # Numeric missing → median
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col].fillna(df[col].median(), inplace=True)

    # Categorical missing → mode
    for col in df.select_dtypes(include=['object']).columns:
        df[col].fillna(df[col].mode()[0], inplace=True)

    return df

def feature_engineering(df):
    print("Performing feature engineering...")

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        df = df.sort_values(by='date')

        df['month'] = df['date'].dt.month
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_year'] = df['date'].dt.dayofyear

        print("Adding lag + rolling features...")

        for col in ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']:
            if col in df.columns:
                df[f'{col}_lag1'] = df[col].shift(1)

        for col in ['PM2.5', 'PM10']:
            if col in df.columns:
                df[f'{col}_roll3'] = df[col].rolling(3).mean()
                df[f'{col}_roll7'] = df[col].rolling(7).mean()

        df = df.dropna()
        df.drop(columns=['date'], inplace=True)

    # Interaction features
    if 'PM2.5' in df.columns and 'PM10' in df.columns:
        df['PM_Ratio'] = df['PM2.5'] / (df['PM10'] + 1e-6)

    if 'NO2' in df.columns and 'SO2' in df.columns:
        df['NO2_SO2_Ratio'] = df['NO2'] / (df['SO2'] + 1e-6)

    cols_to_drop = ['city', 'location_id', 'Confidence_Score', 'fire_count']
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True, errors='ignore')

    return df


def train_models(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    numeric_features = X.select_dtypes(include=[np.number]).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    models = {
        'Logistic Regression': {
            'model': LogisticRegression(max_iter=1000),
            'params': {'clf__C': [0.1, 1, 10]}
        },
        'Decision Tree': {
            'model': DecisionTreeClassifier(),
            'params': {
                'clf__max_depth': [10, 20, None],
                'clf__min_samples_split': [2, 5],
                'clf__min_samples_leaf': [1, 2]
            }
        },
        'Random Forest': {
            'model': RandomForestClassifier(),
            'params': {
                'clf__n_estimators': [100, 200],
                'clf__max_depth': [10, 20, None]
            }
        },
        'XGBoost': {
            'model': XGBClassifier(eval_metric='mlogloss'),
            'params': {
                'clf__n_estimators': [100, 200],
                'clf__max_depth': [5, 7],
                'clf__learning_rate': [0.05, 0.1],
            }
        }
    }

    best_score = 0
    best_model = None
    results = {}

    for name, config in models.items():
        print(f"\n🔵 Training {name}...")

        pipeline = ImbPipeline([
            ('preprocess', preprocessor),
            ('smote', SMOTE()),
            ('clf', config['model'])
        ])

        search = RandomizedSearchCV(
            pipeline,
            config['params'],
            cv=3,
            n_iter=5,
            scoring='f1_weighted',
            random_state=42
        )

        search.fit(X_train, y_train)
        preds = search.predict(X_test)

        f1 = f1_score(y_test, preds, average='weighted')
        print(f"{name} F1 Score = {f1}")

        results[name] = search.best_estimator_

        if f1 > best_score:
            best_score = f1
            best_model = search.best_estimator_

    return results, best_model



def main():

    # ✔ Fixed correct path (CSV)
    data_path = r"C:\Users\Lenovo\ai_powered_enviroScan\labeled_pollution_data.csv"

    df = load_and_clean_data(data_path)
    df = feature_engineering(df)

    if 'Source' not in df.columns:
        raise ValueError(" ERROR: Target column 'Source' missing!")

    X = df.drop(columns=['Source'])
    y = df['Source']

    le = LabelEncoder()
    y = le.fit_transform(y)

    results, best_model = train_models(X, y)

    output_dir = r"C:\Users\Lenovo\ai_powered_enviroScan\models"
    os.makedirs(output_dir, exist_ok=True)

    for name, model in results.items():
        filename = name.lower().replace(" ", "_") + ".pkl"
        joblib.dump(model, os.path.join(output_dir, filename))
        print(f"✔ Saved {name}")

    joblib.dump(le, os.path.join(output_dir, "label_encoder.pkl"))

    print("\n TRAINING COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
