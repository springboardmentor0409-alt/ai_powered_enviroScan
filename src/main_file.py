import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
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

# Suppress warnings
warnings.filterwarnings('ignore')


def load_and_clean_data(filepath):
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)

    # Drop duplicates
    df = df.drop_duplicates()

    # 1. Handle 0s as missing for specific columns
    cols_with_zeros_as_missing = [
        'PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3',
        'temp', 'humidity', 'wind_speed', 'wind_dir'
    ]

    for col in cols_with_zeros_as_missing:
        if col in df.columns:
            # Replace 0 with NaN
            df[col] = df[col].replace(0, np.nan)
            # Impute with median
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"Imputed 0s/NaNs in {col} with median: {median_val}")

    # 2. Handle fire_min_dist_km: fill blank with 0
    if 'fire_min_dist_km' in df.columns:
        df['fire_min_dist_km'] = df['fire_min_dist_km'].fillna(0)
        print("Filled missing fire_min_dist_km with 0")

    # Handle remaining missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    print(f"Loaded dataframe with shape: {df.shape} and columns: {list(df.columns)}")
    return df


def feature_engineering(df):
    print("Performing feature engineering...")

    # Date features
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])

        # Sort by date for time-series features
        df = df.sort_values(by='date')

        df['month'] = df['date'].dt.month
        df['day_of_year'] = df['date'].dt.dayofyear
        df['day_of_week'] = df['date'].dt.dayofweek

        # --- Advanced Time-Series Features ---
        print("Adding Lag and Rolling features...")
        # Lags (Previous day values)
        for col in ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']:
            if col in df.columns:
                df[f'{col}_lag1'] = df[col].shift(1)

        # Rolling Averages (3-day and 7-day)
        for col in ['PM2.5', 'PM10']:
            if col in df.columns:
                df[f'{col}_roll3'] = df[col].rolling(window=3).mean()
                df[f'{col}_roll7'] = df[col].rolling(window=7).mean()

        # Drop rows with NaNs created by shifting/rolling
        df = df.dropna()

        df = df.drop(columns=['date'])

    # Interaction features
    if 'PM2.5' in df.columns and 'PM10' in df.columns:
        df['PM_Ratio'] = df['PM2.5'] / (df['PM10'] + 1e-6)

    if 'NO2' in df.columns and 'SO2' in df.columns:
        df['NO2_SO2_Ratio'] = df['NO2'] / (df['SO2'] + 1e-6)

    # 3. Drop columns (if present)
    cols_to_drop = ['city', 'location_id', 'Confidence_Score', 'fire_count']
    present_to_drop = [c for c in cols_to_drop if c in df.columns]
    if present_to_drop:
        df = df.drop(columns=present_to_drop, errors='ignore')
    print(f"Dropped columns (if any): {present_to_drop}")

    print(f"Feature engineered dataframe shape: {df.shape}")
    return df


def train_models(X, y):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Preprocessing
    numeric_features = X.select_dtypes(include=[np.number]).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    print(f"Numeric features: {list(numeric_features)}")
    print(f"Categorical features: {list(categorical_features)}")

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ])

    # Models to train
    models = {
        'Logistic Regression': {
            'model': LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42),
            'params': {
                'clf__C': [0.1, 1.0, 10.0],
            }
        },
        'Decision Tree': {
            'model': DecisionTreeClassifier(random_state=42),
            'params': {
                'clf__max_depth': [10, 20, 30, None],
                'clf__min_samples_split': [2, 5, 10],
                'clf__min_samples_leaf': [1, 2, 4]
            }
        },
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42, n_jobs=-1),
            'params': {
                'clf__n_estimators': [100, 200, 300],
                'clf__max_depth': [10, 20, 30, None],
                'clf__min_samples_split': [2, 5, 10],
                'clf__min_samples_leaf': [1, 2, 4]
            }
        },
        'XGBoost': {
            'model': XGBClassifier(eval_metric='mlogloss', random_state=42, n_jobs=-1, use_label_encoder=False,verbosity=0,tree_method='hist'),
            'params': {
                'clf__n_estimators': [100, 200, 300],
                'clf__learning_rate': [0.05, 0.1, 0.2],
                'clf__max_depth': [5, 7, 10],
                'clf__min_child_weight': [1, 3, 5],
                'clf__reg_alpha': [0, 0.1, 1.0],
                'clf__reg_lambda': [1.0, 2.0]
            }
        }
    }

    results = {}
    best_model = None
    best_score = 0

    for name, config in models.items():
        print(f"\nTraining {name}...")

        # Use ImbPipeline to include SMOTE
        pipeline = ImbPipeline([
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('clf', config['model'])
        ])

        # Hyperparameter tuning
        search = RandomizedSearchCV(
            pipeline,
            config['params'],
            n_iter=8,
            cv=3,
            scoring='f1_weighted',
            random_state=42,
            n_jobs=-1
        )

        search.fit(X_train, y_train)

        # Evaluate
        y_pred = search.predict(X_test)
        report = classification_report(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        # Check for overfitting (Train vs Test)
        train_score = search.score(X_train, y_train)

        print(f"Best Params: {search.best_params_}")
        print(f"Train F1 Score: {train_score:.4f}")
        print(f"Test F1 Score (Weighted): {f1:.4f}")
        print("Classification Report:\n", report)

        results[name] = {
            'model': search.best_estimator_,
            'score': f1,
            'report': report,
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }

        if f1 > best_score:
            best_score = f1
            best_model = search.best_estimator_

    return results, best_model, X_test, y_test


def main():
    # Resolve project paths relative to this script (safer than hard-coding absolute paths)
    this_dir = os.path.dirname(os.path.abspath(__file__))   # src/
    project_root = os.path.abspath(os.path.join(this_dir, ".."))  # project root

    # file in your data/ folder (you renamed it to this)
    data_filename = "labeled_pollution_data.csv"
    data_path = os.path.join(project_root, "data", data_filename)

    print(f"Script path: {os.path.join(this_dir, 'main_file.py')}")
    print(f"Project root: {project_root}")
    print(f"Trying to load data from: {data_path}")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    # 1. Load and Clean
    df = load_and_clean_data(data_path)

    # 2. Feature Engineering
    df = feature_engineering(df)

    # 3. Prepare X and y
    target_col = 'Source'
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found. Available columns: {list(df.columns)}")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Encode target
    le = LabelEncoder()
    y = le.fit_transform(y)
    print("Target Classes:", le.classes_)

    # 4. Train Models
    results, best_model, X_test, y_test = train_models(X, y)

    # 5. Save Results into project folder
    output_dir = os.path.join(project_root, "models")
    os.makedirs(output_dir, exist_ok=True)

    for name, data in results.items():
        safe_name = name.lower().replace(" ", "_")
        model_path = os.path.join(output_dir, f'{safe_name}_model.pkl')
        joblib.dump(data['model'], model_path)
        print(f"Saved {name} to {model_path}")

    joblib.dump(le, os.path.join(output_dir, 'label_encoder.pkl'))
    print(f"Label encoder saved to {output_dir}")


if __name__ == "__main__":
    main()
