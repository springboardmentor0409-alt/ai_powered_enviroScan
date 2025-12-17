# src/models/train_models.py
import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# XGBoost is optional; code will run without it if not installed
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

# Path to your labeled dataset (adjust if your file is in a different location)
DATA_PATH = os.path.join("data", "labeled_pollution_data.csv")
MODEL_DIR = "models"
MODEL_OUT = os.path.join(MODEL_DIR, "best_model.pkl")


def load_data(path=DATA_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")
    df = pd.read_csv(path)
    # Ensure date is parsed (not strictly required for training after we drop it)
    if "date" in df.columns:
        try:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        except Exception:
            pass
    return df


def prepare_features(df, target_col="Source"):
    """
    Prepare X, y for model training.
    - Drops datetime column
    - Saves original target for label encoding
    - One-hot encodes categorical columns
    - Keeps only numeric columns for X
    Returns: X (DataFrame), y (np.array), label_encoder, feature_columns (list)
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataframe columns: {df.columns.tolist()}")

    df_orig = df.copy()

    # Drop columns that should not be used as raw numeric inputs
    drop_cols = ["Source", "Confidence_Score", "date"]
    drop_cols = [c for c in drop_cols if c in df_orig.columns]
    df = df_orig.drop(columns=drop_cols, errors="ignore")

    # If Season exists, include it in encoding (get_dummies covers it)
    # One-hot encode all object / category columns
    df = pd.get_dummies(df, drop_first=True)

    # Label encode target
    le = LabelEncoder()
    y = le.fit_transform(df_orig[target_col].astype(str))

    # Keep only numeric dtypes for X (float, int, uint8 from get_dummies)
    X = df.select_dtypes(include=["int64", "float64", "uint8"]).copy()

    # As a safety: align index of X and y
    if len(X) != len(y):
        X = X.reset_index(drop=True)
        y = np.array(y).reshape(-1)

    feature_columns = X.columns.tolist()

    print(f"[prepare_features] X shape: {X.shape}, y shape: {y.shape}")
    print(f"[prepare_features] Number of features: {len(feature_columns)}")

    return X, y, le, feature_columns


def train_all_models(X_train, X_test, y_train, y_test):
    """
    Train a collection of models, evaluate, and return results dict.
    results[name] = {"model": estimator, "accuracy": acc, "report": report, "confusion": cm}
    """
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=150, n_jobs=-1, random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=500, random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
    }

    if XGB_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=42,
        )

    results = {}

    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        report = classification_report(y_test, preds)
        cm = confusion_matrix(y_test, preds)

        print(f"{name} accuracy: {acc:.4f}")
        results[name] = {
            "model": model,
            "accuracy": acc,
            "report": report,
            "confusion": cm
        }

    return results


def save_best_model(results, label_encoder, feature_columns, out_path=MODEL_OUT):
    best_name = max(results.keys(), key=lambda k: results[k]["accuracy"])
    best_model = results[best_name]["model"]
    best_acc = results[best_name]["accuracy"]

    os.makedirs(MODEL_DIR, exist_ok=True)

    payload = {
        "model": best_model,
        "label_encoder": label_encoder,
        "feature_columns": feature_columns,
        "model_name": best_name,
        "accuracy": best_acc
    }

    joblib.dump(payload, out_path)
    print(f"\nSaved best model '{best_name}' (acc={best_acc:.4f}) to: {out_path}")
    return out_path


def main():
    print("Loading dataset...")
    df = load_data(DATA_PATH)

    print("Preparing features...")
    X, y, le, feature_columns = prepare_features(df, target_col="Source")

    print("Splitting train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training models (this may take a bit)...")
    results = train_all_models(X_train, X_test, y_train, y_test)

    print("\nSelecting best model and saving...")
    save_best_model(results, le, feature_columns)

    # Print summary reports for best and all models
    print("\n--- Summary of results ---")
    for name, out in results.items():
        print(f"\nModel: {name}\nAccuracy: {out['accuracy']:.4f}")
        print(out["report"])

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
