# src/models/predict.py
import joblib
import pandas as pd
import numpy as np
import os

MODEL_PATH = "models/best_model.pkl"

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    label_encoder = bundle["label_encoder"]
    feature_columns = bundle["feature_columns"]
    
    return model, label_encoder, feature_columns


def preprocess_input(input_data, feature_columns):
    """
    input_data: dict
    Takes a dictionary of input features and converts to a dataframe aligned with training columns.
    """
    df = pd.DataFrame([input_data])

    # Convert categorical → dummy
    df = pd.get_dummies(df, drop_first=True)

    # Add missing columns
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0  # neutral
    
    # Drop extra columns (align exactly)
    df = df[feature_columns]

    return df


def predict_pollution_source(input_data):
    model, label_encoder, feature_columns = load_model()

    X = preprocess_input(input_data, feature_columns)
    preds = model.predict(X)
    probs = model.predict_proba(X)

    predicted_class = label_encoder.inverse_transform(preds)[0]
    confidence = np.max(probs)

    return predicted_class, float(confidence)
