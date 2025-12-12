# main.py
import uvicorn
from fastapi import FastAPI, HTTPException
from pathlib import Path
from typing import Any
import traceback

import sys
from pathlib import Path

# Add the current directory (backend) to sys.path to allow imports when running from root
sys.path.append(str(Path(__file__).resolve().parent))

from model_loader import load_artifacts_and_models, get_models, find_model_by_name, load_label_encoder_for_model
from utilities import PollutionInput, preprocess_input

app = FastAPI(title="EnviroScan Pollution Source Prediction")

@app.on_event("startup")
def startup_event():
    # Try to detect project base dir (one level above backend/)
    base_dir = Path(__file__).resolve().parent.parent
    try:
        load_artifacts_and_models(base_dir=base_dir)
    except Exception:
        traceback.print_exc()
        # startup continues; endpoints will warn if models missing

@app.get("/models")
def list_models():
    models = get_models()
    return {"models": list(models.keys())}

@app.post("/predict/{model_name}")
def predict(model_name: str, input_data: PollutionInput):
    # Find model
    real_key, selected_model = find_model_by_name(model_name)
    if selected_model is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found. Available: {list(get_models().keys())}")

    try:
        df = preprocess_input(input_data)

        # Some scikit-learn models expect a specific set of columns.
        # Here we pass the full dataframe row — ensure training used the same pipeline/columns.
        X = df  # if you need specific columns: X = df[feature_list]

        # If model supports predict_proba, return a best confidence
        try:
            preds = selected_model.predict(X)
            prediction_idx = preds[0]
        except Exception:
            # Sometimes models expect numpy array
            prediction_idx = selected_model.predict(X.values)[0]

        # Try to decode using label encoder if available
        base_dir = Path(__file__).resolve().parent.parent
        le = load_label_encoder_for_model(base_dir=base_dir, model_dir_name=(real_key or model_name).lower().replace(" ", "_"))
        if le is not None:
            try:
                class_name = le.inverse_transform([prediction_idx])[0]
            except Exception:
                class_name = str(prediction_idx)
        else:
            class_name = str(prediction_idx)

        # Confidence
        confidence = "N/A"
        try:
            if hasattr(selected_model, "predict_proba"):
                probs = selected_model.predict_proba(X if hasattr(X, "values") else X.values)
                # take max probability of the predicted class (row 0)
                maxp = max(probs[0].tolist())
                confidence = float(maxp)
        except Exception:
            pass

        return {
            "model": real_key or model_name,
            "prediction": class_name,
            "confidence": confidence
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # run with: python backend/main.py
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
