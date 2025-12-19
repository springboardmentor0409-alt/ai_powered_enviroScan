# main.py
import uvicorn
from fastapi import FastAPI, HTTPException
from pathlib import Path
import traceback
import sys

# --- FIX: Add project root to system path so 'backend' module is found ---
# 1. Add 'backend' folder (for local imports like model_loader)
sys.path.append(str(Path(__file__).resolve().parent))
# 2. Add Project Root (so Uvicorn can find 'backend.main')
sys.path.append(str(Path(__file__).resolve().parent.parent))
# ------------------------------------------------------------------------

# Imports
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

@app.get("/")
def read_root():
    return {
        "status": "active",
        "message": "EnviroScan API is running successfully.",
        "endpoints": [
            "/models (GET)",
            "/predict/{model_name} (POST)"
        ]
    }

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
        # Preprocess
        df = preprocess_input(input_data)
        X = df 

        # Prediction Logic
        try:
            preds = selected_model.predict(X)
            prediction_idx = preds[0]
        except Exception:
            prediction_idx = selected_model.predict(X.values)[0]

        # Label Decoding
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
    # This string "backend.main:app" requires the root dir to be in sys.path
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)