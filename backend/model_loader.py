# model_loader.py
import joblib
from pathlib import Path
from typing import Dict, Any, Optional
import traceback

_artifacts: Dict[str, Any] = {}
_models: Dict[str, Any] = {}

def load_artifacts_and_models(base_dir: Optional[Path] = None) -> None:
    """
    Load preprocessing artifacts and ML models into module-level containers.
    base_dir should point to project root (one level above backend/).
    """
    global _artifacts, _models

    try:
        if base_dir is None:
            # assume this file lives in backend/
            base_dir = Path(__file__).resolve().parent.parent

        # Artifacts
        artifact_path = base_dir / "models" / "artifacts" / "preprocessor.joblib"
        if artifact_path.exists():
            _artifacts = joblib.load(artifact_path)
            print(f"[model_loader] Preprocessing artifacts loaded from {artifact_path}")
        else:
            print(f"[model_loader] WARNING: Artifacts not found at {artifact_path}")

        # Random Forest (primary model)
        rf_path = base_dir / "models" / "random_forest" / "random_forest.joblib"
        if rf_path.exists():
            _models["Random Forest"] = joblib.load(rf_path)
            print(f"[model_loader] Random Forest loaded from {rf_path}")
        else:
            print(f"[model_loader] WARNING: Random Forest model not found at {rf_path}")

        # Try to load other models dynamically if present
        for model_dir in (base_dir / "models").iterdir():
            if model_dir.is_dir() and model_dir.name.lower() != "artifacts":
                possible = model_dir / f"{model_dir.name}.joblib"
                # fallback to common names
                if possible.exists():
                    try:
                        m = joblib.load(possible)
                        _models[model_dir.name] = m
                        print(f"[model_loader] Loaded model {model_dir.name} from {possible}")
                    except Exception:
                        # ignore loading failures for optional models
                        traceback.print_exc()
    except Exception as e:
        print("[model_loader] Exception while loading artifacts/models:")
        traceback.print_exc()
        raise e

def get_artifacts() -> Dict[str, Any]:
    return _artifacts

def get_models() -> Dict[str, Any]:
    return _models

def find_model_by_name(name: str):
    """
    Find a model by exact key or by normalized key (case-insensitive, underscores).
    Returns (real_key, model) or (None, None)
    """
    norm = name.lower().replace(" ", "_")
    for k, m in _models.items():
        if k.lower() == name.lower() or k.lower().replace(" ", "_") == norm:
            return k, m
    return None, None

def load_label_encoder_for_model(base_dir: Optional[Path] = None, model_dir_name: str = "random_forest"):
    """
    Attempt to load label encoder from models/<model_dir_name>/label_encoder.joblib
    """
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent.parent
    le_path = base_dir / "models" / model_dir_name / "label_encoder.joblib"
    if le_path.exists():
        try:
            le = joblib.load(le_path)
            return le
        except Exception:
            traceback.print_exc()
            return None
    return None
