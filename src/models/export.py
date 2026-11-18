"""Exports the final tuned model + scaler + feature schema for the API to load."""

from __future__ import annotations

import json

import joblib

from src.config import MODELS_DIR
from src.features.extractor import FEATURE_COLUMNS


def export_model(model, scaler, model_name: str, extra_metadata: dict | None = None) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / "final_model.joblib")
    joblib.dump(scaler, MODELS_DIR / "final_scaler.joblib")

    metadata = {
        "model_name": model_name,
        "feature_columns": FEATURE_COLUMNS,
        **(extra_metadata or {}),
    }
    with open(MODELS_DIR / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"Exported model '{model_name}' and scaler to {MODELS_DIR}")


def load_exported_model():
    """Loads the exported model/scaler/feature schema. Returns None if not yet exported."""
    model_path = MODELS_DIR / "final_model.joblib"
    scaler_path = MODELS_DIR / "final_scaler.joblib"
    meta_path = MODELS_DIR / "model_metadata.json"
    if not (model_path.exists() and scaler_path.exists() and meta_path.exists()):
        return None

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    with open(meta_path) as f:
        metadata = json.load(f)

    from src.pipeline import LoadedModel
    return LoadedModel(
        model=model,
        scaler=scaler,
        feature_columns=metadata["feature_columns"],
        name=metadata["model_name"],
    )
