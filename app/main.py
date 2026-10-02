from pathlib import Path
import json
import joblib
import numpy as np

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# duong dan
BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "artifacts" / "breast_cancer_svm.joblib"
META_PATH = BASE_DIR / "artifacts" / "metadata.json"

# load model va metadata
model = joblib.load(MODEL_PATH)

with open(META_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

# tao FASTAPI APP
app = FastAPI(
    title="Breast Cancer SVM API",
    version=metadata["model_version"],
    description="Educational demonstration only"
)

class PredictionRequest(BaseModel):
    features: dict[str, float] = Field(
        ...,
        description="Exactly 30 named numeric features"
    )

class PredictionResponse(BaseModel):
    predicted_class: int
    predicted_label: str
    probability_malignant: float
    probability_benign: float
    model_version: str
    warning: str

def build_vector(payload: PredictionRequest) -> np.ndarray:
    expected = metadata["feature_names"]
    received = set(payload.features)
    missing = sorted(set(expected) - received)
    extra = sorted(received - set(expected))

    if missing or extra:
        raise HTTPException(
            status_code=422,
            detail={
                "missing": missing,
                "extra": extra
            }
        )

    values = np.array(
        [[payload.features[name] for name in expected]],
        dtype=float
    )

    if not np.isfinite(values).all():
        raise HTTPException(
            status_code=422,
            detail="Features must be finite numbers"
        )

    return values

# health
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_version": metadata["model_version"]
    }

# metadata
@app.get("/metadata")
def get_metadata():
    return metadata

# predict
@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest):
    x = build_vector(payload)
    predicted_class = int(model.predict(x)[0])
    probabilities = model.predict_proba(x)[0]
    classes = list(model.named_steps["svc"].classes_)
    probability_map = {
        int(c): float(p)
        for c, p in zip(classes, probabilities)
    }

    return PredictionResponse(
        predicted_class=predicted_class,
        predicted_label=metadata[
            "class_mapping"
        ][str(predicted_class)],
        probability_malignant=probability_map[0],
        probability_benign=probability_map[1],
        model_version=metadata[
            "model_version"
        ],
        warning=metadata[
            "warning"
        ]
    )

# /
@app.get("/")
def root():
    return {
        "service": "Breast Cancer SVM API",
        "docs": "/docs",
        "warning": metadata["warning"]
    }
