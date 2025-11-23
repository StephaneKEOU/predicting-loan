import os
import json
from pathlib import Path
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# This file lives in frontend/core/config.py
FRONTEND_DIR = Path(__file__).resolve().parents[1]
SCHEMAS_DIR = FRONTEND_DIR / "schemas"
MODELS_DIR = FRONTEND_DIR / "models"

THRESHOLDS_PATH = MODELS_DIR / "thresholds.json"

# Thresholds loader

def load_thresholds() -> dict:
    if THRESHOLDS_PATH.exists():
        with THRESHOLDS_PATH.open("r") as f:
            return json.load(f)
    return {
        "default_threshold": 0.20,
        "low_medium_threshold": 0.05,
        "medium_high_threshold":0.20,
        "labels":{
            "low": "Low risk (PD < 5%)",
            "medium": "Medium risk (5% ≤ PD < 20%)",
            "high": "High risk (PD ≥ 20%)",
        },
    }

THRESHOLDS = load_thresholds()

# config
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///predictions.db")
DECISION_THRESHOLD = float(os.getenv("DECISION_THRESHOLD", THRESHOLDS.get("default_threshold", 0.20)))

LOW_MEDIUM_THRESHOLD = float(THRESHOLDS.get("low_medium_threshold", 0.05))
MEDIUM_HIGH_THRESHOLD = float(THRESHOLDS.get("medium_high_threshold", 0.20))
RISK_LABELS = THRESHOLDS["labels"]
FEATURE_SCHEMA = str(SCHEMAS_DIR / "feature_schema.json")
MODEL_PATH = str(MODELS_DIR / "base_model.pkl")

# API Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "https://my-api-app-726608232851.europe-west1.run.app")
API_PREDICT_ENDPOINT = f"{API_BASE_URL}/api/v1/predict"
