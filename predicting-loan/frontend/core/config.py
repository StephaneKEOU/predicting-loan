import os
from pathlib import Path
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# This file lives in frontend/core/config.py
FRONTEND_DIR = Path(__file__).resolve().parents[1]
SCHEMAS_DIR  = FRONTEND_DIR / "schemas"
MODELS_DIR   = "models"

DATABASE_URL       = os.getenv("DATABASE_URL", "sqlite:///predictions.db")
DECISION_THRESHOLD = float(os.getenv("DECISION_THRESHOLD", "0.5"))
FEATURE_SCHEMA     = str(SCHEMAS_DIR / "feature_schema.json")
MODEL_PATH         = str("baseline_rf.pkl")
