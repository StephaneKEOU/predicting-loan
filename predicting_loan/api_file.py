from fastapi import FastAPI, HTTPException
import pandas as pd
from enum import Enum
from typing_extensions import Annotated
from pydantic import BaseModel, Field, conint
import joblib
from pathlib import Path
import logging

# -------------------------------------------------
# Logging setup
# -------------------------------------------------
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
logger = logging.getLogger("loan_api")

app = FastAPI(
    title="Loan Default Prediction API",
    version="0.1.0"
)

# -------------------------------------------------
# Load model once at startup
# -------------------------------------------------

MODEL_PATH = Path(__file__).resolve().parent / "models" / "logistic_model.pkl" ## TO DO: Change to "base_model.pkl"

try:
    model = joblib.load(MODEL_PATH)
    logger.info(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    logger.exception(f"Failed to load model from {MODEL_PATH}")
    raise e

# -------------------------------------------------
# Request Schema
# -------------------------------------------------

class EmploymentTypeEnum(str, Enum):
    full_time = "Full-time"
    part_time = "Part-time"
    self_employed = "Self-employed"
    unemployed = "Unemployed"

class LoanApplication(BaseModel):
    loan_amount: Annotated[float, Field(gt=0, example=5000)]
    months_employed: Annotated[int, Field(ge=0, le=600, example=24)]
    age: Annotated[int, Field(ge=18, le=100, example=35)]
    income: Annotated[float, Field(ge=0, example=60000)]
    interest_rate: Annotated[float, Field(gt=0, le=100, example=5.5)]
    employment_type: Annotated[EmploymentTypeEnum, Field(example=EmploymentTypeEnum.full_time.value)]

# -------------------------------------------------
# Endpoints
# -------------------------------------------------
@app.get("/")
def root():
    return {"message": "Welcome to the Loan Default Prediction API"}

@app.get("/api/v1/health")
def health():
    """
    Simple health endpoint.
    """
    return {
        "status": "ok",
        "model_loaded": True if model else False
    }

@app.post("/api/v1/predict")
def predict(payload: LoanApplication):

    logger.info(f"Received request: {payload.model_dump()}")

    X = pd.DataFrame([
        {"LoanAmount": payload.loan_amount,
         "MonthsEmployed": payload.months_employed,
         "Age": payload.age,
         "Income": payload.income,
         "InterestRate": payload.interest_rate,
         "EmploymentType": payload.employment_type}
    ])

    try:
        pred = int(model.predict(X)[0])
        proba_default = float(model.predict_proba(X)[0][1])
    except Exception as e:
        logger.exception("Model prediction failed")
        # Return a 500 with a friendly JSON error
        raise HTTPException(
            status_code=500,
            detail=f"Model prediction failed. {str(e)}",
        ) from e

    logger.info(f"Prediction: {pred}, Probability of Default: {proba_default}")

    return {"prediction": pred,
            "probability": proba_default}
