from fastapi import FastAPI, HTTPException
import pandas as pd
from enum import Enum
from typing_extensions import Annotated
from pydantic import BaseModel, Field, conint
import joblib
from pathlib import Path
import logging
from typing import List

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

MODEL_PATH = Path(__file__).resolve().parent / "models" / "base_model.pkl"

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
    loan_amount: Annotated[int, Field(gt=0, example=5000)]
    months_employed: Annotated[int, Field(ge=0, le=600, example=24)]
    age: Annotated[int, Field(ge=18, le=100, example=35)]
    income: Annotated[int, Field(ge=0, example=60000)]
    interest_rate: Annotated[float, Field(gt=0, le=100, example=5.5)]
    employment_type: Annotated[EmploymentTypeEnum, Field(example=EmploymentTypeEnum.full_time.value)]

class OptimizeLoanRequest(LoanApplication):
    desired_probability: Annotated[float, Field(gt=0, lt=1, example=0.2,
                                                description="Maxium acceptable probability of default")]

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

@app.post("/api/v1/batch_predict")
def batch_predict(payload: List[LoanApplication]):

    logger.info(f"Received batch request with {len(payload)} records")

    X = pd.DataFrame([
        {
         "LoanAmount": p.loan_amount,
         "MonthsEmployed": p.months_employed,
         "Age": p.age,
         "Income": p.income,
         "InterestRate": p.interest_rate,
         "EmploymentType": p.employment_type
         } for p in payload
    ])

    try:
        preds = model.predict(X)
        probas_default = model.predict_proba(X)[:, 1]
    except Exception as e:
        logger.exception("Batch model prediction failed")
        # Return a 500 with a friendly JSON error
        raise HTTPException(
            status_code=500,
            detail=f"Batch model prediction failed. {str(e)}",
        ) from e

    logger.info(f"Porbas type: {type(probas_default)}")

    # Convert to plain Python types so FastAPI can JSON-encode them
    preds_list = [int(p) for p in preds]
    probas_list = [float(p) for p in probas_default]

    logger.info(f"After conversion type: {type(probas_default)}")

    return {
        "results": [
            {
                "input": p.model_dump(),
                "prediction": pred,
                "probability": proba
                }
            for p, pred, proba in zip(payload, preds_list, probas_list)
        ]
    }

MAX_LOAN_AMOUNT = 10_000_000

@app.post("/api/v1/optimize_loan_amount")
def optimize_loan_amount(payload: OptimizeLoanRequest):

    loan_amount = payload.loan_amount
    desired_probability = payload.desired_probability

    X = pd.DataFrame([{
        "LoanAmount": loan_amount,
        "MonthsEmployed": payload.months_employed,
        "Age": payload.age,
        "Income": payload.income,
        "InterestRate": payload.interest_rate,
        "EmploymentType": payload.employment_type
    }])

    initial_proba_default = model.predict_proba(X)[0][1]

    logger.info(
        f"Optimizing loan amount for desired probability: {desired_probability}, "
        f"Initial loan amount: {loan_amount}, Initial probability of default: {initial_proba_default}"
    )

    def proba_for_amount(amount: int) -> float:
        X['LoanAmount'] = amount
        return model.predict_proba(X)[0][1]

    if initial_proba_default <= desired_probability:
        # Increase scenario: what is the maximum loan amount that keeps probability <= desired_probability
        logger.info("Initial loan amount already meets desired probability. Searching for maximum loan amount.")

        proba_default = initial_proba_default
        safe_high = loan_amount
        high = loan_amount * 2

        while high <= MAX_LOAN_AMOUNT:
            proba_default = proba_for_amount(high)
            if proba_default <= desired_probability:
                safe_high = high
                high *= 2
            else:
                break

        if high > MAX_LOAN_AMOUNT and proba_for_amount(MAX_LOAN_AMOUNT) <= desired_probability:
            optimal_loan_amount = float(MAX_LOAN_AMOUNT)
            logger.info(f"Even at cap, probability is below threshold. Returning cap={optimal_loan_amount}.")
            return {"optimal_loan_amount": round(optimal_loan_amount, 2)}

        low = safe_high
        high = max(high, MAX_LOAN_AMOUNT)

    else:
        # Decrease scenario: what is the optimal loan amount that keeps probability <= desired_probability
        logger.info("Initial loan amount exceeds desired probability. Searching for optimal loan amount.")
        low = 0
        high = loan_amount

    optimal_loan_amount = None

    while low <= high:

        mid = (low + high) // 2

        proba_default = proba_for_amount(mid)

        if proba_default > desired_probability:
            high = mid - 1
        else:
            optimal_loan_amount = mid
            low = mid + 1

    if optimal_loan_amount is None:
        raise HTTPException(
            status_code=404,
            detail="No loan amount found that meets the desired probability criteria."
        )

    logger.info(f"Optimal loan amount found: {optimal_loan_amount:.2f}")

    return {"optimal_loan_amount": optimal_loan_amount,
            "probability": proba_for_amount(optimal_loan_amount)}
