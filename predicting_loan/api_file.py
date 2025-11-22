from fastapi import FastAPI
import pickle
import pandas as pd
from enum import Enum
from typing_extensions import Annotated
from pydantic import BaseModel, Field, conint

class EmploymentTypeEnum(str, Enum):
    full_time = "Full-time"
    part_time = "Part-time"
    self_employed = "Self-employed"
    unemployed = "Unemployed"

app = FastAPI()

with open('predicting_loan/models/logistic_model.pkl', 'rb') as file:
    model = pickle.load(file)

class LoanApplication(BaseModel):
    loan_amount: Annotated[float, Field(gt=0, example=5000)]
    months_employed: Annotated[int, Field(ge=0, le=600, example=24)]
    age: Annotated[int, Field(ge=18, le=100, example=35)]
    income: Annotated[float, Field(ge=0, example=60000)]
    interest_rate: Annotated[float, Field(gt=0, le=100, example=5.5)]
    employment_type: Annotated[EmploymentTypeEnum, Field(example=EmploymentTypeEnum.full_time)]

@app.get("/")
def root():
    return {"message": "Hello world"}

@app.post("/api/v1/predict")
def predict(payload: LoanApplication):

    X = pd.DataFrame([
        {"LoanAmount": payload.loan_amount,
         "MonthsEmployed": payload.months_employed,
         "Age": payload.age,
         "Income": payload.income,
         "InterestRate": payload.interest_rate,
         "EmploymentType": payload.employment_type}
    ])

    pred = int(model.predict(X)[0])
    proba = float(model.predict_proba(X)[0][1])

    return {"prediction": pred,
            "probability": proba}
