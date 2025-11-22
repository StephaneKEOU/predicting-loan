import os
import requests
import numpy as np
import pandas as pd
from typing import Optional, Tuple
import streamlit as st

from core.config import (
    DECISION_THRESHOLD,
    LOW_MEDIUM_THRESHOLD,
    MEDIUM_HIGH_THRESHOLD,
    API_PREDICT_ENDPOINT
)

# Name mapping from frontend to API
# API only accepts these 6 fields based on API schema
FIELD_MAPPING = {
    "MonthEmployed": "months_employed",
    "Age": "age",
    "Income": "income",
    "LoanAmount": "loan_amount",
    "InterestRate": "interest_rate",
    "EmploymentType": "employment_type",
}

def convert_to_api_format(row: pd.Series) -> dict:
    """Convert a pandas Series from frontend format to API"""
    api_data = {}
    for frontend_name, api_name in FIELD_MAPPING.items():
        if frontend_name in row.index:
            api_data[api_name] = row[frontend_name]
    return api_data

class ModelClient:
    def __init__(self, api_url: Optional[str] = None) -> None:
        self.threshold = DECISION_THRESHOLD
        self.api_url = api_url or API_PREDICT_ENDPOINT
        self.use_api = True #Using always API now
    
    def _predict_single(self, row: pd.Series) -> Tuple[float, int]:
        """
        Single API request.
        Returns: probability, label
        """
        try:
            api_data = convert_to_api_format(row)
            response = requests.post(
                self.api_url,
                json=api_data,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                },
                timeout=10
            )
            response.raise_for_status()
            result = response.json()
            
            # API returns: {"prediction": 0, "probability": 0.313}
            prob = float(result.get("probability", 0.0))
            label = int(result.get("prediction", 0))
            
            return prob, label
        except requests.exceptions.HTTPError as e:
            # More detailed error handling
            status_code = e.response.status_code if e.response else "Unknown"
            error_text = e.response.text if e.response else str(e)
            error_msg = f"API HTTP Error {status_code}: {error_text}"
            
            # Show error in Streamlit if available
            try:
                st.error(f"⚠️ API Error ({status_code}): {error_text[:200]}")
            except:
                print(f"API request failed: {error_msg}")
            
            # Fallback to dummy prediction
            s = row.select_dtypes(include=["number"]).sum()
            prob = float(1.0 / (1.0 + np.exp(-(s % 5) / 5.0)))
            label = int(prob >= self.threshold)
            return prob, label
        except requests.exceptions.RequestException as e:
            # Fallback to dummy prediction on API error
            error_msg = f"API request failed: {str(e)}"
            try:
                st.warning(f"⚠️ {error_msg}. Using fallback prediction.")
            except:
                print(error_msg)
            s = row.select_dtypes(include=["number"]).sum()
            prob = float(1.0 / (1.0 + np.exp(-(s % 5) / 5.0)))
            label = int(prob >= self.threshold)
            return prob, label
        
    def predict(self, X: pd.DataFrame):
        """
        Predict using the API for each row in the DataFrame.
        Returns:
        probs: probability of default (float 0..1) numpy array
        labels: 1 = High Risk, 0 = Low Risk (DECISION_THRESHOLD)numpy array
        risk_bands: "Low"/"Medium"/"High" numpy array
        """
        probs = []
        labels = []
        
        # Process
        for idx, row in X.iterrows():
            prob, label = self._predict_single(row)
            probs.append(prob)
            labels.append(label)
            
        probs = np.array(probs, dtype=float)
        labels = np.array(labels, dtype=int)
        
        # Calculate risk bands
        risk_bands = []
        for p in probs:
            if p < LOW_MEDIUM_THRESHOLD:
                risk_bands.append("Low")
            elif p < MEDIUM_HIGH_THRESHOLD:
                risk_bands.append("Medium")
            else:
                risk_bands.append("High")
                
        return probs, labels, np.array(risk_bands, dtype=object)