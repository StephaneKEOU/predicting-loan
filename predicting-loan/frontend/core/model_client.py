import os
import joblib
import numpy as np
import pandas as pd

from core.config import (MODEL_PATH, DECISION_THRESHOLD, LOW_MEDIUM_THRESHOLD, MEDIUM_HIGH_THRESHOLD)

class ModelClient:
    def __init__(self) -> None:
        self.threshold = DECISION_THRESHOLD
        self.use_dummy = not os.path.exists(MODEL_PATH)
        self.model = None
        if not self.use_dummy:
            try:
                self.model = joblib.load(MODEL_PATH)
            except Exception:
                # fall back to dummy if load fails
                self.use_dummy = True

    def predict(self, X: pd.DataFrame):
        """
        Returns:
        probs: probability of default (float 0..1)
        labels: 1 = High Risk, 0 = Low Risk (DECISION_THRESHOLD)
        risk_bands: "Low" / "Medium / "High" for the UI
        """
        if self.use_dummy:
            # deterministic mock using numeric sum; OK for FE dev
            s = X.select_dtypes(include=["number"]).sum(axis=1)
            probs = 1.0 / (1.0 + np.exp(-(s % 5) / 5.0))
        else:
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(X)[:, 1]
            elif hasattr(self.model, "decision_function"):
                scores = self.model.decision_function(X)
                probs = 1.0 / (1.0 + np.exp(-scores))
            else:
                # worst case: hard labels; fabricate soft probs
                labels = self.model.predict(X)
                probs = np.where(labels == 1, 0.51, 0.49).astype(float)

    # high risk label MVP
        labels = (probs >= self.threshold).astype(int)

    # Low / Medium / High banding
        risk_bands = []
        for p in probs :
            if p < LOW_MEDIUM_THRESHOLD:
                risk_bands.append("Low")
            elif p < MEDIUM_HIGH_THRESHOLD:
                risk_bands.append("Medium")
            else:
                risk_bands.append("High")
        return probs, labels, np.array(risk_bands, dtype=object)