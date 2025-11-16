import json
from datetime import datetime

import pandas as pd
import streamlit as st

from core.schema_loader import load_schema
from core.config import FEATURE_SCHEMA, DECISION_THRESHOLD
from core.model_client import ModelClient
from core.db import init_db, log

st.set_page_config(page_title="Loan Default — MVP", layout="wide")

# Load schema and model
schema = load_schema(FEATURE_SCHEMA)
model = ModelClient()
init_db()

# Sidebar
st.sidebar.title("Loan Default MVP")
mode = st.sidebar.radio("Mode", ["Single Input", "Batch CSV"])
st.sidebar.write("Threshold: **{:.2f}**".format(DECISION_THRESHOLD))

# Title
st.title("Loan Default Risk Prediction")

def coerce_and_order(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DataFrame has all required feature columns in correct order and dtype-ish."""
    cols = []
    for f in schema["features"]:
        name = f["name"]
        dtype = f.get("dtype", "float")
        default = f.get("default", None)

        if name not in df.columns:
            df[name] = default

        if dtype in ("float", "int"):
            df[name] = pd.to_numeric(df[name], errors="coerce")
        elif dtype == "category":
            df[name] = df[name].astype("string")
        else:
            df[name] = df[name].astype("string")

        cols.append(name)
    return df[cols]

if mode == "Single Input":
    st.subheader("Single Applicant")
    c1, c2 = st.columns(2)
    values = {}
    for i, f in enumerate(schema["features"]):
        target = c1 if i % 2 == 0 else c2
        name = f["name"]
        dtype = f.get("dtype", "float")
        default = f.get("default", 0)

        if dtype == "int":
            values[name] = target.number_input(name, value=int(default or 0), step=1)
        elif dtype == "float":
            values[name] = target.number_input(name, value=float(default or 0.0))
        elif dtype == "category":
            choices = f.get("choices", [])
            idx = 0
            if choices and default in choices:
                idx = choices.index(default)
            values[name] = target.selectbox(name, choices, index=idx if choices else 0)
        else:
            values[name] = target.text_input(name, value=str(default or ""))

    if st.button("Predict"):
        X = coerce_and_order(pd.DataFrame([values]))
        probs, labels = model.predict(X)
        prob = float(probs[0])
        label = int(labels[0])
        st.metric("Probability of Default", "{:.2%}".format(prob))
        st.write("Prediction:", "**High Risk**" if label == 1 else "Low Risk")
        st.dataframe(X.assign(prob_default=probs, pred_label=labels))
        log(datetime.utcnow().isoformat(), "single", json.dumps(values), prob, label)

else:
    st.subheader("Batch CSV")
    up = st.file_uploader("Upload CSV", type=["csv"])
    if up is not None:
        raw = pd.read_csv(up)
        st.caption("Preview")
        st.dataframe(raw.head())
        X = coerce_and_order(raw.copy())
        probs, labels = model.predict(X)
        out = X.assign(prob_default=probs, pred_label=labels)
        st.success("Batch scoring complete.")
        st.dataframe(out.head())
        st.download_button(
            "Download scored CSV",
            out.to_csv(index=False).encode("utf-8"),
            "scored.csv",
            "text/csv"
        )
        for _, r in out.iterrows():
            payload = json.dumps({k: r[k] for k in X.columns})
            log(datetime.utcnow().isoformat(), "batch", payload, float(r["prob_default"]), int(r["pred_label"]))
