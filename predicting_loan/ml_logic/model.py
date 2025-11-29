import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from pathlib import Path
from predicting_loan.frontend.core.config import MODEL_PATH
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score,
    confusion_matrix, RocCurveDisplay, roc_auc_score
)

from data import SimpleMissingHandler, load_data

from preprocessor import get_preprocessor

# -------------------------------------------------------
# LOAD AND PREPARE DATA
# -------------------------------------------------------
def load_and_prepare_data():
    """
    Loads data, handles missing values, and identifies features to be used for modeling.
    Returns feature matrix X and target vector y.
    """
    df = load_data()
    handler = SimpleMissingHandler(how="auto")
    df_fixed = handler.fix_missing(df)

    binary_cols = ["HasMortgage", "HasDependents", "HasCoSigner"]
    for col in binary_cols:
        df_fixed[col] = df_fixed[col].map({"Yes": 1, "No": 0})

    # TARGET + DROP USELESS COLUMNS
    df_fixed = df_fixed.drop(columns=["LoanID"])     # ID column not useful

    # Feature selection based on 'feature_importance.py' analysis
    features = ["Age", "Income", "LoanAmount", "MonthsEmployed", "InterestRate", "EmploymentType"]

    X = df_fixed[features]
    y = df_fixed["Default"]

    return X, y

# -------------------------------------------------------
# MODEL FACTORY
# -------------------------------------------------------
def get_models():
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "RandomForest": RandomForestClassifier(random_state=42, n_jobs=-1)
    }

# -------------------------------------------------------
# TRAINING FUNCTION
# -------------------------------------------------------
def train_model(X, y):
    """
    Trains baseline models and return results and ROC data.
    Not used to save final model.
    """

    preprocessor = get_preprocessor()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    models = get_models()
    results = {}
    roc_data = {}

    for name, model in models.items():
        pipe = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        # FIT MODEL
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        # ---- Inspect probability distribution for LogisticRegression ----
        if name == "LogisticRegression":
            percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
            p_vals = np.percentile(y_proba, percentiles)

            print("\n===LogisticRegression predicted default probability percentiles (test set):")
            for p, v in zip(percentiles, p_vals):
                print(f"  {p:2d}th percentile: {v:.4f}")

            def proba_for_amount(amount):

                df = pd.DataFrame([{
                    "LoanAmount": amount,
                    "MonthsEmployed": 24,
                    "Age": 35,
                    "Income": 60000,
                    "InterestRate": 5.5,
                    "EmploymentType": "Full-time",
                    #Extra columns (not used)
                    "CreditScore": 700,
                    "DTIRatio": 0.44,
                    "LoanTerm": 36,
                    "NumCreditLines": 5,
                    "Education": "Bachelors",
                    "LoanPurpose": "Debt Consolidation",
                    "MaritalStatus": "Single",
                    "HasMortgage": 0,
                    "HasDependents": 0,
                    "HasCoSigner": 0
                }])
                return pipe.predict_proba(df)[0, 1]  # prob of default

            print("\n=== Logistic Regression predicted default probabilities for various loan amounts:")
            for amt in [10, 1_000, 5_000, 10_000, 50_000, 100_000, 250_000, 500_000, 1_000_000]:
                print(f"{amt} |", proba_for_amount(amt))

        # ----------------------------------------------------------------------

        # METRICS
        results[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "y_pred": y_pred,
        }

        roc_data[name] = (y_test, y_proba)

    return results, roc_data, y_test

# -------------------------------------------------------
# SAVE PIPELINE TO PKL
# -------------------------------------------------------

def save_logistic_pipeline(X, y):
    """
    Takes full data, fits LogisticRegression pipeline, and saves it to MODEL_PATH.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from predicting_loan.ml_logic.preprocessor import get_preprocessor

    preprocessor = get_preprocessor()

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000)),
    ])

    pipe.fit(X, y)

    path = Path(MODEL_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, path)
    print(f" Saved LogisticRegression pipeline to {path} ({path.stat().st_size} bytes)")

# -------------------------------------------------------
# CONFUSION MATRIX
# -------------------------------------------------------
def plot_confusion_matrices(y_test, results):

    os.makedirs("outputs", exist_ok=True)

    for name, metrics in results.items():
        cm = confusion_matrix(y_test, metrics["y_pred"])

        fig, ax = plt.subplots()
        ax.imshow(cm, cmap="Blues")
        ax.set_title(f"Confusion Matrix - {name}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha='center', va='center')

        plt.savefig(f"outputs/confusion_matrix_{name}.png", dpi=150)
        plt.close()


# -------------------------------------------------------
# ROC CURVE
# -------------------------------------------------------
def plot_roc_curves(roc_data):
    fig, ax = plt.subplots()

    for name, (yt, yp) in roc_data.items():
        RocCurveDisplay.from_predictions(yt, yp, name=name, ax=ax)

    ax.set_title("ROC Curve - LR vs RF vs DT")
    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/roc_curve.png", dpi=150)
    plt.close()


# -------------------------------------------------------
# PRINT SUMMARY
# -------------------------------------------------------
def print_summary(results):
    print("\n===== BASELINE MODEL RESULTS =====\n")
    for name, m in results.items():
        print(f"{name}:")
        print(f"  Accuracy  : {m['accuracy']:.4f}")
        print(f"  Precision : {m['precision']:.4f}")
        print(f"  Recall    : {m['recall']:.4f}")
        print(f"  F1 Score  : {m['f1']:.4f}")
        print(f"  ROC-AUC   : {m['roc_auc']:.4f}\n")

    print("➡️ Outputs saved in ./outputs/")


# -------------------------------------------------------
# RUN EVERYTHING
# -------------------------------------------------------
if __name__ == "__main__":

    X, y = load_and_prepare_data()
    results, roc_data, y_test = train_model(X, y)
    plot_confusion_matrices(y_test, results)
    plot_roc_curves(roc_data)
    print_summary(results)

    # Fit on full data and save Logistic Regression for frontend
    save_logistic_pipeline(X, y)
