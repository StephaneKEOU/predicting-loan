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



from preprocessor import preprocess_data


df = load_data()
handler = SimpleMissingHandler(how="auto")
df_fixed = handler.fix_missing(df)

# binary_cols = ["HasMortgage", "HasDependents", "HasCoSigner"]
#for col in binary_cols:
 #   df[col] = df[col].map({"Yes": 1, "No": 0})

# TARGET + DROP USELESS COLUMNS
df = df.drop(columns=["LoanID"])     # ID column not useful
print (df_fixed.head())


X = df.drop(columns=["Default"])
y = df["Default"]


categorical_cols = ['EmploymentType']

numerical_cols = [
    'Age', 'Income', 'LoanAmount',  'MonthsEmployed',
     'InterestRate',
]



X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
# -------------------------------------------------------
# MODEL FACTORY
# -------------------------------------------------------
def get_models():
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "RandomForest": RandomForestClassifier(random_state=42, n_jobs=-1)
    }

preprocessor = preprocess_data(df_fixed)
# -------------------------------------------------------
# TRAINING FUNCTION
# -------------------------------------------------------
def train_model():
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

    return results, roc_data


# -------------------------------------------------------
# CONFUSION MATRIX
# -------------------------------------------------------
def plot_confusion_matrices(results):
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
# SAVE PIPELINE TO PKL
# -------------------------------------------------------

def save_logistic_pipeline(X, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from predicting_loan.ml_logic.preprocessor import preprocess_data

    preprocessor = preprocess_data(X)

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])

    pipe.fit(X, y)

    path = Path(MODEL_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, path)
    print(f" Saved LogisticRegression pipeline to {path} ({path.stat().st_size} bytes)")

# -------------------------------------------------------
# RUN EVERYTHING
# -------------------------------------------------------
if __name__ == "__main__":
    results, roc_data = train_model()
    plot_confusion_matrices(results)
    plot_roc_curves(roc_data)
    print_summary(results)

    # Fit on full data and save Logistic Regression for frontend
    save_logistic_pipeline(X, y)
