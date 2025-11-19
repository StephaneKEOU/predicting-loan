import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score,
    confusion_matrix, RocCurveDisplay, roc_auc_score
)


# ----------------------------------------------------
# 1. LOAD DATA
# ----------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath('raw_data'))  # folder where script is
DATA_PATH = os.path.join(BASE_DIR,  "raw_data", "Loan_default.csv")
df = pd.read_csv(DATA_PATH)


print("Data loaded. Shape:", df.shape)

# ----------------------------------------------------
# 2. TARGET + DROP USELESS COLUMNS
# ----------------------------------------------------

df = df.drop(columns=["LoanID"])     # ID column not useful

# Drop rows with missing default
df = df.dropna(subset=["Default"])

# Check for missing values
print("\nMissing Values Count:")
print(df.isnull().sum())
print("\nData Types:")
print(df.dtypes)
print("\nDescriptive stats:")
print(df.describe())

binary_cols = ['HasMortgage', 'HasDependents', 'HasCoSigner']

binary_map = {"yes": 1, "no": 0, "y": 1, "n": 0, "1": 1, "0": 0}

for col in binary_cols:
    if df[col].dtype == object:
        df[col] = df[col].astype(str).str.lower().map(binary_map)


X = df.drop(columns=["Default"])
y = df["Default"]
print(df.head())
# ----------------------------------------------------
# 3. DEFINE COLUMN TYPES
# ----------------------------------------------------
categorical_cols =  ['Education', 'EmploymentType', 'MaritalStatus', 'LoanPurpose']

numerical_cols= ['Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed', 'NumCreditLines', 'InterestRate',
                 'LoanTerm', 'DTIRatio', 'HasDependents', 'HasCoSigner', 'HasMortgage']



print("Categorical columns:", categorical_cols)
print("Numerical columns:", numerical_cols)

# ----------------------------------------------------
# 4. PREPROCESSING PIPELINE
# ----------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numerical_cols),

        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical_cols)
    ]
)

# ----------------------------------------------------
# 5. TRAIN / TEST SPLIT
# ----------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# ----------------------------------------------------
# 6. BASELINE MODELS
# ----------------------------------------------------
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42, n_jobs=-1)
}

results = {}
roc_curves = {}

# ----------------------------------------------------
# 7. TRAINING + METRICS
# ----------------------------------------------------
for name, model in models.items():

    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipe.fit(X_train, y_train)



    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    prec = precision_score(y_test, y_pred)

    results[name] = (acc, f1, rec, auc, prec)
    roc_curves[name] = (y_test, y_proba)

    # ---------------- Confusion Matrix ----------------
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    ax.imshow(cm, cmap="Blues")
    ax.set_title(f"Confusion Matrix - {name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha='center', va='center')

    os.makedirs("outputs", exist_ok=True)
    plt.savefig(f"outputs/confusion_matrix_{name}.png", dpi=150)
    plt.close()

# ----------------------------------------------------
# 8. ROC CURVE
# ----------------------------------------------------
fig, ax = plt.subplots()

for name, (yt, yp) in roc_curves.items():
    RocCurveDisplay.from_predictions(yt, yp, name=name, ax=ax)

ax.set_title("ROC Curve  Log Reg vs Random Forest vs Decision Tree")

plt.savefig("outputs/roc_curve.png", dpi=150)
plt.close()

# ----------------------------------------------------
# 9. PRINT SUMMARY
# ----------------------------------------------------
print("\n===== BASELINE MODEL RESULTS =====\n")
for name, (acc, f1, rec, auc,prec) in results.items():
    print(f"{name}:")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}\n")
    print(f"  Precision  : {prec:.4f}\n")

print("➡️ Confusion matrices and ROC curve saved in ./outputs/")
