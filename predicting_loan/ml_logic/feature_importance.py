import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from data import load_data, SimpleMissingHandler
from preprocessor import get_preprocessor

from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score,
    confusion_matrix, RocCurveDisplay, roc_auc_score
)


BINARY_COLS = ["HasMortgage", "HasDependents", "HasCoSigner"]


def get_models():
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "RandomForest": RandomForestClassifier(random_state=42, n_jobs=-1)
    }


def prepare_data():
    """
    Load data, fix missing values, convert Yes/No, and split X/y.
    Mirrors model.py logic.
    """
    df = load_data()
    handler = SimpleMissingHandler(how="auto")
    df = handler.fix_missing(df)

    # Convert Yes/No to 0/1
    for col in BINARY_COLS:
        df[col] = df[col].map({"Yes": 1, "No": 0})

    # Drop ID column
    df = df.drop(columns=["LoanID"])

    X = df.drop(columns=["Default"])
    y = df["Default"]

    return train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)


def get_feature_names(preprocessor):
    feature_names = []

    for name, transformer, cols in preprocessor.transformers_:
        if name == "remainder":
            continue

        # Case 1: Pipeline (standard, robust, onehot, binary)
        if hasattr(transformer, "named_steps"):
            # OneHotEncoder case
            if "encoder" in transformer.named_steps:
                ohe = transformer.named_steps["encoder"]
                ohe_features = ohe.get_feature_names_out(cols)
                feature_names.extend(ohe_features)
            else:
                # Scalers, imputers → no expansion
                feature_names.extend(cols)
        else:
            # passthrough or unknown transformer
            feature_names.extend(cols)

    return feature_names


def show_tree_importances(name, model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    print(f"\n=== Feature importances for {name} ===")
    for idx in indices[:20]:  # top 20
        print(f"{feature_names[idx]:40s}  {importances[idx]:.4f}")


def show_logreg_importances(name, model, feature_names):
    coef = model.coef_[0]  # binary classification
    abs_coef = np.abs(coef)
    indices = np.argsort(abs_coef)[::-1]

    print(f"\n=== Feature importances (|coef|) for {name} ===")
    for idx in indices[:20]:
        print(f"{feature_names[idx]:40s}  coef={coef[idx]:.4f}  |coef|={abs_coef[idx]:.4f}")


def grouped_importance(feature_names, coefficients):
    """
    Groups one-hot encoded features back to their original feature names
    and sums the absolute coefficients.
    """
    df = pd.DataFrame({
        "feature": feature_names,
        "coef": coefficients,
        "abs_coef": np.abs(coefficients)
    })

    def base(name):
        # Example: EmploymentType_Full-time → EmploymentType
        return name.split("_")[0]

    df["group"] = df["feature"].apply(base)

    grouped = df.groupby("group")["abs_coef"].mean().sort_values(ascending=False)
    return grouped

def evaluate_without_feature(X_train, X_test, y_train, y_test, feature_to_remove):
    cols = [c for c in X_train.columns if c != feature_to_remove]
    X_train_reduced = X_train[cols]
    X_test_reduced = X_test[cols]

    preprocessor = get_preprocessor()
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000))
    ])

    pipe.fit(X_train_reduced, y_train)
    y_proba = pipe.predict_proba(X_test_reduced)[:, 1]
    auc = roc_auc_score(y_test, y_proba)

    return auc


def main():
    X_train, X_test, y_train, y_test = prepare_data()

    models = get_models()

    for name, base_model in models.items():
        print(f"\n\n##### Training {name} for feature importance #####")

        preprocessor = get_preprocessor(for_training=True)

        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", base_model),
        ])

        pipe.fit(X_train, y_train)

        fitted_preprocessor = pipe.named_steps["preprocessor"]
        fitted_model = pipe.named_steps["model"]

        feature_names = get_feature_names(fitted_preprocessor)

        if hasattr(fitted_model, "feature_importances_"):
            show_tree_importances(name, fitted_model, feature_names)
        elif hasattr(fitted_model, "coef_"):
            show_logreg_importances(name, fitted_model, feature_names)
        else:
            print(f"{name}: model does not expose feature_importances_ or coef_.")

        if hasattr(fitted_model, "coef_") and name == "LogisticRegression":
            group_imp = grouped_importance(feature_names, fitted_model.coef_[0])
            print("\n=== Grouped Feature Importance (sum of |coefficients|) ===")
            print(group_imp)

if __name__ == "__main__":
    main()
