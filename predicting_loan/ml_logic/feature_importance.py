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
from preprocessor import preprocess_data


BINARY_COLS = ["HasMortgage", "HasDependents", "HasCoSigner"]


def get_models():
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "RandomForest": RandomForestClassifier(random_state=42, n_jobs=-1),
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


def get_feature_names(preprocessor, X_sample: pd.DataFrame):
    """
    Try to get transformed feature names from the preprocessor.
    """
    try:
        # sklearn >= 1.0
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        # Fallback: generic names
        Xt = preprocessor.transform(X_sample)
        feature_names = [f"feat_{i}" for i in range(Xt.shape[1])]
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


def main():
    X_train, X_test, y_train, y_test = prepare_data()

    models = get_models()

    for name, base_model in models.items():
        print(f"\n\n##### Training {name} for feature importance #####")

        preprocessor = preprocess_data(X_train)

        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", base_model),
        ])

        pipe.fit(X_train, y_train)

        fitted_preprocessor = pipe.named_steps["preprocessor"]
        fitted_model = pipe.named_steps["model"]

        feature_names = get_feature_names(fitted_preprocessor, X_train)

        if hasattr(fitted_model, "feature_importances_"):
            show_tree_importances(name, fitted_model, feature_names)
        elif hasattr(fitted_model, "coef_"):
            show_logreg_importances(name, fitted_model, feature_names)
        else:
            print(f"{name}: model does not expose feature_importances_ or coef_.")


if __name__ == "__main__":
    main()
