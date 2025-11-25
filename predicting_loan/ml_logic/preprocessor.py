import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (OneHotEncoder,StandardScaler,RobustScaler)


def preprocess_data(X: pd.DataFrame) -> ColumnTransformer:
    """
    Builds an preprocessing pipeline for the loan default dataset.

    - Converts binary Yes/No columns to 0/1 BEFORE calling this function.
    - Scales skewed numeric features with RobustScaler.
    - Scales normal numeric features with StandardScaler.
    - OneHot encodes nominal categorical variables.

    Returns:
        ColumnTransformer ready to fit/transform.
    """

    # --- Feature Groups  ---

    robust_numeric = ["Income", "LoanAmount", "MonthsEmployed"]

    standard_numeric = [
        "Age",
        "InterestRate",
    ]

    onehot_cats = [
        "EmploymentType",
    ]

    # --- Pipelines ---

    robust_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", RobustScaler())
    ])

    standard_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    onehot_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    # --- ColumnTransformer ---

    preprocessor = ColumnTransformer(
        transformers=[
            ("robust_num", robust_pipeline, robust_numeric),
            ("standard_num", standard_pipeline, standard_numeric),
            ("onehot_cat", onehot_pipeline, onehot_cats),
        ]
    )

    return preprocessor
