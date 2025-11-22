import pandas as pd
<<<<<<< HEAD
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (OneHotEncoder,StandardScaler,RobustScaler)
=======
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, OrdinalEncoder



>>>>>>> origin/master


def preprocess_data(X: pd.DataFrame) -> ColumnTransformer:
    """
<<<<<<< HEAD
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
        "CreditScore",
        "InterestRate",
        "DTIRatio",
        "NumCreditLines",
        "LoanTerm",
        "HasMortgage",
        "HasDependents",
        "HasCoSigner",
    ]

    onehot_cats = [
        "Education",
        "EmploymentType",
        "MaritalStatus",
        "LoanPurpose",
    ]

    # --- Pipelines ---
=======
    Builds a ColumnTransformer that applies:
    - Robust/Standard scaling for numerical columns
    - Ordinal encoding for ordinal/binary categories
    - OneHot encoding for nominal categories
    """

    # === Feature Groups ===

    # Numeric (Robust scaling: skewed or wide-range)
    robust_numeric = ["Income", "LoanAmount", "MonthsEmployed"]

    # Numeric (Standard scaling)
    standard_numeric = ["Age", "CreditScore", "InterestRate", "DTIRatio"]

    # Ordinal categorical (natural order or few integers)
    ordinal_cats = ["LoanTerm", "NumCreditLines", "HasMortgage", "HasDependents", "HasCoSigner"]

    # Nominal categorical (no order, low cardinality)
    onehot_cats = ["Education", "EmploymentType", "MaritalStatus", "LoanPurpose"]

    # === Pipelines ===
>>>>>>> origin/master

    robust_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", RobustScaler())
    ])

    standard_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

<<<<<<< HEAD
=======
    ordinal_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder())
    ])

>>>>>>> origin/master
    onehot_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

<<<<<<< HEAD
    # --- ColumnTransformer ---

    preprocessor = ColumnTransformer(
        transformers=[
            ("robust_num", robust_pipeline, robust_numeric),
            ("standard_num", standard_pipeline, standard_numeric),
            ("onehot_cat", onehot_pipeline, onehot_cats),
        ]
    )
=======
    # === ColumnTransformer ===

    preprocessor = ColumnTransformer(transformers=[
        ("robust_num", robust_pipeline, robust_numeric),
        ("standard_num", standard_pipeline, standard_numeric),
        ("ordinal_cat", ordinal_pipeline, ordinal_cats),
        ("onehot_cat", onehot_pipeline, onehot_cats)
    ])
>>>>>>> origin/master

    return preprocessor
