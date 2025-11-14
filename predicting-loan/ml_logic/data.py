import os
import warnings
warnings.filterwarnings("ignore")

import kagglehub
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
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score,
    confusion_matrix, RocCurveDisplay, roc_auc_score
)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:

    print("Data loaded. Shape:", df.shape)

    # ----------------------------------------------------
    # 2. TARGET + DROP USELESS COLUMNS
    # ----------------------------------------------------

    df = df.drop(columns=["LoanID"])     # ID column not useful

    # Drop rows with missing Default
    df = df.dropna(subset=['Default'])

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



    print("✅ data cleaned")

    return df


BASE_DIR = os.path.dirname(os.path.abspath('raw_data'))  # folder where script is
DATA_PATH = os.path.join(BASE_DIR,  "raw_data", "Loan_default.csv")
df = pd.read_csv(DATA_PATH)

clean_data (df)
