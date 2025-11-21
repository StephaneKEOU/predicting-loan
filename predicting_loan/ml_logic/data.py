import os
import pandas as pd
import kagglehub


def load_data(path: str = "raw_data/Loan_default.csv"):
    if os.path.exists(path):
        return pd.read_csv(path)

    print(f"Local file not found at {path}. Pulling dataset from Kaggle...")

    kaggle_path = kagglehub.dataset_download("nikhil1e9/loan-default")


    csv_files = [
        f for f in os.listdir(kaggle_path)
        if f.lower().endswith(".csv")
    ]

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV file found inside Kaggle dataset folder: {kaggle_path}"
        )

    kaggle_csv = os.path.join(kaggle_path, csv_files[0])

    print(f"Using Kaggle dataset file: {kaggle_csv}")

    return pd.read_csv(kaggle_csv)





class SimpleMissingHandler:
    def __init__(self, how="auto", constant_value=None):

        self.how = how
        self.constant_value = constant_value

    def check_missing(self, df):
        """Prints missing values per column (if any)."""
        missing = df.isnull().sum()
        total_missing = missing.sum()

        if total_missing == 0:
            print("No missing values in the dataset.")
        else:
            print("Missing values found:")
            print(missing[missing > 0])

    def fix_missing(self, df):
        """Returns a new DataFrame with missing values handled based on the method."""
        df = df.copy()

        if self.how == "drop":
            return df.dropna()

        for col in df.columns:
            if df[col].isnull().any():
                if self.how == "mean":
                    df[col] = df[col].fillna(df[col].mean())
                elif self.how == "median":
                    df[col] = df[col].fillna(df[col].median())
                elif self.how == "mode":
                    df[col] = df[col].fillna(df[col].mode()[0])
                elif self.how == "constant":
                    df[col] = df[col].fillna(self.constant_value)
                elif self.how == "auto":
                    if df[col].dtype == "object":
                        df[col] = df[col].fillna(df[col].mode()[0])
                    else:
                        df[col] = df[col].fillna(df[col].median())

        return df
