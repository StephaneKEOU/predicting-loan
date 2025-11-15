import pandas as pd
import os

def load_data(path: str = "raw_data/Loan_default.csv"):

    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find data at: {path}")

    df = pd.read_csv(path)
    return df




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
