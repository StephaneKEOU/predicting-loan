import pandas as pd
from predicting_loan.ml_logic.data import load_data, SimpleMissingHandler
from predicting_loan.ml_logic.preprocessor import preprocess_data

def main():
    # 1. Load data (local or Kaggle)
    df = load_data()

    # 2. Handle missing values
    handler = SimpleMissingHandler(how="auto")
    df = handler.fix_missing(df)

    # 3. Drop ID column
    df = df.drop(columns=["LoanID"])

    # 4. Convert Yes/No -> 0/1 for binary cols
    binary_cols = ["HasMortgage", "HasDependents", "HasCoSigner"]
    for col in binary_cols:
        df[col] = df[col].map({"Yes": 1, "No": 0})

    # 5. Split features / target
    X = df.drop(columns=["Default"])

    # 6. Build and fit-transform preprocessor
    preprocessor = preprocess_data(X)
    Xt = preprocessor.fit_transform(X)

    print("Preprocessor ran successfully with SimpleMissingHandler")
    print("Shape:", Xt.shape)
    print("Type:", type(Xt))

if __name__ == "__main__":
    main()
