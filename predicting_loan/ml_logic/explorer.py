import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def explore_data(df):
    print("\n--- Basic Info ---")
    print(df.info())

    print("\n--- First 5 Rows ---")
    print(df.head())

    print("\n--- Missing Values ---")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.any() else "No missing values.")

    print("\n--- Summary Stats ---")
    print(df.describe(include='all').T)

    print("\n--- Correlation Heatmap ---")
    num = df.select_dtypes(include=["number"])
    if not num.empty:
        sns.heatmap(num.corr(), annot=True, cmap="coolwarm")
        plt.title("Correlation Matrix")
        plt.tight_layout()
        plt.show()
    else:
        print("No numeric columns to show.")

    if "Default" in df.columns:
        print("\n--- Target Distribution (Default) ---")
        df["Default"].value_counts(normalize=True).plot(kind="bar")
        plt.title("Default Class Balance")
        plt.xlabel("Class")
        plt.ylabel("Proportion")
        plt.grid(axis="y")
        plt.tight_layout()
        plt.show()


def plot_default_by_category(df, cat_col):
    if "Default" not in df.columns or cat_col not in df.columns:
        return
    pd.crosstab(df[cat_col], df["Default"], normalize='index').plot(kind='bar', stacked=True)
    plt.title(f"Default Rate by {cat_col}")
    plt.ylabel("Proportion")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import pandas as pd

    df = pd.read_csv("raw_data/Loan_default.csv")
    plot_default_by_category(df,"Education")
    explore_data(df)
