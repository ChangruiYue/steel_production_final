# scripts/02_data_preprocessing.py

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


def remove_duplicates(df):
    """
    Remove duplicate records
    """
    before = df.shape[0]
    df = df.drop_duplicates()
    after = df.shape[0]
    print(f"Duplicate rows processed: removed {before - after} rows")
    return df


def handle_missing_values(df, strategy="mean"):
    """
    Handle missing values (mean / median imputation)
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    if strategy == "mean":
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif strategy == "median":
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    else:
        raise ValueError("strategy must be 'mean' or 'median'")

    print("Missing value handling completed (numeric features)")
    return df


class DetectOutliersIQR(BaseEstimator, TransformerMixin):
    """
    Detect outliers using the IQR method and replace them
    with the median learned from the training set
    sklearn-style Transformer
    """

    def __init__(self, factor=1.5):
        """
        Parameters
        ----------
        factor : float
            IQR multiplier, default is 1.5
        """
        self.factor = factor

    def fit(self, X, y=None):
        """
        Learn IQR boundaries and medians from the training data
        """
        X = pd.DataFrame(X)
        self.stats_ = {}

        for col in X.columns:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1

            lower = Q1 - self.factor * IQR
            upper = Q3 + self.factor * IQR
            median = X[col].median()

            self.stats_[col] = {
                "lower": lower,
                "upper": upper,
                "median": median
            }

        return self  # sklearn requires fit to return self

    def transform(self, X):
        """
        Replace outliers using statistics learned from the training set
        """
        if not hasattr(self, "stats_"):
            raise RuntimeError("fit() must be called before transform()")

        X = pd.DataFrame(X).copy()

        for col, s in self.stats_.items():
            mask = (X[col] < s["lower"]) | (X[col] > s["upper"])
            X.loc[mask, col] = s["median"]

        return X


def encode_categorical_variables(df):
    """
    Convert categorical variables to numerical variables
    using One-Hot Encoding
    """
    categorical_cols = df.select_dtypes(include=["object"]).columns

    if len(categorical_cols) > 0:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        print(f"Categorical variables encoded: {list(categorical_cols)}")
    else:
        print("No categorical variables detected")

    return df


def check_data_consistency(df):
    """
    Check data consistency
    """
    print("========== Data Consistency Check ==========")
    print(f"Data shape: {df.shape}")
    print("Data types:")
    print(df.dtypes.value_counts())


if __name__ == "__main__":
    pass