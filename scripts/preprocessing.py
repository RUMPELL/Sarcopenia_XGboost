import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def make_scaler():
    """The scaler used throughout the pipeline (Min-Max to [0, 1])."""
    return MinMaxScaler()


def load_raw(csv_path, label_col="Label"):
    """Read data from CSV and split into an unscaled feature matrix X and labels y."""
    df = pd.read_csv(csv_path)
    X = df.drop(columns=[label_col]).values
    y = df[label_col].values
    return X, y


def load_and_scale(csv_path, label_col="Label"):
    """Read data from CSV, split into X, y, and apply Min-Max scaling to the whole file.

    Convenience helper for whole-dataset exploration. The training pipeline
    scales inside each fold via `scripts.fold_preprocessing.FoldPreprocessor`.
    """
    X, y = load_raw(csv_path, label_col)
    X_scaled = make_scaler().fit_transform(X)
    return X_scaled, y
