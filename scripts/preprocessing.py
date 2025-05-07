import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_and_scale(csv_path, label_col="Label"):
    """
    Read data from CSV, split into X, y, and apply MinMax Scaling.
    """
    df = pd.read_csv(csv_path)
    X = df.drop(columns=[label_col]).values
    y = df[label_col].values
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y
