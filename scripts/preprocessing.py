import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_and_scale(csv_path, label_col="Label"):
    """
    CSV에서 데이터를 읽고 X, y로 분리한 뒤 MinMax Scaling을 적용합니다.
    """
    df = pd.read_csv(csv_path)
    X = df.drop(columns=[label_col]).values
    y = df[label_col].values
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y
