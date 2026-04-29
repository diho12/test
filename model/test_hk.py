import pandas as pd
import numpy as np
import xgboost as xgb
import torch
import torch.nn as nn

from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error


# =========================
# FEATURES
# =========================

FEATURES = [
    'DO', 'Temperature', 'pH', 'Salinity',
    'NH3', 'PO4', 'BOD5', 'COD',
    'TSS', 'Coliform', 'Alkalinity', 'Transparency'
]


# =========================
# LOAD + CLEAN
# =========================

def load_data(csv_path):

    df = pd.read_csv(csv_path)

    df = df[['Station', 'Quarter'] + FEATURES]

    df['Date'] = pd.to_datetime(df['Quarter'])

    df = df.sort_values(['Station', 'Date'])

    df[FEATURES] = df.groupby('Station')[FEATURES].transform(
        lambda x: x.interpolate().fillna(x.median())
    )

    return df


# =========================
# OLD MODEL
# =========================

def prepare_lag(df, lags=[1, 4]):

    df = df.copy()

    for col in FEATURES:
        for lag in lags:
            df[f"{col}_lag{lag}"] = df.groupby(
                'Station'
            )[col].shift(lag)

    df['quarter'] = df['Date'].dt.quarter

    df = df.dropna()

    lag_cols = [
        c for c in df.columns
        if "lag" in c
    ]

    X = df[lag_cols + ['quarter']].values

    y = df[FEATURES].values

    return X, y


def train_old_model(X, y):

    model = MultiOutputRegressor(

        xgb.XGBRegressor(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:squarederror'
        )
    )

    model.fit(X, y)

    pred = model.predict(X)

    rmse = np.sqrt(
        mean_squared_error(
            y,
            pred,
            multioutput="raw_values"
        )
    )

    return rmse


# =========================
# LSTM
# =========================

class LSTMEncoder(nn.Module):

    def __init__(self, input_size, hidden=32):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size,
            hidden,
            batch_first=True
        )

    def forward(self, x):

        out, (h, c) = self.lstm(x)

        return h[-1]


# =========================
# SEQUENCE
# =========================

def prepare_seq(df, seq_len=4):

    seqs = []
    targets = []
    quarters = []

    for st, g in df.groupby("Station"):

        g = g.reset_index(drop=True)

        values = g[FEATURES].values

        for i in range(seq_len, len(g)):

            seqs.append(
                values[i-seq_len:i]
            )

            targets.append(
                values[i]
            )

            quarters.append(
                g.loc[i, 'Date'].quarter
            )

    X = np.array(seqs)
    y = np.array(targets)
    q = np.array(quarters).reshape(-1, 1)

    return X, y, q


# =========================
# TRAIN LSTM
# =========================

def train_lstm(X, epochs=30):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    X_t = torch.tensor(
        X,
        dtype=torch.float32
    ).to(device)

    model = LSTMEncoder(
        X.shape[2]
    ).to(device)

    opt = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    for e in range(epochs):

        emb = model(X_t)

        loss = emb.pow(2).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

    return model


def get_emb(model, X):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    X_t = torch.tensor(
        X,
        dtype=torch.float32
    ).to(device)

    with torch.no_grad():

        e = model(X_t)

    return e.cpu().numpy()


# =========================
# NEW MODEL
# =========================

def train_new_model(X_seq, y, q):

    lstm = train_lstm(X_seq)

    emb = get_emb(
        lstm,
        X_seq
    )

    X = np.hstack([emb, q])

    model = MultiOutputRegressor(

        xgb.XGBRegressor(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
        )
    )

    model.fit(X, y)

    pred = model.predict(X)

    rmse = np.sqrt(
        mean_squared_error(
            y,
            pred,
            multioutput="raw_values"
        )
    )

    return rmse


# =========================
# MAIN
# =========================

if __name__ == "__main__":

    CSV = "hk_cobia_quarterly_21vars.csv"

    df = load_data(CSV)

    # OLD

    X_old, y_old = prepare_lag(df)

    rmse_old = train_old_model(
        X_old,
        y_old
    )

    # NEW

    X_seq, y_seq, q = prepare_seq(df)

    rmse_new = train_new_model(
        X_seq,
        y_seq,
        q
    )

    print("\n==== RESULT ====\n")

    for i, f in enumerate(FEATURES):

        print(
            f,
            "old:", round(rmse_old[i], 3),
            "new:", round(rmse_new[i], 3),
        )

    print(
        "\nMean RMSE old:",
        rmse_old.mean()
    )

    print(
        "Mean RMSE new:",
        rmse_new.mean()
    )