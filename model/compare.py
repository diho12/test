import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.models import Model

from pathlib import Path

# ==========================
# CONFIG
# ==========================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

DATA_PATH = PROJECT_DIR / "data/calcofi/bottle_and_cast.csv"

SEQ_LEN = 6
TARGET_COL = "O2ml_L"

FEATURE_COLS = [
    "Depthm",
    "T_degC",
    "Salnty",
    "Lat_Dec",
    "Lon_Dec",
]

# ==========================
# LOAD DATA
# ==========================

df = pd.read_csv(DATA_PATH)

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.sort_values("Date")

# time features
df["month"] = df["Date"].dt.month
df["month_sin"] = np.sin(2*np.pi*df["month"]/12)
df["month_cos"] = np.cos(2*np.pi*df["month"]/12)

FEATURE_COLS += ["month_sin","month_cos"]

df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])

print("Rows:", len(df))

# ==========================
# SCALE
# ==========================

feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

df[FEATURE_COLS] = feature_scaler.fit_transform(df[FEATURE_COLS])
df[[TARGET_COL]] = target_scaler.fit_transform(df[[TARGET_COL]])

# ==========================
# BUILD SEQUENCE BY STATION
# ==========================

def build_sequences(df):

    X_seq = []
    X_tab = []
    y = []

    for (_, depth), group in df.groupby(["Sta_ID","Depthm"]):

        group = group.sort_values("Date")

        data = group[FEATURE_COLS + [TARGET_COL]].values

        if len(data) < SEQ_LEN:
            continue

        for i in range(len(data) - SEQ_LEN):

            X_seq.append(data[i:i+SEQ_LEN, :-1])
            X_tab.append(data[i+SEQ_LEN, :-1])
            y.append(data[i+SEQ_LEN, -1])

    return np.array(X_seq), np.array(X_tab), np.array(y)


X_seq, X_tab, y = build_sequences(df)

print("Sequence shape:", X_seq.shape)
print("Tabular shape:", X_tab.shape)

# ==========================
# TRAIN TEST SPLIT
# ==========================

split = int(len(X_seq) * 0.8)

X_seq_train = X_seq[:split]
X_seq_test = X_seq[split:]

X_tab_train = X_tab[:split]
X_tab_test = X_tab[split:]

y_train = y[:split]
y_test = y[split:]

# ==========================
# BASELINE XGBOOST
# ==========================

xgb_base = xgb.XGBRegressor(
    n_estimators=800,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    n_jobs=-1
)

print("Training XGBoost baseline...")

xgb_base.fit(X_tab_train, y_train)

pred_base = xgb_base.predict(X_tab_test)

# ==========================
# LSTM EMBEDDING
# ==========================

inputs = Input(shape=(SEQ_LEN, len(FEATURE_COLS)))

x = LSTM(32)(inputs)

embedding = Dense(8, activation="relu")(x)

output = Dense(1)(embedding)

lstm_model = Model(inputs, output)

lstm_model.compile(
    optimizer="adam",
    loss="mse"
)

print("Training LSTM...")

lstm_model.fit(
    X_seq_train,
    y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.2,
    verbose=1
)

# extract embedding

feature_extractor = Model(
    inputs=lstm_model.input,
    outputs=embedding
)

train_embed = feature_extractor.predict(X_seq_train)
test_embed = feature_extractor.predict(X_seq_test)

# ==========================
# HYBRID MODEL
# ==========================

X_train_hybrid = np.concatenate([X_tab_train, train_embed], axis=1)
X_test_hybrid = np.concatenate([X_tab_test, test_embed], axis=1)

xgb_hybrid = xgb.XGBRegressor(
    n_estimators=800,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    n_jobs=-1
)

print("Training Hybrid XGBoost...")

xgb_hybrid.fit(X_train_hybrid, y_train)

pred_hybrid = xgb_hybrid.predict(X_test_hybrid)

# ==========================
# EVALUATION
# ==========================

true = target_scaler.inverse_transform(y_test.reshape(-1,1))
pred_base = target_scaler.inverse_transform(pred_base.reshape(-1,1))
pred_hybrid = target_scaler.inverse_transform(pred_hybrid.reshape(-1,1))


def evaluate(name, y_true, y_pred):

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"\n{name}")
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R2:", r2)


print("\n===== RESULTS =====")

evaluate("XGBoost baseline", true, pred_base)
evaluate("XGBoost + LSTM embedding", true, pred_hybrid)