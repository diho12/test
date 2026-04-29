import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.models import Model

import matplotlib.pyplot as plt
from pathlib import Path

# =============================
# CONFIG
# =============================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

OUTPUT_DIR = PROJECT_DIR / "model" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = PROJECT_DIR / "data" / "calcofi"
DATA_PATH = DATA_DIR / "bottle_and_cast.csv"

SEQ_LEN = 12
TARGET_COL = "O2ml_L"

# =============================
# LOAD DATA
# =============================

df = pd.read_csv(DATA_PATH)

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.sort_values("Date")

# =============================
# TIME FEATURES
# =============================

df["Day"] = df["Date"].dt.month
df["Month"] = df["Date"].dt.day
df["Year"] = df["Date"].dt.year
df["DayOfYear"] = df["Date"].dt.dayofyear

# cyclic encoding
df["month_sin"] = np.sin(2*np.pi*df["Month"]/12)
df["month_cos"] = np.cos(2*np.pi*df["Month"]/12)

df["doy_sin"] = np.sin(2*np.pi*df["DayOfYear"]/365)
df["doy_cos"] = np.cos(2*np.pi*df["DayOfYear"]/365)

# =============================
# SPATIAL FEATURES
# =============================

df["lat_sin"] = np.sin(np.radians(df["Lat_Dec"]))
df["lat_cos"] = np.cos(np.radians(df["Lat_Dec"]))

df["lon_sin"] = np.sin(np.radians(df["Lon_Dec"]))
df["lon_cos"] = np.cos(np.radians(df["Lon_Dec"]))

# =============================
# PHYSICS FEATURES
# =============================

df["temp_salinity"] = df["T_degC"] * df["Salnty"]
df["temp_depth"] = df["T_degC"] * df["Depthm"]
df["salinity_depth"] = df["Salnty"] * df["Depthm"]

# density proxy
df["density_proxy"] = df["Salnty"] - 0.2 * df["T_degC"]

# =============================
# LAG FEATURES
# =============================

df["O2_lag1"] = df[TARGET_COL].shift(1)
df["O2_lag2"] = df[TARGET_COL].shift(2)
df["O2_lag3"] = df[TARGET_COL].shift(3)

# =============================
# FEATURE LIST
# =============================

FEATURE_COLS = [
    "Depthm",
    "T_degC",
    "Salnty",

    "lat_sin","lat_cos",
    "lon_sin","lon_cos",

    "month_sin","month_cos",
    "doy_sin","doy_cos",

    "temp_salinity",
    "temp_depth",
    "salinity_depth",
    "density_proxy",

    "O2_lag1",
    "O2_lag2",
    "O2_lag3"
]

df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])

print("Rows after feature engineering:", len(df))

# =============================
# SCALE
# =============================

feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

df[FEATURE_COLS] = feature_scaler.fit_transform(df[FEATURE_COLS])
df[[TARGET_COL]] = target_scaler.fit_transform(df[[TARGET_COL]])

# =============================
# BUILD SEQUENCE
# =============================

def build_sequences(df, features, target, seq_len):

    X = []
    y = []

    data = df[features + [target]].values

    for i in range(len(data) - seq_len):

        X.append(data[i:i+seq_len, :-1])
        y.append(data[i+seq_len, -1])

    return np.array(X), np.array(y)

X, y = build_sequences(df, FEATURE_COLS, TARGET_COL, SEQ_LEN)

print("X:", X.shape)
print("y:", y.shape)

# =============================
# TRAIN TEST SPLIT
# =============================

split = int(len(X) * 0.8)

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

# =============================
# LSTM MODEL
# =============================

inputs = Input(shape=(SEQ_LEN, len(FEATURE_COLS)))

x = LSTM(64, return_sequences=True)(inputs)
x = LSTM(32)(x)

embedding = Dense(16, activation="relu")(x)

output = Dense(1)(embedding)

lstm_model = Model(inputs, output)

lstm_model.compile(
    optimizer="adam",
    loss="mse"
)

print(lstm_model.summary())

# =============================
# TRAIN LSTM
# =============================

lstm_model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2
)

# =============================
# EXTRACT LSTM EMBEDDING
# =============================

feature_extractor = Model(
    inputs=lstm_model.input,
    outputs=embedding
)

train_embed = feature_extractor.predict(X_train)
test_embed = feature_extractor.predict(X_test)

print("Embedding shape:", train_embed.shape)

# =============================
# PREPARE XGBOOST INPUT
# =============================

X_train_flat = X_train.reshape(len(X_train), -1)
X_test_flat = X_test.reshape(len(X_test), -1)

X_train_final = np.concatenate([X_train_flat, train_embed], axis=1)
X_test_final = np.concatenate([X_test_flat, test_embed], axis=1)

# =============================
# XGBOOST MODEL
# =============================

xgb_model = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    n_jobs=-1,
    random_state=42
)

print("Training XGBoost...")

xgb_model.fit(X_train_final, y_train)

# =============================
# PREDICTION
# =============================

pred = xgb_model.predict(X_test_final)

pred = target_scaler.inverse_transform(pred.reshape(-1,1))
true = target_scaler.inverse_transform(y_test.reshape(-1,1))

# =============================
# EVALUATION
# =============================

mae = mean_absolute_error(true, pred)
rmse = np.sqrt(mean_squared_error(true, pred))
r2 = r2_score(true, pred)

print("\n===== HYBRID RESULT =====")

print("MAE:", mae)
print("RMSE:", rmse)
print("R2:", r2)

# =============================
# PLOT RESULT
# =============================

plt.figure(figsize=(10,5))

plt.plot(true[:200], label="True")
plt.plot(pred[:200], label="Hybrid LSTM+XGB")

plt.legend()
plt.title("Hybrid Model Prediction")

plt.show()