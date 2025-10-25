import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta
import warnings
warnings.filterwarnings("ignore")

# -----------------------------
# STEP 1: LOAD DATA
# -----------------------------
file_path = "Oil well.xlsx"

# Try reading without skipping rows first
df = pd.read_excel(file_path)

# If columns are mostly numeric, first row might actually be the header
if all([str(c).replace('.', '', 1).isdigit() for c in df.columns]):
    df = pd.read_excel(file_path, header=None)
    df.columns = df.iloc[0]  # Use first row as header
    df = df.drop(0).reset_index(drop=True)

# Clean up column names
df.columns = df.columns.astype(str).str.strip().str.replace('"', '', regex=False).str.replace('\n', '', regex=False)

print("📋 Columns in your file after cleanup:")
print(df.columns.tolist())

# Try to detect the date column automatically
possible_date_cols = [c for c in df.columns if 'date' in c.lower() or 'prd' in c.lower()]
if possible_date_cols:
    date_col = possible_date_cols[0]
else:
    # Fall back to first column
    date_col = df.columns[0]
    print(f"⚠️ Using '{date_col}' as date column (auto-detected)")

df.rename(columns={date_col: "DATEPRD"}, inplace=True)

# Rename other columns if names partially match known terms
rename_map = {}
for col in df.columns:
    col_low = col.lower()
    if "oil" in col_low:
        rename_map[col] = "BORE_OIL_VOL"
    elif "gas" in col_low:
        rename_map[col] = "GAS_VOL"
    elif "water cut" in col_low or ("water" in col_low and "%" in col_low):
        rename_map[col] = "WATER_CUT"
    elif "water" in col_low:
        rename_map[col] = "WATER_VOL"
    elif "pressure" in col_low:
        rename_map[col] = "RES_PRESS"
    elif "dynamic" in col_low or "dyn" in col_low:
        rename_map[col] = "DYN_LEVEL"
    elif "work" in col_low:
        rename_map[col] = "WORK_HOURS"
    elif "liquid" in col_low or "liq" in col_low:
        rename_map[col] = "LIQ_VOL"

df.rename(columns=rename_map, inplace=True)

# Convert date
df["DATEPRD"] = pd.to_datetime(df["DATEPRD"], errors="coerce")

# Drop invalid rows
df = df.dropna(subset=["DATEPRD"]).sort_values("DATEPRD")

# If target column missing, take second column as fallback
if "BORE_OIL_VOL" not in df.columns:
    print("⚠️ 'BORE_OIL_VOL' not found. Using first numeric column as target.")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        df.rename(columns={num_cols[0]: "BORE_OIL_VOL"}, inplace=True)

print("\n✅ Data loaded successfully:")
print(df.head())

# -----------------------------
# STEP 2: SELECT FEATURES
# -----------------------------
target_col = "BORE_OIL_VOL"

feature_cols = [
    "BORE_OIL_VOL",
    "GAS_VOL",
    "WATER_VOL",
    "WATER_CUT",
    "RES_PRESS",
    "DYN_LEVEL",
    "WORK_HOURS",
    "LIQ_VOL"
]

# Keep only columns that exist
feature_cols = [c for c in feature_cols if c in df.columns]
if target_col not in df.columns:
    raise ValueError("❌ Target column 'BORE_OIL_VOL' not found in dataset.")

lookback = 30
forecast_days = 30

# Scale features
scalers = {}
scaled_df = pd.DataFrame(index=df.index)
for col in feature_cols:
    scaler = MinMaxScaler()
    scaled_df[col] = scaler.fit_transform(df[[col]].fillna(method='ffill')).ravel()
    scalers[col] = scaler

# -----------------------------
# STEP 3: CREATE SEQUENCES
# -----------------------------
X, y = [], []
for i in range(lookback, len(scaled_df)):
    X.append(scaled_df[feature_cols].iloc[i - lookback:i].values)
    y.append(scaled_df[target_col].iloc[i])

X, y = np.array(X), np.array(y)
print(f"\n✅ Created sequences: X shape = {X.shape}, y shape = {y.shape}")

# -----------------------------
# STEP 4: TRAIN/TEST SPLIT
# -----------------------------
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# -----------------------------
# STEP 5: BUILD LSTM
# -----------------------------
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(64, return_sequences=False),
    Dense(32, activation='relu'),
    Dense(1)
])
model.compile(optimizer="adam", loss="mse")
early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

# -----------------------------
# STEP 6: TRAIN
# -----------------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=16,
    callbacks=[early_stop],
    verbose=1
)

# -----------------------------
# STEP 7: PREDICT TEST
# -----------------------------
pred_scaled = model.predict(X_test)
pred = scalers[target_col].inverse_transform(pred_scaled)
actual = scalers[target_col].inverse_transform(y_test.reshape(-1, 1))

plt.figure(figsize=(10,5))
plt.plot(df["DATEPRD"].iloc[-len(y_test):], actual, label="Actual")
plt.plot(df["DATEPRD"].iloc[-len(pred):], pred, label="Predicted", color="orange")
plt.title("Oil Production Prediction (Test Set)")
plt.xlabel("Date")
plt.ylabel("Oil Volume (m³/day)")
plt.legend()
plt.show()

# -----------------------------
# STEP 8: FORECAST FUTURE
# -----------------------------
last_seq = scaled_df[feature_cols].iloc[-lookback:].values
current_seq = last_seq.copy()
future_predictions = []

for _ in range(forecast_days):
    pred = model.predict(current_seq.reshape(1, lookback, len(feature_cols)))[0][0]
    future_predictions.append(pred)
    new_row = current_seq[-1].copy()
    new_row[feature_cols.index(target_col)] = pred
    current_seq = np.vstack((current_seq[1:], new_row))

future_predictions = scalers[target_col].inverse_transform(
    np.array(future_predictions).reshape(-1, 1)
)

last_date = df["DATEPRD"].iloc[-1]
future_dates = [last_date + timedelta(days=i+1) for i in range(forecast_days)]

forecast_df = pd.DataFrame({
    "DATEPRD": future_dates,
    "FORECAST_OIL_VOL_m3_day": future_predictions.flatten()
})

print("\n📈 Next 30-day Forecast:")
print(forecast_df.head())

plt.figure(figsize=(10,5))
plt.plot(df["DATEPRD"], df["BORE_OIL_VOL"], label="Historical")
plt.plot(forecast_df["DATEPRD"], forecast_df["FORECAST_OIL_VOL_m3_day"],
         label="Forecast", color="orange")
plt.title("Future Oil Production Forecast")
plt.xlabel("Date")
plt.ylabel("Oil Volume (m³/day)")
plt.legend()
plt.show()

forecast_df.to_csv("oil_forecast_results.csv", index=False)
print("\n✅ Forecast saved to 'oil_forecast_results.csv'")
