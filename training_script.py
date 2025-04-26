import os
import hopsworks
api_key = "Nlb2ywFqaAR7w07G.j0RReaMnKTpLSSAVhQbzEV9dqSf10BtIc1V2s1An1AUQRcUbk0C5YnNHk1c8Wfip"
project_name = "AQI_Dua"
os.environ["HOPSWORKS_API_KEY"] = api_key

# Step 1: Connect to Hopsworks and load feature data
project = hopsworks.login(api_key_value=os.environ["HOPSWORKS_API_KEY"])
fs = project.get_feature_store()

# ------------------------
# 1. Imports
# ------------------------
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, ReLU
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# ------------------------
# 2. Data Preparation
# ------------------------
# Retrieve feature group (your starting code)
fg = fs.get_feature_group("pollution_pm10_features_git", version=1)
uploaded_df = fg.read()

print(f"\nNumber of records in feature store (before cleaning): {len(uploaded_df)}")

uploaded_df = uploaded_df.drop(columns=["timestamp"])
# Assuming uploaded_df is already loaded and cleaned as in your original code
# Example for how the dataframe looks after cleaning (use your dataset here)
uploaded_df = uploaded_df.dropna().reset_index(drop=True)
target = 'pm10_log'
features = ['pm10_log_rollmean3', 'pm10_log_lag1', 'pm10_log_rollmean6', 'pm2_5',
            'pm10_log_rollmean12', 'pm10_log_lag3', 'co', 'pm10_log_rollmean24',
            'aqi', 'pm10_log_lag24', 'pm10_log_lag6', 'no2', 'pm10_rolling7',
            'no', 'windsp', 'temp', 'pres', 'o3', 'nh3', 'relhumidity', 'so2']

X = uploaded_df[features].values
y = uploaded_df[target].values.reshape(-1, 1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Scaling Features
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# Scaling Target
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# Reshape for LSTM: (batch_size, seq_length, input_size)
X_train_scaled = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
X_test_scaled = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])

# ------------------------
# 3. Model Definition
# ------------------------

model = Sequential()
model.add(LSTM(64, input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2]), return_sequences=False))
model.add(ReLU())
model.add(Dense(32))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# ------------------------
# 4. Training
# ------------------------

history = model.fit(X_train_scaled, y_train_scaled, epochs=80, batch_size=32, validation_data=(X_test_scaled, y_test_scaled))

# Plot training loss
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Training Loss')
plt.title("Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
plt.show()

# ------------------------
# 5. Evaluation
# ------------------------

# Predictions
y_pred_scaled = model.predict(X_test_scaled)

# Inverse scale the predictions
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test = scaler_y.inverse_transform(y_test_scaled)

# Inverse log (expm1)
y_pred_final = np.expm1(y_pred)
y_test_final = np.expm1(y_test)

# Calculate the numerical metrics
mae = mean_absolute_error(y_test_final, y_pred_final)
mse = mean_squared_error(y_test_final, y_pred_final)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_final, y_pred_final)

# Print the evaluation metrics
print("\nLSTM Model Evaluation Metrics (TensorFlow):")
print(f"  - Mean Absolute Error (MAE): {mae:.2f}")
print(f"  - Mean Squared Error (MSE): {mse:.2f}")
print(f"  - Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"  - R-squared (R2 Score): {r2:.3f}")

# Plot Predictions vs True
plt.figure(figsize=(10,6))
plt.plot(y_test_final[:100], label='True PM10')
plt.plot(y_pred_final[:100], label='Predicted PM10')
plt.title('PM10 Predictions vs True Values')
plt.xlabel('Time Step')
plt.ylabel('PM10')
plt.legend()
plt.grid()
plt.show()

import joblib
import os
import hopsworks

# Save the model and scalers separately
os.makedirs("pm10_model_dir", exist_ok=True)

# Save the trained LSTM model using TensorFlow's own saving method
model.save("pm10_model_dir/model.h5")  # Save in HDF5 format

# Save the scalers
joblib.dump(scaler_X, "pm10_model_dir/scaler_X.pkl")
joblib.dump(scaler_y, "pm10_model_dir/scaler_y.pkl")

# Login to Hopsworks and get the model registry
project = hopsworks.login()
model_registry = project.get_model_registry()

# STEP 1: Create model metadata
lstm_model = model_registry.python.create_model(
    name="pm10_LSTM_model",
    metrics={"rmse": rmse, "mae": mae, "r2": r2},
    description="LSTM model for PM10 prediction"
)

# STEP 2: Upload model files (just point to the directory where files are saved)
lstm_model.save("pm10_model_dir")
