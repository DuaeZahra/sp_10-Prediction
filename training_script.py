# training_script.py
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import hopsworks
api_key = "Nlb2ywFqaAR7w07G.j0RReaMnKTpLSSAVhQbzEV9dqSf10BtIc1V2s1An1AUQRcUbk0C5YnNHk1c8Wfip"
project_name = "AQI_Dua"
os.environ["HOPSWORKS_API_KEY"] = api_key

# Step 1: Connect to Hopsworks and load feature data
project = hopsworks.login(api_key_value=os.environ["HOPSWORKS_API_KEY"])
fs = project.get_feature_store()

# fg = fs.get_feature_group(name="pollution_pm10_features", version=1)
print(fs.get_feature_groups())  # to see available feature groups
# clean_df = fg.read()

# Drop timestamp as it's not a feature
clean_df = clean_df.drop(columns=["timestamp"])
target = 'pm10_log'
features = ['pm10_log_rollmean3', 'pm10_log_lag1', 'pm10_log_rollmean6', 'pm2_5',
            'pm10_log_rollmean12', 'pm10_log_lag3', 'co', 'pm10_log_rollmean24',
            'aqi', 'pm10_log_lag24', 'pm10_log_lag6', 'no2', 'pm10_rolling7',
            'no', 'WindSp', 'Temp', 'Pres', 'o3', 'nh3', 'RelHumidity', 'so2']
split_ratio = 0.8
split_index = int(len(clean_df) * split_ratio)

X = clean_df[features]
y = clean_df['pm10_log']
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

ridge = Ridge(alpha=1.0, random_state=42)
ridge.fit(X_train_scaled, y_train)  # y_train should be pm10_log

y_pred_log = ridge.predict(X_test_scaled)

y_pred_pm10 = np.expm1(y_pred_log)
y_test_pm10 = np.expm1(y_test)  # y_test is still in log scale

rmse = np.sqrt(mean_squared_error(y_test_pm10, y_pred_pm10))
mae = mean_absolute_error(y_test_pm10, y_pred_pm10)
r2 = r2_score(y_test_pm10, y_pred_pm10)
accuracy = 100 - np.mean(np.abs((y_test_pm10 - y_pred_pm10) / (y_test_pm10 + 1e-6))) * 100

print("\n Ridge Regression (Original PM10 Scale):")
print(f"  - Mean Squared Error: {rmse:.2f}")
print(f"  - Mean Absolute Error: {mae:.2f}")
print(f"  - R²: {r2:.2f}")
print(f"  - Accuracy ≈ {accuracy:.2f}%")

# Step 5: Save and register model
joblib.dump(ridge, 'ridge_model.pkl')

model_registry = project.get_model_registry()
model = model_registry.python.create_model(
    name="pm10_ridge_model",
    metrics={"rmse": rmse, "mae": mae, "r2": r2},
    description="Ridge regression model for PM10 prediction"
)
model.save("ridge_model.pkl")

