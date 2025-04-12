# training_script.py
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import hopsworks

# Step 1: Connect to Hopsworks and load feature data
project = hopsworks.login(api_key_value=os.environ["HOPSWORKS_API_KEY"])
fs = project.get_feature_store()

fg = fs.get_feature_group(name="pollution_pm10_features", version=1)
df = fg.read()

# Step 2: Preprocess
df["pm10_log"] = np.log1p(df["pm10"])
features = ["pm2_5", "no2", "so2", "o3", "co", "nh3"]  # adjust as needed
X = df[features]
y = df["pm10_log"]

# Train/test split
split_idx = int(0.8 * len(df))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 3: Train Ridge Regression
ridge_model = Ridge(alpha=1.0, random_state=42)
ridge_model.fit(X_train_scaled, y_train)

# Step 4: Predict + evaluate
y_pred_log = ridge_model.predict(X_test_scaled)
y_pred_pm10 = np.expm1(y_pred_log)
y_test_pm10 = np.expm1(y_test)

rmse = np.sqrt(mean_squared_error(y_test_pm10, y_pred_pm10))
mae = mean_absolute_error(y_test_pm10, y_pred_pm10)
r2 = r2_score(y_test_pm10, y_pred_pm10)

print(f"✅ RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.2f}")

# Step 5: Save and register model
joblib.dump(ridge_model, 'ridge_model.pkl')

model_registry = project.get_model_registry()
model = model_registry.python.create_model(
    name="pm10_ridge_model",
    metrics={"rmse": rmse, "mae": mae, "r2": r2},
    description="Ridge regression model for PM10 prediction"
)
model.save("ridge_model.pkl")

