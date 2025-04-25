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
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# ------------------------
# 2. Data Preparation
# ------------------------

# Retrieve feature group (your starting code)
fg = fs.get_feature_group("pollution_pm10_features_git", version=1)
uploaded_df = fg.read()

print(f"\nNumber of records in feature store (before cleaning): {len(uploaded_df)}")

uploaded_df = uploaded_df.drop(columns=["timestamp"])
target = 'pm10_log'
features = ['pm10_log_rollmean3', 'pm10_log_lag1', 'pm10_log_rollmean6', 'pm2_5',
            'pm10_log_rollmean12', 'pm10_log_lag3', 'co', 'pm10_log_rollmean24',
            'aqi', 'pm10_log_lag24', 'pm10_log_lag6', 'no2', 'pm10_rolling7',
            'no', 'windsp', 'temp', 'pres', 'o3', 'nh3', 'relhumidity', 'so2']

uploaded_df = uploaded_df.dropna().reset_index(drop=True)
print(f"\nNumber of records after cleaning: {len(uploaded_df)}")

# Prepare X and y
X = uploaded_df[features].values
y = uploaded_df[target].values.reshape(-1, 1)  # make it 2D

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
# Here, seq_length = 1 (because no actual sequences given), but we can reshape to (batch_size, 1, features)
X_train_scaled = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
X_test_scaled = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])

# Convert to Tensors
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32).to(device)

# ------------------------
# 3. Model Definition
# ------------------------

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # get last output
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out

input_size = len(features)  # 20 features
model = LSTMModel(input_size=input_size).to(device)

criterion = nn.SmoothL1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ------------------------
# 4. Training Loop
# ------------------------

num_epochs = 80
train_losses = []

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    loss.backward()
    optimizer.step()

    train_losses.append(loss.item())

    if (epoch+1) % 10 == 0:
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')

# Plot training loss
plt.figure(figsize=(8,5))
plt.plot(train_losses)
plt.title("Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
plt.show()

# ------------------------
# 5. Evaluation
# ------------------------

model.eval()
with torch.no_grad():
    y_pred_scaled = model(X_test_tensor).cpu().numpy()
    y_test_scaled = y_test_tensor.cpu().numpy()

# Inverse scale the predictions
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test = scaler_y.inverse_transform(y_test_scaled)

# Inverse log (expm1)
y_pred_final = np.expm1(y_pred)
y_test_final = np.expm1(y_test)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Calculate the numerical metrics
mae = mean_absolute_error(y_test_final, y_pred_final)
mse = mean_squared_error(y_test_final, y_pred_final)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_final, y_pred_final)

import joblib
import os
import hopsworks

# Save the model and scaler separately
os.makedirs("pm10_model_dir", exist_ok=True)

# Save the trained LSTM model (using torch's state_dict to save the model)
torch.save(model.state_dict(), "pm10_model_dir/model.pth")

# Save the scalers
joblib.dump(scaler_X, "pm10_model_dir/scaler_X.pkl")
joblib.dump(scaler_y, "pm10_model_dir/scaler_y.pkl")

model_registry = project.get_model_registry()

# STEP 1: Create model metadata
lstm_model = model_registry.python.create_model(
    name="pm10_LSTM_model",
    metrics={"rmse": rmse, "mae": mae, "r2": r2},
    description="LSTM model for PM10 prediction"
)

# STEP 2: Upload model files (just point to the directory where files are saved)
lstm_model.save("pm10_model_dir")
