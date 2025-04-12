# feature_script.py
import requests
import pandas as pd
from datetime import datetime, timedelta
import hopsworks
from hsml.schema import Schema
from hsml.feature_group import FeatureGroup
from meteostat import Point, Hourly
import time

CITY = "Lahore"
LAT, LON = 31.5497, 74.3436
API_KEY = "99150b830abb62c066787c5a95ff9ad9"
TOTAL_DAYS = 180
CHUNK_SIZE = 5

# Step 1: Fetch historical air pollution data
end_time = datetime.utcnow()
all_pollution_records = []

for _ in range(0, TOTAL_DAYS, CHUNK_SIZE):
    chunk_end = int(end_time.timestamp())
    chunk_start = int((end_time - timedelta(days=CHUNK_SIZE)).timestamp())

    print(f"[Pollution] Fetching from {datetime.utcfromtimestamp(chunk_start)} to {datetime.utcfromtimestamp(chunk_end)}")

    url = (
        f"http://api.openweathermap.org/data/2.5/air_pollution/history"
        f"?lat={LAT}&lon={LON}&start={chunk_start}&end={chunk_end}&appid={API_KEY}"
    )

    response = requests.get(url)
    if response.status_code != 200:
        print(f"[Pollution] Failed to fetch data for chunk starting {datetime.utcfromtimestamp(chunk_start)}")
        end_time -= timedelta(days=CHUNK_SIZE)
        time.sleep(1)
        continue

    data = response.json()
    for item in data.get("list", []):
        components = item["components"]
        all_pollution_records.append({
            "timestamp": datetime.utcfromtimestamp(item["dt"]),
            "pm2_5": components.get("pm2_5"),
            "pm10": components.get("pm10"),
            "no2": components.get("no2"),
            "no": components.get("no"),
            "so2": components.get("so2"),
            "o3": components.get("o3"),
            "co": components.get("co"),
            "nh3": components.get("nh3"),
            "aqi": item["main"]["aqi"]
        })

    end_time -= timedelta(days=CHUNK_SIZE)
    time.sleep(1)

df_pollution = pd.DataFrame(all_pollution_records)
df_pollution['timestamp'] = pd.to_datetime(df_pollution['timestamp'])
df_pollution = df_pollution.sort_values("timestamp")

# Step 2: Fetch weather data using Meteostat
start_date = df_pollution['timestamp'].min()
end_date = df_pollution['timestamp'].max()

lahore = Point(LAT, LON)
print(f"[Weather] Fetching from {start_date} to {end_date}")
weather_data = Hourly(lahore, start_date, end_date).fetch()

if weather_data.empty:
    raise ValueError("No weather data returned from Meteostat. Try changing the date range or check Meteostat availability.")

# Preprocess weather data
weather_data = weather_data.reset_index()
weather_data = weather_data.rename(columns={"time": "timestamp"})

# Filter and rename relevant weather features
weather_data = weather_data[[
    "timestamp", "temp", "dwpt", "rhum", "wspd", "pres"
]].rename(columns={
    "temp": "Temp",
    "dwpt": "DewPoint",
    "rhum": "RelHumidity",
    "wspd": "WindSp",
    "pres": "Pres"
})

# Step 3: Merge pollution and weather data on timestamp
df_pollution = pd.merge_asof(
    df_pollution.sort_values("timestamp"),
    weather_data.sort_values("timestamp"),
    on="timestamp",
    direction="nearest",
    tolerance=pd.Timedelta("1h")
)

import os

api_key = "Nlb2ywFqaAR7w07G.j0RReaMnKTpLSSAVhQbzEV9dqSf10BtIc1V2s1An1AUQRcUbk0C5YnNHk1c8Wfip"
project_name = "AQI_Dua"

os.environ["HOPSWORKS_API_KEY"] = api_key

project = hopsworks.login(api_key_value=os.environ["HOPSWORKS_API_KEY"], project=project_name)
fs = project.get_feature_store()

# Create or get feature group
feature_group = fs.get_or_create_feature_group(
    name="pollution_pm10_features_git",
    version=1,
    description="PM10 and weather features hourly",
    primary_key=["timestamp"],
    event_time="timestamp"
)

df_pollution["timestamp"] = pd.to_datetime(df_pollution["timestamp"])
df_pollution = df_pollution.sort_values(by='timestamp')
df_pollution = df_pollution.reset_index(drop=True)
df_pollution['pm10_log'] = np.log1p(df_pollution['pm10'])
df_pollution["hour"] = df_pollution["timestamp"].dt.hour
df_pollution['month'] = df_pollution['timestamp'].dt.month
df_pollution["dayofweek"] = df_pollution["timestamp"].dt.dayofweek
df_pollution["is_weekend"] = df_pollution["dayofweek"].isin([5, 6]).astype(int)

lags = [1, 3, 6, 24]               # in hours
windows = [3, 6, 12, 24]           # rolling window (hourly basis)
for lag in lags:
    df_pollution[f"pm10_log_lag{lag}"] = df_pollution["pm10_log"].shift(lag)
for window in windows:
    df_pollution[f"pm10_log_rollmean{window}"] = df_pollution["pm10_log"].rolling(window=window).mean()
clean_df = df_pollution.dropna().reset_index(drop=True)
clean_df = clean_df.sort_values(by='timestamp')

# Insert the merged data into the feature group
feature_group.insert(clean_df, write_options={"wait_for_job": True})
