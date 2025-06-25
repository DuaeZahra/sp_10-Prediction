# sp_10-Prediction

**Real-time PM10 Prediction Pipeline**
A complete machine learning pipeline for forecasting PM10 (particulate matter) air quality levels using historical and real-time pollution & weather data. Built during a Data Science Internship at 10Pearls Pakistan.

**Project Overview**
This project predicts PM10 levels in Lahore using time-series forecasting techniques. It collects live data, engineers features, trains an LSTM model, and displays predictions through a web app.

**Tech Stack**
Languages: Python
ML Models: LSTM, Random Forest, XGBoost, Ridge
Libraries: Pandas, NumPy, Scikit-learn, TensorFlow, Matplotlib, Seaborn
MLOps Tools: Hopsworks Feature Store, GitHub Actions (CI/CD)
Data Sources: OpenWeatherMap API, Meteostat API

**Features**
Real-time data fetching from APIs
Time-series feature engineering (lag, rolling, temporal encodings)
LSTM model for 3-day PM10 forecast
Hopsworks integration for feature storage & model registry
CI/CD automation for data ingestion & model retraining
Web app for prediction display (table + plot)
