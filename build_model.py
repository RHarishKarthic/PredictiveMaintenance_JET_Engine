
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import os

# Set plotting style
sns.set_theme(style="whitegrid")

print("--- 1. LOADING DATA ---")
index_names = ['unit_nr', 'time_cycles']
setting_names = ['setting_1', 'setting_2', 'setting_3']
sensor_names = ['s_{}'.format(i) for i in range(1, 22)]
col_names = index_names + setting_names + sensor_names

# Load FD001
train_df = pd.read_csv('CMAPSSData/train_FD001.txt', sep=r'\s+', header=None, names=col_names)
test_df = pd.read_csv('CMAPSSData/test_FD001.txt', sep=r'\s+', header=None, names=col_names)
y_test = pd.read_csv('CMAPSSData/RUL_FD001.txt', sep=r'\s+', header=None, names=['RUL'])

print(f"Training data: {train_df.shape}")
print(f"Testing data: {test_df.shape}")

# --- 2. PREPROCESSING ---
print("\n--- 2. PREPROCESSING ---")

def add_remaining_useful_life(df):
    max_cycle = df.groupby('unit_nr')['time_cycles'].transform('max')
    df['RUL'] = max_cycle - df['time_cycles']
    return df

train_df = add_remaining_useful_life(train_df)

# Drop constant sensors
unique_counts = train_df.nunique()
constant_cols = unique_counts[unique_counts <= 1].index
train_df.drop(columns=constant_cols, inplace=True)
test_df.drop(columns=constant_cols, inplace=True)
print(f"Dropped constant columns: {list(constant_cols)}")

# Keep track of features
remaining_sensors = [s for s in sensor_names if s in train_df.columns]
remaining_settings = [s for s in setting_names if s in train_df.columns]
features = remaining_settings + remaining_sensors

# --- 3. FEATURE ENGINEERING (Rolling Features) ---
print("\n--- 3. FEATURE ENGINEERING ---")

def add_rolling_features(df, sensors, window=5):
    for s in sensors:
        df[s + '_rolling_mean'] = df.groupby('unit_nr')[s].transform(lambda x: x.rolling(window).mean())
        df[s + '_rolling_std'] = df.groupby('unit_nr')[s].transform(lambda x: x.rolling(window).std())
    return df.fillna(0)

train_df = add_rolling_features(train_df, remaining_sensors)
test_df = add_rolling_features(test_df, remaining_sensors)

# Update features to include rolling ones
features = features + [s + '_rolling_mean' for s in remaining_sensors] + [s + '_rolling_std' for s in remaining_sensors]

# Standardize
scaler = StandardScaler()
train_df[features] = scaler.fit_transform(train_df[features])
from xgboost import XGBRegressor
import os
import joblib

# ... (previous imports)

# --- 4. MODEL BUILDING ---
print("\n--- 4. MODEL BUILDING & TRAINING ---")

# Training Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(train_df[features], train_df['RUL'])
print("Random Forest trained.")

# Training XGBoost
xgb = XGBRegressor(n_estimators=100, learning_rate=0.05, n_jobs=-1)
xgb.fit(train_df[features], train_df['RUL'])
print("XGBoost trained.")

# Save everything for the web app
joblib.dump(xgb, 'xgb_model.joblib')
joblib.dump(scaler, 'scaler.joblib')
metadata = {
    'features': features,
    'remaining_sensors': remaining_sensors,
    'remaining_settings': remaining_settings
}
joblib.dump(metadata, 'metadata.joblib')
print("Model, Scaler, and Metadata saved successfully!")

# --- 5. EVALUATION ---
print("\n--- 5. EVALUATION ---")

# For test evaluation, we use the last cycle of each engine unit to predict current RUL
test_last = test_df.groupby('unit_nr').last().reset_index()

y_pred_rf = rf.predict(test_last[features])
y_pred_xgb = xgb.predict(test_last[features])

# RMSE and MAE
def print_metrics(y_true, y_pred, name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    print(f"[{name}] RMSE: {rmse:.2f} | MAE: {mae:.2f}")

print_metrics(y_test, y_pred_rf, "Random Forest")
print_metrics(y_test, y_pred_xgb, "XGBoost")

# --- 6. VISUALIZATION ---
print("\n--- 6. VISUALIZATION ---")

plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label='Actual RUL', color='black', linewidth=2, marker='o')
plt.plot(y_pred_rf, label='Random Forest Prediction', color='blue', linestyle='--', alpha=0.8)
plt.plot(y_pred_xgb, label='XGBoost Prediction', color='green', linestyle=':', alpha=0.8)
plt.title('RUL Prediction Comparison (FD001)')
plt.xlabel('Engine Unit')
plt.ylabel('Remaining Cycles')
plt.legend()
plt.tight_layout()
plt.savefig('rul_prediction_comparison.png')
print("Saved comparison plot: rul_prediction_comparison.png")

# Save a sensor trend plot
plt.figure(figsize=(10, 5))
unit1 = train_df[train_df['unit_nr'] == 1]
plt.plot(unit1['time_cycles'], unit1['s_11_rolling_mean'], label='Sensor 11 (Smoothed)')
plt.title('Engine Unit 1: Sensor 11 Degradation Trend')
plt.xlabel('Cycles')
plt.ylabel('Normalized Sensor Value')
plt.savefig('sensor_trend.png')
print("Saved sensor trend plot: sensor_trend.png")

print("\n--- SUCCESS ---")
print("Models have been built and evaluated. Plots are generated.")
