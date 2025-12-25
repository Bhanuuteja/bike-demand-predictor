"""Train cluster-based models and compute EDA metrics for the web app."""
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb

# Setup
Path("artifacts/cluster_models").mkdir(parents=True, exist_ok=True)
dataset_dir = Path("Dataset")
csv_files = sorted(dataset_dir.glob("*.csv"))

print("=" * 60)
print("TRAINING CLUSTER-BASED MODELS & COMPUTING EDA METRICS")
print("=" * 60)

# Load all data
print(f"\nLoading {len(csv_files)} CSV files...")
dfs = []
for csv_file in csv_files:
    try:
        df = pd.read_csv(csv_file)
        dfs.append(df)
    except Exception as e:
        print(f"  Skipped {csv_file.name}: {e}")

bike = pd.concat(dfs, ignore_index=True)
print(f"✓ Loaded {len(bike):,} rows")

# Basic preprocessing
bike['starttime'] = pd.to_datetime(bike['starttime'], errors='coerce')
bike = bike.dropna(subset=['starttime', 'start station name'])

# Hourly aggregation by station
hourly_by_station = bike.groupby([
    bike['starttime'].dt.floor('H'),
    'start station name'
]).size().reset_index(name='demand')
hourly_by_station.rename(columns={'starttime': 'hour_floor'}, inplace=True)

# Load station cluster mapping
with open("artifacts/station_cluster_mapping.json") as f:
    station_mapping = json.load(f)

hourly_by_station['cluster'] = hourly_by_station['start station name'].map(station_mapping)

# Feature engineering
hourly_by_station['month'] = hourly_by_station['hour_floor'].dt.month
hourly_by_station['day_of_week'] = hourly_by_station['hour_floor'].dt.dayofweek
hourly_by_station['hour_of_day'] = hourly_by_station['hour_floor'].dt.hour
hourly_by_station['day_of_month'] = hourly_by_station['hour_floor'].dt.day
hourly_by_station['week_of_year'] = hourly_by_station['hour_floor'].dt.isocalendar().week
hourly_by_station['is_weekend'] = (hourly_by_station['day_of_week'] >= 5).astype(int)

# Cyclical encoding
hourly_by_station['hour_sin'] = np.sin(2 * np.pi * hourly_by_station['hour_of_day'] / 24)
hourly_by_station['hour_cos'] = np.cos(2 * np.pi * hourly_by_station['hour_of_day'] / 24)
hourly_by_station['dow_sin'] = np.sin(2 * np.pi * hourly_by_station['day_of_week'] / 7)
hourly_by_station['dow_cos'] = np.cos(2 * np.pi * hourly_by_station['day_of_week'] / 7)
hourly_by_station['month_sin'] = np.sin(2 * np.pi * hourly_by_station['month'] / 12)
hourly_by_station['month_cos'] = np.cos(2 * np.pi * hourly_by_station['month'] / 12)

# Lag and rolling features (per station)
hourly_by_station = hourly_by_station.sort_values(['start station name', 'hour_floor'])
for station in hourly_by_station['start station name'].unique():
    mask = hourly_by_station['start station name'] == station
    hourly_by_station.loc[mask, 'lag_1h'] = hourly_by_station.loc[mask, 'demand'].shift(1)
    hourly_by_station.loc[mask, 'lag_24h'] = hourly_by_station.loc[mask, 'demand'].shift(24)
    hourly_by_station.loc[mask, 'lag_168h'] = hourly_by_station.loc[mask, 'demand'].shift(168)
    hourly_by_station.loc[mask, 'rolling_mean_6h'] = hourly_by_station.loc[mask, 'demand'].rolling(6, min_periods=1).mean()
    hourly_by_station.loc[mask, 'rolling_mean_24h'] = hourly_by_station.loc[mask, 'demand'].rolling(24, min_periods=1).mean()
    hourly_by_station.loc[mask, 'rolling_std_24h'] = hourly_by_station.loc[mask, 'demand'].rolling(24, min_periods=1).std()
    hourly_by_station.loc[mask, 'rolling_max_24h'] = hourly_by_station.loc[mask, 'demand'].rolling(24, min_periods=1).max()
    hourly_by_station.loc[mask, 'rolling_min_24h'] = hourly_by_station.loc[mask, 'demand'].rolling(24, min_periods=1).min()

# Holiday indicators
hourly_by_station['is_holiday'] = 0
hourly_by_station['is_nye'] = ((hourly_by_station['month'] == 12) & (hourly_by_station['day_of_month'] == 31)).astype(int)
hourly_by_station['is_morning_rush'] = ((hourly_by_station['hour_of_day'] >= 7) & (hourly_by_station['hour_of_day'] <= 9)).astype(int)
hourly_by_station['is_evening_rush'] = ((hourly_by_station['hour_of_day'] >= 16) & (hourly_by_station['hour_of_day'] <= 19)).astype(int)

hourly_by_station = hourly_by_station.dropna()

feature_cols = [
    'month', 'day_of_week', 'hour_of_day', 'day_of_month', 'week_of_year',
    'is_weekend', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
    'month_sin', 'month_cos', 'lag_1h', 'lag_24h', 'lag_168h',
    'rolling_mean_6h', 'rolling_mean_24h', 'rolling_std_24h',
    'rolling_max_24h', 'rolling_min_24h', 'is_holiday', 'is_nye',
    'is_morning_rush', 'is_evening_rush'
]

# Train cluster-specific models
print("\n" + "=" * 60)
print("TRAINING CLUSTER-BASED MODELS")
print("=" * 60)

cluster_models = {}
cluster_scalers = {}
cluster_metrics = {}

for cluster in ['High Demand', 'Medium Demand', 'Low Demand']:
    print(f"\n{cluster}:")
    cluster_data = hourly_by_station[hourly_by_station['cluster'] == cluster]
    
    if len(cluster_data) < 100:
        print(f"  ⚠ Insufficient data ({len(cluster_data)} rows), skipping")
        continue
    
    X = cluster_data[feature_cols].values
    y = cluster_data['demand'].values
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Train XGBoost
    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mae = np.mean(np.abs(y_pred - y_test))
    r2 = 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))
    
    print(f"  Samples: {len(cluster_data):,}")
    print(f"  Test MAE: {mae:.2f}")
    print(f"  Test R²: {r2:.4f}")
    
    # Save model and scaler with protocol 4 for cross-version compatibility
    model_path = f"artifacts/cluster_models/{cluster.lower().replace(' ', '_')}_model.joblib"
    scaler_path = f"artifacts/cluster_models/{cluster.lower().replace(' ', '_')}_scaler.joblib"
    joblib.dump(model, model_path, protocol=4)
    joblib.dump(scaler, scaler_path, protocol=4)
    
    cluster_models[cluster] = model_path
    cluster_scalers[cluster] = scaler_path
    cluster_metrics[cluster] = {"mae": float(mae), "r2": float(r2), "samples": len(cluster_data)}

# Compute EDA metrics for dashboard
print("\n" + "=" * 60)
print("COMPUTING EDA METRICS")
print("=" * 60)

eda_metrics = {
    "total_trips": int(bike.shape[0]),
    "total_stations": len(bike['start station name'].unique()),
    "date_range": {
        "start": str(bike['starttime'].min().date()),
        "end": str(bike['starttime'].max().date())
    },
    "hourly_stats": {
        "mean": float(hourly_by_station['demand'].mean()),
        "std": float(hourly_by_station['demand'].std()),
        "min": int(hourly_by_station['demand'].min()),
        "max": int(hourly_by_station['demand'].max())
    },
    "peak_hours": {
        "morning_rush": int(hourly_by_station[hourly_by_station['is_morning_rush'] == 1]['demand'].mean()),
        "evening_rush": int(hourly_by_station[hourly_by_station['is_evening_rush'] == 1]['demand'].mean()),
        "off_peak": int(hourly_by_station[(hourly_by_station['is_morning_rush'] == 0) & (hourly_by_station['is_evening_rush'] == 0)]['demand'].mean())
    },
    "weekend_vs_weekday": {
        "weekday": int(hourly_by_station[hourly_by_station['is_weekend'] == 0]['demand'].mean()),
        "weekend": int(hourly_by_station[hourly_by_station['is_weekend'] == 1]['demand'].mean())
    },
    "monthly_trends": {month: int(hourly_by_station[hourly_by_station['month'] == month]['demand'].mean()) for month in range(1, 13)},
    "cluster_stats": cluster_metrics
}

# Save EDA metrics
with open("artifacts/eda_metrics.json", "w") as f:
    json.dump(eda_metrics, f, indent=2)

# Save feature columns
with open("artifacts/feature_cols.json", "w") as f:
    json.dump(feature_cols, f, indent=2)

print("\n✓ EDA metrics saved to artifacts/eda_metrics.json")
print("✓ Feature columns saved to artifacts/feature_cols.json")
print("\n" + "=" * 60)
print("✓ CLUSTER-BASED MODELS TRAINED & SAVED")
print("=" * 60)
print(f"\nModels ready for deployment:")
for cluster in ['High Demand', 'Medium Demand', 'Low Demand']:
    print(f"  • {cluster}: {cluster_models.get(cluster, 'NOT TRAINED')}")
