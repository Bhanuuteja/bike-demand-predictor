"""Quick script to train and save the XGBoost model for Streamlit deployment."""
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Setup
Path("artifacts").mkdir(exist_ok=True)
dataset_dir = Path("Dataset")
csv_files = sorted(dataset_dir.glob("*.csv"))

print(f"Loading {len(csv_files)} CSV files...")

# Load and process data
dfs = []
for csv_file in csv_files:
    try:
        df = pd.read_csv(csv_file)
        dfs.append(df)
    except Exception as e:
        print(f"  Skipped {csv_file.name}: {e}")

bike = pd.concat(dfs, ignore_index=True)
print(f"Loaded {len(bike)} rows")

# Basic preprocessing
bike['started_at'] = pd.to_datetime(bike['started_at'], errors='coerce')
bike = bike.dropna(subset=['started_at'])

# Hourly aggregation
hourly_demand = bike.groupby(bike['started_at'].dt.floor('H')).size().reset_index(name='demand')
hourly_demand.rename(columns={'started_at': 'hour_floor'}, inplace=True)

# Create features
hourly_demand['month'] = hourly_demand['hour_floor'].dt.month
hourly_demand['day_of_week'] = hourly_demand['hour_floor'].dt.dayofweek
hourly_demand['hour_of_day'] = hourly_demand['hour_floor'].dt.hour
hourly_demand['day_of_month'] = hourly_demand['hour_floor'].dt.day
hourly_demand['week_of_year'] = hourly_demand['hour_floor'].dt.isocalendar().week
hourly_demand['is_weekend'] = (hourly_demand['day_of_week'] >= 5).astype(int)

# Cyclical encoding
hourly_demand['hour_sin'] = np.sin(2 * np.pi * hourly_demand['hour_of_day'] / 24)
hourly_demand['hour_cos'] = np.cos(2 * np.pi * hourly_demand['hour_of_day'] / 24)
hourly_demand['dow_sin'] = np.sin(2 * np.pi * hourly_demand['day_of_week'] / 7)
hourly_demand['dow_cos'] = np.cos(2 * np.pi * hourly_demand['day_of_week'] / 7)
hourly_demand['month_sin'] = np.sin(2 * np.pi * hourly_demand['month'] / 12)
hourly_demand['month_cos'] = np.cos(2 * np.pi * hourly_demand['month'] / 12)

# Lag features
hourly_demand['lag_1h'] = hourly_demand['demand'].shift(1)
hourly_demand['lag_24h'] = hourly_demand['demand'].shift(24)
hourly_demand['lag_168h'] = hourly_demand['demand'].shift(168)

# Rolling features
hourly_demand['rolling_mean_6h'] = hourly_demand['demand'].rolling(window=6, min_periods=1).mean()
hourly_demand['rolling_mean_24h'] = hourly_demand['demand'].rolling(window=24, min_periods=1).mean()
hourly_demand['rolling_std_24h'] = hourly_demand['demand'].rolling(window=24, min_periods=1).std()
hourly_demand['rolling_max_24h'] = hourly_demand['demand'].rolling(window=24, min_periods=1).max()
hourly_demand['rolling_min_24h'] = hourly_demand['demand'].rolling(window=24, min_periods=1).min()

# Holiday indicators (simple placeholders)
hourly_demand['is_holiday'] = 0
hourly_demand['is_nye'] = ((hourly_demand['month'] == 12) & (hourly_demand['day_of_month'] == 31)).astype(int)
hourly_demand['is_morning_rush'] = ((hourly_demand['hour_of_day'] >= 7) & (hourly_demand['hour_of_day'] <= 9)).astype(int)
hourly_demand['is_evening_rush'] = ((hourly_demand['hour_of_day'] >= 16) & (hourly_demand['hour_of_day'] <= 19)).astype(int)

# Drop NaN
hourly_demand_clean = hourly_demand.dropna()

# Feature and target
feature_cols = [
    'month', 'day_of_week', 'hour_of_day', 'day_of_month', 'week_of_year',
    'is_weekend', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
    'month_sin', 'month_cos', 'lag_1h', 'lag_24h', 'lag_168h',
    'rolling_mean_6h', 'rolling_mean_24h', 'rolling_std_24h',
    'rolling_max_24h', 'rolling_min_24h', 'is_holiday', 'is_nye',
    'is_morning_rush', 'is_evening_rush'
]

X = hourly_demand_clean[feature_cols]
y = hourly_demand_clean['demand']

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train XGBoost
print("\nTraining XGBoost model...")
xgb_model = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
xgb_model.fit(X_train, y_train)
print("✓ Model trained")

# Evaluate
y_pred = xgb_model.predict(X_test)
mae = np.mean(np.abs(y_pred - y_test))
print(f"  Test MAE: {mae:.2f}")

# Save
model_path = "artifacts/xgboost_v1.joblib"
joblib.dump(xgb_model, model_path)
print(f"\n✓ Model saved to {model_path}")
print(f"Ready for Streamlit deployment!")
