"""Generate station-level EDA metrics for enhanced dashboard."""
import pandas as pd
import numpy as np
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# Load all data
dataset_dir = Path("Dataset")
csv_files = sorted(dataset_dir.glob("JC-*.csv"))

print("=" * 60)
print("GENERATING STATION-LEVEL EDA METRICS")
print("=" * 60)

print(f"\nLoading {len(csv_files)} CSV files...")
dfs = []
for i, csv_file in enumerate(csv_files, 1):
    try:
        df = pd.read_csv(csv_file, low_memory=False)
        dfs.append(df)
        if i % 10 == 0:
            print(f"  ✓ Loaded {i}/{len(csv_files)} files")
    except Exception as e:
        print(f"  ⚠ {csv_file.name}: {str(e)[:50]}")

bike = pd.concat(dfs, ignore_index=True)
print(f"✓ Loaded {len(bike):,} total rows")

# Handle column name variations
if 'starttime' in bike.columns:
    bike['started_at'] = pd.to_datetime(bike['starttime'], errors='coerce')
elif 'started_at' in bike.columns:
    bike['started_at'] = pd.to_datetime(bike['started_at'], errors='coerce')

if 'end station name' in bike.columns:
    bike['end_station_name'] = bike['end station name']
elif 'ended_at' not in bike.columns and 'stoptime' in bike.columns:
    bike['ended_at'] = pd.to_datetime(bike['stoptime'], errors='coerce')

# Clean data
bike = bike.dropna(subset=['started_at', 'start station name'])
print(f"✓ Cleaned to {len(bike):,} rows")

# Extract temporal features
bike['hour'] = bike['started_at'].dt.hour
bike['day_of_week'] = bike['started_at'].dt.dayofweek
bike['month'] = bike['started_at'].dt.month
bike['date'] = bike['started_at'].dt.date

print(f"\nDate range: {bike['started_at'].min().date()} to {bike['started_at'].max().date()}")
print(f"Processing {bike['start station name'].nunique()} unique stations...\n")

# Create station-level EDA
station_eda = {}
station_counter = 0

for station in sorted(bike['start station name'].unique()):
    station_data = bike[bike['start station name'] == station]
    
    # Only include stations with sufficient data (>100 trips)
    if len(station_data) < 100:
        continue
    
    station_counter += 1
    total_trips = len(station_data)
    
    # Basic stats
    unique_days = station_data['date'].nunique()
    avg_trips_per_day = total_trips / unique_days if unique_days > 0 else 0
    
    # Hourly pattern
    hourly_dist = station_data.groupby('hour').size()
    peak_hour = int(hourly_dist.idxmax())
    peak_hour_trips = int(hourly_dist.max())
    
    # Rush hour analysis
    morning_rush = len(station_data[(station_data['hour'] >= 7) & (station_data['hour'] <= 9)])
    evening_rush = len(station_data[(station_data['hour'] >= 16) & (station_data['hour'] <= 19)])
    
    # Day of week pattern
    dow_dist = station_data.groupby('day_of_week').size()
    busiest_dow = int(dow_dist.idxmax())
    dow_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Weekend vs Weekday
    weekend_trips = len(station_data[station_data['day_of_week'] >= 5])
    weekday_trips = total_trips - weekend_trips
    weekend_ratio = weekend_trips / total_trips if total_trips > 0 else 0
    
    # Seasonal pattern (monthly)
    monthly_dist = station_data.groupby('month').size()
    peak_month = int(monthly_dist.idxmax())
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # User type (if available)
    if 'user type' in station_data.columns:
        member_ratio = (station_data['user type'] == 'Subscriber').sum() / total_trips
    elif 'member_casual' in station_data.columns:
        member_ratio = (station_data['member_casual'] == 'member').sum() / total_trips
    else:
        member_ratio = 0.7  # Default
    
    # Trip duration (if available)
    avg_trip_duration = 0
    if 'tripduration' in station_data.columns:
        avg_trip_duration = station_data['tripduration'].mean() / 60  # Convert to minutes
    elif 'ended_at' in station_data.columns:
        try:
            station_data['trip_duration_sec'] = (station_data['ended_at'] - station_data['started_at']).dt.total_seconds()
            avg_trip_duration = station_data['trip_duration_sec'].mean() / 60
        except:
            avg_trip_duration = 0
    
    # Store data
    station_eda[station] = {
        'total_trips': int(total_trips),
        'avg_trips_per_day': float(avg_trips_per_day),
        'peak_hour': peak_hour,
        'peak_hour_trips': peak_hour_trips,
        'morning_rush_trips': int(morning_rush),
        'evening_rush_trips': int(evening_rush),
        'weekend_ratio': float(weekend_ratio),
        'busiest_day': dow_names[busiest_dow],
        'peak_month': month_names[peak_month - 1],
        'member_ratio': float(member_ratio),
        'avg_trip_duration_min': float(avg_trip_duration),
        'hourly_distribution': {int(k): int(v) for k, v in hourly_dist.to_dict().items()},
        'daily_distribution': {int(k): int(v) for k, v in dow_dist.to_dict().items()},
        'monthly_distribution': {int(k): int(v) for k, v in monthly_dist.to_dict().items()}
    }
    
    if station_counter % 10 == 0:
        print(f"  ✓ Processed {station_counter} stations...")

# Save station EDA
artifacts_dir = Path("artifacts")
artifacts_dir.mkdir(exist_ok=True)

with open(artifacts_dir / "station_level_eda.json", "w") as f:
    json.dump(station_eda, f, indent=2, default=str)

print(f"\n{'=' * 60}")
print(f"✓ STATION-LEVEL EDA GENERATED")
print(f"{'=' * 60}")
print(f"Processed: {len(station_eda)} stations with sufficient data")
print(f"Saved to: artifacts/station_level_eda.json")

# Show top 10 stations
print(f"\nTop 10 Stations by Trip Volume:")
sorted_stations = sorted(station_eda.items(), key=lambda x: x[1]['total_trips'], reverse=True)
for i, (station, stats) in enumerate(sorted_stations[:10], 1):
    peak_h = stats['peak_hour']
    print(f"  {i:2d}. {station:45s} | {stats['total_trips']:>7,} trips | Peak: {peak_h:02d}:00 | Member: {stats['member_ratio']*100:.0f}%")
