"""Generate station-to-cluster mapping for Streamlit app."""
import pandas as pd
import numpy as np
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# Load data
dataset_dir = Path("Dataset")
csv_files = sorted(dataset_dir.glob("*.csv"))

print("Loading dataset for station extraction...")
dfs = []
for csv_file in csv_files[:10]:  # Sample first 10 files to get unique stations
    try:
        df = pd.read_csv(csv_file, usecols=['start station name', 'starttime'])
        dfs.append(df)
        print(f"  ✓ Loaded {csv_file.name}")
    except Exception as e:
        print(f"  ⚠ Skipped {csv_file.name}: {e}")

bike = pd.concat(dfs, ignore_index=True)
print(f"Loaded {len(bike)} rows")

# Get unique stations
stations = bike['start station name'].dropna().unique()
stations = sorted(list(set(stations)))
print(f"Found {len(stations)} unique stations")

# Simple clustering by station name prefix/region (can be enhanced)
# For now, map stations to clusters based on frequency
station_demand = bike.groupby('start station name').size().reset_index(name='total_trips')
station_demand = station_demand.sort_values('total_trips', ascending=False)

# Create artificial clusters (High, Medium, Low demand)
high_demand_threshold = station_demand['total_trips'].quantile(0.75)
low_demand_threshold = station_demand['total_trips'].quantile(0.25)

station_demand['cluster'] = 'Medium Demand'
station_demand.loc[station_demand['total_trips'] >= high_demand_threshold, 'cluster'] = 'High Demand'
station_demand.loc[station_demand['total_trips'] < low_demand_threshold, 'cluster'] = 'Low Demand'

# Save mapping - use station names as keys
station_mapping = {}
for idx, row in station_demand.iterrows():
    station_mapping[row['start station name']] = row['cluster']

with open("artifacts/station_cluster_mapping.json", "w") as f:
    json.dump(station_mapping, f, indent=2)
with open("artifacts/station_cluster_mapping.json", "w") as f:
    json.dump(station_mapping, f, indent=2)

print(f"\n✓ Station-to-cluster mapping saved to artifacts/station_cluster_mapping.json")
print(f"  High Demand: {len(station_demand[station_demand['cluster'] == 'High Demand'])} stations")
print(f"  Medium Demand: {len(station_demand[station_demand['cluster'] == 'Medium Demand'])} stations")
print(f"  Low Demand: {len(station_demand[station_demand['cluster'] == 'Low Demand'])} stations")
