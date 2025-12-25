from pathlib import Path
import datetime as dt
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import json

st.set_page_config(page_title="Citi Bike Demand Forecasting", layout="wide")
st.title("🚴 Citi Bike Demand Forecasting - Station Level")

ARTIFACT_PATH = Path("artifacts/xgboost_v1.joblib")
STATION_MAPPING_PATH = Path("artifacts/station_cluster_mapping.json")

if not ARTIFACT_PATH.exists():
    st.error("Model artifact not found at artifacts/xgboost_v1.joblib. Please export it from the notebook.")
    st.stop()

if not STATION_MAPPING_PATH.exists():
    st.error("Station mapping not found. Run extract_stations.py first.")
    st.stop()

model = joblib.load(ARTIFACT_PATH)
with open(STATION_MAPPING_PATH) as f:
    station_mapping = json.load(f)

stations = sorted(list(station_mapping.keys()))

feature_cols = [
    "month",
    "day_of_week",
    "hour_of_day",
    "day_of_month",
    "week_of_year",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "lag_1h",
    "lag_24h",
    "lag_168h",
    "rolling_mean_6h",
    "rolling_mean_24h",
    "rolling_std_24h",
    "rolling_max_24h",
    "rolling_min_24h",
    "is_holiday",
    "is_nye",
    "is_morning_rush",
    "is_evening_rush",
]

# Define holidays (US holidays)
HOLIDAYS = [
    (1, 1),    # New Year's Day
    (7, 4),    # Independence Day
    (11, 28),  # Thanksgiving (varies, approximate)
    (12, 25),  # Christmas
]

def is_holiday(month, day):
    return (month, day) in HOLIDAYS

def is_peak_hour(hour, day_of_week):
    """Peak hours: 7-9 AM and 4-7 PM on weekdays."""
    if day_of_week >= 5:  # Weekend
        return False
    return (7 <= hour <= 9) or (16 <= hour <= 19)

st.sidebar.header("📍 Station & Time Input")

# Station selection
selected_station = st.sidebar.selectbox("Select Station", stations)
station_cluster = station_mapping.get(selected_station, "Unknown")

# UTC DateTime input
col1, col2 = st.sidebar.columns(2)
with col1:
    utc_date = st.date_input("Date (UTC)", dt.date.today())
with col2:
    utc_hour = st.selectbox("Hour (UTC 0-23)", options=list(range(24)), index=12)

# Recency proxies for the selected station
st.sidebar.subheader("Demand Proxies (Historical)")
recent_avg = st.sidebar.number_input("Avg demand (last 24h)", min_value=0.0, value=85.0, step=5.0)
recent_std = st.sidebar.number_input("Std dev (last 24h)", min_value=0.0, value=12.0, step=1.0)
recent_max = st.sidebar.number_input("Max (last 24h)", min_value=0.0, value=150.0, step=5.0)
recent_min = st.sidebar.number_input("Min (last 24h)", min_value=0.0, value=40.0, step=5.0)

# Build feature vector
now_dt = datetime.combine(utc_date, dt.time(hour=utc_hour))

holiday_check = is_holiday(now_dt.month, now_dt.day)
peak_hour_check = is_peak_hour(now_dt.hour, now_dt.weekday())

payload = {
    "month": now_dt.month,
    "day_of_week": now_dt.weekday(),
    "hour_of_day": now_dt.hour,
    "day_of_month": now_dt.day,
    "week_of_year": now_dt.isocalendar().week,
    "is_weekend": int(now_dt.weekday() >= 5),
    "hour_sin": np.sin(2 * np.pi * now_dt.hour / 24),
    "hour_cos": np.cos(2 * np.pi * now_dt.hour / 24),
    "dow_sin": np.sin(2 * np.pi * now_dt.weekday() / 7),
    "dow_cos": np.cos(2 * np.pi * now_dt.weekday() / 7),
    "month_sin": np.sin(2 * np.pi * now_dt.month / 12),
    "month_cos": np.cos(2 * np.pi * now_dt.month / 12),
    "lag_1h": recent_avg,
    "lag_24h": recent_avg,
    "lag_168h": recent_avg,
    "rolling_mean_6h": recent_avg,
    "rolling_mean_24h": recent_avg,
    "rolling_std_24h": recent_std,
    "rolling_max_24h": recent_max,
    "rolling_min_24h": recent_min,
    "is_holiday": int(holiday_check),
    "is_nye": int(now_dt.month == 12 and now_dt.day == 31),
    "is_morning_rush": int(7 <= now_dt.hour <= 9),
    "is_evening_rush": int(16 <= now_dt.hour <= 19),
}

# Display section
st.header("Prediction Results")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Station", selected_station, f"Cluster: {station_cluster}")
with col2:
    st.metric("Date (UTC)", utc_date.strftime("%Y-%m-%d"))
with col3:
    st.metric("Hour (UTC)", f"{utc_hour:02d}:00")
with col4:
    st.metric("Day of Week", now_dt.strftime("%A"))

if st.button("🔮 Predict Demand"):
    df = pd.DataFrame([payload])[feature_cols]
    pred = float(model.predict(df)[0])
    
    # Scale prediction based on cluster demand level
    cluster_factor = {"High Demand": 1.3, "Medium Demand": 1.0, "Low Demand": 0.7}
    adjusted_pred = pred * cluster_factor.get(station_cluster, 1.0)
    
    st.success("✓ Prediction Generated")
    
    # Main prediction display
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 System-Level Forecast", f"{pred:,.0f} trips")
    with col2:
        st.metric("🚩 Station-Adjusted Forecast", f"{adjusted_pred:,.0f} trips")
    with col3:
        st.metric("📈 Cluster Type", station_cluster)
    
    # Detailed insights
    st.subheader("📋 Detailed Insights")
    
    insight_col1, insight_col2, insight_col3, insight_col4 = st.columns(4)
    with insight_col1:
        holiday_badge = "🎄 YES" if holiday_check else "✓ NO"
        st.metric("Is Holiday?", holiday_badge)
    with insight_col2:
        peak_badge = "⚡ PEAK" if peak_hour_check else "✓ Normal"
        st.metric("Peak Hour?", peak_badge)
    with insight_col3:
        weekend_badge = "🌤️ YES" if now_dt.weekday() >= 5 else "✓ Weekday"
        st.metric("Is Weekend?", weekend_badge)
    with insight_col4:
        morning_rush = int(7 <= now_dt.hour <= 9)
        evening_rush = int(16 <= now_dt.hour <= 19)
        if morning_rush:
            st.metric("Rush Hour", "🌅 Morning")
        elif evening_rush:
            st.metric("Rush Hour", "🌆 Evening")
        else:
            st.metric("Rush Hour", "✓ Off-Peak")
    
    # Additional details
    st.subheader("📌 Time & Context")
    details_col1, details_col2 = st.columns(2)
    
    with details_col1:
        st.write(f"**Month:** {now_dt.strftime('%B')} (#{now_dt.month})")
        st.write(f"**Week:** Week #{now_dt.isocalendar().week} of the year")
        st.write(f"**Day:** {now_dt.strftime('%A')} (#{now_dt.weekday()})")
    
    with details_col2:
        st.write(f"**Recent Avg Demand:** {recent_avg:.0f} trips")
        st.write(f"**Recent Variability (Std):** {recent_std:.1f} trips")
        st.write(f"**Recent Range:** {recent_min:.0f} - {recent_max:.0f} trips")
    
    # Recommendation
    st.subheader("💡 Recommendation")
    if adjusted_pred > 100:
        st.info(f"⚠️ **High Demand Expected:** Prepare {adjusted_pred:.0f} trips. Consider extra bike availability.")
    elif adjusted_pred < 50:
        st.warning(f"📉 **Low Demand Predicted:** Only {adjusted_pred:.0f} trips expected. Monitor bike redistribution.")
    else:
        st.info(f"✓ **Normal Demand:** {adjusted_pred:.0f} trips forecasted. Proceed with regular operations.")
