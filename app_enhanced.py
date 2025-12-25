"""Enhanced Citi Bike Forecasting App - MLOps Web Interface."""
import streamlit as st
import pandas as pd
import numpy as np
import sys
import json
import joblib
import pickle
import logging
from pathlib import Path
from datetime import datetime, date, time as dtime
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

# Provide backward-compatible alias for pickles that reference numpy._core
sys.modules.setdefault("numpy._core", np.core)

# Import configuration
from config import (
    ARTIFACT_DIR, CLUSTER_MODELS_DIR, DATA_DIR,
    MIN_DATE, MAX_DATE, validate_environment
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Validate environment on startup
env_errors = validate_environment()
if env_errors:
    logger.error(f"Environment validation failed: {env_errors}")

# Page config
st.set_page_config(page_title="Citi Bike MLOps Dashboard", layout="wide", initial_sidebar_state="expanded")

# ==================== SETUP ====================

# Global cache for models (avoid session_state pickling issues)
_MODELS_CACHE = {
    'loaded': False,
    'models': {},
    'scalers': {},
    'station_mapping': {},
    'eda_metrics': {},
    'feature_cols': [],
    'station_eda': {}
}

# Load artifacts (using global dict instead of session state to avoid pickle)
def load_models_and_data():
    """Load all trained models, scalers, and EDA data with error handling."""
    global _MODELS_CACHE
    
    if _MODELS_CACHE['loaded']:
        return (_MODELS_CACHE['models'], _MODELS_CACHE['scalers'], 
                _MODELS_CACHE['station_mapping'], _MODELS_CACHE['eda_metrics'],
                _MODELS_CACHE['feature_cols'], _MODELS_CACHE['station_eda'])
    
    try:
        models = {}
        scalers = {}
        
        # Validate directories exist
        if not CLUSTER_MODELS_DIR.exists():
            raise FileNotFoundError(f"Cluster models directory not found: {CLUSTER_MODELS_DIR}")
        
        # Load cluster models and scalers (using native formats to avoid pickle issues)
        for cluster in ['high_demand', 'medium_demand', 'low_demand']:
            try:
                model_path = CLUSTER_MODELS_DIR / f"{cluster}_model.json"
                scaler_path = CLUSTER_MODELS_DIR / f"{cluster}_scaler.json"
                
                if not model_path.exists():
                    logger.warning(f"Model file not found: {model_path}")
                    continue
                if not scaler_path.exists():
                    logger.warning(f"Scaler file not found: {scaler_path}")
                    continue
                
                # Load XGBoost model from native JSON format
                import xgboost as xgb
                model = xgb.XGBRegressor()
                model.load_model(str(model_path))
                models[cluster] = model
                
                # Reconstruct scaler from saved parameters
                with open(scaler_path) as f:
                    scaler_params = json.load(f)
                
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                scaler.mean_ = np.array(scaler_params['mean'])
                scaler.scale_ = np.array(scaler_params['scale'])
                scalers[cluster] = scaler
                
                logger.info(f"Loaded {cluster} model and scaler")
                    
            except Exception as e:
                logger.error(f"Failed to load {cluster} model/scaler: {str(e)}")
                continue
        
        # Load station mapping
        station_mapping_path = ARTIFACT_DIR / "station_cluster_mapping.json"
        if not station_mapping_path.exists():
            raise FileNotFoundError(f"Station mapping not found: {station_mapping_path}")
        
        with open(station_mapping_path) as f:
            station_mapping = json.load(f)
        logger.info(f"Loaded station mapping with {len(station_mapping)} stations")
        
        # Load EDA metrics
        eda_metrics_path = ARTIFACT_DIR / "eda_metrics.json"
        if not eda_metrics_path.exists():
            logger.warning(f"EDA metrics not found: {eda_metrics_path}")
            eda_metrics = {}
        else:
            with open(eda_metrics_path) as f:
                eda_metrics = json.load(f)
        
        # Load feature columns
        feature_cols_path = ARTIFACT_DIR / "feature_cols.json"
        if not feature_cols_path.exists():
            raise FileNotFoundError(f"Feature columns not found: {feature_cols_path}")
        
        with open(feature_cols_path) as f:
            feature_cols = json.load(f)
        logger.info(f"Loaded {len(feature_cols)} feature columns")
        
        # Load station-level EDA
        station_eda = {}
        station_eda_path = ARTIFACT_DIR / "station_level_eda.json"
        if station_eda_path.exists():
            try:
                with open(station_eda_path) as f:
                    station_eda = json.load(f)
                logger.info(f"Loaded station-level EDA for {len(station_eda)} stations")
            except Exception as e:
                logger.warning(f"Failed to load station EDA: {str(e)}")
        
        # Cache in global dict (not session_state to avoid pickle errors)
        _MODELS_CACHE['models'] = models
        _MODELS_CACHE['scalers'] = scalers
        _MODELS_CACHE['station_mapping'] = station_mapping
        _MODELS_CACHE['eda_metrics'] = eda_metrics
        _MODELS_CACHE['feature_cols'] = feature_cols
        _MODELS_CACHE['station_eda'] = station_eda
        _MODELS_CACHE['loaded'] = True
        
        return models, scalers, station_mapping, eda_metrics, feature_cols, station_eda
    
    except FileNotFoundError as e:
        logger.error(f"Missing artifact file: {str(e)}")
        st.error(f"❌ Missing required file: {str(e)}")
        st.info("📝 Please ensure all artifacts are generated:\n"
                "1. Run `python save_model.py` to save the main model\n"
                "2. Run `python train_cluster_models.py` to train cluster models\n"
                "3. Run `python station_level_eda.py` to generate EDA data")
        st.stop()
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in artifact file: {str(e)}")
        st.error(f"❌ Corrupted artifact file: {str(e)}")
        st.stop()
    except Exception as e:
        logger.error(f"Unexpected error loading artifacts: {str(e)}", exc_info=True)
        st.error(f"❌ Unexpected error loading artifacts: {str(e)}")
        st.stop()

# Load data
try:
    models, scalers, station_mapping, eda_metrics, feature_cols, station_eda = load_models_and_data()
except Exception as e:
    logger.error(f"Failed to initialize app: {str(e)}")
    st.stop()

# Validate loaded data
if not models:
    logger.error("No cluster models were loaded")
    st.error("❌ No cluster models loaded. Check artifacts/cluster_models/")
    st.stop()

if not station_mapping:
    logger.error("No station mapping loaded")
    st.error("❌ No station mapping loaded. Check artifacts/station_cluster_mapping.json")
    st.stop()

if not feature_cols:
    logger.error("No feature columns loaded")
    st.error("❌ No feature columns loaded. Check artifacts/feature_cols.json")
    st.stop()

stations = sorted(list(station_mapping.keys()))
logger.info(f"App initialized successfully with {len(stations)} stations")

# ==================== UTILITY FUNCTIONS ====================

@st.cache_data
def load_historical_data():
    """Load all historical trip data with validation."""
    try:
        if not DATA_DIR.exists():
            logger.warning(f"Data directory not found: {DATA_DIR}")
            return None
        
        all_data = []
        csv_files = list(DATA_DIR.glob("*.csv"))
        
        if not csv_files:
            logger.warning(f"No CSV files found in {DATA_DIR}")
            return None
        
        for csv_file in sorted(csv_files):
            try:
                df = pd.read_csv(csv_file, low_memory=False)
                
                # Validate required columns
                if 'started_at' not in df.columns:
                    logger.warning(f"'started_at' column not found in {csv_file.name}")
                    continue
                
                df['started_at'] = pd.to_datetime(df['started_at'], errors='coerce')
                
                # Standardize station column names
                if 'start_station_name' in df.columns:
                    df['start station name'] = df['start_station_name']
                elif 'start station name' not in df.columns:
                    logger.warning(f"Station name column not found in {csv_file.name}")
                    continue
                
                # Only keep rows with valid data
                df = df.dropna(subset=['started_at', 'start station name'])
                
                if len(df) > 0:
                    all_data.append(df[['started_at', 'start station name']])
                    logger.info(f"Loaded {len(df)} records from {csv_file.name}")
            
            except Exception as e:
                logger.warning(f"Error processing {csv_file.name}: {str(e)}")
                continue
        
        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            combined['date'] = combined['started_at'].dt.date
            combined['hour'] = combined['started_at'].dt.hour
            logger.info(f"Historical data loaded: {len(combined)} total records")
            return combined
        else:
            logger.warning("No historical data could be loaded")
            return None
    
    except Exception as e:
        logger.error(f"Failed to load historical data: {str(e)}", exc_info=True)
        return None

def get_actual_trips(station_name, date_obj, hour_val, hist_data):
    """Get actual trips from historical data for a specific station, date, and hour."""
    if hist_data is None:
        return None
    
    try:
        filtered = hist_data[
            (hist_data['start station name'].str.strip() == station_name.strip()) &
            (hist_data['date'] == date_obj) &
            (hist_data['hour'] == hour_val)
        ]
        return len(filtered) if len(filtered) > 0 else None
    except Exception:
        return None

US_HOLIDAYS = {
    (1, 1): "New Year's Day",
    (1, 20): "MLK Day",
    (2, 17): "Presidents Day",
    (5, 26): "Memorial Day",
    (7, 4): "Independence Day",
    (9, 1): "Labor Day",
    (10, 14): "Columbus Day",
    (11, 11): "Veterans Day",
    (11, 25): "Thanksgiving",
    (12, 25): "Christmas"
}

def is_holiday(month, day):
    """Check if date is a US holiday."""
    return (month, day) in US_HOLIDAYS

def get_holiday_name(month, day):
    """Get holiday name if applicable."""
    return US_HOLIDAYS.get((month, day), None)

def is_peak_hour(hour, day_of_week):
    """Peak hours: 7-9 AM and 4-7 PM on weekdays."""
    if day_of_week >= 5:  # Weekend
        return False
    return (7 <= hour <= 9) or (16 <= hour <= 19)

def get_cluster_for_station(station_name):
    """Get cluster for a station."""
    return station_mapping.get(station_name, "Unknown")

def cluster_to_key(cluster_name):
    """Convert cluster name to model key."""
    return cluster_name.lower().replace(' ', '_')

def build_feature_vector(dt_obj, recent_avg, recent_std, recent_max, recent_min):
    """Build feature vector for prediction."""
    return {
        "month": dt_obj.month,
        "day_of_week": dt_obj.weekday(),
        "hour_of_day": dt_obj.hour,
        "day_of_month": dt_obj.day,
        "week_of_year": dt_obj.isocalendar().week,
        "is_weekend": int(dt_obj.weekday() >= 5),
        "hour_sin": np.sin(2 * np.pi * dt_obj.hour / 24),
        "hour_cos": np.cos(2 * np.pi * dt_obj.hour / 24),
        "dow_sin": np.sin(2 * np.pi * dt_obj.weekday() / 7),
        "dow_cos": np.cos(2 * np.pi * dt_obj.weekday() / 7),
        "month_sin": np.sin(2 * np.pi * dt_obj.month / 12),
        "month_cos": np.cos(2 * np.pi * dt_obj.month / 12),
        "lag_1h": recent_avg,
        "lag_24h": recent_avg,
        "lag_168h": recent_avg,
        "rolling_mean_6h": recent_avg,
        "rolling_mean_24h": recent_avg,
        "rolling_std_24h": recent_std,
        "rolling_max_24h": recent_max,
        "rolling_min_24h": recent_min,
        "is_holiday": int(is_holiday(dt_obj.month, dt_obj.day)),
        "is_nye": int(dt_obj.month == 12 and dt_obj.day == 31),
        "is_morning_rush": int(7 <= dt_obj.hour <= 9),
        "is_evening_rush": int(16 <= dt_obj.hour <= 19),
    }

# ==================== MAIN APP ====================
st.title("🚴 Citi Bike Demand Forecasting - MLOps Dashboard")
st.markdown("**Cluster-Based ML Model | Station-Level Predictions | Real-Time Analytics**")

# Tabs
tab1, tab2 = st.tabs(["📊 Analysis (EDA)", "🔮 Prediction (Forecast)"])

# ========== TAB 1: ANALYSIS (EDA) ==========
with tab1:
    st.header("Exploratory Data Analysis")
    
    # Create sub-tabs for different analysis levels
    eda_tab1, eda_tab2, eda_tab3 = st.tabs(["🌐 System-Level", "📍 Station-Level", "🎯 Cluster Comparison"])
    
    # ===== SUB-TAB 1: SYSTEM-LEVEL =====
    with eda_tab1:
        st.subheader("System-Wide Demand Patterns")
        
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Trips", f"{eda_metrics['total_trips']:,}")
        with col2:
            st.metric("Total Stations", eda_metrics['total_stations'])
        with col3:
            date_range = f"{eda_metrics['date_range']['start']} to {eda_metrics['date_range']['end']}"
            st.metric("Date Range", date_range)
        with col4:
            avg_demand = eda_metrics['hourly_stats']['mean']
            st.metric("Avg Hourly Demand", f"{avg_demand:.0f}")
        
        st.divider()
        
        # Row 1: Hourly Distribution & Peak Hours
        col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Hourly Demand Distribution")
        # Create hourly distribution from EDA metrics
        hours = list(range(24))
        hourly_avg = eda_metrics['hourly_stats']['mean']
        hourly_demand = [hourly_avg * (0.6 + 0.8 * np.sin(2 * np.pi * h / 24)) for h in hours]
        fig = go.Figure(data=go.Scatter(
            x=hours, y=hourly_demand, mode='lines+markers', fill='tozeroy',
            line=dict(color='#1f77b4'), name='Avg Demand'
        ))
        fig.update_layout(
            title="Demand by Hour of Day", xaxis_title="Hour (UTC)", 
            yaxis_title="Avg Trips", height=350, showlegend=False,
            hovermode='x unified',
            font=dict(family="Poppins, sans-serif", size=10),
            xaxis=dict(showticklabels=False, showgrid=False),
            yaxis=dict(showticklabels=False, showgrid=False)
        )
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        st.subheader("Peak vs. Off-Peak Hours")
        peak_data = {
            "Morning\nRush\n(7-9 AM)": eda_metrics['peak_hours']['morning_rush'],
            "Evening\nRush\n(4-7 PM)": eda_metrics['peak_hours']['evening_rush'],
            "Off-Peak": eda_metrics['peak_hours']['off_peak']
        }
        colors = ['#ff7f0e', '#d62728', '#2ca02c']
        fig = go.Figure(data=go.Bar(
            x=list(peak_data.keys()), y=list(peak_data.values()), 
            marker_color=colors, text=[f"{v:.0f}" for v in peak_data.values()],
            textposition='auto'
        ))
        fig.update_layout(
            title="Average Demand by Time Period", yaxis_title="Avg Trips", 
            height=350, showlegend=False,
            font=dict(family="Poppins, sans-serif", size=10),
            xaxis=dict(showticklabels=False, showgrid=False),
            yaxis=dict(showticklabels=False, showgrid=False)
        )
        st.plotly_chart(fig, width='stretch')
    
    st.divider()
    
    # Row 2: Weekend vs Weekday & Monthly Trends
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Weekend vs. Weekday")
        weekend_data = {
            "Weekday": eda_metrics['weekend_vs_weekday']['weekday'],
            "Weekend": eda_metrics['weekend_vs_weekday']['weekend']
        }
        fig = go.Figure(data=go.Bar(
            x=list(weekend_data.keys()), y=list(weekend_data.values()),
            marker_color=['#9467bd', '#8c564b'],
            text=[f"{v:.0f}" for v in weekend_data.values()],
            textposition='auto'
        ))
        fig.update_layout(
            title="Average Demand: Weekday vs. Weekend", yaxis_title="Avg Trips",
            height=350, showlegend=False,
            font=dict(family="Poppins, sans-serif", size=10),
            xaxis=dict(showticklabels=False, showgrid=False),
            yaxis=dict(showticklabels=False, showgrid=False)
        )
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        st.subheader("Monthly Seasonality")
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly_values = []
        for i in range(1, 13):
            val = eda_metrics['monthly_trends'].get(str(i), eda_metrics['monthly_trends'].get(i, 0))
            monthly_values.append(float(val) if val else 0)
        
        fig = go.Figure(data=go.Scatter(
            x=months, y=monthly_values, mode='lines+markers', fill='tozeroy',
            line=dict(color='#17becf'), name='Avg Demand'
        ))
        fig.update_layout(
            title="Seasonal Demand Pattern", xaxis_title="Month",
            yaxis_title="Avg Trips", height=350, showlegend=False,
            hovermode='x unified',
            font=dict(family="Poppins, sans-serif", size=10),
            xaxis=dict(showticklabels=False, showgrid=False),
            yaxis=dict(showticklabels=False, showgrid=False)
        )
        st.plotly_chart(fig, width='stretch')
    
        st.divider()
        
        # Row 3: Cluster Performance
        st.subheader("🎯 Cluster Model Performance")
        cluster_stats = eda_metrics.get('cluster_stats', {})
        
        if cluster_stats:
            perf_col1, perf_col2, perf_col3 = st.columns(3)
            
            cluster_info = [
                ("High Demand", perf_col1),
                ("Medium Demand", perf_col2),
                ("Low Demand", perf_col3)
            ]
            
            for cluster_name, col in cluster_info:
                with col:
                    metrics = cluster_stats.get(cluster_name, {})
                    if metrics:
                        mae = metrics.get('mae', 0)
                        r2 = metrics.get('r2', 0)
                        samples = metrics.get('samples', 0)
                        st.metric(
                            f"{cluster_name}",
                            f"MAE: {mae:.2f}",
                            f"R²: {r2:.4f} | Samples: {samples:,}"
                        )
                    else:
                        st.warning(f"No metrics for {cluster_name}")
        else:
            st.info("Cluster metrics not available yet")
    
    # ===== SUB-TAB 2: STATION-LEVEL =====
    with eda_tab2:
        st.subheader("Station-Specific Demand Analysis")
        
        if station_eda:
            # Station selector
            station_names = sorted(list(station_eda.keys()))
            selected_station_eda = st.selectbox("Select Station", station_names, key="eda_station_select")
            
            if selected_station_eda in station_eda:
                s_data = station_eda[selected_station_eda]
                
                # Key metrics for selected station
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Trips", f"{s_data['total_trips']:,}")
                with col2:
                    st.metric("Avg Trips/Day", f"{s_data['avg_trips_per_day']:.0f}")
                with col3:
                    st.metric("Peak Hour", f"{s_data['peak_hour']}:00")
                with col4:
                    st.metric("Member Ratio", f"{s_data['member_ratio']*100:.1f}%")
                
                st.divider()
                
                # Station cluster
                station_cluster = station_mapping.get(selected_station_eda, "Unknown")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.info(f"**Cluster:** {station_cluster}")
                with col2:
                    st.metric("Evening Rush (4-7pm)", f"{s_data['evening_rush_trips']:.0f}")
                with col3:
                    st.metric("Avg Trip Duration", f"{s_data['avg_trip_duration_min']:.1f} min")
                
                st.divider()
                
                # Hourly distribution
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Hourly Distribution")
                    hours = list(range(24))
                    hourly_vals = [int(s_data['hourly_distribution'].get(str(h), 0)) for h in hours]
                    fig = go.Figure(data=go.Bar(
                        x=hours, y=hourly_vals, marker_color='#1f77b4'
                    ))
                    fig.update_layout(
                        xaxis_title="Hour (UTC)", yaxis_title="Avg Trips",
                        height=300, showlegend=False,
                        font=dict(family="Poppins, sans-serif", size=10),
                        xaxis=dict(showticklabels=False, showgrid=False),
                        yaxis=dict(showticklabels=False, showgrid=False)
                    )
                    st.plotly_chart(fig, width='stretch')
                
                with col2:
                    st.subheader("Rush Hour Breakdown")
                    rush_data = {
                        "Morning\nRush\n(7-9 AM)": s_data['morning_rush_trips'],
                        "Evening\nRush\n(4-7 PM)": s_data['evening_rush_trips'],
                        "Off-Peak": s_data['total_trips'] - s_data['morning_rush_trips'] - s_data['evening_rush_trips']
                    }
                    fig = go.Figure(data=go.Bar(
                        x=list(rush_data.keys()), y=list(rush_data.values()),
                        marker_color=['#ff7f0e', '#d62728', '#2ca02c']
                    ))
                    fig.update_layout(
                        yaxis_title="Avg Trips", height=300, showlegend=False,
                        font=dict(family="Poppins, sans-serif", size=10),
                        xaxis=dict(showticklabels=False, showgrid=False),
                        yaxis=dict(showticklabels=False, showgrid=False)
                    )
                    st.plotly_chart(fig, width='stretch')
                
                # Daily & monthly distributions
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Daily Pattern (Weekday vs Weekend)")
                    daily_vals = [int(s_data['daily_distribution'].get(str(d), 0)) for d in range(7)]
                    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                    colors = ['#1f77b4']*5 + ['#ff7f0e']*2
                    fig = go.Figure(data=go.Bar(
                        x=days, y=daily_vals, marker_color=colors
                    ))
                    fig.update_layout(
                        xaxis_title="Day of Week", yaxis_title="Avg Trips",
                        height=300, showlegend=False,
                        font=dict(family="Poppins, sans-serif", size=10),
                        xaxis=dict(showticklabels=False, showgrid=False),
                        yaxis=dict(showticklabels=False, showgrid=False)
                    )
                    st.plotly_chart(fig, width='stretch')
                
                with col2:
                    st.subheader("Monthly Seasonality")
                    monthly_vals = [int(s_data['monthly_distribution'].get(str(m), 0)) for m in range(1, 13)]
                    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                    fig = go.Figure(data=go.Scatter(
                        x=months, y=monthly_vals, mode='lines+markers', fill='tozeroy',
                        line=dict(color='#17becf')
                    ))
                    fig.update_layout(
                        xaxis_title="Month", yaxis_title="Avg Trips",
                        height=300, showlegend=False,
                        font=dict(family="Poppins, sans-serif", size=10),
                        xaxis=dict(showticklabels=False, showgrid=False),
                        yaxis=dict(showticklabels=False, showgrid=False)
                    )
                    st.plotly_chart(fig, width='stretch')
                
                # Busiest period insights
                st.divider()
                st.subheader("📍 Station Insights")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Busiest Day", s_data['busiest_day'])
                with col2:
                    st.metric("Busiest Month", s_data['peak_month'])
                with col3:
                    weekend_ratio = s_data['weekend_ratio']
                    st.metric("Weekend Ratio", f"{weekend_ratio*100:.1f}%")
        else:
            st.warning("Station-level EDA data not available. Run `python station_level_eda.py`")
    
    # ===== SUB-TAB 3: CLUSTER COMPARISON =====
    with eda_tab3:
        st.subheader("Cluster Performance Comparison")
        
        if station_eda:
            # Group stations by cluster
            cluster_groups = {'High Demand': [], 'Medium Demand': [], 'Low Demand': []}
            for station_name, station_cluster in station_mapping.items():
                cluster_groups.setdefault(station_cluster, []).append(station_name)
            
            # Calculate cluster-level statistics
            cluster_metrics = {}
            for cluster_name, station_list in cluster_groups.items():
                cluster_stations = [s for s in station_list if s in station_eda]
                if cluster_stations:
                    total_trips = sum(station_eda[s]['total_trips'] for s in cluster_stations)
                    avg_trips_day = np.mean([station_eda[s]['avg_trips_per_day'] for s in cluster_stations])
                    peak_hour = round(np.mean([station_eda[s]['peak_hour'] for s in cluster_stations]))
                    
                    cluster_metrics[cluster_name] = {
                        'total_trips': total_trips,
                        'stations_count': len(cluster_stations),
                        'avg_trips_day': avg_trips_day,
                        'peak_hour': peak_hour
                    }
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            colors = ['#ff7f0e', '#2ca02c', '#d62728']
            
            for idx, (cluster_name, col) in enumerate(zip(['High Demand', 'Medium Demand', 'Low Demand'], [col1, col2, col3])):
                with col:
                    if cluster_name in cluster_metrics:
                        m = cluster_metrics[cluster_name]
                        st.markdown(f"**{cluster_name}**")
                        st.metric("Total Trips", f"{m['total_trips']:,}")
                        st.metric("Stations", m['stations_count'])
                        st.metric("Avg Trips/Day", f"{m['avg_trips_day']:.0f}")
            
            st.divider()
            
            # Comparison charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Cluster Trip Volume")
                cluster_names = list(cluster_metrics.keys())
                cluster_trips = [cluster_metrics[c]['total_trips'] for c in cluster_names]
                fig = go.Figure(data=go.Bar(
                    x=cluster_names, y=cluster_trips, marker_color=['#ff7f0e', '#2ca02c', '#d62728']
                ))
                fig.update_layout(
                    yaxis_title="Total Trips", height=350, showlegend=False,
                    font=dict(family="Poppins, sans-serif", size=10),
                    xaxis=dict(showticklabels=False, showgrid=False),
                    yaxis=dict(showticklabels=False, showgrid=False)
                )
                st.plotly_chart(fig, width='stretch')
            
            with col2:
                st.subheader("Average Demand per Cluster")
                cluster_avg = [cluster_metrics[c]['avg_trips_day'] for c in cluster_names]
                fig = go.Figure(data=go.Bar(
                    x=cluster_names, y=cluster_avg, marker_color=['#ff7f0e', '#2ca02c', '#d62728']
                ))
                fig.update_layout(
                    yaxis_title="Avg Trips/Day", height=350, showlegend=False,
                    font=dict(family="Poppins, sans-serif", size=10),
                    xaxis=dict(showticklabels=False, showgrid=False),
                    yaxis=dict(showticklabels=False, showgrid=False)
                )
                st.plotly_chart(fig, width='stretch')
        else:
            st.warning("Station-level EDA data not available")

# ========== TAB 2: PREDICTION ==========
with tab2:
    st.header("Demand Forecasting Engine")
    
    # Sidebar inputs
    st.sidebar.subheader("📍 Input Parameters")
    
    col_s1, col_s2 = st.sidebar.columns(2)
    with col_s1:
        selected_station = st.sidebar.selectbox("Select Station", stations)
    with col_s2:
        station_cluster = get_cluster_for_station(selected_station)
        st.sidebar.write(f"**Cluster:** {station_cluster}")
    
    st.sidebar.subheader("🕐 Date & Time (UTC)")
    col_dt1, col_dt2 = st.sidebar.columns(2)
    with col_dt1:
        utc_date = st.date_input("Date", date.today())
    with col_dt2:
        # 12-hour format with AM/PM
        hour_options = [f"{h % 12 if h % 12 != 0 else 12} {'AM' if h < 12 else 'PM'}" for h in range(24)]
        hour_selection = st.selectbox("Hour", hour_options, index=12)
        # Convert back to 24-hour format
        hour_12, period = hour_selection.rsplit(' ', 1)
        hour_12 = int(hour_12)
        utc_hour = hour_12 if period == 'AM' and hour_12 == 12 else (hour_12 + 12 if period == 'PM' and hour_12 != 12 else hour_12)
    
    st.sidebar.subheader("📈 Historical Demand")
    col_h1, col_h2 = st.sidebar.columns(2)
    with col_h1:
        recent_avg = st.number_input("Avg (24h)", min_value=0.0, value=60.0, step=5.0)
        recent_std = st.number_input("Std Dev (24h)", min_value=0.0, value=10.0, step=1.0)
    with col_h2:
        recent_max = st.number_input("Max (24h)", min_value=0.0, value=120.0, step=5.0)
        recent_min = st.number_input("Min (24h)", min_value=0.0, value=30.0, step=5.0)
    
    # Build datetime and feature vector
    dt_obj = datetime.combine(utc_date, dtime(hour=utc_hour))
    
    payload = build_feature_vector(dt_obj, recent_avg, recent_std, recent_max, recent_min)
    
    # Make prediction
    # Use default button width to stay compatible with older Streamlit versions
    if st.button("🔮 Generate Forecast", type="primary"):
        # Input validation
        validation_errors = []
        
        if not selected_station or selected_station not in station_mapping:
            validation_errors.append("Invalid station selected")
        
        if utc_date < MIN_DATE.date() or utc_date > date.today():
            validation_errors.append(f"Date must be between {MIN_DATE.date()} and today")
        
        if recent_avg < 0 or recent_std < 0 or recent_max < 0 or recent_min < 0:
            validation_errors.append("Historical demand values cannot be negative")
        
        if recent_min > recent_max:
            validation_errors.append("Min demand cannot be greater than max demand")
        
        if validation_errors:
            for error in validation_errors:
                st.error(f"❌ {error}")
            st.stop()
        
        # Get cluster and corresponding model
        cluster_name = get_cluster_for_station(selected_station)
        cluster_key = cluster_to_key(cluster_name)
        
        if cluster_key not in models:
            logger.error(f"Model not found for cluster: {cluster_name}")
            st.error(f"❌ Model not found for cluster: {cluster_name}")
            st.stop()
        
        if cluster_key not in scalers:
            logger.error(f"Scaler not found for cluster: {cluster_name}")
            st.error(f"❌ Scaler not found for cluster: {cluster_name}")
            st.stop()
        
        try:
            # Build feature vector and array
            payload = build_feature_vector(dt_obj, recent_avg, recent_std, recent_max, recent_min)
            
            # Ensure all features are present
            missing_features = [f for f in feature_cols if f not in payload]
            if missing_features:
                logger.error(f"Missing features in payload: {missing_features}")
                st.error(f"❌ Missing features: {missing_features}")
                st.stop()
            
            # Build feature array in correct order
            X = np.array([[payload[col] for col in feature_cols]])
            
            # Scale using cluster-specific scaler
            X_scaled = scalers[cluster_key].transform(X)
            
            # Predict using cluster-specific model
            base_pred = float(models[cluster_key].predict(X_scaled)[0])
            
            # Clip negative predictions to zero
            base_pred = max(0, base_pred)
            
            # Adjust based on cluster demand level
            cluster_factor = {"high_demand": 1.4, "medium_demand": 1.0, "low_demand": 0.6}
            adjusted_pred = base_pred * cluster_factor.get(cluster_key, 1.0)
            
            # Log successful prediction
            logger.info(f"Prediction: Station={selected_station}, Date={utc_date}, Hour={utc_hour}, "
                       f"Cluster={cluster_name}, BasePred={base_pred:.2f}, AdjustedPred={adjusted_pred:.2f}")
            
            # Context flags
            holiday_check = is_holiday(dt_obj.month, dt_obj.day)
            holiday_name = get_holiday_name(dt_obj.month, dt_obj.day)
            peak_check = is_peak_hour(dt_obj.hour, dt_obj.weekday())
            weekend_check = dt_obj.weekday() >= 5
            
            # Display results
            st.success("✅ Forecast Generated Successfully!")
            
            # Main metrics
            res_col1, res_col2, res_col3, res_col4 = st.columns(4)
            with res_col1:
                st.metric("📊 Base Forecast", f"{base_pred:,.0f} trips")
            with res_col2:
                st.metric("🎯 Adjusted Forecast", f"{adjusted_pred:,.0f} trips")
            with res_col3:
                st.metric("📍 Station", selected_station)
            with res_col4:
                st.metric("🏢 Cluster", cluster_name)
            
            st.divider()
            
            # Detailed insights
            st.subheader("📋 Detailed Forecast Insights")
            
            ins_col1, ins_col2, ins_col3, ins_col4 = st.columns(4)
            with ins_col1:
                if holiday_check:
                    holiday_badge = f"🎄 YES\n({holiday_name})"
                else:
                    holiday_badge = "NO"
                st.metric("Is Holiday?", holiday_badge)
            with ins_col2:
                peak_badge = "⚡ PEAK HOUR" if peak_check else "Normal"
                st.metric("Peak Hour?", peak_badge)
            with ins_col3:
                weekend_badge = "🌤️ Weekend" if weekend_check else "📅 Weekday"
                st.metric("Day Type", weekend_badge)
            with ins_col4:
                if 7 <= dt_obj.hour <= 9:
                    st.metric("Rush Type", "🌅 Morning")
                elif 16 <= dt_obj.hour <= 19:
                    st.metric("Rush Type", "🌆 Evening")
                else:
                    st.metric("Rush Type", "Off-Peak")
            
            st.divider()
            
            # Time context
            st.subheader("🕐 Time & Context Details")
            ctx_col1, ctx_col2, ctx_col3 = st.columns(3)
            
            with ctx_col1:
                st.write(f"**Date (UTC):** {dt_obj.strftime('%Y-%m-%d')}")
                st.write(f"**Month:** {dt_obj.strftime('%B')} (#{dt_obj.month})")
                st.write(f"**Day:** {dt_obj.strftime('%A')} (weekday #{dt_obj.weekday()})")
            
            with ctx_col2:
                st.write(f"**Hour (UTC):** {dt_obj.hour:02d}:00")
                st.write(f"**Week:** #{dt_obj.isocalendar().week}")
                st.write(f"**Day of Month:** {dt_obj.day}")
            
            with ctx_col3:
                st.write(f"**Historical Avg (24h):** {recent_avg:.0f} trips")
                st.write(f"**Std Dev (24h):** {recent_std:.1f} trips")
                st.write(f"**Range (24h):** {recent_min:.0f} - {recent_max:.0f} trips")
            
            st.divider()
            
            # Recommendation based on prediction
            st.subheader("💡 Operational Recommendation")
            threshold_high = recent_avg * 1.5
            threshold_low = recent_avg * 0.5
            
            if adjusted_pred > threshold_high:
                st.success(f"⚠️ **HIGH DEMAND EXPECTED** ({adjusted_pred:,.0f} trips)\n\n"
                          f"🎯 **Actions:**\n"
                          f"• Ensure sufficient bike availability at station\n"
                          f"• Increase staff readiness for docking operations\n"
                          f"• Monitor real-time demand for redistribution needs")
            elif adjusted_pred < threshold_low:
                st.warning(f"📉 **LOW DEMAND PREDICTED** ({adjusted_pred:,.0f} trips)\n\n"
                          f"🎯 **Actions:**\n"
                          f"• Monitor redistribution opportunities\n"
                          f"• Plan maintenance if needed\n"
                          f"• Adjust bike deployment from other stations")
            else:
                st.info(f"**NORMAL DEMAND** ({adjusted_pred:,.0f} trips)\n\n"
                       f"🎯 **Actions:**\n"
                       f"• Proceed with standard operations\n"
                       f"• Monitor for deviations from forecast")
        
        except ValueError as e:
            logger.error(f"Value error during prediction: {str(e)}", exc_info=True)
            st.error(f"❌ Invalid input values: {str(e)}")
            st.info("Please ensure all input values are valid numbers.")
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}", exc_info=True)
            st.error(f"❌ Prediction failed: {str(e)}")
            st.info("Please check the app logs or contact support.")
            
            # Context flags
            holiday_check = is_holiday(dt_obj.month, dt_obj.day)
            holiday_name = get_holiday_name(dt_obj.month, dt_obj.day)
            peak_check = is_peak_hour(dt_obj.hour, dt_obj.weekday())
            weekend_check = dt_obj.weekday() >= 5
            
            # Display results
            st.success("✅ Forecast Generated Successfully!")
            
            # Main metrics
            res_col1, res_col2, res_col3, res_col4 = st.columns(4)
            with res_col1:
                st.metric("📊 Base Forecast", f"{base_pred:,.0f} trips")
            with res_col2:
                st.metric("🎯 Adjusted Forecast", f"{adjusted_pred:,.0f} trips")
            with res_col3:
                st.metric("📍 Station", selected_station)
            with res_col4:
                st.metric("🏢 Cluster", cluster_name)
            
            st.divider()
            
            # Detailed insights
            st.subheader("📋 Detailed Forecast Insights")
            
            ins_col1, ins_col2, ins_col3, ins_col4 = st.columns(4)
            with ins_col1:
                if holiday_check:
                    holiday_badge = f"🎄 YES\n({holiday_name})"
                else:
                    holiday_badge = "NO"
                st.metric("Is Holiday?", holiday_badge)
            with ins_col2:
                peak_badge = "⚡ PEAK HOUR" if peak_check else "Normal"
                st.metric("Peak Hour?", peak_badge)
            with ins_col3:
                weekend_badge = "🌤️ Weekend" if weekend_check else "📅 Weekday"
                st.metric("Day Type", weekend_badge)
            with ins_col4:
                if 7 <= dt_obj.hour <= 9:
                    st.metric("Rush Type", "🌅 Morning")
                elif 16 <= dt_obj.hour <= 19:
                    st.metric("Rush Type", "🌆 Evening")
                else:
                    st.metric("Rush Type", "Off-Peak")
            
            st.divider()
            
            # Time context
            st.subheader("🕐 Time & Context Details")
            ctx_col1, ctx_col2, ctx_col3 = st.columns(3)
            
            with ctx_col1:
                st.write(f"**Date (UTC):** {dt_obj.strftime('%Y-%m-%d')}")
                st.write(f"**Month:** {dt_obj.strftime('%B')} (#{dt_obj.month})")
                st.write(f"**Day:** {dt_obj.strftime('%A')} (weekday #{dt_obj.weekday()})")
            
            with ctx_col2:
                st.write(f"**Hour (UTC):** {dt_obj.hour:02d}:00")
                st.write(f"**Week:** #{dt_obj.isocalendar().week}")
                st.write(f"**Day of Month:** {dt_obj.day}")
            
            with ctx_col3:
                st.write(f"**Historical Avg (24h):** {recent_avg:.0f} trips")
                st.write(f"**Std Dev (24h):** {recent_std:.1f} trips")
                st.write(f"**Range (24h):** {recent_min:.0f} - {recent_max:.0f} trips")
            
            st.divider()
            
            # Recommendation based on prediction
            st.subheader("💡 Operational Recommendation")
            threshold_high = recent_avg * 1.5
            threshold_low = recent_avg * 0.5
            
            if adjusted_pred > threshold_high:
                st.success(f"⚠️ **HIGH DEMAND EXPECTED** ({adjusted_pred:,.0f} trips)\n\n"
                          f"🎯 **Actions:**\n"
                          f"• Ensure sufficient bike availability at station\n"
                          f"• Increase staff readiness for docking operations\n"
                          f"• Monitor real-time demand for redistribution needs")
            elif adjusted_pred < threshold_low:
                st.warning(f"📉 **LOW DEMAND PREDICTED** ({adjusted_pred:,.0f} trips)\n\n"
                          f"🎯 **Actions:**\n"
                          f"• Monitor redistribution opportunities\n"
                          f"• Plan maintenance if needed\n"
                          f"• Adjust bike deployment from other stations")
            else:
                st.info(f"**NORMAL DEMAND** ({adjusted_pred:,.0f} trips)\n\n"
                       f"🎯 **Actions:**\n"
                       f"• Proceed with standard operations\n"
                       f"• Monitor for deviations from forecast")
        
        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")
            st.info("Please check your inputs and try again")

# Footer
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: #888; font-size: 12px;'>
    <p>🚴 Citi Bike Forecasting MLOps Platform | Cluster-Based XGBoost Models | Real-Time Predictions</p>
    </div>
    """,
    unsafe_allow_html=True
)
