# Citi Bike MLOps Web Application

A production-ready machine learning forecasting platform for Citi Bike demand prediction using cluster-based models.

## 🚀 Quick Start

### 1. Train Cluster-Based Models
```bash
python train_cluster_models.py
```
This will:
- Load all 69 CSV files from the `Dataset/` folder
- Train 3 cluster-specific XGBoost models (High/Medium/Low demand)
- Generate EDA metrics and visualizations
- Save artifacts to `artifacts/cluster_models/`

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Web App
```bash
streamlit run app_enhanced.py
```
Open http://localhost:8501 in your browser.

---

## 📊 Application Features

### **Tab 1: Analysis (EDA)**
- **Key Metrics**: Total trips, stations, date range, average demand
- **Hourly Distribution**: Demand trends across 24 hours
- **Peak Hours Analysis**: Morning rush, evening rush, off-peak comparison
- **Seasonal Patterns**: Weekend vs. weekday, monthly trends
- **Model Performance**: R² and MAE for each cluster

### **Tab 2: Prediction (Forecast)**
- **Station Selection**: Choose from 51 Citi Bike stations
- **UTC DateTime Input**: Select date and hour in UTC
- **Historical Context**: Input recent demand stats (avg, std, min, max)
- **Cluster-Based Forecasting**: Uses cluster-specific XGBoost model
- **Detailed Output**:
  - Base and adjusted demand forecasts
  - Holiday/peak hour detection
  - Weekday/weekend classification
  - Rush hour identification (morning/evening)
  - Operational recommendations

---

## 🏗️ Architecture

```
artifacts/
├── xgboost_v1.joblib              # System-level model (legacy)
├── station_cluster_mapping.json    # Station → Cluster mapping
├── eda_metrics.json                # EDA statistics & trends
├── feature_cols.json               # Feature column names
└── cluster_models/
    ├── high_demand_model.joblib    # High-demand cluster model
    ├── high_demand_scaler.joblib   # High-demand feature scaler
    ├── medium_demand_model.joblib  # Medium-demand cluster model
    ├── medium_demand_scaler.joblib
    ├── low_demand_model.joblib     # Low-demand cluster model
    └── low_demand_scaler.joblib
```

---

## 🐳 Docker Deployment

### Build Docker Image
```bash
docker build -t citi-bike-mlops .
```

### Run Container
```bash
docker run -it --rm -p 8501:8501 citi-bike-mlops
```

### Deploy to Render/Fly.io
1. Push to GitHub
2. Create Web Service in Render (Docker)
3. Set Port: 8501
4. Deploy!

---

## 📈 Model Details

| Cluster | Samples | MAE | R² |
|---------|---------|-----|-----|
| High Demand | See eda_metrics.json | Computed | Computed |
| Medium Demand | See eda_metrics.json | Computed | Computed |
| Low Demand | See eda_metrics.json | Computed | Computed |

---

## 🔧 Configuration

- **Feature Engineering**: 24 features including cyclical encoding, lags, and rolling statistics
- **Model**: XGBoost Regressor (300 estimators, depth=6, learning_rate=0.1)
- **Train/Test Split**: 80/20 stratified by time
- **Scaling**: StandardScaler per cluster

---

## 📁 Files Overview

| File | Purpose |
|------|---------|
| `train_cluster_models.py` | Train cluster-specific models |
| `app_enhanced.py` | Main Streamlit web application |
| `save_model.py` | Legacy script for system-level model |
| `extract_stations.py` | Extract station names & mapping |
| `streamlit_app.py` | Basic Streamlit app (use app_enhanced.py) |
| `Dockerfile` | Container configuration |
| `requirements.txt` | Python dependencies |

---

## 🎯 Next Steps

1. Run `python train_cluster_models.py` to generate models
2. Start web app with `streamlit run app_enhanced.py`
3. Test predictions with different stations and times
4. Deploy to Render/Fly for production
5. Monitor predictions and retrain monthly

---

**Status**: ✅ Ready for MLOps deployment
