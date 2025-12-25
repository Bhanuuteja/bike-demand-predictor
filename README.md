# 🚲 Citi Bike Demand Forecasting - MLOps Dashboard

A production-ready machine learning application for predicting bike-sharing demand using cluster-based XGBoost models with an interactive Streamlit interface.

## 🌟 Features

- **Cluster-Based Predictions** - Separate models for high, medium, and low demand stations
- **Interactive Dashboard** - Real-time exploratory data analysis and forecasting
- **Station-Level Analytics** - Detailed insights for individual bike stations
- **Production Hardened** - Comprehensive error handling, logging, and validation
- **Docker Ready** - Containerized for consistent deployment anywhere

## 📊 Tech Stack

- **Frontend:** Streamlit
- **ML Framework:** XGBoost, scikit-learn
- **Data Processing:** Pandas, NumPy
- **Visualization:** Plotly
- **Deployment:** Docker + Render
- **Language:** Python 3.11

## 🚀 Quick Start

### Local Development (Without Docker)

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app_enhanced.py
```

### Local Development (With Docker)

```bash
# Build and run with Docker Compose
docker-compose up

# Or build manually
docker build -t bike-predictor .
docker run -p 8501:8501 bike-predictor
```

Access at: **http://localhost:8501**

## 📁 Project Structure

```
ML_Final_Project/
├── app_enhanced.py           # Main Streamlit application
├── config.py                 # Configuration management
├── train_cluster_models.py   # Model training script
├── station_level_eda.py      # EDA generation
├── save_model.py             # Model persistence
├── extract_stations.py       # Station data extraction
├── requirements.txt          # Python dependencies
├── Dockerfile               # Docker container definition
├── docker-compose.yml       # Docker Compose configuration
├── render.yaml              # Render deployment config
├── artifacts/               # Trained models and metadata
│   ├── cluster_models/      # Cluster-specific models
│   ├── station_cluster_mapping.json
│   ├── eda_metrics.json
│   └── feature_cols.json
└── Dataset/                 # Historical trip data (2020-2024)
```

## 🔧 Configuration

Environment variables (`.env`):

```bash
MODEL_PATH=artifacts/xgboost_v1.joblib
CLUSTER_MODELS_DIR=artifacts/cluster_models
DATA_DIR=Dataset
LOG_LEVEL=INFO
MIN_DATE=2020-01-01
MAX_DATE=2024-04-30
```

## 🎯 Model Details

### Architecture
- **Base Model:** XGBoost Regressor
- **Strategy:** Cluster-based modeling (High/Medium/Low demand)
- **Features:** 24 engineered features including temporal, cyclical, and lag features

### Features
- Temporal: hour, day of week, month, week of year
- Cyclical: sine/cosine transformations for hour, day, month
- Lag features: 1h, 24h, 168h rolling statistics
- Calendar: holidays, weekends, rush hours
- Historical: rolling mean, std, max, min

### Performance
- **High Demand Cluster:** MAE < 5.0, R² > 0.85
- **Medium Demand Cluster:** MAE < 3.0, R² > 0.80
- **Low Demand Cluster:** MAE < 2.0, R² > 0.75

## 📊 Data

- **Source:** Citi Bike Jersey City Trip Data (2020-2024)
- **Records:** 4M+ bike trips
- **Stations:** 50+ bike stations
- **Features:** Station, timestamp, user type, trip duration

## 🐳 Deployment

### Render (Recommended)

```bash
# Push to GitHub
git push origin main

# Render auto-deploys from GitHub
# Uses render.yaml configuration
```

### Manual Deployment

```bash
# Build Docker image
docker build -t bike-predictor .

# Tag for registry
docker tag bike-predictor:latest your-registry/bike-predictor:latest

# Push to registry
docker push your-registry/bike-predictor:latest

# Deploy anywhere (AWS, Azure, GCP, DigitalOcean, etc.)
```

## 🛠️ Development

### Training Models

```bash
# Generate station-level EDA
python station_level_eda.py

# Train cluster models
python train_cluster_models.py

# Save main model
python save_model.py
```

### Testing

```bash
# Test locally
streamlit run app_enhanced.py

# Test with Docker
docker-compose up

# Check logs
tail -f app.log
```

## 📈 Production Features

✅ **Error Handling** - Comprehensive try/catch with user-friendly messages  
✅ **Logging** - File-based logging with timestamps  
✅ **Input Validation** - Date ranges, numerical bounds, feature validation  
✅ **Caching** - Model and data caching for performance  
✅ **Configuration** - Environment-based config management  
✅ **Health Checks** - Docker health monitoring  
✅ **Security** - Non-root user in container  

## 🎓 Use Cases

- **Operations Planning** - Optimize bike distribution across stations
- **Demand Forecasting** - Predict future demand for capacity planning
- **Resource Allocation** - Staff deployment during peak hours
- **Maintenance Scheduling** - Plan maintenance during low-demand periods

## 📝 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions welcome! Please open an issue or submit a pull request.

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Built with ❤️ for smarter urban mobility**
