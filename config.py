"""Configuration module for Citi Bike MLOps App."""
import os
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Paths
BASE_DIR = Path(__file__).parent
MODEL_PATH = Path(os.getenv('MODEL_PATH', 'artifacts/xgboost_v1.joblib'))
CLUSTER_MODELS_DIR = Path(os.getenv('CLUSTER_MODELS_DIR', 'artifacts/cluster_models'))
DATA_DIR = Path(os.getenv('DATA_DIR', 'Dataset'))
ARTIFACT_DIR = Path('artifacts')

# Date ranges
MIN_DATE = datetime.strptime(os.getenv('MIN_DATE', '2020-01-01'), '%Y-%m-%d')
MAX_DATE = datetime.strptime(os.getenv('MAX_DATE', '2025-09-30'), '%Y-%m-%d')

# Logging
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FILE = 'app.log'

# Default values
DEFAULT_BASELINE_DEMAND = 50
DEFAULT_STATIONS = ["5 Corners Library", "Downtown Station", "Central Park"]

# Validation
REQUIRED_FILES = {
    'model': MODEL_PATH,
    'cluster_models': CLUSTER_MODELS_DIR,
    'artifacts': ARTIFACT_DIR
}

def validate_environment():
    """Validate that required directories exist."""
    errors = []
    
    if not ARTIFACT_DIR.exists():
        errors.append(f"Missing artifacts directory: {ARTIFACT_DIR}")
    
    if not CLUSTER_MODELS_DIR.exists():
        errors.append(f"Missing cluster models directory: {CLUSTER_MODELS_DIR}")
    
    # DATA_DIR is optional for serving predictions; skip strict validation
    return errors
