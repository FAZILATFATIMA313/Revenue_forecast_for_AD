# src/config.py
import os
from pathlib import Path

# dotenv is optional: make sure importing config never fails during scoring
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except ImportError:
    load_dotenv = None  # noqa: F401

# Project paths
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
PICKLE_DIR = ROOT_DIR / "pickle"
OUTPUT_DIR = ROOT_DIR / "output"
SRC_DIR = ROOT_DIR / "src"

# Create directories if they don't exist
for dir_path in [DATA_DIR, PICKLE_DIR, OUTPUT_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# LLM Configuration (Groq)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")

# Forecasting defaults
DEFAULT_PERIODS = [30, 60, 90]
QUANTILES = [0.1, 0.5, 0.9]

# Model parameters
RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))
LGBM_PARAMS = {
    'max_depth': 3,
    'num_leaves': 7,
    'min_data_in_leaf': 20,
    'lambda_l1': 0.5,
    'lambda_l2': 0.5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': RANDOM_SEED,
    'verbose': -1
}

# Anomaly detection
ANOMALY_IQR_MULTIPLIER = 1.5
ANOMALY_WINDOW_DAYS = 30

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Shared campaign-type normalization mapping (training + inference MUST use the same set)
# Ensures feature-name consistency between src/features.py and src/generate_features.py.
CAMPAIGN_TYPE_MAPPING = {
    'SEARCH': 'search',
    'Search': 'search',
    'PERFORMANCE_MAX': 'pmax',
    'PerformanceMax': 'pmax',
    'DISPLAY': 'display',
    'Display': 'display',
    'VIDEO': 'video',
    'Video': 'video',
    'DEMAND_GEN': 'demand_gen',
    'SHOPPING': 'shopping',
    'Shopping': 'shopping',
    'Audience': 'audience',
    'unknown': 'other',
    '': 'other',
}

# All known normalized ctype buckets — used by generate_features.py to emit zero-filled columns
# even when a type has no rows in the feature window, preventing silent feature-name drift.
KNOWN_CTYPE_BUCKETS = ['search', 'pmax', 'display', 'video', 'demand_gen', 'shopping', 'audience', 'other']
