"""
Configuration Module
====================

Central configuration for the new product forecasting system.
All constants, parameters, and paths are defined here.
"""

from pathlib import Path
import os
import sys

# Add parent directory to path to allow importing from parent config
sys.path.append(str(Path(__file__).parent.parent))

try:
    from ..config import SWEDISH_HOLIDAYS_PATH
except ImportError:
    # Fallback if relative import fails (e.g. when running as script)
    # This might happen if planogram is not treated as a package
    SWEDISH_HOLIDAYS_PATH = None

# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent

# Data directories (raw data is in data/raw/, processed will be in data/processed/)
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
MODEL_DIR = PROJECT_ROOT / 'moaaz_trend' / 'artifact'

# Swedish holidays file
if SWEDISH_HOLIDAYS_PATH:
    HOLIDAYS_FILE = Path(SWEDISH_HOLIDAYS_PATH)
else:
    raise ImportError("Could not resolve SWEDISH_HOLIDAYS_PATH from parent config, and no fallback is provided for HOLIDAYS_FILE.")

# Note: Directories are created on-demand when files are saved
# (e.g., in processor.py, save_utils.py, etc.)
# This prevents creating empty directories when just importing the config

# ============================================================================
# DATA PARAMETERS
# ============================================================================

# Years to load
DATA_YEARS = list(range(2017, 2026))  # 2017-2025

# Date range filters
MIN_DATE = "2018-01-01"
MAX_DATE = "2025-08-31"

# Essential columns to keep after cleaning
ESSENTIAL_COLUMNS = [
    'machine_key',
    'ean',
    'local_timestamp',
    'price',
    'position',
    'product_name',
    'provider',
    'category',
    'subcategory',
    'machine_eva_group',
    'machine_sub_group',
    'refiller',
    'customer_id',
    'purchase_price_kr'
]

# ============================================================================
# FEATURE ENGINEERING FLAGS
# ============================================================================

# Feature categories - all disabled initially, enable incrementally
FEATURE_CATEGORIES = {
    'cold_start': False,
    'historical_sales': False,
    'temporal': False,
    'product_machine': False,
    'interactions': False
}

# ============================================================================
# MODEL PARAMETERS
# ============================================================================

# XGBoost parameters (from research baseline)
XGBOOST_PARAMS = {
    'n_estimators': 300,
    'max_depth': 9,
    'learning_rate': 0.0369,
    'subsample': 0.92,
    'colsample_bytree': 0.99,
    'reg_alpha': 10,
    'reg_lambda': 0,
    'min_child_weight': 20,
    'random_state': 42,
    'tree_method': 'hist',
    'n_jobs': -1
}

# Optimized hyperparameters from VendTrend experiment
# These are the relevant params that can be used in our architecture
VENDTREND_OPTIMISED_PARAMS = {
    'objective': 'reg:absoluteerror',
    'learning_rate': 0.03686698762504045,
    'max_depth': 9,
    'subsample': 0.9648307299541896,
    'colsample_bytree': 0.9944938162251564,
    'reg_alpha': 10,
    'reg_lambda': 0,
    'min_child_weight': 20
}

# Prediction parameters
WEEKS_TO_PREDICT = 4

# Reproducibility
RANDOM_SEED = 42

# ============================================================================
# FILTERING THRESHOLDS
# ============================================================================

TEST_PRICE_THRESHOLD = 2  # Remove transactions ≤ 2 SEK (test data)
MIN_TRANSACTIONS_PER_MACHINE = 10  # Remove low-activity machines
CORRELATION_THRESHOLD = 0.95  # Remove redundant features

