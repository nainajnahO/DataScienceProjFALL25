"""
Fallback Configuration
Constants and parameters for the fallback architecture.

Broadly speaking, the variables in this file dictate the order in which fallback levels are tried,
and the specificity of each level.
"""

from pathlib import Path

# HIERARCHY LEVELS (DEFINE THE SPECIFICITY OF LEVELS IN ORDER)
# -----------------------------------------------------------

# Machine aggregation levels (most specific to least specific)
MACHINE_LEVELS = ['machine_id', 'machine_sub_group', 'machine_eva_group', 'ALL']

# Product aggregation levels (most specific to least specific)
PRODUCT_LEVELS = ['product_name', 'subcategory', 'category']

# THRESHOLD PARAMETERS
# -----------------------------------------------------------

# Minimum number of transactions required for each fallback level
MIN_SAMPLE_THRESHOLDS = {
    'machine_id': {
        'product_name': 15,
        'subcategory': 20,
        'category': 30
    },
    'machine_sub_group': {
        'product_name': 50,
        'subcategory': 100,
        'category': 150
    },
    'machine_eva_group': {
        'product_name': 100,
        'subcategory': 150,
        'category': 200
    },
    'ALL': {
        'product_name': 500,
        'subcategory': 500,
        'category': 500
    }
}

# Default minimum sample size (used if a specific threshold not defined)
DEFAULT_MIN_SAMPLES = 15

# Minimum number of unique weeks required
MIN_WEEKS_THRESHOLD = 4

# CONFIDENCE WEIGHTS
# -----------------------------------------------------------

# Confidence multiplier for each fallback level (0-11)
# Higher levels (more general) have lower confidence
# These levels are just symbolic and carry no evidence, but rather a
# good measure of how specific each level is and what the product of the multiplier has become.

CONFIDENCE_WEIGHTS = {
    0: 0.90,  # Level 0: Most specific (the product is found in the machine)
    1: 0.85,
    2: 0.80,
    3: 0.75,
    4: 0.70,
    5: 0.65,
    6: 0.60,
    7: 0.50,
    8: 0.40,
    9: 0.30,
    10: 0.20,
    11: 0.10,  # Level 11: Most general (any product of the same category in any machine)
}

# DATA PATHS
# -----------------------------------------------------------

# Base directory for the RBM project
BASE_DIR = Path(__file__).parent.parent

# Directory containing processed sales data
PROCESSED_DATA_DIR = BASE_DIR.parent.parent / "processed"

# PAIR MAPPING
# -----------------------------------------------------------

# Map pair number (1-12) to (machine_level, product_level) tuple
PAIR_MAPPING = {
    1: ('machine_id', 'product_name'),
    2: ('machine_id', 'subcategory'),
    3: ('machine_id', 'category'),
    4: ('machine_sub_group', 'product_name'),
    5: ('machine_sub_group', 'subcategory'),
    6: ('machine_sub_group', 'category'),
    7: ('machine_eva_group', 'product_name'),
    8: ('machine_eva_group', 'subcategory'),
    9: ('machine_eva_group', 'category'),
    10: ('ALL', 'product_name'),
    11: ('ALL', 'subcategory'),
    12: ('ALL', 'category'),
}
