"""
Configuration for Location-Based Forecasting

Centralized configuration for geographic scoring, weighting strategies,
and prediction parameters.
"""

import os

# Geographic Similarity Weights
# ----------------------------------------

GEO_SIMILARITY_WEIGHTS = {
    'distance': 0.1,  # Direct distance between locations
    'ica_profile': 0.45,  # Similarity in ICA proximity
    'company_profile': 0.45  # Similarity in company proximity
}

# ICA Proximity Configuration
# ----------------------------------------
# Number of nearest ICA stores to consider
ICA_K_NEAREST = 3

# Maximum distance to consider for ICA proximity (km)
# Beyond this distance, ICA score = 0
MAX_ICA_DISTANCE_KM = 15.0

# Company Proximity Configuration
# ----------------------------------------

# Search radius for nearby companies (km)
COMPANY_RADIUS_KM = 2.0

# Inner and outer radius for density calculations
COMPANY_INNER_RADIUS_KM = 0.5
COMPANY_OUTER_RADIUS_KM = 2.0

# Prediction Configuration
# ----------------------------------------

# Default weighting strategy for predictions
# Options: 'inverse_distance', 'gaussian', 'top_k'
DEFAULT_WEIGHTING_STRATEGY = 'inverse_distance'

# Maximum distance to consider reference machines (km)
# Machines beyond this distance are excluded
MAX_REFERENCE_DISTANCE_KM = 100.0

# Minimum number of reference machines required for prediction
MIN_REFERENCE_MACHINES = 1

# Gaussian weighting parameter (if using gaussian strategy)
GAUSSIAN_SIGMA = 2.0

# Top-k parameter (if using top_k strategy)
TOP_K_MACHINES = 5

# Minimum similarity threshold for reference machines
# Machines with similarity below this are excluded
MIN_SIMILARITY_THRESHOLD = 0.4

# Category-Specific Configuration
# ----------------------------------------

# Different parameters for different machine categories
CATEGORY_CONFIGS = {
    'WORK': {
        'max_distance': 100.0,
        'min_machines': 1,
        'ica_weight': 0.45,
        'company_weight': 0.45
    },
    'GYM': {
        'max_distance': 100.0,
        'min_machines': 1,
        'ica_weight': 0.45,
        'company_weight': 0.45
    },
    'SCHOOLS': {
        'max_distance': 100.0,
        'min_machines': 1,
        'ica_weight': 0.45,
        'company_weight': 0.45
    },
    'MALL': {
        'max_distance': 100.0,
        'min_machines': 1,
        'ica_weight': 0.45,
        'company_weight': 0.45
    },
    'SPORTS GROUNDS': {
        'max_distance': 100.0,
        'min_machines': 1,
        'ica_weight': 0.45,
        'company_weight': 0.45
    }
}

# Fallback configuration for categories not explicitly defined
DEFAULT_CATEGORY_CONFIG = {
    'max_distance': 100.0,
    'min_machines': 1,
    'ica_weight': 0.45,
    'company_weight': 0.45
}

# Confidence Scoring
# ----------------------------------------

# Weights for confidence score components
CONFIDENCE_WEIGHTS = {
    'similarity': 0.5,  # Average geo-similarity of reference machines
    'sample_size': 0.3,  # Number of reference machines used
    'consistency': 0.2  # Consistency of sales across reference machines
}

# Minimum confidence threshold to consider prediction reliable
MIN_CONFIDENCE_THRESHOLD = 0.4

# Data Paths (Relative to LocationBasedForecasting directory)
# ----------------------------------------

DEFAULT_ICA_STORES_PATH = 'data/mock_ica_stores.parquet'
DEFAULT_COMPANIES_PATH = 'data/mock_companies.parquet'


def get_category_config(category: str) -> dict:
    """
    Get configuration for a specific machine category.

    Args:
        category: Machine category (e.g., 'WORK', 'GYM', 'SCHOOLS')

    Returns:
        Dictionary with category-specific configuration
    """
    return CATEGORY_CONFIGS.get(category, DEFAULT_CATEGORY_CONFIG)


def get_geo_similarity_weights(category: str = None) -> dict:
    """
    Get geographic similarity weights, optionally customized by category.

    Args:
        category: Optional machine category for category-specific weights

    Returns:
        Dictionary with weights for distance, ica_profile, company_profile
    """
    if category and category in CATEGORY_CONFIGS:
        cat_config = CATEGORY_CONFIGS[category]
        # Build weights from category config
        ica_w = cat_config.get('ica_weight', GEO_SIMILARITY_WEIGHTS['ica_profile'])
        company_w = cat_config.get('company_weight', GEO_SIMILARITY_WEIGHTS['company_profile'])

        # Distance weight is remainder
        distance_w = 1.0 - ica_w - company_w

        return {
            'distance': distance_w,
            'ica_profile': ica_w,
            'company_profile': company_w
        }

    return GEO_SIMILARITY_WEIGHTS.copy()
