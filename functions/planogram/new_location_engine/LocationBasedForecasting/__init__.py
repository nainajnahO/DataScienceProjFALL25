"""
Location-Based Forecasting

Predict sales for products at new locations using geo-weighted sales data
from similar machine categories.
"""

# Core engine
from .location_forecaster import LocationBasedForecaster

# High-level API
from .prediction_pipeline import (
    predict_sales_at_location,
    compare_locations,
    get_best_location,
    load_default_ica_stores,
    load_default_companies
)

# Geographic utilities
from .geo_utils import (
    haversine_distance,
    find_nearest_points,
    find_points_within_radius,
    calculate_distance_matrix,
    get_bounding_box
)

# Proximity scoring
from .ica_proximity import (
    calculate_ica_proximity_score,
    calculate_ica_density_score,
    compare_ica_proximity
)

from .company_proximity import (
    calculate_company_proximity_score,
    calculate_employee_density_score,
    compare_company_proximity,
    get_nearby_employee_stats
)

# Geographic similarity
from .geo_similarity import (
    calculate_geo_similarity,
    calculate_similarity_to_multiple,
    get_top_similar_locations
)

# Weighting strategies
from .weighting_strategies import (
    inverse_distance_weighting,
    gaussian_weighting,
    top_k_weighting,
    adaptive_weighting,
    get_strategy,
    STRATEGIES
)

# Configuration
from .config import (
    GEO_SIMILARITY_WEIGHTS,
    ICA_K_NEAREST,
    MAX_ICA_DISTANCE_KM,
    COMPANY_RADIUS_KM,
    DEFAULT_WEIGHTING_STRATEGY,
    MAX_REFERENCE_DISTANCE_KM,
    MIN_REFERENCE_MACHINES,
    CATEGORY_CONFIGS,
    get_category_config,
    get_geo_similarity_weights
)

__all__ = [
    # Core engine
    'LocationBasedForecaster',

    # High-level API
    'predict_sales_at_location',
    'compare_locations',
    'get_best_location',
    'load_default_ica_stores',
    'load_default_companies',

    # Geographic utilities
    'haversine_distance',
    'find_nearest_points',
    'find_points_within_radius',
    'calculate_distance_matrix',
    'get_bounding_box',

    # Proximity scoring
    'calculate_ica_proximity_score',
    'calculate_ica_density_score',
    'compare_ica_proximity',
    'calculate_company_proximity_score',
    'calculate_employee_density_score',
    'compare_company_proximity',
    'get_nearby_employee_stats',

    # Geographic similarity
    'calculate_geo_similarity',
    'calculate_similarity_to_multiple',
    'get_top_similar_locations',

    # Weighting strategies
    'inverse_distance_weighting',
    'gaussian_weighting',
    'top_k_weighting',
    'adaptive_weighting',
    'get_strategy',
    'STRATEGIES',

    # Configuration
    'GEO_SIMILARITY_WEIGHTS',
    'ICA_K_NEAREST',
    'MAX_ICA_DISTANCE_KM',
    'COMPANY_RADIUS_KM',

    'DEFAULT_WEIGHTING_STRATEGY',
    'MAX_REFERENCE_DISTANCE_KM',
    'MIN_REFERENCE_MACHINES',
    'CATEGORY_CONFIGS',
    'get_category_config',
    'get_geo_similarity_weights',
]
