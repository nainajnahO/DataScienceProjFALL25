# Core engine
from .fallback_engine import FallbackEngine, quick_fallback

# Strategies
from .strategies import (
    PRODUCT_FIRST_STRATEGY,
    MACHINE_FIRST_STRATEGY,
    LOCATION_AFFINITY_STRATEGY,
    STRATEGY_REGISTRY,
    get_strategy,
    list_strategies,
    register_custom_strategy,
    compare_strategies,
    visualize_strategy,
)

# Query functions
from .queries import (
    query_machine_key_product,
    query_machine_key_subcategory,
    query_machine_key_category,
    query_subgroup_product,
    query_subgroup_subcategory,
    query_subgroup_category,
    query_evagroup_product,
    query_evagroup_subcategory,
    query_evagroup_category,
    query_all_product,
    query_all_subcategory,
    query_all_category,
    QUERY_FUNCTIONS,
)

# Configuration
from .config import (
    MACHINE_LEVELS,
    PRODUCT_LEVELS,
    MIN_SAMPLE_THRESHOLDS,
    DEFAULT_MIN_SAMPLES,
    CONFIDENCE_WEIGHTS,
    PAIR_MAPPING,
)

# Define public API
__all__ = [
    # Main engine
    'FallbackEngine',
    'quick_fallback',

    # Strategies
    'PRODUCT_FIRST_STRATEGY',
    'MACHINE_FIRST_STRATEGY',
    'LOCATION_AFFINITY_STRATEGY',
    'STRATEGY_REGISTRY',
    'get_strategy',
    'list_strategies',
    'register_custom_strategy',
    'compare_strategies',
    'visualize_strategy',

    # Query functions
    'query_machine_key_product',
    'query_machine_key_subcategory',
    'query_machine_key_category',
    'query_subgroup_product',
    'query_subgroup_subcategory',
    'query_subgroup_category',
    'query_evagroup_product',
    'query_evagroup_subcategory',
    'query_evagroup_category',
    'query_all_product',
    'query_all_subcategory',
    'query_all_category',
    'QUERY_FUNCTIONS',

    # Configuration
    'MACHINE_LEVELS',
    'PRODUCT_LEVELS',
    'MIN_SAMPLE_THRESHOLDS',
    'DEFAULT_MIN_SAMPLES',
    'CONFIDENCE_WEIGHTS',
    'PAIR_MAPPING',
]

# Package metadata
__version__ = '1.0.0'
__author__ = 'RBM Team'
__description__ = 'Hierarchical fallback architecture for rule-based models'
