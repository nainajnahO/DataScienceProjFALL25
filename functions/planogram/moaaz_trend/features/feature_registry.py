"""
Feature Registry
================

Central registry for feature groups and their columns.
Manages feature group combinations and provides utilities for feature retrieval.

Usage:
    from src.features.feature_registry import FEATURE_GROUPS, get_feature_columns
    
    # Get all features for BASE group
    features = get_feature_columns(['BASE'])
    
    # Get features for multiple groups
    features = get_feature_columns(['BASE', 'TEMPORAL'])
"""

from typing import List, Dict


# ============================================================================
# FEATURE GROUP DEFINITIONS
# ============================================================================

FEATURE_GROUPS = {
    'BASE': {
        'numeric': ['week_sin', 'week_cos', 'price_mean', 'purchase_price_kr', 'working_days'],
        'categorical': ['category', 'subcategory', 'machine_eva_group', 'machine_sub_group']
    },
    'TEMPORAL': {
        'numeric': [
            'season_fall', 'season_spring', 'season_summer', 'season_winter',
            'is_midsummer_period', 'is_summer_holiday', 'is_dark_period',
            'is_sportlov', 'is_autumn_break', 'is_payday_period',
            'is_new_year_resolution_period'
        ],
        'categorical': []
    },
    'PRODUCT': {
        'numeric': [
            'product_category_price_mean_lag_1',
            'product_subcategory_price_mean_lag_1',
            'product_price_vs_category_ratio',
            'product_price_vs_subcategory_ratio',
            'product_price_gap_to_next_cheapest_category',
            'product_price_gap_to_next_cheapest_subcategory',
            'product_is_round_or_zero_price',
            'profit_margin'
        ],
        'categorical': []
    },
    'MACHINE': {
        'numeric': [
            'machine_avg_price_lag_1',
            'machine_category_price_machine_mean_lag_1',
            'machine_subcategory_price_machine_mean_lag_1',
            'machine_price_vs_machine_mean_ratio',
            'machine_price_vs_category_machine_ratio',
            'machine_price_vs_subcategory_machine_ratio',
            'machine_price_gap_to_next_cheapest_category',
            'machine_price_gap_to_next_cheapest_subcategory',
            'machine_total_sales',
            'machine_avg_sales_per_product_lag_1',
            'machine_sales_volatility_lag_1',
            'machine_product_count_lag_1',
            'machine_category_count_lag_1',
            'machine_subcategory_count_lag_1',
            'machine_min_price_lag_1',
            'machine_max_price_lag_1',
            'machine_category_sales_lag_1',
            'machine_subcategory_sales_lag_1',
            'machine_category_mix',
            'machine_subcategory_mix',
            'machine_product_share',
            'category_product_share_current_week',
            'subcategory_product_share_current_week',
            'num_category_competitors_current_week',
            'num_subcategory_competitors_current_week'
        ],
        'categorical': []
    },
    'HISTORICAL_SALES': {
        'numeric': [
            'product_sales_lag_1', 'product_sales_lag_2', 'product_sales_lag_3', 'product_sales_lag_4',
            'product_sales_ewma_4', 'product_sales_ewma_8',
            'product_sales_rolling_std_4', 'product_sales_rolling_std_8',
            'product_machine_sales_lag_1', 'product_machine_sales_lag_2',
            'product_machine_sales_lag_3', 'product_machine_sales_lag_4',
            'product_machine_sales_ewma_4', 'product_machine_sales_ewma_8',
            'product_machine_sales_rolling_std_4', 'product_machine_sales_rolling_std_8',
            'product_sales_lag_1_imputation_level', 'product_sales_lag_1_imputation_std',
            'product_sales_lag_2_imputation_level', 'product_sales_lag_2_imputation_std',
            'product_sales_lag_3_imputation_level', 'product_sales_lag_3_imputation_std',
            'product_sales_lag_4_imputation_level', 'product_sales_lag_4_imputation_std',
            'product_machine_sales_lag_1_imputation_level', 'product_machine_sales_lag_1_imputation_std',
            'product_machine_sales_lag_2_imputation_level', 'product_machine_sales_lag_2_imputation_std',
            'product_machine_sales_lag_3_imputation_level', 'product_machine_sales_lag_3_imputation_std',
            'product_machine_sales_lag_4_imputation_level', 'product_machine_sales_lag_4_imputation_std'
        ],
        'categorical': []
    },
    'PRODUCT_LIFECYCLE': {
        'numeric': [
            'weeks_since_launch',
            'weeks_in_machine',
            'is_early_lifecycle',
            'is_new_product',
            'is_mature_product',
            'product_age_log',
            'launch_timing'
        ],
        'categorical': []
    },
    'BRAND': {
        'numeric': [
            # Global brand performance (lagged)
            'brand_total_sales_lag_1',
            'brand_avg_sales_per_product_lag_1',
            'brand_product_count_lag_1',
            'brand_category_count_lag_1',
            'brand_machine_penetration_lag_1',
            'brand_price_premium_lag_1',
            'brand_stability_lag_1',
            'brand_sales_ewma_4',
            # Brand-machine distribution (current week)
            'brand_machine_share',
            'brand_machine_product_count',
            'brand_category_machine_share',
            'brand_slot_representation',
            # Brand-machine sales/volatility (lagged)
            'brand_machine_sales_lag_1',
            'brand_machine_avg_sales_lag_1',
            'brand_machine_sales_std_lag_1',
            'brand_machine_sales_ewma_4'
        ],
        'categorical': []
    }
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_feature_columns(feature_groups: List[str] = ['BASE']) -> List[str]:
    """
    Get all feature column names for specified groups.
    
    Args:
        feature_groups: List of feature group names to include
        
    Returns:
        List of all feature column names
    """
    all_features = []
    
    for group_name in feature_groups:
        if group_name not in FEATURE_GROUPS:
            raise ValueError(f"Unknown feature group: {group_name}")
        
        group_def = FEATURE_GROUPS[group_name]
        
        # Add numeric features
        if 'numeric' in group_def:
            all_features.extend(group_def['numeric'])
        
        # Add categorical features
        if 'categorical' in group_def:
            all_features.extend(group_def['categorical'])
    
    return list(set(all_features))  # Remove duplicates


def get_numeric_features(feature_groups: List[str] = ['BASE']) -> List[str]:
    """Get only numeric features for specified groups."""
    numeric_features = []
    
    for group_name in feature_groups:
        if group_name in FEATURE_GROUPS and 'numeric' in FEATURE_GROUPS[group_name]:
            numeric_features.extend(FEATURE_GROUPS[group_name]['numeric'])
    
    return list(set(numeric_features))


def get_categorical_features(feature_groups: List[str] = ['BASE']) -> List[str]:
    """Get only categorical features for specified groups."""
    categorical_features = []
    
    for group_name in feature_groups:
        if group_name in FEATURE_GROUPS and 'categorical' in FEATURE_GROUPS[group_name]:
            categorical_features.extend(FEATURE_GROUPS[group_name]['categorical'])
    
    return list(set(categorical_features))


def register_feature_group(group_name: str, numeric_cols: List[str], categorical_cols: List[str]):
    """
    Dynamically register a new feature group after creation.
    
    Args:
        group_name: Name of the feature group
        numeric_cols: List of numeric column names
        categorical_cols: List of categorical column names
    """
    FEATURE_GROUPS[group_name] = {
        'numeric': numeric_cols,
        'categorical': categorical_cols
    }


def get_feature_groups_summary(feature_groups: List[str] = ['BASE']) -> str:
    """
    Get a human-readable summary of feature groups.
    
    Args:
        feature_groups: List of feature group names
        
    Returns:
        String summary for use in plot titles, etc.
    """
    return ', '.join(feature_groups)

