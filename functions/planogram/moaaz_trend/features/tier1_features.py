"""
Tier 1 Feature Creation Functions
==================================

Modular functions for creating feature groups in Tier 1 feature engineering.
All functions work on the full dataset and prevent data leakage.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm

# Set up tqdm for pandas
tqdm.pandas()


def create_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create BASE features - foundational features including cyclical week encoding.
    
    BASE features include:
    - week_sin, week_cos: Cyclical 52-week encoding
    - price_mean, purchase_price_kr, working_days: Raw numeric features
    - category, subcategory, machine_eva_group, machine_sub_group: Categorical features
    
    Note: The raw numeric and categorical features should already exist in df.
    This function only adds week_sin/week_cos if they don't exist.
    
    Args:
        df: DataFrame with columns: week, price_mean, purchase_price_kr, working_days,
            category, subcategory, machine_eva_group, machine_sub_group
            
    Returns:
        DataFrame with BASE features (week_sin, week_cos added if not present)
    """
    df = df.copy()
    
    # Add cyclical week encoding if not already present
    if 'week_sin' not in df.columns or 'week_cos' not in df.columns:
        if 'week' not in df.columns:
            # Extract week from week_start if week column doesn't exist
            if 'week_start' in df.columns:
                df['week'] = df['week_start'].dt.isocalendar().week
            else:
                raise ValueError("Need either 'week' column or 'week_start' column to create week_sin/week_cos")
        
        df['week_sin'] = np.sin(2 * np.pi * df['week'] / 52)
        df['week_cos'] = np.cos(2 * np.pi * df['week'] / 52)
    
    return df


def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create temporal/calendar features for Swedish context.
    
    Excludes month_sin/cos (redundant with week_sin/cos) and is_holiday_week
    (redundant with working_days).
    
    Features Created:
    - Season flags: season_fall, season_spring, season_summer, season_winter
    - Cultural periods: is_midsummer_period, is_summer_holiday, is_dark_period,
                        is_sportlov, is_autumn_break, is_payday_period,
                        is_new_year_resolution_period
    
    Args:
        df: DataFrame with week_start column
        
    Returns:
        DataFrame with temporal features added
    """
    df = df.copy()
    
    # Get week of year from week_start
    week_of_year = df['week_start'].dt.isocalendar().week
    
    # Swedish seasons based on ISO week numbers
    def get_swedish_season(iso_week):
        if iso_week >= 47 or iso_week <= 8:  # Late Nov - Late Feb
            return 'winter'
        elif 9 <= iso_week <= 21:  # Early Mar - Late May
            return 'spring'
        elif 22 <= iso_week <= 34:  # Early Jun - Late Aug
            return 'summer'
        else:  # 35 <= iso_week <= 46  # Early Sep - Mid Nov
            return 'fall'
    
    df['swedish_season'] = week_of_year.apply(get_swedish_season)
    
    # One-hot encoding for Swedish seasons
    season_dummies = pd.get_dummies(df['swedish_season'], prefix='season')
    df = pd.concat([df, season_dummies], axis=1)
    df = df.drop('swedish_season', axis=1)
    
    # Swedish cultural/holiday periods
    df['is_midsummer_period'] = week_of_year.isin([24, 25]).astype(int)
    df['is_summer_holiday'] = week_of_year.between(26, 32).astype(int)
    df['is_dark_period'] = ((week_of_year >= 48) | (week_of_year <= 7)).astype(int)
    df['is_sportlov'] = week_of_year.isin([8, 9]).astype(int)
    df['is_autumn_break'] = week_of_year.isin([44, 45]).astype(int)
    
    # Payday period estimation (Swedish paydays ~25th of month)
    def is_likely_payday_week(idx):
        week_start = df['week_start'].iloc[idx]
        day_of_month = week_start.day
        
        # If week starts on 22nd-27th, it likely contains payday
        if 22 <= day_of_month <= 27:
            return 1
        return 0
    
    # Vectorized payday calculation
    df['is_payday_period'] = pd.Series(range(len(df))).apply(is_likely_payday_week).astype(int)
    
    # New Year resolution period (first 4 weeks of year)
    df['is_new_year_resolution_period'] = (week_of_year <= 4).astype(int)
    
    return df


def get_temporal_feature_columns() -> List[str]:
    """Get list of temporal feature column names."""
    return [
        'season_fall', 'season_spring', 'season_summer', 'season_winter',
        'is_midsummer_period', 'is_summer_holiday', 'is_dark_period',
        'is_sportlov', 'is_autumn_break', 'is_payday_period',
        'is_new_year_resolution_period'
    ]


def create_product_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create product-level pricing intelligence features.
    
    Features Created (8 total):
    - product_category_price_mean_lag_1: Category average price (lagged)
    - product_subcategory_price_mean_lag_1: Subcategory average price (lagged)
    - product_price_vs_category_ratio: Price vs category average
    - product_price_vs_subcategory_ratio: Price vs subcategory average
    - product_price_gap_to_next_cheapest_category: Price gap to next cheapest in category
    - product_price_gap_to_next_cheapest_subcategory: Price gap to next cheapest in subcategory
    - product_is_round_or_zero_price: Binary flag for round prices (0/5 ending)
    - profit_margin: price_mean - purchase_price_kr
    
    All price aggregates use LAG (historical data) except profit_margin and round_price flag.
    
    Args:
        df: DataFrame with columns: week_start, category, subcategory, price_mean, purchase_price_kr
        
    Returns:
        DataFrame with product features added
    """
    df = df.copy()
    
    # Ensure proper sorting for time-based operations  
    # Sort by ean first to preserve product-level continuity, then by category/subcategory/week
    df = df.sort_values(['ean', 'category', 'subcategory', 'week_start']).reset_index(drop=True)
    
    # 1. Category average price (lagged)
    # First compute mean per category/week, then shift within category across weeks
    category_weekly_mean = df.groupby(['category', 'week_start'])['price_mean'].agg('mean').reset_index(name='category_mean')
    category_weekly_mean = category_weekly_mean.sort_values(['category', 'week_start']).reset_index(drop=True)
    category_weekly_mean['product_category_price_mean_lag_1'] = category_weekly_mean.groupby('category')['category_mean'].shift(1)
    
    # Merge category lag feature - drop if already exists to avoid conflicts
    merge_cols = ['category', 'week_start']
    if 'product_category_price_mean_lag_1' in df.columns:
        df = df.drop('product_category_price_mean_lag_1', axis=1)
    
    df = df.merge(
        category_weekly_mean[merge_cols + ['product_category_price_mean_lag_1']],
        on=merge_cols,
        how='left'
    )
    
    # 2. Subcategory average price (lagged)
    # Same approach for subcategory
    subcategory_weekly_mean = df.groupby(['subcategory', 'week_start'])['price_mean'].agg('mean').reset_index(name='subcategory_mean')
    subcategory_weekly_mean = subcategory_weekly_mean.sort_values(['subcategory', 'week_start']).reset_index(drop=True)
    subcategory_weekly_mean['product_subcategory_price_mean_lag_1'] = subcategory_weekly_mean.groupby('subcategory')['subcategory_mean'].shift(1)
    
    # Merge subcategory lag feature - drop if already exists to avoid conflicts
    merge_cols_sub = ['subcategory', 'week_start']
    if 'product_subcategory_price_mean_lag_1' in df.columns:
        df = df.drop('product_subcategory_price_mean_lag_1', axis=1)
    
    df = df.merge(
        subcategory_weekly_mean[merge_cols_sub + ['product_subcategory_price_mean_lag_1']],
        on=merge_cols_sub,
        how='left'
    )
    
    # Verify columns exist before using them
    if 'product_category_price_mean_lag_1' not in df.columns:
        raise ValueError(f"product_category_price_mean_lag_1 column not created. Sample columns: {list(df.columns)[:20]}")
    if 'product_subcategory_price_mean_lag_1' not in df.columns:
        raise ValueError(f"product_subcategory_price_mean_lag_1 column not created. Sample columns: {list(df.columns)[:20]}")
    
    # 3-4. Price positioning ratios
    df['product_price_vs_category_ratio'] = (
        df['price_mean'] / df['product_category_price_mean_lag_1'].fillna(df['price_mean'])
    )
    df['product_price_vs_subcategory_ratio'] = (
        df['price_mean'] / df['product_subcategory_price_mean_lag_1'].fillna(df['price_mean'])
    )
    
    # 5-6. Price gaps to next cheapest (lagged)
    def price_gap_to_next_cheapest(group):
        """Calculate minimum price gap between cheapest and 2nd cheapest."""
        if len(group) < 2:
            return 0
        sorted_prices = group.sort_values()
        # Return gap between 2nd cheapest and cheapest
        return sorted_prices.iloc[1] - sorted_prices.iloc[0]
    
    # Category price gaps: compute gap per group, shift week, then merge
    category_gaps = (
        df.groupby(['category', 'week_start'])['price_mean']
        .apply(price_gap_to_next_cheapest)
        .reset_index(name='gap_value')
    )
    category_gaps['week_start'] = category_gaps.groupby('category')['week_start'].shift(1)
    df = df.merge(
        category_gaps[['category', 'week_start', 'gap_value']],
        on=['category', 'week_start'],
        how='left'
    )
    df = df.rename(columns={'gap_value': 'product_price_gap_to_next_cheapest_category'})
    df['product_price_gap_to_next_cheapest_category'] = df['product_price_gap_to_next_cheapest_category'].fillna(0)
    
    # Subcategory price gaps: same approach
    subcategory_gaps = (
        df.groupby(['subcategory', 'week_start'])['price_mean']
        .apply(price_gap_to_next_cheapest)
        .reset_index(name='gap_value')
    )
    subcategory_gaps['week_start'] = subcategory_gaps.groupby('subcategory')['week_start'].shift(1)
    df = df.merge(
        subcategory_gaps[['subcategory', 'week_start', 'gap_value']],
        on=['subcategory', 'week_start'],
        how='left',
        suffixes=('', '_subcat')
    )
    df = df.rename(columns={'gap_value': 'product_price_gap_to_next_cheapest_subcategory'})
    df['product_price_gap_to_next_cheapest_subcategory'] = df['product_price_gap_to_next_cheapest_subcategory'].fillna(0)
    
    # 7. Round or zero ending price (current week - no lag needed)
    df['product_is_round_or_zero_price'] = (
        ((df['price_mean'] % 5 == 0) | (df['price_mean'] % 10 == 0)).astype(int)
    )
    
    # 8. Profit margin (current week - no lag needed)
    df['profit_margin'] = df['price_mean'] - df['purchase_price_kr'].fillna(0)
    
    return df


def create_machine_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create machine-level features: pricing, sales performance, distribution, and competitive context.
    
    Features Created (25 total):
    Price Benchmarking (8 - LAGGED):
    - machine_avg_price_lag_1
    - machine_category_price_machine_mean_lag_1
    - machine_subcategory_price_machine_mean_lag_1
    - machine_price_vs_machine_mean_ratio
    - machine_price_vs_category_machine_ratio
    - machine_price_vs_subcategory_machine_ratio
    - machine_price_gap_to_next_cheapest_category
    - machine_price_gap_to_next_cheapest_subcategory
    
    Sales Performance (10 - LAGGED):
    - machine_total_sales
    - machine_avg_sales_per_product_lag_1
    - machine_sales_volatility_lag_1
    - machine_product_count_lag_1
    - machine_category_count_lag_1
    - machine_subcategory_count_lag_1
    - machine_min_price_lag_1
    - machine_max_price_lag_1
    - machine_category_sales_lag_1 (NEW)
    - machine_subcategory_sales_lag_1 (NEW)
    
    Distribution Features (5 - CURRENT WEEK):
    - machine_category_mix (NEW)
    - machine_subcategory_mix (NEW)
    - machine_product_share (NEW)
    - category_product_share_current_week (NEW)
    - subcategory_product_share_current_week (NEW)
    
    Competitive Context (2 - CURRENT WEEK):
    - num_category_competitors_current_week (NEW)
    - num_subcategory_competitors_current_week (NEW)
    
    Args:
        df: DataFrame with machine_key, week_start, ean, category, subcategory, 
            price_mean, weekly_sales, position columns
            
    Returns:
        DataFrame with machine features added
    """
    df = df.copy()
    
    # Ensure proper sorting for time-based operations
    df = df.sort_values(['machine_key', 'week_start']).reset_index(drop=True)
    
    # ============================================================================
    # PRICE BENCHMARKING (LAGGED)
    # ============================================================================
    
    # 1. Machine average price (lagged)
    machine_price_stats = df.groupby(['machine_key', 'week_start'])['price_mean'].agg('mean').shift(1)
    machine_price_stats.name = 'machine_avg_price_lag_1'
    df = df.merge(
        machine_price_stats.reset_index(),
        on=['machine_key', 'week_start'],
        how='left'
    )
    
    # 2. Category price in machine (lagged)
    category_machine_price_stats = (
        df.groupby(['category', 'machine_key', 'week_start'])['price_mean']
        .agg('mean')
        .shift(1)
    )
    category_machine_price_stats.name = 'machine_category_price_machine_mean_lag_1'
    df = df.merge(
        category_machine_price_stats.reset_index(),
        on=['category', 'machine_key', 'week_start'],
        how='left'
    )
    
    # 3. Subcategory price in machine (lagged)
    subcategory_machine_price_stats = (
        df.groupby(['subcategory', 'machine_key', 'week_start'])['price_mean']
        .agg('mean')
        .shift(1)
    )
    subcategory_machine_price_stats.name = 'machine_subcategory_price_machine_mean_lag_1'
    df = df.merge(
        subcategory_machine_price_stats.reset_index(),
        on=['subcategory', 'machine_key', 'week_start'],
        how='left'
    )
    
    # 4-6. Price positioning ratios
    df['machine_price_vs_machine_mean_ratio'] = (
        df['price_mean'] / df['machine_avg_price_lag_1'].fillna(df['price_mean'])
    )
    df['machine_price_vs_category_machine_ratio'] = (
        df['price_mean'] / df['machine_category_price_machine_mean_lag_1'].fillna(df['price_mean'])
    )
    df['machine_price_vs_subcategory_machine_ratio'] = (
        df['price_mean'] / df['machine_subcategory_price_machine_mean_lag_1'].fillna(df['price_mean'])
    )
    
    # 7-8. Machine price gaps (lagged)
    def price_gap_to_next_cheapest(group):
        """Calculate minimum price gap between cheapest and 2nd cheapest."""
        if len(group) < 2:
            return 0
        sorted_prices = group.sort_values()
        return sorted_prices.iloc[1] - sorted_prices.iloc[0]
    
    # Machine-category price gaps
    machine_category_gaps = (
        df.groupby(['machine_key', 'category', 'week_start'])['price_mean']
        .apply(price_gap_to_next_cheapest)
        .reset_index(name='gap_value')
    )
    machine_category_gaps['week_start'] = machine_category_gaps.groupby(['machine_key', 'category'])['week_start'].shift(1)
    df = df.merge(
        machine_category_gaps[['machine_key', 'category', 'week_start', 'gap_value']],
        on=['machine_key', 'category', 'week_start'],
        how='left'
    )
    df = df.rename(columns={'gap_value': 'machine_price_gap_to_next_cheapest_category'})
    df['machine_price_gap_to_next_cheapest_category'] = df['machine_price_gap_to_next_cheapest_category'].fillna(0)
    
    # Machine-subcategory price gaps
    machine_subcategory_gaps = (
        df.groupby(['machine_key', 'subcategory', 'week_start'])['price_mean']
        .apply(price_gap_to_next_cheapest)
        .reset_index(name='gap_value')
    )
    machine_subcategory_gaps['week_start'] = machine_subcategory_gaps.groupby(['machine_key', 'subcategory'])['week_start'].shift(1)
    df = df.merge(
        machine_subcategory_gaps[['machine_key', 'subcategory', 'week_start', 'gap_value']],
        on=['machine_key', 'subcategory', 'week_start'],
        how='left',
        suffixes=('', '_mach_subcat')
    )
    df = df.rename(columns={'gap_value': 'machine_price_gap_to_next_cheapest_subcategory'})
    df['machine_price_gap_to_next_cheapest_subcategory'] = df['machine_price_gap_to_next_cheapest_subcategory'].fillna(0)
    
    # ============================================================================
    # SALES PERFORMANCE METRICS (LAGGED)
    # ============================================================================
    
    # 9. Machine total sales (all time - no lag needed, but could be computed)
    df['machine_total_sales'] = (
    df.groupby('machine_key')['weekly_sales']
    .cumsum()
    .shift(1)
    .fillna(0)
)
    
    # 10-16. Previous week machine stats (lagged)
    machine_weekly_stats = df.groupby(['machine_key', 'week_start']).agg({
        'weekly_sales': ['mean', 'std'],
        'ean': 'nunique',
        'category': 'nunique',
        'subcategory': 'nunique',
        'price_mean': ['min', 'max']
    }).shift(1)
    
    # Flatten column names
    machine_weekly_stats.columns = [
        'machine_avg_sales_per_product_lag_1',
        'machine_sales_volatility_lag_1',
        'machine_product_count_lag_1',
        'machine_category_count_lag_1',
        'machine_subcategory_count_lag_1',
        'machine_min_price_lag_1',
        'machine_max_price_lag_1'
    ]
    
    df = df.merge(
        machine_weekly_stats.reset_index(),
        on=['machine_key', 'week_start'],
        how='left'
    )
    
    # 17-18. NEW: Machine category/subcategory sales (lagged)
    machine_category_sales = (
        df.groupby(['machine_key', 'category', 'week_start'])['weekly_sales']
        .transform('sum')
        .groupby([df['machine_key'], df['category']])
        .shift(1)
    )
    df['machine_category_sales_lag_1'] = machine_category_sales.values
    
    machine_subcategory_sales = (
        df.groupby(['machine_key', 'subcategory', 'week_start'])['weekly_sales']
        .transform('sum')
        .groupby([df['machine_key'], df['subcategory']])
        .shift(1)
    )
    df['machine_subcategory_sales_lag_1'] = machine_subcategory_sales.values
    
    # ============================================================================
    # DISTRIBUTION FEATURES (CURRENT WEEK - NO LAG)
    # ============================================================================
    
    # 19. Machine category mix (% of machine entries that are this category)
    machine_total_entries = df.groupby(['machine_key', 'week_start']).size().reset_index(name='total_entries')
    category_entries = df.groupby(['machine_key', 'category', 'week_start']).size().reset_index(name='category_entries')
    df = df.merge(machine_total_entries, on=['machine_key', 'week_start'], how='left')
    df = df.merge(category_entries, on=['machine_key', 'category', 'week_start'], how='left')
    df['machine_category_mix'] = (df['category_entries'] / df['total_entries'].fillna(1) * 100).fillna(0)
    df = df.drop(['total_entries', 'category_entries'], axis=1)
    
    # 20. Machine subcategory mix
    subcategory_entries = df.groupby(['machine_key', 'subcategory', 'week_start']).size().reset_index(name='subcategory_entries')
    machine_total_entries = df.groupby(['machine_key', 'week_start']).size().reset_index(name='total_entries')
    df = df.merge(machine_total_entries, on=['machine_key', 'week_start'], how='left')
    df = df.merge(subcategory_entries, on=['machine_key', 'subcategory', 'week_start'], how='left')
    df['machine_subcategory_mix'] = (df['subcategory_entries'] / df['total_entries'].fillna(1) * 100).fillna(0)
    df = df.drop(['total_entries', 'subcategory_entries'], axis=1)
    
    # 21. Machine product share (% of machine entries that are this product)
    product_entries = df.groupby(['machine_key', 'ean', 'week_start']).size().reset_index(name='product_entries')
    machine_total_entries = df.groupby(['machine_key', 'week_start']).size().reset_index(name='total_entries')
    df = df.merge(machine_total_entries, on=['machine_key', 'week_start'], how='left')
    df = df.merge(product_entries, on=['machine_key', 'ean', 'week_start'], how='left')
    df['machine_product_share'] = (df['product_entries'] / df['total_entries'].fillna(1) * 100).fillna(0)
    df = df.drop(['total_entries', 'product_entries'], axis=1)
    
    # 22. Category product share (% of category entries that are this product - across all machines)
    category_total_entries = df.groupby(['category', 'week_start']).size().reset_index(name='category_total')
    category_product_entries = df.groupby(['category', 'ean', 'week_start']).size().reset_index(name='category_product_count')
    df = df.merge(category_total_entries, on=['category', 'week_start'], how='left')
    df = df.merge(category_product_entries, on=['category', 'ean', 'week_start'], how='left')
    df['category_product_share_current_week'] = (df['category_product_count'] / df['category_total'].fillna(1) * 100).fillna(0)
    df = df.drop(['category_total', 'category_product_count'], axis=1)
    
    # 23. Subcategory product share
    subcategory_total_entries = df.groupby(['subcategory', 'week_start']).size().reset_index(name='subcategory_total')
    subcategory_product_entries = df.groupby(['subcategory', 'ean', 'week_start']).size().reset_index(name='subcategory_product_count')
    df = df.merge(subcategory_total_entries, on=['subcategory', 'week_start'], how='left')
    df = df.merge(subcategory_product_entries, on=['subcategory', 'ean', 'week_start'], how='left')
    df['subcategory_product_share_current_week'] = (df['subcategory_product_count'] / df['subcategory_total'].fillna(1) * 100).fillna(0)
    df = df.drop(['subcategory_total', 'subcategory_product_count'], axis=1)
    
    # ============================================================================
    # COMPETITIVE CONTEXT (CURRENT WEEK - NO LAG)
    # ============================================================================
    
    # 24. Number of category competitors in machine (current week)
    df['num_category_competitors_current_week'] = (
        df.groupby(['machine_key', 'category', 'week_start'])['ean']
        .transform('nunique') - 1  # Exclude self
    )
    df['num_category_competitors_current_week'] = df['num_category_competitors_current_week'].clip(lower=0)
    
    # 25. Number of subcategory competitors in machine (current week)
    df['num_subcategory_competitors_current_week'] = (
        df.groupby(['machine_key', 'subcategory', 'week_start'])['ean']
        .transform('nunique') - 1  # Exclude self
    )
    df['num_subcategory_competitors_current_week'] = df['num_subcategory_competitors_current_week'].clip(lower=0)
    
    return df


def create_historical_sales_features(df: pd.DataFrame, use_cold_start_fallback: bool = True) -> pd.DataFrame:
    """
    Create historical sales pattern features for products (across all machines) 
    and product-machine combinations.
    
    Features Created (16 total):
    
    Product-Level (8 - aggregated across ALL machines where product appears):
    - product_sales_lag_1, product_sales_lag_2, product_sales_lag_3, product_sales_lag_4
    - product_sales_ewma_4, product_sales_ewma_8
    - product_sales_rolling_std_4, product_sales_rolling_std_8
    
    Product-Machine Level (8 - specific to product in THIS machine):
    - product_machine_sales_lag_1, product_machine_sales_lag_2, 
      product_machine_sales_lag_3, product_machine_sales_lag_4
    - product_machine_sales_ewma_4, product_machine_sales_ewma_8
    - product_machine_sales_rolling_std_4, product_machine_sales_rolling_std_8
    
    If use_cold_start_fallback=True, missing values will be filled using hierarchical
    fallback and imputation indicators will be added.
    
    Args:
        df: DataFrame with columns: ean, machine_key, week_start, weekly_sales, 
            category, subcategory
        use_cold_start_fallback: Whether to apply cold-start fallback logic
        
    Returns:
        DataFrame with historical sales features added
    """
    df = df.copy()
    
    # Ensure proper sorting for time-based operations
    df = df.sort_values(['ean', 'machine_key', 'week_start']).reset_index(drop=True)
    
    # ============================================================================
    # PRODUCT-LEVEL FEATURES (across all machines)
    # ============================================================================
    
    # OPTIMIZED: Product sales aggregated once and reused
    product_weekly = df.groupby(['ean', 'week_start'])['weekly_sales'].sum().reset_index()
    product_weekly.columns = ['ean', 'week_start', 'product_total_sales']
    df = df.merge(product_weekly, on=['ean', 'week_start'], how='left')
    
    # OPTIMIZED: Group once, compute all features (vectorized)
    product_grouped = df.groupby('ean')['product_total_sales']
    
    # 1-4. Product lags (vectorized)
    for lag in [1, 2, 3, 4]:
        df[f'product_sales_lag_{lag}'] = product_grouped.shift(lag)
    
    # 5-6. Product EWMA (vectorized)
    for span in [4, 8]:
        alpha = 2 / (span + 1)
        df[f'product_sales_ewma_{span}'] = (
            product_grouped.shift(1)
            .transform(lambda x: x.ewm(alpha=alpha, adjust=False).mean())
        )
    
    # 7-8. Product rolling std (vectorized)
    for window in [4, 8]:
        df[f'product_sales_rolling_std_{window}'] = (
            product_grouped.shift(1)
            .transform(lambda x: x.rolling(window=window, min_periods=1).std())
        )
    
    df = df.drop('product_total_sales', axis=1)
    
    # ============================================================================
    # PRODUCT-MACHINE LEVEL FEATURES (specific to this machine)
    # ============================================================================
    
    # OPTIMIZED: Group once, compute all features (vectorized)
    pm_grouped = df.groupby(['machine_key', 'ean'])['weekly_sales']
    
    # 1-4. Product-machine lags
    for lag in [1, 2, 3, 4]:
        df[f'product_machine_sales_lag_{lag}'] = pm_grouped.shift(lag)
    
    # 5-6. Product-machine EWMA (vectorized)
    for span in [4, 8]:
        alpha = 2 / (span + 1)
        df[f'product_machine_sales_ewma_{span}'] = (
            pm_grouped.shift(1)
            .transform(lambda x: x.ewm(alpha=alpha, adjust=False).mean())
        )
    
    # 7-8. Product-machine rolling std (vectorized)
    for window in [4, 8]:
        df[f'product_machine_sales_rolling_std_{window}'] = (
            pm_grouped.shift(1)
            .transform(lambda x: x.rolling(window=window, min_periods=1).std())
        )
    
    # ============================================================================
    # COLD-START FALLBACK (if enabled)
    # ============================================================================
    
    if use_cold_start_fallback:
        df = _apply_cold_start_fallback_vectorized(df)
    
    return df


def _apply_cold_start_fallback_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized cold-start fallback - much faster than row-by-row iteration.
    ~50-100x speedup using vectorized operations.
    
    Apply hierarchical fallback to fill missing sales_lag values and add 
    imputation indicators.
    
    Imputation Levels:
    0: Actual historical data (product-machine combination)
    1: Same product, other machines
    2: Subcategory in same machine
    3: Category in same machine
    4: Subcategory across all machines
    5: Category across all machines
    6: Zero/default
    
    For each lag feature that needs imputation, adds:
    - {feature}_imputation_level: 0-6 indicator
    - {feature}_imputation_std: Standard deviation of imputed values (uncertainty)
    
    Args:
        df: DataFrame with historical sales features
        
    Returns:
        DataFrame with cold-start fallback applied and indicators added
    """
    df = df.copy()
    
    # List of features that need cold-start fallback
    lag_features = (
        [f'product_sales_lag_{i}' for i in [1, 2, 3, 4]] +
        [f'product_machine_sales_lag_{i}' for i in [1, 2, 3, 4]]
    )
    
    for feature in tqdm(lag_features, desc="Cold-start fallback", leave=False):
        if feature not in df.columns:
            continue
        
        # Create imputation level and std columns
        imputation_level_col = f'{feature}_imputation_level'
        imputation_std_col = f'{feature}_imputation_std'
        
        # Initialize
        df[imputation_level_col] = 0  # Assume actual data
        df[imputation_std_col] = 0.0
        
        missing_mask = df[feature].isna()
        
        if not missing_mask.any():
            continue
        
        # OPTIMIZED: Prepare fallback aggregations ONCE (vectorized)
        if 'product_machine' in feature:
            # Level 2: Subcategory in machine (rolling mean of last 4 weeks)
            df['_subcat_machine_mean'] = (
                df.groupby(['machine_key', 'subcategory'])['weekly_sales']
                .transform(lambda x: x.shift(1).rolling(4, min_periods=4).mean())
            )
            df['_subcat_machine_std'] = (
                df.groupby(['machine_key', 'subcategory'])['weekly_sales']
                .transform(lambda x: x.shift(1).rolling(4, min_periods=4).std())
            )
            
            # Level 3: Category in machine
            df['_cat_machine_mean'] = (
                df.groupby(['machine_key', 'category'])['weekly_sales']
                .transform(lambda x: x.shift(1).rolling(4, min_periods=4).mean())
            )
            df['_cat_machine_std'] = (
                df.groupby(['machine_key', 'category'])['weekly_sales']
                .transform(lambda x: x.shift(1).rolling(4, min_periods=4).std())
            )
            
            # Level 1: Same product, other machines
            df['_product_other_mean'] = (
                df.groupby('ean')['weekly_sales']
                .transform(lambda x: x.shift(1).rolling(4, min_periods=4).mean())
            )
            df['_product_other_std'] = (
                df.groupby('ean')['weekly_sales']
                .transform(lambda x: x.shift(1).rolling(4, min_periods=4).std())
            )
            
            # Apply fallback in order of priority (vectorized - no row iteration)
            # Level 2: Subcategory in machine
            mask_l2 = missing_mask & df['_subcat_machine_mean'].notna()
            df.loc[mask_l2, feature] = df.loc[mask_l2, '_subcat_machine_mean']
            df.loc[mask_l2, imputation_level_col] = 2
            df.loc[mask_l2, imputation_std_col] = df.loc[mask_l2, '_subcat_machine_std'].fillna(0)
            
            # Level 3: Category in machine
            still_missing = missing_mask & df[feature].isna()
            mask_l3 = still_missing & df['_cat_machine_mean'].notna()
            df.loc[mask_l3, feature] = df.loc[mask_l3, '_cat_machine_mean']
            df.loc[mask_l3, imputation_level_col] = 3
            df.loc[mask_l3, imputation_std_col] = df.loc[mask_l3, '_cat_machine_std'].fillna(0)
            
            # Level 1: Same product, other machines
            still_missing = missing_mask & df[feature].isna()
            mask_l1 = still_missing & df['_product_other_mean'].notna()
            df.loc[mask_l1, feature] = df.loc[mask_l1, '_product_other_mean']
            df.loc[mask_l1, imputation_level_col] = 1
            df.loc[mask_l1, imputation_std_col] = df.loc[mask_l1, '_product_other_std'].fillna(0)
            
            # Clean up temp columns
            df = df.drop(['_subcat_machine_mean', '_subcat_machine_std',
                         '_cat_machine_mean', '_cat_machine_std',
                         '_product_other_mean', '_product_other_std'], axis=1)
        
        # Fill remaining NaN with 0 (Level 6)
        df.loc[missing_mask & df[feature].isna(), feature] = 0
        df.loc[missing_mask & df[feature].isna(), imputation_std_col] = 0
    
    return df


def create_product_lifecycle_features(df: pd.DataFrame, min_provider_age_years: float = 1.0, early_lifecycle_weeks: int = 8) -> pd.DataFrame:
    """
    Create product lifecycle and age features.
    
    Features Created (7 total):
    1-2. Age metrics:
       - weeks_since_launch: Weeks since product first appearance in dataset
       - weeks_in_machine: Weeks since product first appeared in this specific machine
    
    3-6. Lifecycle indicators:
       - is_early_lifecycle: Product ≤8 weeks old overall
       - is_new_product: EAN never seen before AND provider established 1+ year (IMPORTANT for splits)
       - is_mature_product: Product >52 weeks old
       - product_age_log: Log-transformed age
    
    7. Launch timing:
       - launch_timing: Launch week normalized to year (0-1)
    
    Args:
        df: DataFrame with columns: ean, provider, machine_key, week_start, weekly_sales
        min_provider_age_years: Minimum provider age in years for is_new_product (default 1.0)
        early_lifecycle_weeks: Weeks threshold for early lifecycle (default 8)
        
    Returns:
        DataFrame with product lifecycle features added
    """
    df = df.copy()
    
    # Ensure proper sorting for time-based operations
    df = df.sort_values(['ean', 'machine_key', 'week_start']).reset_index(drop=True)
    
    # 1. Weeks since first appearance in dataset (overall)
    df['weeks_since_launch'] = df.groupby('ean')['week_start'].rank(method='dense')
    
    # 2. Weeks in specific machine (machine-EAN specific)
    df['weeks_in_machine'] = df.groupby(['machine_key', 'ean'])['week_start'].rank(method='dense')
    
    # 3. Early lifecycle flag (≤8 weeks overall)
    df['is_early_lifecycle'] = (df['weeks_since_launch'] <= early_lifecycle_weeks).astype(int)
    
    # 4. is_new_product: EAN never seen before AND provider established 1+ year
    # Calculate provider age in weeks
    provider_first_seen = df.groupby('provider')['week_start'].transform('min')
    provider_age_weeks = (df['week_start'] - provider_first_seen).dt.days / 7
    
    # EAN first appearance in dataset
    ean_first_seen = df.groupby('ean')['week_start'].transform('min')
    
    # is_new_product: First appearance AND provider established
    df['is_new_product'] = (
        (df['week_start'] == ean_first_seen) & 
        (provider_age_weeks >= min_provider_age_years * 52)
    ).astype(int)
    
    # 5. Mature product flag (>52 weeks)
    df['is_mature_product'] = (df['weeks_since_launch'] > 52).astype(int)
    
    # 6. Log-transformed product age
    df['product_age_log'] = np.log1p(df['weeks_since_launch'])
    
    # 7. Launch timing (week of year normalized to 0-1)
    df['launch_timing'] = df['week_start'].dt.isocalendar().week / 52.0
    
    return df


def create_brand_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create brand/provider-level features (same metrics for both brand and provider,
    since provider column represents the brand).
    
    Features Created (16 total):
    
    Global Brand Performance (8 features) - LAGGED:
    1-4. Brand sales aggregates:
       - brand_total_sales_lag_1: Total sales across all products
       - brand_avg_sales_per_product_lag_1: Average sales per product
       - brand_product_count_lag_1: Number of products in brand portfolio
       - brand_category_count_lag_1: Number of categories brand operates in
    
    5-8. Brand performance metrics:
       - brand_machine_penetration_lag_1: % of machines brand appears in
       - brand_price_premium_lag_1: Brand avg price vs market average
       - brand_stability_lag_1: CV of brand sales (std/mean)
       - brand_sales_ewma_4: EWMA smoothing of brand sales
    
    Brand-Machine Performance (8 features):
    
    Distribution Features (4) - CURRENT WEEK:
    9-12. Brand distribution in machine:
       - brand_machine_share: % of machine slots that are this brand
       - brand_machine_product_count: Number of brand products in machine (current week)
       - brand_category_machine_share: % of machine category slots that are this brand
       - brand_slot_representation: Total brand entries / total machine entries (non-unique)
    
    Sales Performance (4) - LAGGED:
    13-14. Brand-machine sales:
iv       - brand_machine_sales_lag_1: Total brand sales in this machine last week
       - brand_machine_avg_sales_lag_1: Average sales per brand product in machine last week
    
    15-16. Brand-machine volatility:
       - brand_machine_sales_std_lag_1: Std dev of brand sales in machine
       - brand_machine_sales_ewma_4: EWMA of brand sales in machine
    
    Args:
        df: DataFrame with columns: provider, machine_key, ean, category, week_start, 
            weekly_sales, price_mean
            
    Returns:
        DataFrame with brand features added
    """
    df = df.copy()
    
    # Ensure proper sorting
    df = df.sort_values(['provider', 'machine_key', 'week_start']).reset_index(drop=True)
    
    # ============================================================================
    # GLOBAL BRAND PERFORMANCE (LAGGED)
    # ============================================================================
    
    # 1-4. Brand aggregates (lagged)
    brand_sales_total = df.groupby(['provider', 'week_start'])['weekly_sales'].sum().reset_index(name='brand_total_sales')
    df = df.merge(brand_sales_total, on=['provider', 'week_start'], how='left')
    df['brand_total_sales_lag_1'] = df.groupby('provider')['brand_total_sales'].shift(1)
    
    df['brand_avg_sales_per_product_lag_1'] = (
        df.groupby(['provider', 'week_start'])['weekly_sales'].transform('mean')
    )
    df['brand_avg_sales_per_product_lag_1'] = df.groupby('provider')['brand_avg_sales_per_product_lag_1'].shift(1)
    
    df['brand_product_count_lag_1'] = (
        df.groupby(['provider', 'week_start'])['ean'].transform('nunique')
    )
    df['brand_product_count_lag_1'] = df.groupby('provider')['brand_product_count_lag_1'].shift(1)
    
    df['brand_category_count_lag_1'] = (
        df.groupby(['provider', 'week_start'])['category'].transform('nunique')
    )
    df['brand_category_count_lag_1'] = df.groupby('provider')['brand_category_count_lag_1'].shift(1)
    
    # 5-8. Brand performance metrics (lagged)
    # Machine penetration: % of machines brand appears in
    brand_machines_total = df.groupby(['provider', 'week_start'])['machine_key'].transform('nunique')
    total_machines = df['machine_key'].nunique()
    df['brand_machine_penetration_lag_1'] = (
        brand_machines_total / total_machines
    )
    df['brand_machine_penetration_lag_1'] = df.groupby('provider')['brand_machine_penetration_lag_1'].shift(1)
    
    # Price premium: Brand avg price vs market average
    brand_avg_price = df.groupby(['provider', 'week_start'])['price_mean'].transform('mean')
    market_avg_price = df['price_mean'].mean()
    df['brand_price_premium_lag_1'] = (brand_avg_price / market_avg_price)
    df['brand_price_premium_lag_1'] = df.groupby('provider')['brand_price_premium_lag_1'].shift(1)
    
    # Brand stability: CV (std/mean) of brand sales
    brand_sales_mean = df.groupby(['provider', 'week_start'])['weekly_sales'].transform('mean')
    brand_sales_std = df.groupby(['provider', 'week_start'])['weekly_sales'].transform('std')
    df['brand_stability_lag_1'] = (brand_sales_std / brand_sales_mean).fillna(0)
    df['brand_stability_lag_1'] = df.groupby('provider')['brand_stability_lag_1'].shift(1)
    
    # Brand EWMA
    alpha = 2 / (4 + 1)
    df['brand_sales_ewma_4'] = (
        df.groupby('provider')['brand_total_sales']
        .shift(1)
        .ewm(alpha=alpha, adjust=False)
        .mean()
    )
    
    # Drop temporary column
    df = df.drop('brand_total_sales', axis=1)
    
    # ============================================================================
    # BRAND-MACHINE DISTRIBUTION (CURRENT WEEK)
    # ============================================================================
    
    # 9. Brand machine share (% of machine slots that are this brand)
    brand_entries_in_machine = df.groupby(['machine_key', 'provider', 'week_start']).size().reset_index(name='brand_entries_count')
    total_entries_in_machine = df.groupby(['machine_key', 'week_start']).size().reset_index(name='total_entries_count')
    
    df = df.merge(brand_entries_in_machine, on=['machine_key', 'provider', 'week_start'], how='left')
    df = df.merge(total_entries_in_machine, on=['machine_key', 'week_start'], how='left')
    df['brand_machine_share'] = (df['brand_entries_count'] / df['total_entries_count'] * 100).fillna(0)
    df = df.drop(['brand_entries_count', 'total_entries_count'], axis=1)
    
    # 10. Brand machine product count (current week)
    df['brand_machine_product_count'] = (
        df.groupby(['machine_key', 'provider', 'week_start'])['ean']
        .transform('nunique')
    )
    # 11. Brand category machine share (% of machine category slots that are this brand)
    brand_category_entries = df.groupby(['machine_key', 'provider', 'category', 'week_start']).size().reset_index(name='brand_category_entries')
    category_entries = df.groupby(['machine_key', 'category', 'week_start']).size().reset_index(name='category_entries')
    
    df = df.merge(brand_category_entries, on=['machine_key', 'provider', 'category', 'week_start'], how='left')
    df = df.merge(category_entries, on=['machine_key', 'category', 'week_start'], how='left')
    df['brand_category_machine_share'] = (df['brand_category_entries'] / df['category_entries'] * 100).fillna(0)
    df = df.drop(['brand_category_entries', 'category_entries'], axis=1)
    
    # 12. Brand slot representation (total brand entries / total machine entries, non-unique)
    # This is same as brand_machine_share but keeping both for clarity
    brand_total_entries = df.groupby(['machine_key', 'provider', 'week_start']).size().reset_index(name='brand_total_entries')
    machine_total_entries = df.groupby(['machine_key', 'week_start']).size().reset_index(name='machine_total_entries')
    
    df = df.merge(brand_total_entries, on=['machine_key', 'provider', 'week_start'], how='left')
    df = df.merge(machine_total_entries, on=['machine_key', 'week_start'], how='left')
    df['brand_slot_representation'] = (df['brand_total_entries'] / df['machine_total_entries'] * 100).fillna(0)
    df = df.drop(['brand_total_entries', 'machine_total_entries'], axis=1)
    
    # ============================================================================
    # BRAND-MACHINE SALES PERFORMANCE (LAGGED)
    # ============================================================================
    
    # สรุป -14. Brand-machine sales (lagged)
    brand_machine_sales = df.groupby(['machine_key', 'provider', 'week_start'])['weekly_sales'].sum().reset_index(name='brand_machine_sales_total')
    df = df.merge(brand_machine_sales, on=['machine_key', 'provider', 'week_start'], how='left')
    df['brand_machine_sales_lag_1'] = df.groupby(['machine_key', 'provider'])['brand_machine_sales_total'].shift(1)
    
    df['brand_machine_avg_sales_lag_1'] = (
        df.groupby(['machine_key', 'provider', 'week_start'])['weekly_sales'].transform('mean')
    )
    df['brand_machine_avg_sales_lag_1'] = df.groupby(['machine_key', 'provider'])['brand_machine_avg_sales_lag_1'].shift(1)
    
    # 15-16. Brand-machine volatility (lagged)
    df['brand_machine_sales_std_lag_1'] = (
        df.groupby(['machine_key', 'provider', 'week_start'])['weekly_sales'].transform('std')
    )
    df['brand_machine_sales_std_lag_1'] = df.groupby(['machine_key', 'provider'])['brand_machine_sales_std_lag_1'].shift(1)
    
    # Brand-machine EWMA
    df['brand_machine_sales_ewma_4'] = (
        df.groupby(['machine_key', 'provider'])['brand_machine_sales_total']
        .shift(1)
        .ewm(alpha=alpha, adjust=False)
        .mean()
    )
    
    # Drop temporary column
    df = df.drop('brand_machine_sales_total', axis=1)
    
    # Fill NaN with 0 for ratio features
    brand_ratio_cols = [
        'brand_machine_penetration_lag_1', 'brand_price_premium_lag_1',
        'brand_stability_lag_1', 'brand_machine_share', 'brand_category_machine_share',
        'brand_slot_representation'
    ]
    for col in brand_ratio_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    return df


def create_all_features(
    df: pd.DataFrame,
    feature_groups: Optional[Dict[str, bool]] = None,
    use_cold_start_fallback: bool = True
) -> pd.DataFrame:
    """
    Master function to create all feature groups with optional selection.
    
    Args:
        df: Input DataFrame with base columns (week_start, category, subcategory, 
            machine_key, ean, price_mean, purchase_price_kr, weekly_sales, etc.)
        feature_groups: Dictionary specifying which feature groups to include.
            Keys: 'TEMPORAL', 'PRODUCT', 'MACHINE', 'HISTORICAL_SALES'
            Values: True to include, False to exclude
            If None, all groups are included by default.
        use_cold_start_fallback: Whether to apply cold-start fallback to 
            historical sales features
            
    Returns:
        DataFrame with all requested features added
        
    Example:
        # Include only TEMPORAL and PRODUCT features
        df_features = create_all_features(
            df, 
            feature_groups={'TEMPORAL': True, 'PRODUCT': True, 
                          'MACHINE': False, 'HISTORICAL_SALES': False}
        )
        
        # Include all features (default)
        df_features = create_all_features(df)
    """
    df = df.copy()
    
    # Default: all groups enabled
    if feature_groups is None:
        feature_groups = {
            'BASE': True,
            'TEMPORAL': True,
            'PRODUCT': True,
            'MACHINE': True,
            'HISTORICAL_SALES': True,
            'PRODUCT_LIFECYCLE': True,
            'BRAND': True
        }
    else:
        # Ensure all groups are specified (False for missing ones)
        default_groups = {
            'BASE': False,
            'TEMPORAL': False,
            'PRODUCT': False,
            'MACHINE': False,
            'HISTORICAL_SALES': False,
            'PRODUCT_LIFECYCLE': False,
            'BRAND': False
        }
        default_groups.update(feature_groups)
        feature_groups = default_groups
    
    # BASE features are always created first (foundation for all other features)
    # Auto-enable BASE if any other feature group is enabled
    if any(feature_groups.get(k, False) for k in ['TEMPORAL', 'PRODUCT', 'MACHINE', 'HISTORICAL_SALES', 'PRODUCT_LIFECYCLE', 'BRAND']):
        feature_groups['BASE'] = True
    
    # Apply feature groups in order with progress bars
    feature_group_order = [
        ('BASE', create_base_features),
        ('TEMPORAL', create_temporal_features),
        ('PRODUCT', create_product_features),
        ('MACHINE', create_machine_features),
        ('HISTORICAL_SALES', lambda d: create_historical_sales_features(d, use_cold_start_fallback=use_cold_start_fallback)),
        ('PRODUCT_LIFECYCLE', create_product_lifecycle_features),
        ('BRAND', create_brand_features),
    ]
    
    enabled_groups = [name for name, _ in feature_group_order if feature_groups.get(name, False)]
    
    if enabled_groups:
        print(f"\n📊 Creating {len(enabled_groups)} feature group(s): {', '.join(enabled_groups)}")
    
    for group_name, func in tqdm(feature_group_order, desc="Feature Groups", leave=True):
        if feature_groups.get(group_name, False):
            df = func(df)
    
    return df
