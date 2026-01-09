"""
Data Splitting Module - Robust Implementation
==============================================

Implements three data splitting strategies for new product forecasting with
improved robustness, scalability, and logical correctness.

Key improvements:
- Cutoff-first approach ensures temporal ordering
- Better handling of edge cases
- More efficient memory usage for large datasets
- Clearer validation and error messages
- Vectorized operations where possible
"""

import pandas as pd
import numpy as np
from typing import Dict, Set, Tuple, Optional
import logging
from pathlib import Path
from ..config import SPLITS_DIR

logger = logging.getLogger(__name__)


def identify_new_products(
    df: pd.DataFrame, 
    min_provider_age_years: float = 1.0
) -> Tuple[Set, pd.DataFrame]:
    """
    Identify new products: EAN never seen before AND provider established ≥1 year.
    
    Args:
        df: Full dataset with columns [week_start, ean, provider, ...]
        min_provider_age_years: Minimum provider age to consider product "new"
        
    Returns:
        (set of new EAN codes, DataFrame with new product metadata)
    """
    logger.info("🔍 IDENTIFYING NEW PRODUCTS")
    logger.info("=" * 50)
    
    # Ensure datetime type
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df['week_start']):
        df['week_start'] = pd.to_datetime(df['week_start'])
    
    # Get provider first appearance (vectorized)
    provider_first_seen = (
        df.groupby('provider', as_index=False)['week_start']
        .min()
        .rename(columns={'week_start': 'provider_first_seen'})
    )
    
    # Get EAN first appearance with metadata
    ean_info = (
        df.groupby('ean', as_index=False)
        .agg({
            'week_start': 'min',
            'provider': 'first',
            'product_name': 'first',
            'category': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]
        })
        .rename(columns={'week_start': 'ean_first_seen'})
    )
    
    # Merge and calculate provider age at EAN launch
    ean_info = ean_info.merge(provider_first_seen, on='provider', how='left')
    ean_info['provider_age_at_launch_days'] = (
        (ean_info['ean_first_seen'] - ean_info['provider_first_seen']).dt.days
    )
    
    # Filter for new products
    min_days = min_provider_age_years * 365
    new_product_info = ean_info[
        ean_info['provider_age_at_launch_days'] >= min_days
    ].copy()
    new_product_eans = set(new_product_info['ean'].values)
    
    # Add sales statistics
    new_product_sales = (
        df[df['ean'].isin(new_product_eans)]
        .groupby('ean', as_index=False)
        .agg({
            'weekly_sales': 'sum',
            'machine_key': 'nunique'
        })
        .rename(columns={
            'weekly_sales': 'total_sales',
            'machine_key': 'unique_machines'
        })
    )
    new_product_info = new_product_info.merge(new_product_sales, on='ean', how='left')
    
    # Log results
    total_products = df['ean'].nunique()
    new_product_ratio = len(new_product_eans) / total_products * 100
    
    logger.info(f"  • Total products: {total_products:,}")
    logger.info(f"  • New products: {len(new_product_eans):,} ({new_product_ratio:.1f}%)")
    logger.info(f"  • Date range: {new_product_info['ean_first_seen'].min()} → "
                f"{new_product_info['ean_first_seen'].max()}")
    
    return new_product_eans, new_product_info


def method_1_all_data_to_new_products(
    df: pd.DataFrame, 
    new_product_eans: Set, 
    target_test_pct: float = 0.15,
    tolerance: float = 0.05
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Method 1: All data for train, only new products for test.
    
    FIXED LOGIC: Use established products for training context, but split new products
    temporally to achieve target test percentage. This maintains the "all data for train"
    concept while ensuring correct test percentage calculation.
    
    Args:
        df: Full dataset
        new_product_eans: Set of new product EANs
        target_test_pct: Target test percentage
        tolerance: Acceptable deviation from target (e.g., 0.05 = ±5%)
        
    Returns:
        (train_df, test_df)
    """
    logger.info("📊 METHOD 1: ALL DATA → NEW PRODUCTS (FIXED)")
    logger.info("=" * 50)
    
    # Ensure datetime
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df['week_start']):
        df['week_start'] = pd.to_datetime(df['week_start'])
    
    # Separate established and new product data
    established_data = df[~df['ean'].isin(new_product_eans)].copy()
    new_product_data = df[df['ean'].isin(new_product_eans)].copy()
    
    logger.info(f"  • Total rows: {len(df):,}")
    logger.info(f"  • Established product rows: {len(established_data):,}")
    logger.info(f"  • New product rows: {len(new_product_data):,}")
    
    if len(new_product_data) == 0:
        raise ValueError("No new product data found in dataset")
    
    # Apply Method 2 logic to new products only
    logger.info("  • Applying temporal split to new products...")
    
    # Get first appearance per EAN
    ean_first_appearance = (
        new_product_data.groupby('ean', as_index=False)['week_start']
        .min()
        .rename(columns={'week_start': 'first_appearance_week'})
    )
    
    # Sort weeks descending to work backwards
    unique_weeks = sorted(ean_first_appearance['first_appearance_week'].unique(), reverse=True)
    
    logger.info(f"  • New product date range: {min(unique_weeks)} → {max(unique_weeks)}")
    logger.info(f"  • Target test size: {target_test_pct:.1%}")
    
    # Find cutoff week by going backwards
    best_cutoff = None
    best_test_pct = 0
    best_distance = float('inf')
    best_test_eans = set()
    
    logger.info(f"  • Searching for optimal cutoff week...")
    
    for i, cutoff_week in enumerate(unique_weeks):
        # Test EANs: those that FIRST APPEAR >= cutoff_week
        test_eans = set(
            ean_first_appearance[
                ean_first_appearance['first_appearance_week'] >= cutoff_week
            ]['ean']
        )
        
        # Calculate test percentage based on TOTAL ROWS (not just new products)
        test_rows = len(new_product_data[new_product_data['ean'].isin(test_eans)])
        train_new_rows = len(new_product_data[~new_product_data['ean'].isin(test_eans)])
        
        if train_new_rows == 0:
            continue
        
        # Calculate total rows including established products
        established_before_cutoff = len(established_data[established_data['week_start'] < cutoff_week])
        total_train_rows = established_before_cutoff + train_new_rows
        total_test_rows = test_rows
        
        test_pct = total_test_rows / (total_train_rows + total_test_rows)
        distance = abs(test_pct - target_test_pct)
        
        # Log every 10th week or when we find a better split
        should_log = (i % 10 == 0) or (distance < best_distance)
        
        if should_log:
            logger.info(f"    Week {cutoff_week.strftime('%Y-%m-%d')}: {len(test_eans)} test EANs, "
                       f"{test_pct:.1%} test size, distance={distance:.3f}")
        
        if distance < best_distance:
            best_distance = distance
            best_cutoff = cutoff_week
            best_test_pct = test_pct
            best_test_eans = test_eans.copy()
            
            # Log when we find a better split
            logger.info(f"    ✓ New best: {test_pct:.1%} test size (target: {target_test_pct:.1%})")
    
    if best_cutoff is None:
        raise ValueError("Could not find valid cutoff week")
    
    logger.info(f"  • Final cutoff: {best_cutoff.strftime('%Y-%m-%d')}")
    logger.info(f"  • Final test size: {best_test_pct:.1%} (target: {target_test_pct:.1%})")
    logger.info(f"  • Test EANs: {len(best_test_eans)}")
    
    # Create final split with best cutoff
    test_eans = best_test_eans
    
    # Split new products
    train_new_products = new_product_data[~new_product_data['ean'].isin(test_eans)].copy()
    test_new_products = new_product_data[new_product_data['ean'].isin(test_eans)].copy()
    
    # CRITICAL FIX: Only include established products that appeared BEFORE cutoff
    # This ensures temporal integrity
    established_before_cutoff = established_data[
        established_data['week_start'] < best_cutoff
    ].copy()
    
    # Combine established data (before cutoff) with new product training data
    train_df = pd.concat([established_before_cutoff, train_new_products], ignore_index=True)
    test_df = test_new_products
    
    logger.info(f"  • Final split created:")
    logger.info(f"    Cutoff week: {best_cutoff.strftime('%Y-%m-%d')}")
    logger.info(f"    Test EANs: {len(test_eans):,}")
    logger.info(f"    Train: {len(train_df):,} rows ({len(established_before_cutoff):,} established + {len(train_new_products):,} new)")
    logger.info(f"    Test: {len(test_df):,} rows (new products only)")
    logger.info(f"    Test % (total rows): {best_test_pct:.1%}")
    logger.info(f"    Train EANs: {train_df['ean'].nunique():,}")
    logger.info(f"    Test EANs: {test_df['ean'].nunique():,}")
    
    return train_df, test_df


def method_2_new_products_only(
    df: pd.DataFrame, 
    new_product_eans: Set, 
    target_test_pct: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Method 2: Only new products for both train and test.
    
    CUTOFF-FIRST approach: Find temporal cutoff, then select which EANs 
    that FIRST APPEAR after cutoff go to test.
    
    Args:
        df: Full dataset
        new_product_eans: Set of new product EANs
        target_test_pct: Target test percentage
        
    Returns:
        (train_df, test_df)
    """
    logger.info("📊 METHOD 2: NEW PRODUCTS ONLY")
    logger.info("=" * 50)
    
    # Ensure datetime
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df['week_start']):
        df['week_start'] = pd.to_datetime(df['week_start'])
    
    # Filter to new products only
    filtered_df = df[df['ean'].isin(new_product_eans)].copy()
    
    if len(filtered_df) == 0:
        raise ValueError("No new product data found in dataset")
    
    logger.info(f"  • Filtered dataset: {len(filtered_df):,} rows")
    logger.info(f"  • Unique new products: {filtered_df['ean'].nunique():,}")
    
    # Get first appearance per EAN
    ean_first_appearance = (
        filtered_df.groupby('ean', as_index=False)['week_start']
        .min()
        .rename(columns={'week_start': 'first_appearance_week'})
    )
    
    # Sort weeks descending to work backwards
    unique_weeks = sorted(ean_first_appearance['first_appearance_week'].unique(), reverse=True)
    
    logger.info(f"  • Date range: {min(unique_weeks)} → {max(unique_weeks)}")
    logger.info(f"  • Target test size: {target_test_pct:.1%}")
    
    # Find cutoff week by going backwards
    best_cutoff = None
    best_test_pct = 0
    best_distance = float('inf')
    
    for cutoff_week in unique_weeks:
        # Test EANs: those that FIRST APPEAR >= cutoff_week
        test_eans = set(
            ean_first_appearance[
                ean_first_appearance['first_appearance_week'] >= cutoff_week
            ]['ean']
        )
        
        # Calculate test percentage
        # Key: ALL data for test EANs goes to test (temporal split within filtered data)
        test_rows = len(filtered_df[filtered_df['ean'].isin(test_eans)])
        train_rows = len(filtered_df[~filtered_df['ean'].isin(test_eans)])
        
        if train_rows == 0:
            continue
        
        test_pct = test_rows / (train_rows + test_rows)
        distance = abs(test_pct - target_test_pct)
        
        if distance < best_distance:
            best_distance = distance
            best_cutoff = cutoff_week
            best_test_pct = test_pct
    
    if best_cutoff is None:
        raise ValueError("Could not find valid cutoff week")
    
    # Create final split with best cutoff
    test_eans = set(
        ean_first_appearance[
            ean_first_appearance['first_appearance_week'] >= best_cutoff
        ]['ean']
    )
    
    train_df = filtered_df[~filtered_df['ean'].isin(test_eans)].copy()
    test_df = filtered_df[filtered_df['ean'].isin(test_eans)].copy()
    
    logger.info(f"  • Cutoff week: {best_cutoff}")
    logger.info(f"  • Test EANs: {len(test_eans):,}")
    logger.info(f"  • Train: {len(train_df):,} rows, {train_df['ean'].nunique():,} EANs")
    logger.info(f"  • Test: {len(test_df):,} rows, {test_df['ean'].nunique():,} EANs")
    logger.info(f"  • Actual test %: {best_test_pct:.1%}")
    
    return train_df, test_df


def method_3_first_weeks_only(
    df: pd.DataFrame, 
    new_product_eans: Set, 
    max_weeks: int = 6, 
    target_test_pct: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Method 3: First N weeks of new products only.
    
    CUTOFF-FIRST approach applied to early weeks data.
    
    Args:
        df: Full dataset
        new_product_eans: Set of new product EANs
        max_weeks: Maximum weeks after launch to include
        target_test_pct: Target test percentage
        
    Returns:
        (train_df, test_df)
    """
    logger.info("📊 METHOD 3: FIRST WEEKS ONLY")
    logger.info("=" * 50)
    
    # Ensure datetime
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df['week_start']):
        df['week_start'] = pd.to_datetime(df['week_start'])
    
    # Filter to new products
    new_product_data = df[df['ean'].isin(new_product_eans)].copy()
    
    if len(new_product_data) == 0:
        raise ValueError("No new product data found")
    
    # Calculate weeks since launch (vectorized)
    ean_first_seen = new_product_data.groupby('ean')['week_start'].transform('min')
    weeks_since_launch = ((new_product_data['week_start'] - ean_first_seen).dt.days // 7)
    
    # Filter to early weeks
    early_mask = weeks_since_launch <= max_weeks
    early_weeks_data = new_product_data[early_mask].copy()
    
    logger.info(f"  • Early weeks data: {len(early_weeks_data):,} rows (≤{max_weeks} weeks)")
    logger.info(f"  • Unique products: {early_weeks_data['ean'].nunique():,}")
    
    if len(early_weeks_data) == 0:
        raise ValueError(f"No data found within first {max_weeks} weeks of new products")
    
    # Get first appearance per EAN (in the early weeks data)
    ean_first_appearance = (
        early_weeks_data.groupby('ean', as_index=False)['week_start']
        .min()
        .rename(columns={'week_start': 'first_appearance_week'})
    )
    
    # Sort weeks descending
    unique_weeks = sorted(ean_first_appearance['first_appearance_week'].unique(), reverse=True)
    
    logger.info(f"  • Date range: {min(unique_weeks)} → {max(unique_weeks)}")
    logger.info(f"  • Target test size: {target_test_pct:.1%}")
    
    # Find cutoff week
    best_cutoff = None
    best_test_pct = 0
    best_distance = float('inf')
    
    for cutoff_week in unique_weeks:
        # Test EANs: those that FIRST APPEAR >= cutoff_week
        test_eans = set(
            ean_first_appearance[
                ean_first_appearance['first_appearance_week'] >= cutoff_week
            ]['ean']
        )
        
        # Calculate test percentage
        test_rows = len(early_weeks_data[early_weeks_data['ean'].isin(test_eans)])
        train_rows = len(early_weeks_data[~early_weeks_data['ean'].isin(test_eans)])
        
        if train_rows == 0:
            continue
        
        test_pct = test_rows / (train_rows + test_rows)
        distance = abs(test_pct - target_test_pct)
        
        if distance < best_distance:
            best_distance = distance
            best_cutoff = cutoff_week
            best_test_pct = test_pct
    
    if best_cutoff is None:
        raise ValueError("Could not find valid cutoff week")
    
    # Create final split
    test_eans = set(
        ean_first_appearance[
            ean_first_appearance['first_appearance_week'] >= best_cutoff
        ]['ean']
    )
    
    train_df = early_weeks_data[~early_weeks_data['ean'].isin(test_eans)].copy()
    test_df = early_weeks_data[early_weeks_data['ean'].isin(test_eans)].copy()
    
    logger.info(f"  • Cutoff week: {best_cutoff}")
    logger.info(f"  • Test EANs: {len(test_eans):,}")
    logger.info(f"  • Train: {len(train_df):,} rows, {train_df['ean'].nunique():,} EANs")
    logger.info(f"  • Test: {len(test_df):,} rows, {test_df['ean'].nunique():,} EANs")
    logger.info(f"  • Actual test %: {best_test_pct:.1%}")
    
    return train_df, test_df


def method_4_vendtrend_comparison(
    df: pd.DataFrame, 
    train_start: str = "2020-01-01",
    train_end: str = "2023-12-31", 
    test_start: str = "2024-01-01",
    test_end: str = "2024-12-31"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Method 4: VendTrend comparison split - Simple date-based split.
    
    Args:
        df: Full dataset
        train_start: Training data start date (inclusive)
        train_end: Training data end date (inclusive)
        test_start: Test data start date (inclusive)
        test_end: Test data end date (inclusive)
        
    Returns:
        (train_df, test_df)
    """
    logger.info("📊 METHOD 4: VENDTREND COMPARISON")
    logger.info("=" * 50)
    
    # Ensure datetime
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df['week_start']):
        df['week_start'] = pd.to_datetime(df['week_start'])
    
    # Convert date strings to datetime
    train_start_dt = pd.to_datetime(train_start)
    train_end_dt = pd.to_datetime(train_end)
    test_start_dt = pd.to_datetime(test_start)
    test_end_dt = pd.to_datetime(test_end)
    
    logger.info(f"  • Train period: {train_start} → {train_end}")
    logger.info(f"  • Test period: {test_start} → {test_end}")
    
    # Create train and test splits
    train_df = df[
        (df['week_start'] >= train_start_dt) & 
        (df['week_start'] <= train_end_dt)
    ].copy()
    
    test_df = df[
        (df['week_start'] >= test_start_dt) & 
        (df['week_start'] <= test_end_dt)
    ].copy()
    
    logger.info(f"  • Train: {len(train_df):,} rows, {train_df['ean'].nunique():,} EANs")
    logger.info(f"  • Test: {len(test_df):,} rows, {test_df['ean'].nunique():,} EANs")
    
    # Calculate test percentage
    test_pct = len(test_df) / (len(train_df) + len(test_df)) if len(train_df) + len(test_df) > 0 else 0
    logger.info(f"  • Test %: {test_pct:.1%}")
    
    return train_df, test_df


def method_5_august_2024_temporal(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Method 5: August 2024 temporal split - splits at the latest week ending in August 2024.
    Test set = all weeks after August 2024 through end of dataset.
    
    Args:
        df: Full dataset with week_start column
        
    Returns:
        (train_df, test_df)
    """
    logger.info("📊 METHOD 5: AUGUST 2024 TEMPORAL SPLIT")
    logger.info("=" * 50)
    
    # Ensure datetime
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df['week_start']):
        df['week_start'] = pd.to_datetime(df['week_start'])
    
    # Find all weeks ending in August 2024
    # Week ending = last day of that week (typically Sunday)
    # For simplicity, we'll use week_start and check if it's in August 2024
    # Then find the maximum week_start that is still in August 2024
    august_2024_df = df[
        (df['week_start'].dt.year == 2024) & 
        (df['week_start'].dt.month == 8)
    ]
    
    if len(august_2024_df) == 0:
        raise ValueError("No data found for August 2024")
    
    # Get the latest week_start in August 2024
    cutoff_week = august_2024_df['week_start'].max()
    
    # Calculate week end (typically 6 days after week_start)
    cutoff_date = cutoff_week + pd.Timedelta(days=6)
    
    logger.info(f"  • Cutoff week start: {cutoff_week}")
    logger.info(f"  • Cutoff date (end of week): {cutoff_date}")
    
    # Train = all data up to and including cutoff_week
    train_df = df[df['week_start'] <= cutoff_week].copy()
    
    # Test = all data after cutoff_week (next week onwards)
    test_df = df[df['week_start'] > cutoff_week].copy()
    
    logger.info(f"  • Train: {len(train_df):,} rows ({train_df['week_start'].min()} to {train_df['week_start'].max()})")
    logger.info(f"  • Test: {len(test_df):,} rows ({test_df['week_start'].min()} to {test_df['week_start'].max()})")
    logger.info(f"  • Train EANs: {train_df['ean'].nunique():,}")
    logger.info(f"  • Test EANs: {test_df['ean'].nunique():,}")
    
    # Calculate test percentage
    test_pct = len(test_df) / (len(train_df) + len(test_df)) if len(train_df) + len(test_df) > 0 else 0
    logger.info(f"  • Test %: {test_pct:.1%}")
    
    return train_df, test_df


def validate_split(
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame, 
    new_product_eans: Set,
    method_name: str = "",
    min_test_pct: float = 0.05,
    max_test_pct: float = 0.50
) -> bool:
    """
    Validate data split meets all requirements.
    
    Args:
        train_df: Training data
        test_df: Test data
        new_product_eans: Set of new product EANs
        method_name: Name of method for logging
        min_test_pct: Minimum acceptable test percentage
        max_test_pct: Maximum acceptable test percentage
        
    Returns:
        True if all checks pass
        
    Raises:
        AssertionError: If any validation fails
    """
    logger.info(f"🔍 VALIDATING {method_name}")
    logger.info("=" * 50)
    
    # Check 1: No EAN overlap
    train_eans = set(train_df['ean'].unique())
    test_eans = set(test_df['ean'].unique())
    overlap = train_eans.intersection(test_eans)
    
    assert len(overlap) == 0, (
        f"EAN overlap detected: {len(overlap)} EANs in both train and test. "
        f"Examples: {list(overlap)[:5]}"
    )
    logger.info(f"  ✓ No EAN overlap: {len(train_eans):,} train, {len(test_eans):,} test")
    
    # Check 2: Temporal ordering - CRITICAL FIX
    # For Methods 2 and 3, we need to check that all train data comes before all test data
    # Since we're using a cutoff on first_appearance_week, train EANs can have data
    # AFTER test EANs start appearing. This is acceptable IF train EANs appeared BEFORE cutoff.
    
    # What we really need to check: no train EAN first appeared after any test EAN first appeared
    if len(train_df) > 0 and len(test_df) > 0:
        # Get first appearance of each EAN in each set
        train_first_appearance = train_df.groupby('ean')['week_start'].min()
        test_first_appearance = test_df.groupby('ean')['week_start'].min()
        
        train_max_first_appearance = train_first_appearance.max()
        test_min_first_appearance = test_first_appearance.min()
        
        assert train_max_first_appearance < test_min_first_appearance, (
            f"Temporal ordering violated: Latest train EAN first appeared {train_max_first_appearance}, "
            f"but earliest test EAN first appeared {test_min_first_appearance}. "
            f"Train EANs must all first appear BEFORE any test EAN first appears."
        )
        logger.info(f"  ✓ Temporal ordering: train EANs first appear ≤ {train_max_first_appearance}, "
                   f"test EANs first appear ≥ {test_min_first_appearance}")
    
    # Check 3: Test only contains new products
    test_non_new = test_df[~test_df['ean'].isin(new_product_eans)]
    assert len(test_non_new) == 0, (
        f"Test contains {len(test_non_new)} rows from non-new products"
    )
    logger.info(f"  ✓ Test contains only new products")
    
    # Check 4: Test size within acceptable range
    total_size = len(train_df) + len(test_df)
    test_pct = len(test_df) / total_size if total_size > 0 else 0
    
    # Special handling for Method 1: calculate test % based on new products only
    if method_name == "Method 1":
        # For Method 1, we need to calculate test % based on new products only
        # since established products are always in training
        new_product_rows_in_train = len(train_df[train_df['ean'].isin(new_product_eans)])
        new_product_rows_in_test = len(test_df)
        total_new_product_rows = new_product_rows_in_train + new_product_rows_in_test
        
        if total_new_product_rows > 0:
            test_pct = new_product_rows_in_test / total_new_product_rows
            logger.info(f"  ✓ Test % (new products only): {test_pct:.1%}")
        else:
            test_pct = 0
            logger.warning("  ⚠️ No new product rows found for percentage calculation")
    
    assert min_test_pct <= test_pct <= max_test_pct, (
        f"Test size {test_pct:.1%} outside range [{min_test_pct:.1%}, {max_test_pct:.1%}]. "
        f"Consider adjusting target_test_pct or using a different method."
    )
    logger.info(f"  ✓ Test size: {test_pct:.1%} (within [{min_test_pct:.1%}, {max_test_pct:.1%}])")
    
    # Check 5: Non-empty datasets
    assert len(train_df) > 0, "Training set is empty"
    assert len(test_df) > 0, "Test set is empty"
    logger.info(f"  ✓ Both sets non-empty")
    
    logger.info("✅ All validation checks passed!")
    return True


def create_all_splits(
    df: pd.DataFrame, 
    target_test_pct: float = 0.15, 
    min_provider_age_years: float = 1.0, 
    max_weeks: int = 6,
    tolerance: float = 0.05,
    save_to_disk: bool = True,
    output_dir: Optional[Path] = None
) -> Dict[str, Optional[Dict[str, pd.DataFrame]]]:
    """
    Create all three data splits with robust error handling.
    
    Args:
        df: Full dataset
        target_test_pct: Target test percentage
        min_provider_age_years: Minimum provider age for new product definition
        max_weeks: Maximum weeks for Method 3
        tolerance: Acceptable deviation from target_test_pct for Method 1
        save_to_disk: Whether to save splits to parquet files
        output_dir: Directory for output files (default: data/splits)
        
    Returns:
        Dict mapping method names to {'train': df, 'test': df} or None if failed
    """
    logger.info("🚀 CREATING ALL DATA SPLITS")
    logger.info("=" * 60)
    
    # Identify new products
    new_product_eans, new_product_info = identify_new_products(
        df, min_provider_age_years
    )
    
    if len(new_product_eans) == 0:
        raise ValueError(
            "No new products identified. Check min_provider_age_years parameter "
            "or data quality."
        )
    
    # Log new product statistics
    total_products = df['ean'].nunique()
    new_product_percentage = len(new_product_eans) / total_products * 100
    logger.info(f"📊 NEW PRODUCT STATISTICS")
    logger.info(f"  • Total unique products: {total_products:,}")
    logger.info(f"  • New products: {len(new_product_eans):,}")
    logger.info(f"  • New product percentage: {new_product_percentage:.1f}%")
    logger.info(f"  • New product rows: {len(df[df['ean'].isin(new_product_eans)]):,}")
    logger.info(f"  • Total rows: {len(df):,}")
    logger.info("")
    
    splits = {}
    
    # Method 1
    logger.info("\n" + "="*60)
    try:
        train_1, test_1 = method_1_all_data_to_new_products(
            df, new_product_eans, target_test_pct, tolerance
        )
        validate_split(train_1, test_1, new_product_eans, "Method 1")
        splits['method_1_all_to_new'] = {'train': train_1, 'test': test_1}
        logger.info("✅ Method 1 completed successfully")
    except Exception as e:
        logger.error(f"❌ Method 1 failed: {str(e)}")
        splits['method_1_all_to_new'] = None
    
    # Method 2
    logger.info("\n" + "="*60)
    try:
        train_2, test_2 = method_2_new_products_only(
            df, new_product_eans, target_test_pct
        )
        validate_split(train_2, test_2, new_product_eans, "Method 2")
        splits['method_2_new_only'] = {'train': train_2, 'test': test_2}
        logger.info("✅ Method 2 completed successfully")
    except Exception as e:
        logger.error(f"❌ Method 2 failed: {str(e)}")
        splits['method_2_new_only'] = None
    
    # Method 3
    logger.info("\n" + "="*60)
    try:
        train_3, test_3 = method_3_first_weeks_only(
            df, new_product_eans, max_weeks, target_test_pct
        )
        validate_split(train_3, test_3, new_product_eans, "Method 3")
        splits['method_3_first_weeks'] = {'train': train_3, 'test': test_3}
        logger.info("✅ Method 3 completed successfully")
    except Exception as e:
        logger.error(f"❌ Method 3 failed: {str(e)}")
        splits['method_3_first_weeks'] = None
    
    # Method 4
    logger.info("\n" + "="*60)
    try:
        train_4, test_4 = method_4_vendtrend_comparison(df)
        # Method 4 doesn't need new product validation since it's date-based
        splits['method_4_vendtrend'] = {'train': train_4, 'test': test_4}
        logger.info("✅ Method 4 completed successfully")
    except Exception as e:
        logger.error(f"❌ Method 4 failed: {str(e)}")
        splits['method_4_vendtrend'] = None
    
    # Save to disk
    if save_to_disk:
        logger.info("\n" + "="*60)
        logger.info("💾 SAVING SPLITS TO DISK")
        logger.info("=" * 60)
        
        if output_dir is None:
            output_dir = SPLITS_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for split_name, split_data in splits.items():
            if split_data is not None:
                train_file = output_dir / f"{split_name}_train.parquet"
                test_file = output_dir / f"{split_name}_test.parquet"
                
                split_data['train'].to_parquet(train_file, compression='snappy')
                split_data['test'].to_parquet(test_file, compression='snappy')
                
                logger.info(f"  ✓ {split_name}")
                logger.info(f"    Train: {train_file}")
                logger.info(f"    Test:  {test_file}")
        
        logger.info(f"\n✅ Splits saved to {output_dir}")
    
    # Log final summary
    logger.info("\n" + "="*60)
    logger.info("📊 FINAL SPLIT SUMMARY")
    logger.info("=" * 60)
    
    successful_splits = 0
    for split_name, split_data in splits.items():
        if split_data is not None:
            successful_splits += 1
            train_count = len(split_data['train'])
            test_count = len(split_data['test'])
            
            # All methods: Test % = test_count / (train_count + test_count)
            test_pct = test_count / (train_count + test_count) if (train_count + test_count) > 0 else 0
            
            logger.info(f"\n✅ {split_name}:")
            logger.info(f"  Train: {train_count:,} records")
            logger.info(f"  Test:  {test_count:,} records")
            logger.info(f"  Test %: {test_pct:.1%}")
            logger.info(f"  Train products: {split_data['train']['ean'].nunique()}")
            logger.info(f"  Test products:  {split_data['test']['ean'].nunique()}")
        else:
            logger.info(f"\n❌ {split_name}: Failed")
    
    logger.info(f"\n🎯 Successfully created {successful_splits}/4 data splits")
    
    return splits