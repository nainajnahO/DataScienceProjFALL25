import logging
from pathlib import Path
from typing import Optional, Union, List
from datetime import timedelta
import pandas as pd
import numpy as np
import itertools

from .models.multi_week_forecaster import MultiWeekForecaster
from .features.tier1_features import create_all_features
from .config import MODEL_DIR
from .data.processor import process_sales_data

logger = logging.getLogger(__name__)

def map_ean_to_product_name(
    df: pd.DataFrame,
    product_df: pd.DataFrame,
    ean_col: str = 'ean',
    product_name_col: str = 'product_name',
    overwrite_existing: bool = False
) -> pd.DataFrame:
    """
    Maps EAN to product_name using product_df and adds product_name column to df.
    
    Args:
        df: DataFrame with 'ean' column to map
        product_df: DataFrame with 'ean' and 'product_name' columns for mapping
        ean_col: Name of EAN column in df (default: 'ean')
        product_name_col: Name of product_name column to add/create (default: 'product_name')
        overwrite_existing: If True, overwrites existing product_name values. If False, only fills NaN values.
        
    Returns:
        DataFrame with product_name column added/updated
    """
    df = df.copy()
    
    # Validate product_df has required columns
    if 'ean' not in product_df.columns or 'product_name' not in product_df.columns:
        logger.error("product_df must have 'ean' and 'product_name' columns")
        return df
    
    # Validate df has EAN column
    if ean_col not in df.columns:
        logger.error(f"df must have '{ean_col}' column")
        return df
    
    # Create EAN -> product_name mapping (drop duplicates to handle non-unique EANs)
    # Keep first occurrence if duplicates exist
    product_df_unique = product_df.drop_duplicates(subset=['ean'], keep='first')
    ean_to_name = product_df_unique.set_index('ean')['product_name'].to_dict()
    
    if len(product_df_unique) < len(product_df):
        duplicates = len(product_df) - len(product_df_unique)
        logger.warning(f"Found {duplicates} duplicate EANs in product_df, using first occurrence for each")
    
    # Add product_name to df
    if product_name_col not in df.columns or overwrite_existing:
        # Map EAN to product_name
        df[product_name_col] = df[ean_col].map(ean_to_name)
        missing_eans = df[product_name_col].isna().sum()
        if missing_eans > 0:
            logger.warning(f"Could not map {missing_eans} EANs to product_name from product_df")
    else:
        # Only fill NaN values if column exists and overwrite_existing is False
        mask = df[product_name_col].isna()
        df.loc[mask, product_name_col] = df.loc[mask, ean_col].map(ean_to_name)
        missing_eans = df[product_name_col].isna().sum()
        if missing_eans > 0:
            logger.warning(f"Could not map {missing_eans} EANs to product_name from product_df")
    
    return df

def prepare_prediction_dataframe(
    history_df: pd.DataFrame,
    product_df: pd.DataFrame,
    machines_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Prepares a dataframe for prediction by creating rows for ALL (machine_key, ean) combinations.
    
    Logic:
    1. Identify the latest week in history.
    2. Create cartesian product of all machines × all products from product_df.
    3. Fill metadata from product_df and history.
    4. Create future rows for the target week (latest + 1).
    5. Append to history_df to allow feature engineering (lags) to work.
    
    Args:
        history_df: Historical sales/snapshot data.
        product_df: Product dataframe with ean and product metadata (category, subcategory, etc.)
        machines_df: Optional machines dataframe. If not provided, uses machines from history.
        lookback_window: Weeks to look back for machine metadata (unused now, kept for compatibility).
        
    Returns:
        DataFrame containing history + 1 future week of placeholder rows for ALL combinations.
    """
    df = history_df.copy()
    
    # Ensure week_start is datetime
    if 'week_start' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['week_start']):
        df['week_start'] = pd.to_datetime(df['week_start'])
        
    # 1. Determine timing
    max_date = df['week_start'].max()
    target_date = max_date + timedelta(weeks=1)
    
    logger.info(f"Preparing prediction data for target week: {target_date.date()} (derived from max history: {max_date.date()})")

    # 2. Get all unique machines (from machines_df if provided, otherwise from history)
    if 'machine_key' in machines_df.columns:
        all_machines = machines_df['machine_key'].unique()
        logger.info(f"Using {len(all_machines)} machines from machines_df")
    else:
        all_machines = df['machine_key'].unique()
        logger.info(f"Using {len(all_machines)} machines from history")
    
    # 3. Get all unique EANs from product_df
    if 'ean' not in product_df.columns:
        logger.error("product_df must have 'ean' column")
        return df, target_date
    
    all_eans = product_df['ean'].dropna().unique()
    logger.info(f"Using {len(all_eans)} products from product_df")
    
    # 4. Create cartesian product of all machines × all products
    all_combinations = pd.DataFrame(
        list(itertools.product(all_machines, all_eans)),
        columns=['machine_key', 'ean']
    )
    logger.info(f"Created {len(all_combinations):,} (machine, product) combinations")
    
    # 5. Get latest machine metadata from history
    latest_machine_state = (
        df.sort_values('week_start')
        .groupby('machine_key')
        .tail(1)
    )
    
    # Select machine-level metadata columns (if they exist)
    machine_metadata_cols = ['machine_key', 'machine_eva_group', 'machine_sub_group', 'refiller', 'customer_id']
    available_machine_cols = [col for col in machine_metadata_cols if col in latest_machine_state.columns]
    
    if available_machine_cols:
        machine_metadata = latest_machine_state[available_machine_cols].drop_duplicates('machine_key')
        # Merge machine metadata
        future_df = all_combinations.merge(machine_metadata, on='machine_key', how='left')
    else:
        future_df = all_combinations.copy()
    
    # 6. Get product metadata from product_df
    product_metadata_cols = ['ean', 'product_name', 'category', 'subcategory', 'provider', 'purchase_price_kr']
    available_product_cols = [col for col in product_metadata_cols if col in product_df.columns]
    
    if available_product_cols:
        product_metadata = product_df[available_product_cols].drop_duplicates('ean')
        # Merge product metadata
        future_df = future_df.merge(product_metadata, on='ean', how='left')
    
    # 7. Try to get price_mean from history for products that have it (for better accuracy)
    # Get latest price for each (machine, ean) combination from history
    latest_prices = (
        df.sort_values('week_start')
        .groupby(['machine_key', 'ean'])
        .tail(1)
        [['machine_key', 'ean', 'price_mean']]
        .drop_duplicates(['machine_key', 'ean'])
    )
    
    if not latest_prices.empty:
        future_df = future_df.merge(latest_prices, on=['machine_key', 'ean'], how='left', suffixes=('', '_from_history'))
        # Use price from history if available, otherwise keep NaN (will be handled by feature engineering)
        if 'price_mean_from_history' in future_df.columns:
            future_df['price_mean'] = future_df['price_mean_from_history'].fillna(future_df.get('price_mean', np.nan))
            future_df = future_df.drop(columns=['price_mean_from_history'])
    
    # 7a. Set confidence_score: Normalized score (0.0-1.0) representing sales history representativeness
    # Combines temporal span (how long product has been sold) and spatial coverage (number of machines)
    # Uses full history (no time restriction) to determine overall representativeness

    # Initialize score to 0.0
    future_df['confidence_score'] = 0.0

    if not df.empty and 'weekly_sales' in df.columns:
        # Use full history to calculate metrics
        sales_history = df[df['weekly_sales'] > 0].copy()
        
        if not sales_history.empty:
            # Calculate product-level metrics (across all machines in full history)
            # 1. Number of unique machines that have sold this product
            product_machine_count = (
                sales_history.groupby('ean')['machine_key']
                .nunique()
                .reset_index()
                .rename(columns={'machine_key': 'machines_with_sales'})
            )
            
            # 2. Temporal span: duration over which product has been sold (first to last week)
            product_temporal_span = (
                sales_history.groupby('ean')['week_start']
                .agg(['min', 'max'])
                .reset_index()
            )
            product_temporal_span['temporal_span_weeks'] = (
                (product_temporal_span['max'] - product_temporal_span['min']).dt.days / 7
            )
            product_temporal_span = product_temporal_span[['ean', 'temporal_span_weeks']]
            
            # Merge product metrics
            product_metrics = product_machine_count.merge(
                product_temporal_span, on='ean', how='outer'
            ).fillna(0)
            
            # Merge product-level metrics to future_df
            future_df = future_df.merge(
                product_metrics,
                on='ean',
                how='left'
            ).fillna(0)
            
            # Calculate normalized scores
            # 1. Product spatial score: machines with sales / normalization factor
            # Products sold in 10+ machines get score of 1.0
            normalization_factor_machines = 10
            future_df['product_spatial_score'] = (
                future_df['machines_with_sales'] / normalization_factor_machines
            ).clip(0, 1)
            
            # 2. Product temporal score: temporal span normalized to fixed max
            # Products sold over 52+ weeks (1 year) get score of 1.0
            # This rewards products with longer sales history
            normalization_factor_time = 52
            future_df['product_temporal_score'] = (
                future_df['temporal_span_weeks'] / normalization_factor_time
            ).clip(0, 1)
            
            # Combine scores: 50% spatial (machines), 50% temporal (duration)
            future_df['confidence_score'] = (
                0.5 * future_df['product_spatial_score'] +
                0.5 * future_df['product_temporal_score']
            ).clip(0, 1)
            
            # Clean up temporary columns
            cols_to_drop = [
                'machines_with_sales', 'temporal_span_weeks',
                'product_spatial_score', 'product_temporal_score'
            ]
            future_df = future_df.drop(columns=[col for col in cols_to_drop if col in future_df.columns])

    # 7b. For remaining missing price_mean values, use average price_mean for that EAN 
    # within the specific machine_sub_group (prices vary by machine subcategory)
    missing_price_mask = future_df['price_mean'].isna()
    
    if missing_price_mask.any() and 'machine_sub_group' in future_df.columns:
        # Calculate average price_mean by (machine_sub_group, ean) from history
        if 'machine_sub_group' in df.columns and 'price_mean' in df.columns:
            avg_prices_by_subgroup = (
                df[df['price_mean'].notna()]
                .groupby(['machine_sub_group', 'ean'])['price_mean']
                .mean()
                .reset_index()
                .rename(columns={'price_mean': 'price_mean_avg'})
            )
            
            if not avg_prices_by_subgroup.empty:
                # Merge average prices for missing values
                future_df = future_df.merge(
                    avg_prices_by_subgroup,
                    on=['machine_sub_group', 'ean'],
                    how='left'
                )
                # Fill missing price_mean with the average for that subgroup-ean combo
                future_df.loc[missing_price_mask, 'price_mean'] = future_df.loc[missing_price_mask, 'price_mean_avg']
                # Drop the temporary column
                if 'price_mean_avg' in future_df.columns:
                    future_df = future_df.drop(columns=['price_mean_avg'])

    
    # 8. Set target date and reset sales columns
    future_df['week_start'] = target_date
    future_df['weekly_sales'] = 0
    
    # Reset other sales/target columns
    cols_to_reset = ['quantity', 'sales', 'local_timestamp_count']
    for col in cols_to_reset:
        if col in future_df.columns:
            future_df[col] = 0
    
    # Ensure time-derived columns are updated
    if 'year' in future_df.columns:
        future_df['year'] = target_date.isocalendar().year
    if 'week' in future_df.columns:
        future_df['week'] = target_date.isocalendar().week
    
    # Set position to a default value (since we're not tracking position anymore)
    if 'position' not in future_df.columns:
        future_df['position'] = None
    
    logger.info(f"Created {len(future_df):,} future prediction rows for {target_date.date()}")
        
    # 9. Concatenate
    # We append the future rows to the history so that feature engineering can calculate lags
    # e.g. lag_1 for the future row will grab the values from 'max_date'
    combined_df = pd.concat([df, future_df], ignore_index=True)
    
    return combined_df, target_date

def moaaz_predict(
    sales_df: pd.DataFrame,
    machines_df: pd.DataFrame,
    product_df: pd.DataFrame,
    model_path: Optional[Union[str, Path]] = None,
    lookback_weeks: Optional[int] = 52
) -> pd.DataFrame:
    """
    Predict sales using historical data from sales_df.
    
    Generates predictions for ALL (machine_key, ean) combinations from product_df.
    
    Args:
        sales_df: Historical sales dataframe
        machines_df: Machines dataframe (used for filtering/consistency and getting machine list)
        product_df: Product dataframe with all EANs to predict for
        model_path: Path to model artifact
        lookback_weeks: Number of weeks of history to use
        
    Returns:
        DataFrame with predictions for the next week for all (machine_key, ean) combinations.
    """
    if model_path is None:
        model_path = MODEL_DIR

    # Make a defensive copy
    machines_df = machines_df.copy()
    if machines_df.empty:
        logger.warning("Empty machines dataframe provided to moaaz_predict")
        return pd.DataFrame()
        
    if sales_df.empty:
        logger.warning("Empty input sales dataframe provided to moaaz_predict")
        return pd.DataFrame()
    
    df_history = sales_df.copy()

    # 1. Load Model
    try:
        forecaster = MultiWeekForecaster.load(Path(model_path))
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        return pd.DataFrame()

    # Filter history to machines in machines_df
    if 'machine_key' in machines_df.columns and 'machine_key' in df_history.columns:
        valid_machine_keys = machines_df['machine_key'].unique()
        df_history = df_history[df_history['machine_key'].isin(valid_machine_keys)].copy()
    else: 
        logger.warning("No machine_key column found. Unable to filter history.")
        return pd.DataFrame()
    
    if df_history.empty:
        logger.warning("No history data available after filtering.")
        return pd.DataFrame()

    # Optimization: Filter to recent history
    if lookback_weeks is not None and 'local_timestamp' in df_history.columns:
        if not pd.api.types.is_datetime64_any_dtype(df_history['local_timestamp']):
            df_history['local_timestamp'] = pd.to_datetime(df_history['local_timestamp'], errors='coerce')
        
        max_date = df_history['local_timestamp'].max()
        cutoff_date = max_date - timedelta(weeks=lookback_weeks)
        
        logger.info(f"Filtering raw sales data: {cutoff_date} onwards")
        df_history = df_history[df_history['local_timestamp'] >= cutoff_date].copy()
    
    # 2. Process Data 
    df_processed = process_sales_data(
        df_history,
        use_cache=False,
        show_progress=False # Reduce noise
    )

    # 3. Prepare Prediction Data (Add Future Rows)
    # This is the critical step: we create the rows for "Next Week" for ALL (machine, product) combinations
    df_combined, target_date = prepare_prediction_dataframe(
        df_processed,
        product_df=product_df,
        machines_df=machines_df
    )

    # 4. Generate Features
    # We generate features on the *combined* dataframe so the future rows get correct lags
    df_features = create_all_features(df_combined, use_cold_start_fallback=True)

    # 5. Slice to Target Week
    # We only want to predict for the rows we just added (the future)
    df_predict = df_features[df_features['week_start'] == target_date].copy()
    
    if df_predict.empty:
        logger.warning(f"No rows generated for target date {target_date}")
        return pd.DataFrame()

    logger.info(f"Generating predictions for {len(df_predict)} items for week of {target_date.date()}")

    # 6. Predict
    preds = forecaster.predict(df_predict)
    
    # 7. Format Output
    # Attach predictions back to the item identifiers
    # MultiWeekForecaster returns a DataFrame with 'pred_week_1', etc.
    # We want to join this with our identifiers
    
    # The forecaster.predict returns a dataframe with same index or separate?
    # Looking at MultiWeekForecaster code: it returns a DataFrame with columns like 'pred_week_1', 'actual_week_1'
    # It uses _prepare_data internally but returns results aligned with input?
    # Wait, forecaster.predict takes df, prepares X, predicts.
    # If we pass the full df_predict, the index should align if we are careful.
    
    # To be safe, we join the predictions back to df_predict metadata
    result_cols = ['machine_key', 'ean', 'week_start', 'confidence_score']
    results = df_predict[result_cols].reset_index(drop=True)
    
    # Verify lengths align
    if len(preds) != len(results):
         logger.error(f"Prediction length mismatch: {len(preds)} vs {len(results)}")
         return pd.DataFrame()
    
    # Filter out actual_week_* columns (only keep pred_week_* columns)
    pred_cols = [col for col in preds.columns if col.startswith('pred_week_')]
    preds_filtered = preds[pred_cols].reset_index(drop=True)
         
    # Combine
    results = pd.concat([results, preds_filtered], axis=1)
    
    return results
