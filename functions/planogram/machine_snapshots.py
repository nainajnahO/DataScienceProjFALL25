import pandas as pd
import numpy as np
from typing import List, Optional, Union, Dict, Any, Tuple
from datetime import datetime, timedelta

from .config import (
    SNAPSHOT_N_DAYS,
    SNAPSHOT_DATE_INTERVAL_DAYS,
    SNAPSHOT_MIN_SALES
)

def get_available_machines(
    sales_data_df: pd.DataFrame,
    min_sales: int = SNAPSHOT_MIN_SALES
) -> List[int]:
    """
    Get list of machine IDs that have sufficient sales data.
    
    Args:
        sales_data_df: Sales data DataFrame
        min_sales: Minimum number of sales records required
    
    Returns:
        List of machine IDs
    """
    if sales_data_df.empty:
        return []
        
    machine_counts = sales_data_df['machine_id'].value_counts()
    valid_machines = machine_counts[machine_counts >= min_sales].index.tolist()
    return sorted(valid_machines)


def generate_snapshot(
    sales_data_df: pd.DataFrame, 
    machine_id: Union[int, str], 
    snapshot_date: str, 
    n_days: int = SNAPSHOT_N_DAYS
) -> pd.DataFrame:
    """
    Generates a single point-in-time snapshot for a specific machine.
    Shows which products were in which positions and their sales in the lookback period.
    
    Args:
        sales_data_df: DataFrame containing sales history
        machine_id: The ID of the machine to snapshot
        snapshot_date: The date of the snapshot (YYYY-MM-DD)
        n_days: Number of days to look back for sales counts
        
    Returns:
        DataFrame containing the snapshot (position, product, sales count, etc.)
    """
    # Ensure machine_id types match
    machine_id_val = int(machine_id) if isinstance(machine_id, str) and str(machine_id).isdigit() else machine_id
    
    # Filter on local_timestamp between YYYY_MM_DD and YYYY_MM_DD - n_days
    # Ensure local_timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(sales_data_df['local_timestamp']):
        sales_data_df = sales_data_df.copy()
        sales_data_df['local_timestamp'] = pd.to_datetime(sales_data_df['local_timestamp'])
    
    # Convert snapshot_date to datetime
    end_date = pd.to_datetime(snapshot_date)
    start_date = end_date - pd.Timedelta(days=n_days)
    
    # If local_timestamp is timezone-aware, make comparison dates timezone-aware too
    if hasattr(sales_data_df['local_timestamp'].dtype, 'tz') and sales_data_df['local_timestamp'].dtype.tz is not None:
        # Get the timezone from the data
        tz = sales_data_df['local_timestamp'].dtype.tz
        # Localize naive timestamps to the same timezone
        if end_date.tzinfo is None:
            end_date = end_date.tz_localize(tz)
        if start_date.tzinfo is None:
            start_date = start_date.tz_localize(tz)
    
    # Filter by date range and machine_id
    data_df = sales_data_df[
        (sales_data_df['local_timestamp'] >= start_date) & 
        (sales_data_df['local_timestamp'] <= end_date) &
        (sales_data_df['machine_id'] == machine_id_val)
    ].copy()
    
    if data_df.empty:
        return pd.DataFrame()

    # Group by machine_id and position to find the last product in that position
    # This assumes the 'last' product in the period is the one currently in the slot
    snapshot_df = data_df.groupby(['machine_id', 'position']).agg({
        'product_name': 'last',
        'provider': 'last',
        'subcategory': 'last',
    }).reset_index()
    
    # Count sales per position (add n_sales column)
    sales_counts = data_df.groupby(['machine_id', 'position']).size().reset_index(name='n_sales')
    snapshot_df = snapshot_df.merge(sales_counts, on=['machine_id', 'position'], how='left')
    
    # Add snapshot date metadata
    snapshot_df['snapshot_date'] = snapshot_date
    
    return snapshot_df


def train_snapshot_model(
    sales_df: pd.DataFrame,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    snapshot_dates: Optional[List[str]] = None,
    machine_ids: Optional[List[Union[int, str]]] = None,
    date_interval_days: int = SNAPSHOT_DATE_INTERVAL_DAYS,
    n_days: int = SNAPSHOT_N_DAYS,
    min_sales: int = SNAPSHOT_MIN_SALES
) -> pd.DataFrame:
    """
    Batch generates snapshots for multiple machines and dates.
    Follows the 'Train' pattern where the output is a comprehensive artifact (DataFrame of snapshots).
    
    Args:
        sales_df: Raw sales DataFrame
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        snapshot_dates: specific list of dates (overrides start/end/interval if provided)
        machine_ids: List of machine IDs to process. If None, auto-discovers based on sales.
        date_interval_days: Interval between snapshots if using start/end range
        n_days: Lookback period for each snapshot
        min_sales: Minimum sales threshold for auto-discovery of machines
        
    Returns:
        pd.DataFrame: Combined snapshots for all requested machines and dates
    """
    # 1. Determine Dates
    dates_to_process = []
    if snapshot_dates:
        dates_to_process = snapshot_dates
    elif start_date and end_date:
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        current = start
        while current <= end:
            dates_to_process.append(current.strftime('%Y-%m-%d'))
            current += timedelta(days=date_interval_days)
    else:
        raise ValueError("Must provide either snapshot_dates list OR start_date and end_date")

    # 2. Determine Machines
    if machine_ids is None:
        machine_ids = get_available_machines(sales_df, min_sales=min_sales)
    
    if not machine_ids:
        return pd.DataFrame()

    # 3. Generate Snapshots
    all_snapshots = []
    
    # Pre-convert timestamp column once for efficiency if needed
    if not pd.api.types.is_datetime64_any_dtype(sales_df['local_timestamp']):
        sales_df = sales_df.copy()
        sales_df['local_timestamp'] = pd.to_datetime(sales_df['local_timestamp'])

    for date in dates_to_process:
        for machine_id in machine_ids:
            try:
                snapshot = generate_snapshot(
                    sales_df,
                    machine_id=machine_id,
                    snapshot_date=date,
                    n_days=n_days
                )
                if not snapshot.empty:
                    all_snapshots.append(snapshot)
            except Exception:
                # Silently skip failed snapshots in batch processing to avoid partial failure
                continue
                
    if not all_snapshots:
        return pd.DataFrame()
        
    return pd.concat(all_snapshots, ignore_index=True)


def predict_snapshot_metrics(
    snapshot_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Analyzes the snapshot artifact to produce summary metrics.
    Follows the 'Predict' pattern where the input is the trained model artifact.
    
    Args:
        snapshot_df: The DataFrame resulting from train_snapshot_model
        
    Returns:
        pd.DataFrame: Summary statistics per machine/date
    """
    snapshot_df = snapshot_df.copy()
    if snapshot_df.empty:
        return pd.DataFrame()
        
    analysis = []
    
    # Group by machine and snapshot_date
    # Ensure snapshot_date exists (it should if coming from generate_snapshot)
    group_cols = ['machine_id']
    if 'snapshot_date' in snapshot_df.columns:
        group_cols.append('snapshot_date')
    
    for name, group in snapshot_df.groupby(group_cols):
        machine_id = name[0] if isinstance(name, tuple) else name
        snapshot_date = name[1] if isinstance(name, tuple) and len(name) > 1 else None
        
        # Handle potential missing columns safely
        subcategory_col = group['subcategory'] if 'subcategory' in group.columns else pd.Series(dtype='object')
        provider_col = group['provider'] if 'provider' in group.columns else pd.Series(dtype='object')
        product_name_col = group['product_name'] if 'product_name' in group.columns else pd.Series(dtype='object')
        sales_col = group['n_sales'] if 'n_sales' in group.columns else pd.Series(dtype='float')
        position_col = group['position'] if 'position' in group.columns else pd.Series(dtype='object')
        
        analysis_record = {
            'machine_id': machine_id,
            'snapshot_date': snapshot_date,
            'n_positions': len(group),
            'total_sales': sales_col.sum(),
            'avg_sales_per_position': sales_col.mean(),
            'max_sales_position': None,
            'min_sales_position': None,
            'n_unique_products': product_name_col.nunique(),
            'n_unique_providers': provider_col.nunique(),
            'n_unique_subcategories': subcategory_col.nunique(),
            'top_subcategory': None,
        }

        if len(group) > 0:
             if not sales_col.empty:
                try:
                    analysis_record['max_sales_position'] = position_col.loc[sales_col.idxmax()]
                    analysis_record['min_sales_position'] = position_col.loc[sales_col.idxmin()]
                except (ValueError, KeyError):
                     pass # Handle cases where idxmax might fail on empty/all-NA series
                     
             if not subcategory_col.empty:
                 mode_result = subcategory_col.mode()
                 if not mode_result.empty:
                     analysis_record['top_subcategory'] = mode_result[0]

        analysis.append(analysis_record)
    
    return pd.DataFrame(analysis)

