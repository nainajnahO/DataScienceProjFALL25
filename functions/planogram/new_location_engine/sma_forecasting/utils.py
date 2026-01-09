"""
Utility functions for data preparation and visualization for SMA forecasting.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path
import sys


def load_sales_data(file_paths: List[str]) -> pd.DataFrame:
    """
    Load sales data from one or more parquet files.

    Args:
        file_paths: List of paths to parquet files

    Returns:
        Combined sales data as a DataFrame
    """
    dataframes = []
    for file_path in file_paths:
        df = pd.read_parquet(file_path)
        dataframes.append(df)

    sales_data = pd.concat(dataframes, ignore_index=True)
    return sales_data


def prepare_weekly_data(sales_data: pd.DataFrame,
                       product: Optional[str] = None,
                       machine: Optional[int] = None,
                       timestamp_col: str = 'local_timestamp',
                       product_col: str = 'product_name',
                       machine_col: str = 'machine_id') -> pd.DataFrame:
    """
    Convert sales data to weekly aggregates.

    Args:
        sales_data: Raw sales data DataFrame
        product: Product name to filter by (optional)
        machine: Machine ID to filter by (optional)
        timestamp_col: Name of the timestamp column
        product_col: Name of the product column
        machine_col: Name of the machine column

    Returns:
        DataFrame with weekly aggregated sales data
    """
    df = sales_data.copy()

    # Convert to date
    df['date'] = pd.to_datetime(df[timestamp_col]).dt.date
    df['date'] = pd.to_datetime(df['date'])

    # Filter by product if specified
    if product is not None:
        df = df[df[product_col] == product]

    # Calculate week start (Monday)
    df['week_start'] = df['date'] - pd.to_timedelta(df['date'].dt.dayofweek, unit='D')

    # Aggregate by week and machine
    group_cols = ['week_start']
    if machine_col in df.columns:
        group_cols.append(machine_col)

    weekly = df.groupby(group_cols).size().reset_index(name='quantity')

    # Filter by machine if specified
    if machine is not None and machine_col in weekly.columns:
        weekly = weekly[weekly[machine_col] == machine]

    weekly = weekly.sort_values('week_start').reset_index(drop=True)

    return weekly


def get_quantity_series(weekly_data: pd.DataFrame,
                       quantity_col: str = 'quantity') -> pd.Series:
    """
    Extract quantity series from weekly data.

    Args:
        weekly_data: Weekly aggregated data
        quantity_col: Name of the quantity column

    Returns:
        Quantity values as a pandas Series
    """
    return weekly_data[quantity_col].reset_index(drop=True)


def plot_forecast(historical_data: pd.Series,
                 forecast_values: List[float],
                 n_weeks_to_show: int = 12,
                 title: str = "Progressive Window SMA Forecast",
                 figsize: Tuple[int, int] = (14, 6)) -> None:
    """
    Visualize historical data and forecasts.

    Args:
        historical_data: Historical quantity values
        forecast_values: Forecasted values
        n_weeks_to_show: Number of historical weeks to display
        title: Plot title
        figsize: Figure size as (width, height)
    """
    plot_data = historical_data.tail(n_weeks_to_show)
    forecast_weeks = len(forecast_values)

    plt.figure(figsize=figsize)
    plt.plot(range(len(plot_data)), plot_data.values, marker='o',
             label='Historical', linewidth=2)
    plt.plot(range(len(plot_data), len(plot_data) + forecast_weeks),
             forecast_values, marker='s', label='Forecast',
             linewidth=2, linestyle='--')
    plt.axvline(x=len(plot_data)-0.5, color='red', linestyle=':',
                linewidth=1.5, label='Forecast Start')

    plt.xlabel('Week')
    plt.ylabel('Quantity Sold')
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_backtest_results(backtest_results: dict,
                         title: str = "Backtest Results",
                         figsize: Tuple[int, int] = (14, 5)) -> None:
    """
    Visualize backtesting results.

    Args:
        backtest_results: Results dictionary from ProgressiveWindowSMA.backtest()
        title: Overall plot title
        figsize: Figure size as (width, height)
    """
    predictions = backtest_results['predictions']
    metrics = backtest_results['metrics']
    n_weeks = len(predictions)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Forecast vs Actual
    ax1.plot(range(n_weeks), predictions['actual'], marker='o',
             label='Actual', linewidth=2)
    ax1.plot(range(n_weeks), predictions['forecast'], marker='s',
             label='Forecast', linewidth=2, linestyle='--')
    ax1.set_xlabel('Week')
    ax1.set_ylabel('Quantity')
    ax1.set_title(f'Forecast vs Actual\nMAE: {metrics["MAE"]:.2f}, RMSE: {metrics["RMSE"]:.2f}')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Errors
    colors = ['red' if e < 0 else 'green' for e in predictions['error']]
    ax2.bar(range(n_weeks), predictions['error'], color=colors, alpha=0.6)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_xlabel('Week')
    ax2.set_ylabel('Error (Actual - Forecast)')
    ax2.set_title('Forecast Errors')
    ax2.grid(alpha=0.3)

    fig.suptitle(title, y=1.02, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def prepare_product_forecast_data(sales_files: List[str],
                                  product: str,
                                  machine: int) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Complete pipeline to load and prepare data for forecasting a specific product-machine pair.

    Args:
        sales_files: List of paths to sales data parquet files
        product: Product name
        machine: Machine ID

    Returns:
        Tuple of (weekly_dataframe, quantity_series)
    """
    # Load data
    sales_data = load_sales_data(sales_files)

    # Prepare weekly data
    weekly_data = prepare_weekly_data(sales_data, product=product, machine=machine)

    # Get quantity series
    quantity_series = get_quantity_series(weekly_data)

    return weekly_data, quantity_series


def print_data_summary(weekly_data: pd.DataFrame,
                      product: str,
                      machine: int,
                      n_tail: int = 5) -> None:
    """
    Print a summary of the prepared data.

    Args:
        weekly_data: Weekly aggregated data
        product: Product name
        machine: Machine ID
        n_tail: Number of recent weeks to display
    """
    print(f"Product: {product}")
    print(f"Machine: {machine}")
    print(f"Weeks of data: {len(weekly_data)}")
    print(f"\nLast {n_tail} weeks:")
    print(weekly_data.tail(n_tail))


# ============================================================================
# Fallback Engine Integration Functions
# ============================================================================


def prepare_weekly_data_from_fallback_result(
    fallback_data: pd.DataFrame,
    timestamp_col: str = 'local_timestamp'
) -> pd.DataFrame:
    """
    Convert fallback result DataFrame to weekly aggregates.

    This function is similar to prepare_weekly_data() but works specifically
    on already-filtered fallback engine results.

    Args:
        fallback_data: Raw transaction data from fallback engine
        timestamp_col: Name of the timestamp column

    Returns:
        DataFrame with week_start and quantity columns
    """
    if len(fallback_data) == 0:
        return pd.DataFrame(columns=['week_start', 'quantity'])

    df = fallback_data.copy()

    # Convert to date
    df['date'] = pd.to_datetime(df[timestamp_col]).dt.date
    df['date'] = pd.to_datetime(df['date'])

    # Calculate week start (Monday)
    df['week_start'] = df['date'] - pd.to_timedelta(df['date'].dt.dayofweek, unit='D')

    # Aggregate by week
    weekly = df.groupby('week_start').size().reset_index(name='quantity')
    weekly = weekly.sort_values('week_start').reset_index(drop=True)

    return weekly


def initialize_fallback_engine(
    sales_df: pd.DataFrame,
    strategy: str = 'product_first',
    **engine_kwargs
) -> Any:
    """
    Helper to initialize a FallbackEngine with common patterns.

    Args:
        sales_df: Sales DataFrame with required columns
        strategy: Strategy name - 'product_first', 'machine_first', or 'location_affinity'
        **engine_kwargs: Additional arguments for FallbackEngine constructor
                        (e.g., min_samples, use_adaptive_thresholds, custom_thresholds)

    Returns:
        Initialized FallbackEngine instance

    Raises:
        ImportError: If FallbackAlgorithm package not available
        ValueError: If strategy name is invalid

    Example:
        >>> engine = initialize_fallback_engine(sales_df, strategy='product_first')
        >>> result = engine.execute_fallback(machine_id=123, product_name='Twix')
    """
    # Add parent directory to path for imports
    parent_dir = Path(__file__).parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))

    try:
        from FallbackAlgorithm import FallbackEngine, get_strategy
    except ImportError:
        raise ImportError(
            "FallbackAlgorithm package not found. "
            "Please ensure it's in your Python path or installed."
        )

    # Convert strategy name to list
    try:
        strategy_list = get_strategy(strategy)
    except (KeyError, ValueError) as e:
        raise ValueError(f"Invalid strategy '{strategy}': {e}")

    # Initialize and return engine
    return FallbackEngine(sales_df, strategy_list, **engine_kwargs)


def check_data_sufficiency(
    data: pd.DataFrame,
    min_transactions: int = 15,
    min_weeks: int = 4
) -> Dict[str, Any]:
    """
    Check if data meets minimum requirements for forecasting.

    Args:
        data: Transaction or weekly data DataFrame
        min_transactions: Minimum number of transactions required
        min_weeks: Minimum number of unique weeks required

    Returns:
        Dictionary with:
            - sufficient: Boolean indicating if data meets both criteria
            - num_transactions: Total number of records
            - num_weeks: Number of unique weeks (if timestamp available)
            - reason: Explanation if insufficient

    Example:
        >>> check = check_data_sufficiency(data, min_transactions=15, min_weeks=4)
        >>> if check['sufficient']:
        ...     print("Data is sufficient for forecasting")
        ... else:
        ...     print(f"Insufficient: {check['reason']}")
    """
    num_transactions = len(data)
    num_weeks = None
    reasons = []

    # Check transaction count
    if num_transactions < min_transactions:
        reasons.append(f"Only {num_transactions} transactions (need {min_transactions})")

    # Check unique weeks if timestamp column exists
    timestamp_cols = ['local_timestamp', 'timestamp', 'date', 'week_start']
    timestamp_col = next((col for col in timestamp_cols if col in data.columns), None)

    if timestamp_col:
        if timestamp_col == 'week_start':
            num_weeks = data['week_start'].nunique()
        else:
            dates = pd.to_datetime(data[timestamp_col]).dt.date
            weeks = pd.to_datetime(dates) - pd.to_timedelta(
                pd.to_datetime(dates).dt.dayofweek, unit='D'
            )
            num_weeks = weeks.nunique()

        if num_weeks < min_weeks:
            reasons.append(f"Only {num_weeks} weeks (need {min_weeks})")

    sufficient = len(reasons) == 0

    return {
        'sufficient': sufficient,
        'num_transactions': num_transactions,
        'num_weeks': num_weeks,
        'reason': '; '.join(reasons) if reasons else 'Data is sufficient'
    }


def plot_forecast_with_metadata(
    historical_data: pd.Series,
    forecast_values: List[float],
    metadata: Dict[str, Any],
    n_weeks_to_show: int = 12,
    figsize: Tuple[int, int] = (14, 6)
) -> None:
    """
    Visualize forecast with metadata information in the title.

    Args:
        historical_data: Historical quantity values
        forecast_values: Forecasted values
        metadata: Metadata dictionary from SMAForecaster
        n_weeks_to_show: Number of historical weeks to display
        figsize: Figure size as (width, height)

    Example:
        >>> result = forecaster.forecast(machine_id=123, product_name='Twix')
        >>> if result['success']:
        ...     plot_forecast_with_metadata(
        ...         historical_data=qty_series,
        ...         forecast_values=result['forecast']['forecast'].tolist(),
        ...         metadata=result['metadata']
        ...     )
    """
    plot_data = historical_data.tail(n_weeks_to_show)
    forecast_weeks = len(forecast_values)

    # Build title with metadata
    data_source = metadata.get('data_source', 'unknown')
    title = f"Progressive Window SMA Forecast\nData Source: {data_source}"

    if data_source == 'fallback' and 'fallback_info' in metadata:
        fb_info = metadata['fallback_info']
        title += f" (Level {fb_info['level']}, Confidence: {fb_info['confidence']:.1%})"

    plt.figure(figsize=figsize)
    plt.plot(range(len(plot_data)), plot_data.values, marker='o',
             label='Historical', linewidth=2)
    plt.plot(range(len(plot_data), len(plot_data) + forecast_weeks),
             forecast_values, marker='s', label='Forecast',
             linewidth=2, linestyle='--')
    plt.axvline(x=len(plot_data)-0.5, color='red', linestyle=':',
                linewidth=1.5, label='Forecast Start')

    plt.xlabel('Week')
    plt.ylabel('Quantity Sold')
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
