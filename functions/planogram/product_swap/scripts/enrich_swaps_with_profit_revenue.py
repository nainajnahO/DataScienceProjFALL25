"""
Enrich Product Swaps with Profit/Revenue Analysis
==================================================

For each product swap, calculate profit and revenue for:
- 10 weeks BEFORE the swap (for the old product)
- 10 weeks AFTER the swap (for the new product)

This enables analysis of the financial impact of product swaps.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta
from typing import Optional
from tqdm import tqdm


def load_sales_data(date_start: Optional[pd.Timestamp] = None, date_end: Optional[pd.Timestamp] = None, data_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load sales data from parquet file(s) for the specified date range.
    
    Args:
        date_start: Start date for filtering sales data (inclusive)
        date_end: End date for filtering sales data (inclusive)
        data_path: Path to sales data file (default: loads from data/sales/)
    
    Returns:
        DataFrame with sales data filtered to the specified date range
    """
    if data_path is None:
        # Load from individual year files in data/sales/
        script_dir = Path(__file__).parent  # scripts/
        project_root = script_dir.parent     # product_swap/
        sales_dir = project_root / "data" / "sales"
        
        # Determine which year files to load based on date range
        sales_files = list(sales_dir.glob("Sales_*_with_profit.parquet"))
        if sales_files:
            # Filter out files before 2020 (we only process data from 2020 onwards)
            sales_files = [f for f in sales_files if int(f.stem.split('_')[1]) >= 2020]
            
            # If date range specified, only load relevant years
            if date_start is not None:
                if date_end is not None:
                    # Both start and end specified - load only needed years
                    years_needed = set(range(max(date_start.year, 2020), date_end.year + 1))  # Enforce 2020 minimum
                    sales_files = [f for f in sales_files if int(f.stem.split('_')[1]) in years_needed]
                else:
                    # Only start specified - load all years from start year onwards
                    min_year = max(date_start.year, 2020)
                    sales_files = [f for f in sales_files if int(f.stem.split('_')[1]) >= min_year]
            
            if sales_files:
                print(f"Loading sales data from {len(sales_files)} year file(s) in {sales_dir}...")
                dfs = []
                for sf in sorted(sales_files):
                    df_year = pd.read_parquet(sf)
                    dfs.append(df_year)
                df = pd.concat(dfs, ignore_index=True)
                print(f"✓ Combined {len(sales_files)} year file(s)")
            else:
                raise FileNotFoundError(f"No sales files found for the required date range in {sales_dir}")
        else:
            # Fallback to old location for backwards compatibility
            sales_file = project_root / "data" / "all_sales_with_profit.parquet"
            if not sales_file.exists():
                raise FileNotFoundError(f"Sales data not found. Expected files in {sales_dir} or {sales_file}")
            print(f"Loading sales data from {sales_file}...")
            df = pd.read_parquet(sales_file)
    else:
        sales_file = data_path
        if not sales_file.exists():
            raise FileNotFoundError(f"Sales data not found: {sales_file}")
        print(f"Loading sales data from {sales_file}...")
        df = pd.read_parquet(sales_file)
    
    # Ensure datetime
    if not pd.api.types.is_datetime64_any_dtype(df['local_timestamp']):
        df['local_timestamp'] = pd.to_datetime(df['local_timestamp'])
    
    # Enforce minimum date of 2020-01-01 (we only process data from 2020 onwards)
    min_sales_date = pd.Timestamp('2020-01-01')
    
    # Check if sales data is timezone-aware and prepare min_sales_date accordingly
    is_tz_aware = hasattr(df['local_timestamp'].dtype, 'tz') and df['local_timestamp'].dtype.tz is not None
    if is_tz_aware:
        tz = df['local_timestamp'].dtype.tz
        if min_sales_date.tzinfo is None:
            min_sales_date = min_sales_date.tz_localize(tz)
    else:
        if min_sales_date.tzinfo is not None:
            min_sales_date = min_sales_date.tz_localize(None)
    
    # Filter to specified date range if provided
    if date_start is not None:
        # Align timezones first
        if is_tz_aware:
            if date_start.tzinfo is None:
                date_start = date_start.tz_localize(tz)
            if date_end is not None and date_end.tzinfo is None:
                date_end = date_end.tz_localize(tz)
        else:
            if date_start.tzinfo is not None:
                date_start = date_start.tz_localize(None)
            if date_end is not None and date_end.tzinfo is not None:
                date_end = date_end.tz_localize(None)
        
        # Enforce 2020 minimum after timezone alignment
        date_start = max(date_start, min_sales_date)
        
        if date_end is not None:
            # Filter to date range
            df = df[(df['local_timestamp'] >= date_start) & (df['local_timestamp'] <= date_end)]
            print(f"  Filtered to date range: {date_start.date()} to {date_end.date()}")
        else:
            # Filter from date_start onwards (no upper limit)
            df = df[df['local_timestamp'] >= date_start]
            print(f"  Filtered from {date_start.date()} onwards: {len(df):,} records")
    else:
        # Even if no date range specified, filter out data before 2020
        df = df[df['local_timestamp'] >= min_sales_date]
        print(f"  Filtered to 2020 onwards: {len(df):,} records")
    
    # Convert machine_id to int to match swaps data
    if df['machine_id'].dtype == 'object':
        df['machine_id'] = pd.to_numeric(df['machine_id'], errors='coerce').astype('Int64')
        print(f"  Converted machine_id to integer type")
    
    print(f"✓ Loaded {len(df):,} sales records")
    print(f"  Date range: {df['local_timestamp'].min()} to {df['local_timestamp'].max()}")
    print(f"  Unique machines: {df['machine_id'].nunique():,}")
    
    return df


def calculate_swap_financials(
    swap_row: pd.Series,
    sales_df: pd.DataFrame,
    weeks_before: int = 4,
    weeks_after: int = 4
) -> dict:
    """
    Calculate profit and revenue for a single swap over specified weeks.
    
    Args:
        swap_row: Row from swaps DataFrame
        sales_df: Sales data
        weeks_before: Number of weeks to look back
        weeks_after: Number of weeks to look forward
    
    Returns:
        Dictionary with financial metrics
    """
    machine_id = swap_row['machine_id']
    position = swap_row['position']
    swap_date = pd.to_datetime(swap_row['swap_date'])
    product_before = swap_row['product_before']
    product_after = swap_row['product_after']
    
    # Convert swap_date to timezone-aware to match sales data
    if swap_date.tzinfo is None:
        # If local_timestamp in sales_df is timezone-aware, match it
        if hasattr(sales_df['local_timestamp'].dtype, 'tz') and sales_df['local_timestamp'].dtype.tz is not None:
            swap_date = swap_date.tz_localize(sales_df['local_timestamp'].dtype.tz)
    
    # Define time periods
    before_start = swap_date - timedelta(weeks=weeks_before)
    before_end = swap_date - timedelta(days=1)  # Exclude swap day
    after_start = swap_date + timedelta(days=1)  # Start day after swap
    after_end = swap_date + timedelta(weeks=weeks_after)
    
    # Filter for this machine-position
    position_sales = sales_df[
        (sales_df['machine_id'] == machine_id) &
        (sales_df['position'] == position)
    ]
    
    # BEFORE period: old product
    before_sales = position_sales[
        (position_sales['local_timestamp'] >= before_start) &
        (position_sales['local_timestamp'] <= before_end) &
        (position_sales['product_name'] == product_before)
    ]
    
    # AFTER period: new product
    after_sales = position_sales[
        (position_sales['local_timestamp'] >= after_start) &
        (position_sales['local_timestamp'] <= after_end) &
        (position_sales['product_name'] == product_after)
    ]
    
    # Calculate metrics using the pre-calculated revenue and profit columns
    # BEFORE metrics
    revenue_before = before_sales['revenue'].sum() if 'revenue' in before_sales.columns else 0
    profit_before = before_sales['profit'].sum() if 'profit' in before_sales.columns else 0
    cost_before = revenue_before - profit_before if revenue_before > 0 else 0
    
    sales_count_before = len(before_sales)
    days_observed_before = (before_end - before_start).days + 1
    
    # AFTER metrics
    revenue_after = after_sales['revenue'].sum() if 'revenue' in after_sales.columns else 0
    profit_after = after_sales['profit'].sum() if 'profit' in after_sales.columns else 0
    cost_after = revenue_after - profit_after if revenue_after > 0 else 0
    
    sales_count_after = len(after_sales)
    days_observed_after = (after_end - after_start).days + 1
    
    # Calculate changes
    revenue_change = revenue_after - revenue_before
    revenue_change_pct = (revenue_change / revenue_before * 100) if revenue_before > 0 else None
    
    profit_change = profit_after - profit_before
    profit_change_pct = (profit_change / profit_before * 100) if profit_before != 0 else None
    
    return {
        # Before metrics (4 weeks)
        'revenue_before_4w': revenue_before,
        'profit_before_4w': profit_before,
        'cost_before_4w': cost_before,
        'sales_count_before_4w': sales_count_before,
        'days_observed_before_4w': days_observed_before,
        'revenue_per_day_before_4w': revenue_before / days_observed_before if days_observed_before > 0 else 0,
        'profit_per_day_before_4w': profit_before / days_observed_before if days_observed_before > 0 else 0,
        
        # After metrics (4 weeks)
        'revenue_after_4w': revenue_after,
        'profit_after_4w': profit_after,
        'cost_after_4w': cost_after,
        'sales_count_after_4w': sales_count_after,
        'days_observed_after_4w': days_observed_after,
        'revenue_per_day_after_4w': revenue_after / days_observed_after if days_observed_after > 0 else 0,
        'profit_per_day_after_4w': profit_after / days_observed_after if days_observed_after > 0 else 0,
        
        # Change metrics
        'revenue_change_4w': revenue_change,
        'revenue_change_pct_4w': revenue_change_pct,
        'profit_change_4w': profit_change,
        'profit_change_pct_4w': profit_change_pct,
        'sales_count_change_4w': sales_count_after - sales_count_before,
    }


def enrich_swaps_with_financials(
    swaps_path: str,
    sales_data_path: Optional[Path] = None,
    output_path: Optional[str] = None,
    weeks_before: int = 4,
    weeks_after: int = 4,
    max_swaps: Optional[int] = None
) -> pd.DataFrame:
    """
    Enrich swap data with profit and revenue analysis.
    
    Args:
        swaps_path: Path to product_swaps.parquet
        sales_data_path: Path to sales data file (optional, auto-detects from data/sales/)
        output_path: Path to save enriched data
        weeks_before: Weeks to analyze before swap
        weeks_after: Weeks to analyze after swap
        max_swaps: Maximum number of swaps to process (for testing)
    
    Returns:
        DataFrame with enriched swap data
    """
    print("="*70)
    print("ENRICHING SWAPS WITH PROFIT/REVENUE DATA")
    print("="*70)
    
    # Load swaps
    print(f"\n1. Loading swaps from {swaps_path}...")
    swaps = pd.read_parquet(swaps_path)
    
    # Ensure swap_date is datetime
    if 'swap_date' in swaps.columns:
        swaps['swap_date'] = pd.to_datetime(swaps['swap_date'])
        swap_date_min = swaps['swap_date'].min()
        swap_date_max = swaps['swap_date'].max()
        print(f"   ✓ Loaded {len(swaps):,} swaps")
        print(f"   ✓ Swap date range: {swap_date_min.date()} to {swap_date_max.date()}")
    else:
        raise ValueError("swap_date column not found in swaps data")
    
    # Limit to max_swaps if specified
    if max_swaps is not None:
        swaps = swaps.head(max_swaps)
        print(f"   ⚠ Limited to first {max_swaps} swaps for testing")
    
    if len(swaps) == 0:
        raise ValueError("No swaps found. Check swap_date column.")
    
    # Load sales data from 2020-01-01 to 2025-05-01 (we only process data in this date range)
    min_sales_date = pd.Timestamp('2020-01-01')
    max_sales_date = pd.Timestamp('2025-05-01')
    
    # Load sales data from 2020 to 2025-05-01
    print(f"\n2. Loading sales data from {min_sales_date.date()} to {max_sales_date.date()}...")
    print(f"   (Will use this data to analyze swaps from {swap_date_min.date()} to {swap_date_max.date()})")
    sales_df = load_sales_data(date_start=min_sales_date, date_end=max_sales_date, data_path=sales_data_path)
    
    # Calculate financials for each swap
    print(f"\n3. Calculating {weeks_before}-week before and {weeks_after}-week after metrics...")
    print(f"   This may take a few minutes for {len(swaps):,} swaps...\n")
    
    financial_data = []
    for idx, row in tqdm(swaps.iterrows(), total=len(swaps), desc="Processing swaps"):
        try:
            financials = calculate_swap_financials(
                row,
                sales_df,
                weeks_before=weeks_before,
                weeks_after=weeks_after
            )
            financial_data.append(financials)
        except Exception as e:
            # If error, append None values
            print(f"   Warning: Error processing swap {idx}: {e}")
            financial_data.append({key: None for key in [
                'revenue_before_4w', 'profit_before_4w', 'cost_before_4w',
                'sales_count_before_4w', 'days_observed_before_4w',
                'revenue_per_day_before_4w', 'profit_per_day_before_4w',
                'revenue_after_4w', 'profit_after_4w', 'cost_after_4w',
                'sales_count_after_4w', 'days_observed_after_4w',
                'revenue_per_day_after_4w', 'profit_per_day_after_4w',
                'revenue_change_4w', 'revenue_change_pct_4w',
                'profit_change_4w', 'profit_change_pct_4w',
                'sales_count_change_4w'
            ]})
    
    # Add financial data to swaps
    financial_df = pd.DataFrame(financial_data)
    enriched_swaps = pd.concat([swaps, financial_df], axis=1)
    
    # Summary statistics
    print("\n" + "="*70)
    print("ENRICHMENT SUMMARY")
    print("="*70)
    print(f"Total swaps processed: {len(enriched_swaps):,}")
    print(f"Swaps with profit data: {enriched_swaps['profit_before_4w'].notna().sum():,}")
    print(f"\nRevenue Statistics (4-week periods):")
    print(f"  Average revenue before: {enriched_swaps['revenue_before_4w'].mean():.2f} SEK")
    print(f"  Average revenue after:  {enriched_swaps['revenue_after_4w'].mean():.2f} SEK")
    print(f"  Average revenue change: {enriched_swaps['revenue_change_4w'].mean():.2f} SEK")
    
    if enriched_swaps['profit_before_4w'].notna().sum() > 0:
        print(f"\nProfit Statistics (4-week periods):")
        print(f"  Average profit before: {enriched_swaps['profit_before_4w'].mean():.2f} SEK")
        print(f"  Average profit after:  {enriched_swaps['profit_after_4w'].mean():.2f} SEK")
        print(f"  Average profit change: {enriched_swaps['profit_change_4w'].mean():.2f} SEK")
    
    print(f"\nSwap Outcomes:")
    revenue_positive = (enriched_swaps['revenue_change_4w'] > 0).sum()
    revenue_negative = (enriched_swaps['revenue_change_4w'] < 0).sum()
    print(f"  Revenue increased: {revenue_positive:,} ({revenue_positive/len(enriched_swaps)*100:.1f}%)")
    print(f"  Revenue decreased: {revenue_negative:,} ({revenue_negative/len(enriched_swaps)*100:.1f}%)")
    
    if enriched_swaps['profit_change_4w'].notna().sum() > 0:
        profit_positive = (enriched_swaps['profit_change_4w'] > 0).sum()
        profit_negative = (enriched_swaps['profit_change_4w'] < 0).sum()
        print(f"  Profit increased:  {profit_positive:,} ({profit_positive/len(enriched_swaps)*100:.1f}%)")
        print(f"  Profit decreased:  {profit_negative:,} ({profit_negative/len(enriched_swaps)*100:.1f}%)")
    
    print("="*70)
    
    # Save enriched data
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        enriched_swaps.to_parquet(output_file, index=False)
        print(f"\n✓ Saved enriched swaps to {output_file}")
    
    return enriched_swaps


def main(max_swaps: Optional[int] = None):
    """Main execution function."""
    # Paths
    script_dir = Path(__file__).parent  # scripts/
    project_root = script_dir.parent     # product_swap/
    swaps_output_dir = project_root / "data" / "swaps"
    swaps_output_dir.mkdir(parents=True, exist_ok=True)
    swaps_path = swaps_output_dir / "product_swaps.parquet"
    
    # Use different output file for test runs
    if max_swaps is not None:
        output_path = swaps_output_dir / f"product_swaps_enriched_test_{max_swaps}.parquet"
    else:
        output_path = swaps_output_dir / "product_swaps_enriched.parquet"
    
    # Run enrichment
    enriched_swaps = enrich_swaps_with_financials(
        swaps_path=swaps_path,
        output_path=output_path,
        weeks_before=4,
        weeks_after=4,
        max_swaps=max_swaps
    )
    
    # Display sample
    print("\n" + "="*70)
    print("SAMPLE ENRICHED SWAPS (First 5)")
    print("="*70)
    
    # Select key columns to display
    display_cols = [
        'machine_id', 'position', 'product_before', 'product_after',
        'revenue_before_4w', 'revenue_after_4w', 'revenue_change_4w',
        'profit_before_4w', 'profit_after_4w', 'profit_change_4w'
    ]
    
    print(enriched_swaps[display_cols].head(5).to_string())
    
    return enriched_swaps


if __name__ == "__main__":
    swaps = main()

