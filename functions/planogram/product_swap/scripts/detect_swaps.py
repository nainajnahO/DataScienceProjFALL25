"""
Product Swap Detection
======================

Detects product swaps from machine_snapshots.parquet by identifying when
a position in a machine has a different product in the next snapshot.

Output: CSV/Parquet file with all detected swaps
"""

import pandas as pd
from pathlib import Path
from typing import Optional


def detect_product_swaps(
    snapshots_path: str,
    output_path: Optional[str] = None,
    min_persistence_days: int = 7
) -> pd.DataFrame:
    """
    Detect product swaps from machine snapshots.
    
    A swap is detected when:
    - Same machine_id and position
    - Different product_name in consecutive snapshots
    
    Args:
        snapshots_path: Path to machine_snapshots.parquet
        output_path: Path to save swap data (optional)
        min_persistence_days: Minimum days between snapshots to consider valid swap
    
    Returns:
        DataFrame with detected swaps
    """
    print("Loading machine snapshots...")
    df = pd.read_parquet(snapshots_path)
    
    print(f"Loaded {len(df):,} snapshot records")
    print(f"Date range: {df['snapshot_date'].min()} to {df['snapshot_date'].max()}")
    print(f"Unique machines: {df['machine_id'].nunique():,}")
    print(f"Unique snapshot dates: {sorted(df['snapshot_date'].unique())}")
    
    # Convert snapshot_date to datetime
    df['snapshot_date'] = pd.to_datetime(df['snapshot_date'])
    
    # Filter to date range: 2020-01-01 to 2025-05-01
    min_date = pd.Timestamp('2020-01-01')
    max_date = pd.Timestamp('2025-05-01')
    df = df[(df['snapshot_date'] >= min_date) & (df['snapshot_date'] <= max_date)].copy()
    print(f"\nFiltered to date range 2020-01-01 to 2025-05-01: {len(df):,} snapshot records")
    print(f"Date range after filter: {df['snapshot_date'].min()} to {df['snapshot_date'].max()}")
    
    # Sort by machine, position, and date
    df = df.sort_values(['machine_id', 'position', 'snapshot_date']).reset_index(drop=True)
    
    print("\nDetecting product swaps...")
    
    # For each machine-position combination, get the previous snapshot
    df['product_before'] = df.groupby(['machine_id', 'position'])['product_name'].shift(1)
    df['provider_before'] = df.groupby(['machine_id', 'position'])['provider'].shift(1)
    df['subcategory_before'] = df.groupby(['machine_id', 'position'])['subcategory'].shift(1)
    df['n_sales_before'] = df.groupby(['machine_id', 'position'])['n_sales'].shift(1)
    df['snapshot_date_before'] = df.groupby(['machine_id', 'position'])['snapshot_date'].shift(1)
    
    # Identify swaps: where product changed
    swaps = df[
        (df['product_before'].notna()) &  # Has a previous snapshot
        (df['product_name'] != df['product_before'])  # Product changed
    ].copy()
    
    # Calculate days between snapshots
    swaps['days_between_snapshots'] = (
        swaps['snapshot_date'] - swaps['snapshot_date_before']
    ).dt.days
    
    # Filter for reasonable time gaps (not too long)
    swaps = swaps[swaps['days_between_snapshots'] >= min_persistence_days].copy()
    
    # Rename columns for clarity
    swaps = swaps.rename(columns={
        'product_name': 'product_after',
        'provider': 'provider_after',
        'subcategory': 'subcategory_after',
        'n_sales': 'n_sales_after',
        'snapshot_date': 'swap_date'
    })
    
    # Select relevant columns
    swap_columns = [
        'machine_id',
        'position',
        'swap_date',
        'snapshot_date_before',
        'days_between_snapshots',
        'product_before',
        'product_after',
        'provider_before',
        'provider_after',
        'subcategory_before',
        'subcategory_after',
        'n_sales_before',
        'n_sales_after'
    ]
    
    swaps = swaps[swap_columns].reset_index(drop=True)
    
    # Add some derived features
    swaps['same_category'] = (swaps['subcategory_before'] == swaps['subcategory_after']).astype(int)
    swaps['same_provider'] = (swaps['provider_before'] == swaps['provider_after']).astype(int)
    swaps['sales_change'] = swaps['n_sales_after'] - swaps['n_sales_before']
    swaps['sales_change_pct'] = (
        (swaps['n_sales_after'] - swaps['n_sales_before']) / swaps['n_sales_before'] * 100
    ).where(swaps['n_sales_before'] > 0, None)
    
    print(f"\n{'='*70}")
    print(f"SWAP DETECTION SUMMARY")
    print(f"{'='*70}")
    print(f"Total swaps detected: {len(swaps):,}")
    print(f"Unique machines with swaps: {swaps['machine_id'].nunique():,}")
    print(f"Average swaps per machine: {len(swaps) / swaps['machine_id'].nunique():.1f}")
    print(f"\nSwap date range: {swaps['swap_date'].min()} to {swaps['swap_date'].max()}")
    print(f"\nSame category swaps: {swaps['same_category'].sum():,} ({swaps['same_category'].mean()*100:.1f}%)")
    print(f"Same provider swaps: {swaps['same_provider'].sum():,} ({swaps['same_provider'].mean()*100:.1f}%)")
    print(f"\nAverage sales change: {swaps['sales_change'].mean():.2f} units")
    print(f"Positive swaps (sales increased): {(swaps['sales_change'] > 0).sum():,} ({(swaps['sales_change'] > 0).mean()*100:.1f}%)")
    print(f"Negative swaps (sales decreased): {(swaps['sales_change'] < 0).sum():,} ({(swaps['sales_change'] < 0).mean()*100:.1f}%)")
    print(f"{'='*70}")
    
    # Save to file if output path provided
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as parquet
        swaps.to_parquet(output_file, index=False)
        print(f"\n✓ Saved {len(swaps):,} swaps to {output_file}")
    
    return swaps


def main():
    """Main execution function."""
    # Paths
    script_dir = Path(__file__).parent  # scripts/
    project_root = script_dir.parent     # product_swap/
    snapshots_path = project_root.parent / "machine_snapshots" / "data" / "machine_snapshots.parquet"
    swaps_output_dir = project_root / "data" / "swaps"
    swaps_output_dir.mkdir(parents=True, exist_ok=True)
    output_path = swaps_output_dir / "product_swaps.parquet"
    
    # Detect swaps
    swaps_df = detect_product_swaps(
        snapshots_path=snapshots_path,
        output_path=output_path,
        min_persistence_days=7
    )
    
    # Display sample swaps
    print("\n" + "="*70)
    print("SAMPLE SWAPS (First 10)")
    print("="*70)
    print(swaps_df.head(10).to_string())
    
    print("\n" + "="*70)
    print("TOP 10 PRODUCTS SWAPPED OUT")
    print("="*70)
    product_out = swaps_df['product_before'].value_counts().head(10)
    for product, count in product_out.items():
        print(f"  {product}: {count} times")
    
    print("\n" + "="*70)
    print("TOP 10 PRODUCTS SWAPPED IN")
    print("="*70)
    product_in = swaps_df['product_after'].value_counts().head(10)
    for product, count in product_in.items():
        print(f"  {product}: {count} times")
    
    return swaps_df


if __name__ == "__main__":
    swaps = main()

