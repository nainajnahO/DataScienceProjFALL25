"""
Sales Data Aggregation with Profit & Revenue Analysis
======================================================

Processes each Sales_{year}.parquet file individually and calculates profit/revenue metrics
for each transaction, saving separate output files for each year.

PROFIT CALCULATION METHOD:
--------------------------
Since the purchase_price_kr column in the raw data is inaccurate, this script
uses an estimated cost of goods sold (COGS) approach:

- Estimated Cost = 70% of selling price
- Profit = 30% of selling price
- Profit Margin = 30%

This provides a consistent and realistic profit estimate across all products.
If you need to adjust the cost percentage, modify the cost_percentage parameter
in the calculate_profit_revenue() function (default: 0.70).

Output: Individual parquet files for each year (Sales_{year}_with_profit.parquet)
         plus a combined product-level aggregate file
"""

import pandas as pd
from pathlib import Path
from typing import Optional, List


# Paths
SCRIPT_DIR = Path(__file__).parent  # scripts/
PROJECT_ROOT = SCRIPT_DIR.parent     # product_swap/
SALES_DATA_DIR = PROJECT_ROOT.parent / "moaaz-prod" / "data" / "raw"


def get_sales_files(years: Optional[List[int]] = None) -> List[Path]:
    """
    Get list of available sales files.
    
    Args:
        years: List of years to get files for (default: 2020-2025)
    
    Returns:
        List of Path objects for existing sales files
    """
    if years is None:
        years = list(range(2020, 2026))  # 2020 through 2025
    
    sales_files = []
    for year in years:
        file_path = SALES_DATA_DIR / f"Sales_{year}.parquet"
        if file_path.exists():
            sales_files.append(file_path)
        else:
            print(f"⚠ Warning: {file_path} not found, skipping...")
    
    if not sales_files:
        raise FileNotFoundError(f"No sales data files found in {SALES_DATA_DIR}")
    
    return sales_files


def calculate_profit_revenue(sales_df: pd.DataFrame, cost_percentage: float = 0.70) -> pd.DataFrame:
    """
    Calculate profit and revenue for each transaction.
    
    PRESERVES ALL ORIGINAL COLUMNS from the raw sales data and adds new columns.
    
    Uses estimated cost of goods sold (COGS) as a percentage of selling price,
    since the purchase_price_kr column in the data is inaccurate.
    
    Default assumption: Cost = 70% of price (30% profit margin)
    
    Profit = price - (price * cost_percentage)
    Revenue = price
    
    Args:
        sales_df: Raw sales data with 'price' column (all original columns preserved)
        cost_percentage: Estimated cost as percentage of price (default: 0.70 = 70%)
    
    Returns:
        DataFrame with ALL original columns PLUS 'profit', 'revenue', 
        'estimated_purchase_price', and 'profit_margin_pct' columns
    """
    print(f"\n💰 Calculating profit and revenue (using {cost_percentage*100:.0f}% estimated cost)...")
    print(f"   Original columns: {len(sales_df.columns)}")
    
    df = sales_df.copy()
    
    # Revenue is simply the price
    df['revenue'] = df['price']
    
    # Estimate purchase price as percentage of selling price
    # Using estimated cost instead of inaccurate purchase_price_kr column
    df['estimated_purchase_price'] = df['price'] * cost_percentage
    
    # Profit = price - estimated cost
    df['profit'] = df['price'] - df['estimated_purchase_price']
    
    # Add profit margin percentage
    df['profit_margin_pct'] = (df['profit'] / df['price'] * 100).where(df['price'] > 0, 0)
    
    print(f"   ✓ Added 4 new columns (revenue, estimated_purchase_price, profit, profit_margin_pct)")
    print(f"   ✓ Total columns: {len(df.columns)}")
    print(f"   ✓ Revenue: min={df['revenue'].min():.2f}, max={df['revenue'].max():.2f}, mean={df['revenue'].mean():.2f} SEK")
    print(f"   ✓ Profit: min={df['profit'].min():.2f}, max={df['profit'].max():.2f}, mean={df['profit'].mean():.2f} SEK")
    print(f"   ✓ Profit margin: ~{df['profit_margin_pct'].mean():.1f}%")
    
    return df


def aggregate_by_product(sales_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate sales data by product with profit and revenue metrics.
    
    Args:
        sales_df: Sales DataFrame with profit and revenue columns
    
    Returns:
        Product-level aggregated DataFrame
    """
    print("\n📊 Aggregating by product...")
    
    # Helper function to get most common value
    def get_most_common(x):
        """Get the most common value, handling empty and all-NaN cases"""
        if len(x) == 0:
            return None
        x_clean = x.dropna()
        if len(x_clean) == 0:
            return None
        value_counts = x_clean.value_counts()
        if len(value_counts) > 0:
            return value_counts.index[0]
        return None
    
    # Aggregate by product_name
    # Note: Can't use 'price' twice in dict, so combine all price aggregations
    product_agg = sales_df.groupby('product_name').agg({
        # Revenue metrics
        'revenue': ['count', 'sum', 'mean', 'median'],  # count gives transaction_count
        
        # Profit metrics
        'profit': ['sum', 'mean', 'median'],
        
        # Profit margin
        'profit_margin_pct': 'mean',
        
        # Price metrics (all in one to avoid duplicate key)
        'price': ['mean', 'median', 'min', 'max'],
        
        # Estimated purchase price (70% of selling price)
        'estimated_purchase_price': 'mean',
        
        # Machine metrics
        'machine_id': 'nunique',
        
        # Product metadata (most common value)
        'ean': 'first',
        'category': get_most_common,
        'subcategory': get_most_common,
        'provider': get_most_common,
    }).reset_index()
    
    # Flatten column names
    product_agg.columns = [
        'product_name',
        'transaction_count',
        'total_revenue', 'avg_revenue', 'median_revenue',
        'total_profit', 'avg_profit', 'median_profit',
        'avg_profit_margin_pct',
        'avg_price', 'median_price', 'min_price', 'max_price',
        'avg_estimated_purchase_price',
        'unique_machines',
        'ean', 'category', 'subcategory', 'provider'
    ]
    
    # Calculate additional metrics
    product_agg['revenue_per_machine'] = product_agg['total_revenue'] / product_agg['unique_machines']
    product_agg['profit_per_machine'] = product_agg['total_profit'] / product_agg['unique_machines']
    product_agg['transactions_per_machine'] = product_agg['transaction_count'] / product_agg['unique_machines']
    
    # Sort by total profit (most profitable first)
    product_agg = product_agg.sort_values('total_profit', ascending=False).reset_index(drop=True)
    
    print(f"   ✓ Aggregated into {len(product_agg):,} unique products")
    print(f"   ✓ Total revenue across all products: {product_agg['total_revenue'].sum():,.2f} SEK")
    print(f"   ✓ Total profit across all products: {product_agg['total_profit'].sum():,.2f} SEK")
    
    return product_agg


def print_summary_statistics(product_df: pd.DataFrame):
    """Print summary statistics about the aggregated data."""
    print("\n" + "="*70)
    print("PRODUCT PROFIT & REVENUE SUMMARY")
    print("="*70)
    print("Note: Profit calculated using 70% estimated cost (30% margin)")
    print("-"*70)
    
    print(f"\n📈 Overall Metrics:")
    print(f"   Total unique products: {len(product_df):,}")
    print(f"   Total transactions: {product_df['transaction_count'].sum():,}")
    print(f"   Total revenue: {product_df['total_revenue'].sum():,.2f} SEK")
    print(f"   Total profit: {product_df['total_profit'].sum():,.2f} SEK")
    print(f"   Overall profit margin: {(product_df['total_profit'].sum() / product_df['total_revenue'].sum() * 100):.2f}%")
    
    print(f"\n🏆 Top 10 Products by Total Profit:")
    top_profit = product_df.head(10)[['product_name', 'total_profit', 'total_revenue', 'avg_profit_margin_pct', 'transaction_count']]
    for idx, row in top_profit.iterrows():
        print(f"   {idx+1}. {row['product_name'][:40]:<40} | Profit: {row['total_profit']:>10,.0f} SEK | Revenue: {row['total_revenue']:>10,.0f} SEK | Margin: {row['avg_profit_margin_pct']:>5.1f}%")
    
    print(f"\n💎 Top 10 Products by Profit Margin:")
    top_margin = product_df.nlargest(10, 'avg_profit_margin_pct')[['product_name', 'avg_profit_margin_pct', 'total_profit', 'transaction_count']]
    for idx, row in top_margin.iterrows():
        print(f"   {idx+1}. {row['product_name'][:40]:<40} | Margin: {row['avg_profit_margin_pct']:>5.1f}% | Profit: {row['total_profit']:>10,.0f} SEK | Sales: {row['transaction_count']:>6,}")
    
    print(f"\n💵 Top 10 Products by Total Revenue:")
    top_revenue = product_df.nlargest(10, 'total_revenue')[['product_name', 'total_revenue', 'total_profit', 'transaction_count']]
    for idx, row in top_revenue.iterrows():
        print(f"   {idx+1}. {row['product_name'][:40]:<40} | Revenue: {row['total_revenue']:>10,.0f} SEK | Profit: {row['total_profit']:>10,.0f} SEK | Sales: {row['transaction_count']:>6,}")
    
    print("="*70)


def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("PROCESS SALES DATA (2020-01-01 to 2025-05-01) WITH PROFIT & REVENUE")
    print("="*70 + "\n")
    
    # Get list of sales files
    sales_files = get_sales_files()
    print(f"📁 Found {len(sales_files)} sales files to process...")
    
    # Create output directories
    sales_output_dir = SCRIPT_DIR / "data" / "sales"
    aggregates_output_dir = SCRIPT_DIR / "data" / "aggregates"
    sales_output_dir.mkdir(parents=True, exist_ok=True)
    aggregates_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Columns to drop
    columns_to_drop = [
        'machine_name', 'refiller', 'customer_id', 'card_brand', 
        'card_type', 'moms', 'too_unspecific', 'machine_group_tag', 'sielaff_id'
    ]
    
    # Process each file individually
    all_processed_dfs = []
    processed_files = []
    
    for file_path in sales_files:
        year = file_path.stem.split('_')[1]  # Extract year from filename
        print(f"\n{'='*70}")
        print(f"Processing Sales_{year}.parquet...")
        print(f"{'='*70}")
        
        # Load the year's data
        df = pd.read_parquet(file_path)
        print(f"   ✓ Loaded: {len(df):,} transactions")
        
        # Filter to date range: 2020-01-01 to 2025-05-01
        max_date = pd.Timestamp('2025-05-01')
        if 'local_timestamp' in df.columns:
            # Ensure datetime type
            if not pd.api.types.is_datetime64_any_dtype(df['local_timestamp']):
                df['local_timestamp'] = pd.to_datetime(df['local_timestamp'])
            
            # Check if timezone-aware and handle accordingly
            is_tz_aware = hasattr(df['local_timestamp'].dtype, 'tz') and df['local_timestamp'].dtype.tz is not None
            if is_tz_aware:
                tz = df['local_timestamp'].dtype.tz
                if max_date.tzinfo is None:
                    max_date = max_date.tz_localize(tz)
            else:
                if max_date.tzinfo is not None:
                    max_date = max_date.tz_localize(None)
            
            # Filter out data after 2025-05-01
            before_filter = len(df)
            df = df[df['local_timestamp'] <= max_date].copy()
            after_filter = len(df)
            if before_filter != after_filter:
                print(f"   ✓ Filtered to max date 2025-05-01: {after_filter:,} transactions (removed {before_filter - after_filter:,})")
        
        # Calculate profit and revenue (preserves all original columns)
        sales_with_metrics = calculate_profit_revenue(df)
        
        # Drop unnecessary columns
        print(f"\n🔧 Cleaning up unnecessary columns...")
        existing_drops = [col for col in columns_to_drop if col in sales_with_metrics.columns]
        if existing_drops:
            sales_with_metrics = sales_with_metrics.drop(columns=existing_drops)
            print(f"   ✓ Dropped {len(existing_drops)} columns: {', '.join(existing_drops)}")
        else:
            print(f"   ℹ No columns to drop")
        
        # Fix data type inconsistencies before saving to parquet
        print(f"\n🔧 Preparing data for parquet format...")
        # Convert columns with mixed types to string to avoid parquet errors
        string_columns = ['machine_id', 'ean']
        for col in string_columns:
            if col in sales_with_metrics.columns:
                sales_with_metrics[col] = sales_with_metrics[col].astype(str)
        
        # Save individual year file
        output_path = sales_output_dir / f"Sales_{year}_with_profit.parquet"
        print(f"\n💾 Saving to: {output_path.relative_to(SCRIPT_DIR)}...")
        sales_with_metrics.to_parquet(output_path, index=False)
        print(f"   ✓ Saved successfully! ({len(sales_with_metrics):,} rows)")
        
        # Keep for aggregate analysis
        all_processed_dfs.append(sales_with_metrics)
        processed_files.append(output_path)
    
    # Create product-level aggregates from all years combined
    print(f"\n{'='*70}")
    print("Creating product-level aggregates from all years...")
    print(f"{'='*70}")
    combined_df = pd.concat(all_processed_dfs, ignore_index=True)
    print(f"   ✓ Combined {len(all_processed_dfs)} years: {len(combined_df):,} total transactions")
    
    product_aggregates = aggregate_by_product(combined_df)
    
    aggregate_output_path = aggregates_output_dir / "product_profit_revenue.parquet"
    product_aggregates.to_parquet(aggregate_output_path, index=False)
    print(f"   ✓ Saved product aggregates to: {aggregate_output_path.relative_to(SCRIPT_DIR)}")
    
    # Print summary statistics
    print_summary_statistics(product_aggregates)
    
    print("\n" + "="*70)
    print("✅ COMPLETE - Output files:")
    print(f"\n📁 Sales data (data/sales/):")
    for output_file in processed_files:
        print(f"   • {output_file.name}")
    print(f"\n📊 Aggregates (data/aggregates/):")
    print(f"   • {aggregate_output_path.name}")
    print(f"\n   Total transactions processed: {len(combined_df):,}")
    print("="*70)
    
    return processed_files, product_aggregates


if __name__ == "__main__":
    output_files, product_df = main()

