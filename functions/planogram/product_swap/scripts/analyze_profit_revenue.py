"""
Quick Analysis Script for Product Profit & Revenue Data
========================================================

Load and analyze the aggregated product profit/revenue data.
Run after aggregate_sales_with_profit.py has created the data files.
"""

import pandas as pd
from pathlib import Path


def main():
    """Quick analysis of product profit and revenue data."""
    
    # Paths
    script_dir = Path(__file__).parent  # scripts/
    project_root = script_dir.parent     # product_swap/
    product_file = project_root / "data" / "aggregates" / "product_profit_revenue.parquet"
    # Try new structure first, fallback to old
    sales_dir = project_root / "data" / "sales"
    if sales_dir.exists() and list(sales_dir.glob("Sales_*_with_profit.parquet")):
        # Load all year files
        sales_files = sorted(sales_dir.glob("Sales_*_with_profit.parquet"))
        print(f"Loading sales data from {len(sales_files)} year files...")
        dfs = [pd.read_parquet(f) for f in sales_files]
        sales_file = None
        sales_df = pd.concat(dfs, ignore_index=True)
    else:
        # Fallback to old location
        sales_file = project_root / "data" / "all_sales_with_profit.parquet"
        sales_df = pd.read_parquet(sales_file) if sales_file.exists() else None
    
    # Check if files exist
    if not product_file.exists():
        print(f"❌ File not found: {product_file}")
        print("   Please run 'python aggregate_sales_with_profit.py' first.")
        return
    
    if sales_df is None:
        print(f"❌ Sales data not found.")
        print("   Please run 'python aggregate_sales_with_profit.py' first.")
        return
    
    print("="*70)
    print("PRODUCT PROFIT & REVENUE ANALYSIS")
    print("="*70 + "\n")
    
    # Load product aggregates
    print(f"📂 Loading product aggregates from {product_file.name}...")
    products = pd.read_parquet(product_file)
    print(f"   ✓ Loaded {len(products):,} products\n")
    
    # Basic statistics
    print("📊 OVERALL STATISTICS")
    print("-" * 70)
    print(f"Total products:         {len(products):>12,}")
    print(f"Total transactions:     {products['transaction_count'].sum():>12,}")
    print(f"Total revenue:          {products['total_revenue'].sum():>12,.2f} SEK")
    print(f"Total profit:           {products['total_profit'].sum():>12,.2f} SEK")
    print(f"Overall margin:         {(products['total_profit'].sum() / products['total_revenue'].sum() * 100):>12.2f}%")
    print(f"Unique machines:        {products['unique_machines'].max():>12,}")
    print()
    
    # Top 10 by profit
    print("🏆 TOP 10 MOST PROFITABLE PRODUCTS")
    print("-" * 70)
    top_profit = products.head(10)
    for i, row in top_profit.iterrows():
        print(f"{i+1:2d}. {row['product_name'][:35]:<35} "
              f"Profit: {row['total_profit']:>10,.0f} SEK  "
              f"Margin: {row['avg_profit_margin_pct']:>5.1f}%  "
              f"Sales: {row['transaction_count']:>7,}")
    print()
    
    # Top 10 by profit margin
    print("💎 TOP 10 HIGHEST PROFIT MARGIN PRODUCTS (min 100 sales)")
    print("-" * 70)
    high_volume = products[products['transaction_count'] >= 100]
    top_margin = high_volume.nlargest(10, 'avg_profit_margin_pct')
    for i, (idx, row) in enumerate(top_margin.iterrows(), 1):
        print(f"{i:2d}. {row['product_name'][:35]:<35} "
              f"Margin: {row['avg_profit_margin_pct']:>5.1f}%  "
              f"Profit: {row['total_profit']:>10,.0f} SEK  "
              f"Sales: {row['transaction_count']:>7,}")
    print()
    
    # Category analysis
    print("📦 PROFIT BY CATEGORY")
    print("-" * 70)
    category_stats = products.groupby('category').agg({
        'total_profit': 'sum',
        'total_revenue': 'sum',
        'transaction_count': 'sum',
        'product_name': 'count'
    }).sort_values('total_profit', ascending=False)
    category_stats['profit_margin'] = (category_stats['total_profit'] / 
                                       category_stats['total_revenue'] * 100)
    
    for category, row in category_stats.head(10).iterrows():
        if pd.notna(category):
            print(f"{str(category)[:20]:<20} "
                  f"Profit: {row['total_profit']:>12,.0f} SEK  "
                  f"Revenue: {row['total_revenue']:>12,.0f} SEK  "
                  f"Margin: {row['profit_margin']:>5.1f}%  "
                  f"Products: {row['product_name']:>4.0f}")
    print()
    
    # Revenue per machine leaders
    print("🎯 TOP 10 REVENUE PER MACHINE")
    print("-" * 70)
    # Filter for products in at least 5 machines to avoid outliers
    multi_machine = products[products['unique_machines'] >= 5]
    top_rpm = multi_machine.nlargest(10, 'revenue_per_machine')
    for i, (idx, row) in enumerate(top_rpm.iterrows(), 1):
        print(f"{i:2d}. {row['product_name'][:35]:<35} "
              f"Rev/Machine: {row['revenue_per_machine']:>10,.0f} SEK  "
              f"Machines: {row['unique_machines']:>4.0f}  "
              f"Avg Sales: {row['transactions_per_machine']:>6.1f}")
    print()
    
    # Provider analysis
    print("🏢 TOP 10 PROVIDERS BY PROFIT")
    print("-" * 70)
    provider_stats = products.groupby('provider').agg({
        'total_profit': 'sum',
        'total_revenue': 'sum',
        'transaction_count': 'sum',
        'product_name': 'count'
    }).sort_values('total_profit', ascending=False)
    provider_stats['profit_margin'] = (provider_stats['total_profit'] / 
                                       provider_stats['total_revenue'] * 100)
    
    for provider, row in provider_stats.head(10).iterrows():
        if pd.notna(provider):
            print(f"{str(provider)[:20]:<20} "
                  f"Profit: {row['total_profit']:>12,.0f} SEK  "
                  f"Margin: {row['profit_margin']:>5.1f}%  "
                  f"Products: {row['product_name']:>4.0f}  "
                  f"Sales: {row['transaction_count']:>8,.0f}")
    print()
    
    # Transaction-level analysis (if file exists)
    if sales_file.exists():
        print("📈 TRANSACTION-LEVEL INSIGHTS")
        print("-" * 70)
        print(f"Loading sample from {sales_file.name}...")
        # Read only a sample for quick analysis
        sales_sample = pd.read_parquet(sales_file, columns=['profit', 'revenue', 
                                                            'profit_margin_pct', 
                                                            'machine_id'])
        print(f"   ✓ Total transactions: {len(sales_sample):,}")
        print(f"\nTransaction Statistics:")
        print(f"   Average profit per sale:  {sales_sample['profit'].mean():>8.2f} SEK")
        print(f"   Median profit per sale:   {sales_sample['profit'].median():>8.2f} SEK")
        print(f"   Average margin per sale:  {sales_sample['profit_margin_pct'].mean():>8.2f}%")
        print(f"   Profitable transactions:  {(sales_sample['profit'] > 0).mean()*100:>8.1f}%")
        print()
    
    print("="*70)
    print("✓ Analysis complete!")
    print("="*70)


if __name__ == "__main__":
    main()

