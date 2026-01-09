"""
Exploratory Data Analysis - Product Swap Patterns
==================================================

Analyze swap patterns to understand what makes a successful swap.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def load_enriched_swaps(path='data/swaps/product_swaps_enriched.parquet'):
    """Load enriched swap data."""
    df = pd.read_parquet(path)
    print(f"Loaded {len(df):,} enriched swaps")
    return df


def basic_statistics(df):
    """Display basic statistics about swaps."""
    print("\n" + "="*70)
    print("BASIC SWAP STATISTICS")
    print("="*70)
    
    print(f"\nTotal swaps: {len(df):,}")
    print(f"Unique machines: {df['machine_id'].nunique():,}")
    print(f"Date range: {df['swap_date'].min()} to {df['swap_date'].max()}")
    
    print(f"\n--- Revenue Statistics (4-week periods) ---")
    print(f"Average revenue before: {df['revenue_before_4w'].mean():.2f} SEK")
    print(f"Average revenue after:  {df['revenue_after_4w'].mean():.2f} SEK")
    print(f"Average revenue change: {df['revenue_change_4w'].mean():.2f} SEK")
    print(f"Median revenue change:  {df['revenue_change_4w'].median():.2f} SEK")
    
    print(f"\n--- Profit Statistics (4-week periods) ---")
    print(f"Average profit before: {df['profit_before_4w'].mean():.2f} SEK")
    print(f"Average profit after:  {df['profit_after_4w'].mean():.2f} SEK")
    print(f"Average profit change: {df['profit_change_4w'].mean():.2f} SEK")
    print(f"Median profit change:  {df['profit_change_4w'].median():.2f} SEK")
    
    print(f"\n--- Swap Outcomes ---")
    revenue_positive = (df['revenue_change_4w'] > 0).sum()
    revenue_negative = (df['revenue_change_4w'] < 0).sum()
    revenue_neutral = (df['revenue_change_4w'] == 0).sum()
    
    print(f"Revenue increased: {revenue_positive:,} ({revenue_positive/len(df)*100:.1f}%)")
    print(f"Revenue decreased: {revenue_negative:,} ({revenue_negative/len(df)*100:.1f}%)")
    print(f"Revenue unchanged: {revenue_neutral:,} ({revenue_neutral/len(df)*100:.1f}%)")
    
    profit_positive = (df['profit_change_4w'] > 0).sum()
    profit_negative = (df['profit_change_4w'] < 0).sum()
    
    print(f"\nProfit increased:  {profit_positive:,} ({profit_positive/len(df)*100:.1f}%)")
    print(f"Profit decreased:  {profit_negative:,} ({profit_negative/len(df)*100:.1f}%)")


def analyze_swap_types(df):
    """Analyze different types of swaps."""
    print("\n" + "="*70)
    print("SWAP TYPE ANALYSIS")
    print("="*70)
    
    # Same vs different category
    print("\n--- Same vs Cross-Category Swaps ---")
    same_cat = df[df['same_category'] == 1]
    diff_cat = df[df['same_category'] == 0]
    
    print(f"Same category swaps: {len(same_cat):,} ({len(same_cat)/len(df)*100:.1f}%)")
    print(f"  Avg revenue change: {same_cat['revenue_change_4w'].mean():.2f} SEK")
    print(f"  Success rate: {(same_cat['revenue_change_4w'] > 0).mean()*100:.1f}%")
    
    print(f"\nCross-category swaps: {len(diff_cat):,} ({len(diff_cat)/len(df)*100:.1f}%)")
    print(f"  Avg revenue change: {diff_cat['revenue_change_4w'].mean():.2f} SEK")
    print(f"  Success rate: {(diff_cat['revenue_change_4w'] > 0).mean()*100:.1f}%")
    
    # Same vs different provider
    print("\n--- Same vs Cross-Provider Swaps ---")
    same_prov = df[df['same_provider'] == 1]
    diff_prov = df[df['same_provider'] == 0]
    
    print(f"Same provider swaps: {len(same_prov):,} ({len(same_prov)/len(df)*100:.1f}%)")
    print(f"  Avg revenue change: {same_prov['revenue_change_4w'].mean():.2f} SEK")
    print(f"  Success rate: {(same_prov['revenue_change_4w'] > 0).mean()*100:.1f}%")
    
    print(f"\nCross-provider swaps: {len(diff_prov):,} ({len(diff_prov)/len(df)*100:.1f}%)")
    print(f"  Avg revenue change: {diff_prov['revenue_change_4w'].mean():.2f} SEK")
    print(f"  Success rate: {(diff_prov['revenue_change_4w'] > 0).mean()*100:.1f}%")


def top_swaps_analysis(df):
    """Analyze best and worst performing swaps."""
    print("\n" + "="*70)
    print("TOP PERFORMING SWAPS")
    print("="*70)
    
    # Top 20 swaps by revenue increase
    top_swaps = df.nlargest(20, 'revenue_change_4w')
    
    print("\n--- Top 20 Swaps by Revenue Increase ---")
    for idx, row in top_swaps.head(20).iterrows():
        print(f"\n{row['product_before']:30s} → {row['product_after']:30s}")
        print(f"  Revenue: {row['revenue_before_4w']:6.0f} → {row['revenue_after_4w']:6.0f} SEK ({row['revenue_change_4w']:+.0f})")
        print(f"  Profit:  {row['profit_before_4w']:6.0f} → {row['profit_after_4w']:6.0f} SEK ({row['profit_change_4w']:+.0f})")
    
    # Worst 10 swaps
    worst_swaps = df.nsmallest(10, 'revenue_change_4w')
    
    print("\n" + "="*70)
    print("WORST PERFORMING SWAPS")
    print("="*70)
    print("\n--- Bottom 10 Swaps by Revenue Decrease ---")
    for idx, row in worst_swaps.head(10).iterrows():
        print(f"\n{row['product_before']:30s} → {row['product_after']:30s}")
        print(f"  Revenue: {row['revenue_before_4w']:6.0f} → {row['revenue_after_4w']:6.0f} SEK ({row['revenue_change_4w']:+.0f})")
        print(f"  Profit:  {row['profit_before_4w']:6.0f} → {row['profit_after_4w']:6.0f} SEK ({row['profit_change_4w']:+.0f})")


def product_analysis(df):
    """Analyze which products are commonly swapped in/out."""
    print("\n" + "="*70)
    print("PRODUCT SWAP PATTERNS")
    print("="*70)
    
    print("\n--- Top 15 Products Swapped OUT ---")
    products_out = df['product_before'].value_counts().head(15)
    for product, count in products_out.items():
        swaps = df[df['product_before'] == product]
        avg_revenue_change = swaps['revenue_change_4w'].mean()
        success_rate = (swaps['revenue_change_4w'] > 0).mean() * 100
        print(f"{product:35s}: {count:4d} times | Avg Δ: {avg_revenue_change:+7.1f} SEK | Success: {success_rate:5.1f}%")
    
    print("\n--- Top 15 Products Swapped IN ---")
    products_in = df['product_after'].value_counts().head(15)
    for product, count in products_in.items():
        swaps = df[df['product_after'] == product]
        avg_revenue_change = swaps['revenue_change_4w'].mean()
        success_rate = (swaps['revenue_change_4w'] > 0).mean() * 100
        print(f"{product:35s}: {count:4d} times | Avg Δ: {avg_revenue_change:+7.1f} SEK | Success: {success_rate:5.1f}%")


def category_analysis(df):
    """Analyze swap patterns by category."""
    print("\n" + "="*70)
    print("CATEGORY ANALYSIS")
    print("="*70)
    
    print("\n--- Categories Swapped IN: Performance ---")
    cat_performance = df.groupby('subcategory_after').agg({
        'revenue_change_4w': ['count', 'mean', 'median'],
        'profit_change_4w': 'mean'
    }).round(2)
    
    cat_performance.columns = ['Count', 'Avg_Rev_Change', 'Med_Rev_Change', 'Avg_Profit_Change']
    cat_performance = cat_performance.sort_values('Avg_Rev_Change', ascending=False)
    
    print("\nTop 15 Categories by Average Revenue Change:")
    print(cat_performance.head(15).to_string())
    
    print("\n\nBottom 10 Categories by Average Revenue Change:")
    print(cat_performance.tail(10).to_string())


def temporal_analysis(df):
    """Analyze swap patterns over time."""
    print("\n" + "="*70)
    print("TEMPORAL ANALYSIS")
    print("="*70)
    
    # Add temporal features
    df['swap_month'] = pd.to_datetime(df['swap_date']).dt.month
    df['swap_week'] = pd.to_datetime(df['swap_date']).dt.isocalendar().week
    
    print("\n--- Swaps by Month ---")
    monthly = df.groupby('swap_month').agg({
        'revenue_change_4w': ['count', 'mean'],
        'profit_change_4w': 'mean'
    }).round(2)
    
    monthly.columns = ['Count', 'Avg_Rev_Change', 'Avg_Profit_Change']
    
    month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                   7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
    monthly.index = monthly.index.map(month_names)
    
    print(monthly.to_string())


def sales_volume_analysis(df):
    """Analyze how previous sales volume affects swap outcomes."""
    print("\n" + "="*70)
    print("SALES VOLUME ANALYSIS")
    print("="*70)
    
    # Create sales volume bins
    df['sales_bin_before'] = pd.cut(df['sales_count_before_4w'], 
                                     bins=[0, 5, 10, 20, 50, 1000],
                                     labels=['0-5', '6-10', '11-20', '21-50', '50+'])
    
    print("\n--- Swap Outcomes by Previous Sales Volume ---")
    volume_analysis = df.groupby('sales_bin_before').agg({
        'revenue_change_4w': ['count', 'mean'],
        'revenue_before_4w': 'mean',
        'revenue_after_4w': 'mean'
    }).round(2)
    
    volume_analysis.columns = ['Count', 'Avg_Rev_Change', 'Avg_Rev_Before', 'Avg_Rev_After']
    
    print(volume_analysis.to_string())
    
    # Success rate by volume
    print("\n--- Success Rate by Previous Sales Volume ---")
    for sales_bin in df['sales_bin_before'].cat.categories:
        subset = df[df['sales_bin_before'] == sales_bin]
        success_rate = (subset['revenue_change_4w'] > 0).mean() * 100
        print(f"{sales_bin:6s}: {success_rate:5.1f}% success rate ({len(subset):,} swaps)")


def main():
    """Run complete EDA analysis."""
    print("="*70)
    print("PRODUCT SWAP EXPLORATORY DATA ANALYSIS")
    print("="*70)
    
    # Load data
    df = load_enriched_swaps()
    
    # Run analyses
    basic_statistics(df)
    analyze_swap_types(df)
    top_swaps_analysis(df)
    product_analysis(df)
    category_analysis(df)
    temporal_analysis(df)
    sales_volume_analysis(df)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nKey Insights:")
    print("1. Check which product/category swaps perform best")
    print("2. Consider sales volume when making swap decisions")
    print("3. Same-category vs cross-category swap success rates")
    print("4. Seasonal patterns in swap performance")
    

if __name__ == "__main__":
    main()




