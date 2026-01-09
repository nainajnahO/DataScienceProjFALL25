import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, TypeAlias

# This module is used to calculate the location score for a given product and location type.
# The location score is calculated based on the product performance metrics for the given location type.
# The metrics are calculated based on the sales data and the location type.
# The location score is used to determine the best location for a given product.

# This is used to calculate the score for an entire machine given a 
# specific location aswell.

# For example, if we have a machine in a gym and we want to calculate the 
# score for that machine, we can use this function to calculate the score for 
# that machine given the location type 'gym'. Products, for example like 
# protein bars, are more likely to be sold in a gym than in a supermarket, 
# and thus the score for that machine would be higher. Compared to Coca-Cola
# which is more likely to be sold in a supermarket than in a gym. 

# Scoring Weights
# The weights are used to calculate the location score.
WEIGHT_FREQUENCY = 0.40 # The frequency of the product in the location
WEIGHT_REVENUE_PER_MACHINE = 0.30 # The revenue per machine in the location
WEIGHT_TOTAL_REVENUE = 0.20 # The total revenue in the location
WEIGHT_PENETRATION = 0.10 # The penetration of the product in the location

DEFAULT_FALLBACK_SCORE = 0.5 # The default fallback score

# Type Alias for the Location Model (just a dictionary of performance DataFrames)
LocationModel: TypeAlias = Dict[str, pd.DataFrame]

# This function is used to normalize the scores to a 0-1 range.
def normalize_scores(scores: pd.Series, method: str = 'min_max') -> pd.Series:

    if scores.empty:
        return scores

    if method == 'min_max':
        min_score = scores.min()
        max_score = scores.max()
        if max_score == min_score:
            return pd.Series([0.5] * len(scores), index=scores.index)
        return (scores - min_score) / (max_score - min_score)
    elif method == 'percentile':
        ranked = scores.rank(pct=True, method='max')
        if len(ranked) > 0:
            max_rank = ranked.max()
            if max_rank < 1.0:
                ranked = ranked / max_rank
        return ranked
    else:
        raise ValueError(f"Unknown normalization method: {method}")

# This function is used to analyze the product performance by location.
def analyze_product_performance_by_location(
    sales_df: pd.DataFrame,
    location_type: str,
    min_transactions: int = 10,
    pre_filtered: bool = False
) -> pd.DataFrame:
   # This function is used to analyze the product performance by location.
    if not pre_filtered:
        if 'machine_eva_group' not in sales_df.columns:
            raise ValueError("Sales data must contain 'machine_eva_group' column")
        
        location_sales = sales_df[sales_df['machine_eva_group'] == location_type].copy()
    else:
        # Data is already filtered (from groupby), just use it directly
        location_sales = sales_df.copy()
    
    if len(location_sales) == 0:
        return pd.DataFrame()
    
    # Ensure purchase_price_kr is filled
    if 'purchase_price_kr' in location_sales.columns:
        location_sales['purchase_price_kr'] = location_sales['purchase_price_kr'].fillna(location_sales['price'])
    else:
        location_sales['purchase_price_kr'] = location_sales['price']

    location_sales['profit'] = location_sales['price'] - location_sales['purchase_price_kr']
    
    def get_most_common(x):
        if len(x) == 0:
            return None
        x_clean = x.dropna()
        if len(x_clean) == 0:
            return None
        # value_counts().index[0] is faster than mode() for this purpose usually
        try:
            return x_clean.value_counts().index[0]
        except IndexError:
            return None
    
    # Aggregation
    product_metrics = location_sales.groupby('product_name').agg({
        'price': ['count', 'sum', 'mean'],
        'profit': 'sum',
        'machine_key': 'nunique',
        'ean': 'first',
        'category': get_most_common,
        'subcategory': get_most_common,
        'provider': get_most_common
    }).reset_index()
    
    # Flatten columns
    product_metrics.columns = [
        'product_name', 'transaction_count', 'total_revenue', 'avg_price', 
        'total_profit', 'unique_machines', 'ean', 'category', 'subcategory', 'provider'
    ]
    
    # Calculate derived metrics
    product_metrics['transaction_frequency'] = (
        product_metrics['transaction_count'] / product_metrics['unique_machines']
    )
    product_metrics['avg_revenue_per_machine'] = (
        product_metrics['total_revenue'] / product_metrics['unique_machines']
    )
    product_metrics['avg_profit_per_machine'] = (
        product_metrics['total_profit'] / product_metrics['unique_machines']
    )
    
    # Filter by min transactions
    product_metrics = product_metrics[
        product_metrics['transaction_count'] >= min_transactions
    ].copy()
    
    product_metrics = product_metrics.sort_values('total_revenue', ascending=False)
    
    return product_metrics
# This function is used to train the location model. It returns a dictionary of DataFrames for each location type, 
# this dictionary contains the product metrics for each location type. 
# The metrics are calculated based on the sales data and the location type.
def train_location_model(
    sales_df: pd.DataFrame,
    min_transactions: int = 10
) -> LocationModel:
   
    # Validate required columns exist
    required_cols = ['machine_eva_group', 'product_name', 'price', 'machine_key', 
                     'ean', 'category', 'subcategory', 'provider']
    missing_cols = [col for col in required_cols if col not in sales_df.columns]
    if missing_cols:
        raise ValueError(f"Sales data missing required columns: {missing_cols}")
    
    # Filter out rows without machine_eva_group (needed for location scoring)
    merged_sales = sales_df[sales_df['machine_eva_group'].notna()].copy()
    
    if merged_sales.empty:
        return {}

    # Ensure purchase_price_kr exists and is filled
    if 'purchase_price_kr' not in merged_sales.columns:
        merged_sales['purchase_price_kr'] = merged_sales['price']
    else:
        merged_sales['purchase_price_kr'] = merged_sales['purchase_price_kr'].fillna(merged_sales['price'])

    # OPTIMIZATION: Group by location_type once instead of filtering 10 times
    # This is much faster than filtering the entire DataFrame for each location type
    location_groups = merged_sales.groupby('machine_eva_group', observed=True)
    location_types = merged_sales['machine_eva_group'].unique().tolist()
    
    results = {}
    
    for location_type, location_sales in location_groups:
        # Use the pre-grouped data instead of filtering
        product_metrics = analyze_product_performance_by_location(
            location_sales, location_type, min_transactions, pre_filtered=True
        )
        
        if len(product_metrics) == 0:
            results[location_type] = pd.DataFrame()
            continue
        
        # Calculate scores
        product_metrics['frequency_score'] = normalize_scores(
            product_metrics['transaction_frequency'], method='percentile'
        )
        product_metrics['revenue_per_machine_score'] = normalize_scores(
            product_metrics['avg_revenue_per_machine'], method='percentile'
        )
        product_metrics['total_revenue_score'] = normalize_scores(
            product_metrics['total_revenue'], method='percentile'
        )
        product_metrics['penetration_score'] = normalize_scores(
            product_metrics['unique_machines'], method='percentile'
        )
        
        product_metrics['location_fit_score'] = (
            WEIGHT_FREQUENCY * product_metrics['frequency_score'] +
            WEIGHT_REVENUE_PER_MACHINE * product_metrics['revenue_per_machine_score'] +
            WEIGHT_TOTAL_REVENUE * product_metrics['total_revenue_score'] +
            WEIGHT_PENETRATION * product_metrics['penetration_score']
        )
        
        product_metrics['location_fit_score'] = product_metrics['location_fit_score'].round(3)
        product_metrics = product_metrics.sort_values('location_fit_score', ascending=False)
        
        # Deduplicate by product_name just in case
        product_metrics = product_metrics.drop_duplicates(subset=['product_name'], keep='first')
        
        results[location_type] = product_metrics
    
    return results

# This function is used to predict the location scores for the given products and location types.
# It returns a DataFrame with the location scores for each product and location type.
# The calculation is based on the trained model from the train_location_model function.
def predict_location_scores(
    products_df: pd.DataFrame,
    location_types: List[str],
    trained_model: LocationModel
) -> pd.DataFrame:
  
    products_df = products_df.copy()
    results = trained_model
    
    # 1. Prepare Base Product Data
    # We need a master list of products with their metadata
    base_products = products_df[['product_name', 'category', 'subcategory', 'provider']].copy()
    base_products = base_products.drop_duplicates(subset=['product_name'], keep='first').set_index('product_name')

    # 2. Aggregate Scores into a single DataFrame
    score_frames = []
    for loc_type, df in results.items():
        if not df.empty:
            # Keep only relevant columns
            temp = df[['product_name', 'location_fit_score', 'category']].copy()
            temp['location_type'] = loc_type
            score_frames.append(temp)
    
    if not score_frames:
        # Return empty structure if no scores
        matrix = base_products.reset_index()
        for loc in location_types:
            matrix[loc] = DEFAULT_FALLBACK_SCORE
        return matrix

    all_scores = pd.concat(score_frames, ignore_index=True)

    # 3. Pivot to create Matrix (Product x Location Type)
    product_score_matrix = all_scores.pivot(
        index='product_name', 
        columns='location_type', 
        values='location_fit_score'
    )

    # 4. Calculate Category Averages (Category x Location Type)
    # Group by category and location_type, then unstack to get matrix
    category_avg_matrix = all_scores.groupby(['category', 'location_type'])['location_fit_score'].mean().unstack()

    # 5. Align with Base Products
    # Reindex to include all products from products_df
    final_matrix = product_score_matrix.reindex(base_products.index)

    # 6. Fill Missing Values
    # For each column (location type), fill NaNs
    for loc in location_types:
        if loc not in final_matrix.columns:
            final_matrix[loc] = np.nan
        
        # Mask for missing values in this column
        missing_mask = final_matrix[loc].isna()
        
        if missing_mask.any():
            # Map product -> category -> avg_score
            # Get categories for missing products
            missing_cats = base_products.loc[final_matrix.index[missing_mask], 'category']
            
            # Lookup average scores for these categories at this location
            if loc in category_avg_matrix.columns:
                fill_values = missing_cats.map(category_avg_matrix[loc])
            else:
                fill_values = pd.Series(np.nan, index=missing_cats.index)
            
            # Update the matrix
            final_matrix.loc[missing_mask, loc] = fill_values

    # 7. Final Fallback
    final_matrix = final_matrix.fillna(DEFAULT_FALLBACK_SCORE)

    # 8. Restore Metadata columns
    final_matrix = base_products.join(final_matrix)
    
    return final_matrix.reset_index()
