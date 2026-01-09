
import pandas as pd
from typing import List, Optional

def aggregate_predictions_weighted(
    predictions_df: pd.DataFrame,
    similarity_df: pd.DataFrame,
    product_col: str = 'product_name',
    machine_key_col: str = 'machine_key',
    similarity_col: str = 'similarity'
) -> pd.DataFrame:
    """
    Aggregates predictions using a weighted average based on similarity scores.
    
    Args:
        predictions_df: DataFrame containing predictions (must have pred_week_* columns)
        similarity_df: DataFrame containing machine_key and similarity scores
        product_col: Column name to group by (default: 'product_name')
        machine_key_col: Column name for machine identifier (default: 'machine_key')
        similarity_col: Column name for similarity score (default: 'similarity')
        
    Returns:
        DataFrame with one row per product, containing weighted average predictions.
    """
    
    # Work on a copy to avoid modifying the original dataframe
    df = predictions_df.copy()
    
    # 1. Merge similarity scores into the predictions dataframe
    # Check if similarity is already in df to avoid duplicate columns/suffixes
    if similarity_col not in df.columns:
        if machine_key_col not in df.columns:
            raise ValueError(f"predictions_df must contain '{machine_key_col}' column")
        if machine_key_col not in similarity_df.columns:
            raise ValueError(f"similarity_df must contain '{machine_key_col}' column")
            
        df = df.merge(
            similarity_df[[machine_key_col, similarity_col]], 
            on=machine_key_col, 
            how='left'
        )
    
    # Ensure similarity is numeric and fill NaN with 0 (or drop?)
    # If a machine has no similarity score, it shouldn't contribute to the average if we strictly follow weights.
    # But filtering should have been done beforehand.
    df[similarity_col] = pd.to_numeric(df[similarity_col], errors='coerce').fillna(0)

    # 2. Identify prediction columns
    pred_cols = [col for col in df.columns if col.startswith('pred_week_')]
    
    if not pred_cols:
        # Fallback if no pred_week columns found, maybe return simple aggregation or empty?
        # But assuming this function is specifically for this task:
        return pd.DataFrame()

    # 3. Calculate weighted sums
    # Multiply each prediction by the machine's similarity weight
    for col in pred_cols:
        df[f'{col}_weighted'] = df[col] * df[similarity_col]

    # 4. Define aggregation rules
    # - Weighted columns: sum
    # - Similarity: sum (to divide by later)
    # - Everything else: first (keep metadata)
    agg_rules = {
        col: 'first' for col in df.columns 
        if col != product_col 
        and col not in pred_cols 
        and not col.endswith('_weighted') 
        and col != similarity_col
    }
    
    for col in pred_cols:
        agg_rules[f'{col}_weighted'] = 'sum'
        
    agg_rules[similarity_col] = 'sum'

    # 5. Group and Aggregate
    if product_col not in df.columns:
         raise ValueError(f"predictions_df must contain '{product_col}' column")
         
    aggregated_predictions = df.groupby(product_col).agg(agg_rules).reset_index()

    # 6. Normalize to get the final Weighted Average
    for col in pred_cols:
        # Weighted Sum / Sum of Weights
        # Handle division by zero if sum of weights is 0
        similarity_sum = aggregated_predictions[similarity_col]
        weighted_sum = aggregated_predictions[f'{col}_weighted']
        
        aggregated_predictions[col] = weighted_sum / similarity_sum
        
        # Fill NaNs (div by zero) with 0? Or keep as NaN?
        # Usually 0 sales if no weights found is safer, or keep NaN to indicate no data.
        # Let's keep NaN as it's more truthful, but maybe fill with 0 if requested.
        
        # Remove the temporary weighted sum column
        del aggregated_predictions[f'{col}_weighted']

    # Remove the 'similarity' sum column and machine_key (since it's aggregated)
    del aggregated_predictions[similarity_col]
    if machine_key_col in aggregated_predictions.columns:
        # Since we took 'first', machine_key will be arbitrary. 
        # Typically we drop it after aggregation unless we want a sample machine.
        # But user asked to "preserve all other features".
        # 'first' keeps random machine metadata. That is what user agreed to.
        pass

    # 7. Sort by the first prediction week
    if pred_cols:
        primary_pred_col = sorted(pred_cols)[0]
        aggregated_predictions = aggregated_predictions.sort_values(primary_pred_col, ascending=False)

    return aggregated_predictions
