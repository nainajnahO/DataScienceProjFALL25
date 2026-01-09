# src/models/utils.py

import pandas as pd
import numpy as np
from typing import Dict, List
from sklearn.metrics import mean_absolute_error, r2_score


def calculate_weekly_metrics(predictions: pd.DataFrame, horizons: List[int]) -> pd.DataFrame:
    """
    Calculate per-week metrics across all horizons.
    
    Useful for understanding which weeks are hardest to predict.
    """
    metrics = []
    for horizon in horizons:
        pred_col = f'pred_week_{horizon}'
        actual_col = f'actual_week_{horizon}'
        
        mae = mean_absolute_error(predictions[actual_col], predictions[pred_col])
        rmse = np.sqrt(((predictions[actual_col] - predictions[pred_col])**2).mean())
        r2 = r2_score(predictions[actual_col], predictions[pred_col])
        
        metrics.append({
            'horizon': horizon,
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        })
    
    return pd.DataFrame(metrics)


def compare_strategies(
    predictions_dict: Dict[str, pd.DataFrame], 
    horizons: List[int] = [1, 2, 3, 4]
) -> pd.DataFrame:
    """
    Compare multiple forecasting strategies side-by-side.
    
    Args:
        predictions_dict: {'strategy_name': predictions_df, ...}
        horizons: List of weeks to evaluate
        
    Returns:
        Comparison DataFrame with MAE per strategy per horizon
    """
    comparison = []
    
    for strategy_name, predictions in predictions_dict.items():
        for horizon in horizons:
            pred_col = f'pred_week_{horizon}'
            actual_col = f'actual_week_{horizon}'
            
            mae = mean_absolute_error(predictions[actual_col], predictions[pred_col])
            r2 = r2_score(predictions[actual_col], predictions[pred_col])
            
            comparison.append({
                'strategy': strategy_name,
                'horizon': f'Week +{horizon}',
                'mae': mae,
                'r2': r2
            })
    
    return pd.DataFrame(comparison)


def identify_high_error_products(
    predictions: pd.DataFrame,
    test_df: pd.DataFrame,
    threshold: float = 5.0,
    horizon: int = 1
) -> pd.DataFrame:
    """
    Identify products with consistently high prediction errors.
    
    Args:
        predictions: Predictions DataFrame
        test_df: Original test DataFrame (for product metadata)
        threshold: MAE threshold for "high error"
        horizon: Which week to analyze
        
    Returns:
        DataFrame with high-error products and their characteristics
    """
    pred_col = f'pred_week_{horizon}'
    actual_col = f'actual_week_{horizon}'
    
    predictions['error'] = (predictions[actual_col] - predictions[pred_col]).abs()
    high_error_mask = predictions['error'] > threshold
    
    high_error_indices = predictions[high_error_mask].index
    high_error_products = test_df.loc[high_error_indices, [
        'ean', 'product_name', 'category', 'subcategory', 
        'machine_key', 'machine_group', 'price', 
        'product_age_weeks', 'is_new_launch_product'
    ]].copy()
    
    high_error_products['error'] = predictions.loc[high_error_indices, 'error'].values
    high_error_products['actual_sales'] = predictions.loc[high_error_indices, actual_col].values
    high_error_products['predicted_sales'] = predictions.loc[high_error_indices, pred_col].values
    
    return high_error_products.sort_values('error', ascending=False)


def create_prediction_intervals(
    predictions: pd.DataFrame,
    horizons: List[int] = [1, 2, 3, 4],
    confidence: float = 0.95
) -> pd.DataFrame:
    """
    Create prediction intervals using residual bootstrap.
    
    Provides upper/lower bounds for predictions (useful for inventory planning).
    
    Args:
        predictions: Predictions DataFrame with actual and predicted values
        horizons: Which weeks to compute intervals for
        confidence: Confidence level (0.95 = 95% interval)
        
    Returns:
        DataFrame with lower_bound and upper_bound columns
    """
    results = predictions.copy()
    alpha = 1 - confidence
    
    for horizon in horizons:
        pred_col = f'pred_week_{horizon}'
        actual_col = f'actual_week_{horizon}'
        
        # Calculate residuals
        residuals = results[actual_col] - results[pred_col]
        
        # Bootstrap residuals
        lower_quantile = residuals.quantile(alpha / 2)
        upper_quantile = residuals.quantile(1 - alpha / 2)
        
        # Add intervals to predictions
        results[f'lower_week_{horizon}'] = results[pred_col] + lower_quantile
        results[f'upper_week_{horizon}'] = results[pred_col] + upper_quantile
        
        # Ensure non-negative
        results[f'lower_week_{horizon}'] = results[f'lower_week_{horizon}'].clip(lower=0)
    
    return results


def generate_business_recommendations(
    predictions: pd.DataFrame,
    test_df: pd.DataFrame,
    low_sales_threshold: float = 3.0,
    high_sales_threshold: float = 15.0
) -> pd.DataFrame:
    """
    Generate actionable business recommendations based on predictions.
    
    Args:
        predictions: Predictions DataFrame
        test_df: Original test DataFrame
        low_sales_threshold: Below this = consider discontinuing
        high_sales_threshold: Above this = consider expanding
        
    Returns:
        DataFrame with product-level recommendations
    """
    # Use 4-week average prediction
    avg_prediction = predictions[[f'pred_week_{h}' for h in [1, 2, 3, 4]]].mean(axis=1)
    
    recommendations = test_df[['ean', 'product_name', 'machine_key', 'category']].copy()
    recommendations['predicted_4week_avg'] = avg_prediction.values
    
    def get_recommendation(pred):
        if pred < low_sales_threshold:
            return 'DISCONTINUE'
        elif pred > high_sales_threshold:
            return 'EXPAND'
        else:
            return 'MONITOR'
    
    recommendations['action'] = recommendations['predicted_4week_avg'].apply(get_recommendation)
    
    # Add confidence (based on prediction variability)
    pred_std = predictions[[f'pred_week_{h}' for h in [1, 2, 3, 4]]].std(axis=1)
    recommendations['confidence'] = pd.cut(
        pred_std, 
        bins=[0, 1, 3, np.inf], 
        labels=['HIGH', 'MEDIUM', 'LOW']
    )
    
    return recommendations.sort_values('predicted_4week_avg', ascending=False)