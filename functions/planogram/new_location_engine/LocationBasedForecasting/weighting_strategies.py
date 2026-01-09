"""
Weighting Strategies for Sales Prediction
==========================================

Different strategies for weighting reference machine sales by geographic similarity.
"""

# IMPORTS
import numpy as np
import pandas as pd
from typing import Dict, Any, List


def inverse_distance_weighting(
        geo_similarities: np.ndarray,
        sales_values: np.ndarray
) -> Dict[str, Any]:
    """
    Weight sales by inverse of distance (via similarity scores).

    Machines with higher similarity get proportionally higher weight.
    Simple and intuitive weighting approach.

    Args:
        geo_similarities: Array of similarity scores (0.0-1.0) for each reference machine
        sales_values: Array of sales values from each reference machine

    Returns:
        Dictionary with:
            - predicted_sales: Weighted average sales prediction
            - weights: Individual weights for each machine
            - total_weight: Sum of weights (for normalization)
    """

    # EDGE CASE: NO POINTS
    if len(geo_similarities) == 0:
        return {
            'predicted_sales': 0.0,
            'weights': np.array([]),
            'total_weight': 0.0
        }

    # NORMALIZE WEIGHTS
    total_weight = geo_similarities.sum()
    if total_weight == 0:
        return {
            'predicted_sales': 0.0,
            'weights': geo_similarities,
            'total_weight': 0.0
        }

    normalized_weights = geo_similarities / total_weight

    # WEIGHTED AVERAGE
    predicted_sales = (normalized_weights * sales_values).sum()

    # RETURN
    return {
        'predicted_sales': float(predicted_sales),
        'weights': normalized_weights,
        'total_weight': float(total_weight)
    }


def gaussian_weighting(
        geo_similarities: np.ndarray,
        sales_values: np.ndarray,
        sigma: float = 2.0
) -> Dict[str, Any]:
    """
    Weight sales using Gaussian kernel based on similarity.

    Provides smoother falloff than inverse distance, reducing influence of distant machines.

    Args:
        geo_similarities: Array of similarity scores for each reference machine
        sales_values: Array of sales values from each reference machine
        sigma: Gaussian kernel width parameter (larger = broader influence)

    Returns:
        Dictionary with prediction and weights
    """

    # EDGE CASE: NO POINTS
    if len(geo_similarities) == 0:
        return {
            'predicted_sales': 0.0,
            'weights': np.array([]),
            'total_weight': 0.0
        }

    # CONVERT SIMILARITY TO DISTANCE-LIKE MEASURE (0 = IDENTICAL, 1 = COMPLETELY DIFFERENT)
    distances = 1.0 - geo_similarities

    # GAUSSIAN KERNEL: EXP(-(DISTANCE^2) / (2 * SIGMA^2)
    weights = np.exp(-(distances ** 2) / (2 * sigma ** 2))

    total_weight = weights.sum()

    if total_weight == 0:
        return {
            'predicted_sales': 0.0,
            'weights': weights,
            'total_weight': 0.0
        }

    # NORMALIZE WEIGHTS
    normalized_weights = weights / total_weight
    predicted_sales = (normalized_weights * sales_values).sum()

    # RETURN
    return {
        'predicted_sales': float(predicted_sales),
        'weights': normalized_weights,
        'total_weight': float(total_weight)
    }


def top_k_weighting(
        geo_similarities: np.ndarray,
        sales_values: np.ndarray,
        k: int = 5
) -> Dict[str, Any]:
    """
    Use only the top k most similar machines, with equal or similarity-based weighting.

    Ignores distant machines entirely, focusing on most similar locations.

    Args:
        geo_similarities: Array of similarity scores for each reference machine
        sales_values: Array of sales values from each reference machine
        k: Number of top similar machines to use

    Returns:
        Dictionary with prediction and weights
    """

    # EDGE CASE: NO POINTS
    if len(geo_similarities) == 0:
        return {
            'predicted_sales': 0.0,
            'weights': np.array([]),
            'total_weight': 0.0,
            'k_used': 0
        }

    # GET TOP-K INDEXES
    k_actual = min(k, len(geo_similarities))
    top_k_indices = np.argsort(geo_similarities)[-k_actual:]

    # CREATE WEIGHTS ARRAY
    weights = np.zeros_like(geo_similarities)
    weights[top_k_indices] = geo_similarities[top_k_indices]

    #  NORMALIZE WEIGHTS
    total_weight = weights.sum()
    if total_weight == 0:
        return {
            'predicted_sales': 0.0,
            'weights': weights,
            'total_weight': 0.0,
            'k_used': 0
        }

    normalized_weights = weights / total_weight

    # WEIGHTED AVRAGE
    predicted_sales = (normalized_weights * sales_values).sum()

    # RETURN
    return {
        'predicted_sales': float(predicted_sales),
        'weights': normalized_weights,
        'total_weight': float(total_weight),
        'k_used': k_actual
    }


def adaptive_weighting(
        geo_similarities: np.ndarray,
        sales_values: np.ndarray,
        similarity_threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Adaptive weighting that switches strategy based on similarity distribution.

    - If many high-similarity machines: use top-k
    - If moderate similarities: use Gaussian
    - If low similarities: use inverse distance

    Args:
        geo_similarities: Array of similarity scores
        sales_values: Array of sales values
        similarity_threshold: Threshold for "high similarity"

    Returns:
        Dictionary with prediction, weights, and strategy used
    """

    # EDGE CASE: NO POINTS
    if len(geo_similarities) == 0:
        return {
            'predicted_sales': 0.0,
            'weights': np.array([]),
            'total_weight': 0.0,
            'strategy': 'none'
        }

    # COUNT HIGH-SIMILARITY MACHINES
    high_sim_count = (geo_similarities >= similarity_threshold).sum()

    # CHOOSE WEIGHTING STRATEGY
    if high_sim_count >= 3:
        # Many high-similarity machines: use top-k
        result = top_k_weighting(geo_similarities, sales_values, k=5)
        result['strategy'] = 'top_k'
    elif geo_similarities.mean() >= 0.4:
        # Moderate similarities: use Gaussian
        result = gaussian_weighting(geo_similarities, sales_values, sigma=1.5)
        result['strategy'] = 'gaussian'
    else:
        # Low similarities: use inverse distance
        result = inverse_distance_weighting(geo_similarities, sales_values)
        result['strategy'] = 'inverse_distance'

    return result


# STRATEGIES FOR EASY SELECTION
STRATEGIES = {
    'inverse_distance': inverse_distance_weighting,
    'gaussian': gaussian_weighting,
    'top_k': top_k_weighting,
    'adaptive': adaptive_weighting
}


def get_strategy(strategy_name: str):
    """
    Get weighting strategy function by name.

    Args:
        strategy_name: Name of strategy ('inverse_distance', 'gaussian', 'top_k', 'adaptive')

    Returns:
        Strategy function

    Raises:
        ValueError: If strategy name not found
    """
    if strategy_name not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy_name}. "
                         f"Available: {list(STRATEGIES.keys())}")

    return STRATEGIES[strategy_name]
