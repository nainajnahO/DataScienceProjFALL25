"""
ICA Proximity Scoring
=====================

Calculate proximity scores based on distance to ICA supermarkets.
"""

# IMPORTS
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from .geo_utils import find_nearest_points
from .config import MAX_ICA_DISTANCE_KM


def calculate_ica_proximity_score(
        latitude: float,
        longitude: float,
        ica_stores_df: pd.DataFrame,
        k_nearest: int = 3,
        max_distance_km: float = MAX_ICA_DISTANCE_KM
) -> Dict[str, Any]:
    """
    Calculate ICA proximity score for a location.

    Score is based on the k nearest ICA stores, weighted by inverse distance.
    Locations closer to ICA stores get higher scores.

    Args:
        latitude: Target location latitude
        longitude: Target location longitude
        ica_stores_df: DataFrame with ICA store locations
        k_nearest: Number of nearest ICA stores to consider (default: 3)
        max_distance_km: Maximum distance to consider (beyond this, score = 0)

    Returns:
        Dictionary with:
            - score: Normalized proximity score (0.0-1.0)
            - nearest_stores: List of k nearest ICA stores with distances
            - avg_distance_km: Average distance to k nearest stores

    Example:
    """

    # EDGE CASE: NO POINTS
    if ica_stores_df.empty:
        return {
            'score': 0.0,
            'nearest_stores': [],
            'avg_distance_km': None
        }

    # FIND K NEAREST ICA STORES
    nearest = find_nearest_points(
        latitude, longitude,
        ica_stores_df,
        k=k_nearest
    )

    # EDGE CASE: NO NEAREST STORES
    if nearest.empty:
        return {
            'score': 0.0,
            'nearest_stores': [],
            'avg_distance_km': None
        }

    """
    CALCULATE WEIGHTED SCORE USING INVERSE DISTANCE (WITH NORMALIZATION)
    FORMULA: score = sum(1 / (1 + distance_i)) / k
    """

    distances = nearest['distance_km'].values
    avg_distance = distances.mean()

    # IF ALL STORES ARE TOO FAR AWAY, RETURN ZERO
    if distances.min() > max_distance_km:
        return {
            'score': 0.0,
            'nearest_stores': nearest.to_dict('records'),
            'avg_distance_km': float(avg_distance)
        }

    # CALCULATE INVERSE DISTANCE WEIGHTS
    # ADD 1 TO DISTANCE TO AVOID DIVISION BY ZERO AND REDUCE EXTREME WEIGHTS
    weights = 1.0 / (1.0 + distances)
    raw_proximity_sum = weights.sum()

    # NORMALIZE BY SUM OF WEIGHTS AT IDEAL DISTANCE (VERY CLOSE, ~0.1km)
    # THIS ENSURES SCORE APPROACHES 1.0 FOR VERY CLOSE STORES
    ideal_weights = 1.0 / (1.0 + 0.1)
    normalized_score = raw_proximity_sum / (k_nearest * ideal_weights)

    # CLIP RANGE TO 1 (NECESSARY?)
    score = min(normalized_score, 1.0)

    # APPLY DISTANCE DECAY: REDUCE SCORE OF STORES BEYOND
    decay_factor = np.exp(-avg_distance / max_distance_km)
    score = score * decay_factor

    # RETUN
    return {
        'score': float(score),
        'raw_proximity_sum': float(raw_proximity_sum),
        'nearest_stores': nearest.to_dict('records'),
        'avg_distance_km': float(avg_distance)
    }


def calculate_ica_density_score(
        latitude: float,
        longitude: float,
        ica_stores_df: pd.DataFrame,
        radius_km: float = 5.0
) -> Dict[str, Any]:
    """
    Calculate ICA density score based on number of stores within radius.

    Alternative to k-nearest approach. Useful for understanding local market saturation.

    Args:
        latitude: Target location latitude
        longitude: Target location longitude
        ica_stores_df: DataFrame with ICA store locations
        radius_km: Search radius in kilometers

    Returns:
        Dictionary with:
            - score: Density-based score (0.0-1.0)
            - store_count: Number of ICA stores within radius
            - stores: List of stores within radius
    """
    from .geo_utils import find_points_within_radius

    # EDGE CASE: NO POINTS
    if ica_stores_df.empty:
        return {
            'score': 0.0,
            'store_count': 0,
            'stores': []
        }

    # FIND STORES WITHIN RADIUS
    nearby = find_points_within_radius(
        latitude, longitude,
        ica_stores_df,
        radius_km=radius_km
    )

    # COUNT
    store_count = len(nearby)

    # NORMALIZE SCORE BASED ON TYPICAL URBAN DENSITY
    # ASSUME 5+ STORES WITHIN 5KM = MAXIMUM SCORE (TYPICAL URBAN AREA)
    # TODO: REASSESS ASSUMPTION/METHOD
    max_expected_stores = 5
    score = min(store_count / max_expected_stores, 1.0)

    # RETURN
    return {
        'score': float(score),
        'store_count': store_count,
        'stores': nearby.to_dict('records')
    }


def compare_ica_proximity(
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float,
        ica_stores_df: pd.DataFrame,
        k_nearest: int = 3,
        score1: Optional[Dict[str, Any]] = None
) -> float:
    """
    Compare ICA proximity similarity between two locations.

    Uses RELATIVE SIMILARITY:
    Comparses the raw "Inverse Distance Sum" to find true structural matches.
    Similarity = min(val1, val2) / max(val1, val2)

    Args:
        lat1, lon1: First location coordinates
        lat2, lon2: Second location coordinates
        ica_stores_df: DataFrame with ICA store locations
        k_nearest: Number of nearest stores to consider
        score1: Optional pre-calculated score for location 1

    Returns:
        Similarity score (1.0 = identical raw proximity structure)
    """
    if score1 is None:
        score1 = calculate_ica_proximity_score(lat1, lon1, ica_stores_df, k_nearest)

    score2 = calculate_ica_proximity_score(lat2, lon2, ica_stores_df, k_nearest)

    # EXTRACT RAW PROXIMITY SUMS
    val1 = score1.get('raw_proximity_sum', 0.0)
    val2 = score2.get('raw_proximity_sum', 0.0)

    # HANDLE ZERO CASE
    if val1 == 0 and val2 == 0:
        return 1.0
    if val1 == 0 or val2 == 0:
        return 0.0

    # CALCULATE RATIO
    similarity = min(val1, val2) / max(val1, val2)

    return float(similarity)
