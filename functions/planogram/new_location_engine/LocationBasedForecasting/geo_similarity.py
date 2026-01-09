"""
Geographic Similarity Scoring

Calculate overall geographic similarity between two locations based on
multiple factors: direct distance, ICA proximity similarity, and company proximity similarity.
"""

# IMPORT
import numpy as np
from typing import Dict, Any, Optional
import pandas as pd

from .geo_utils import haversine_distance
from .ica_proximity import compare_ica_proximity, calculate_ica_proximity_score
from .company_proximity import compare_company_proximity, calculate_company_proximity_score


def calculate_geo_similarity(
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float,
        ica_stores_df: pd.DataFrame,
        companies_df: pd.DataFrame,
        distance_weight: float = 0.0,
        ica_weight: float = 0.3,
        company_weight: float = 0.7,
        max_distance_km: float = 50.0,
        target_ica_score: Optional[Dict[str, Any]] = None,
        target_company_score: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Calculate overall geographic similarity between two locations.

    Similarity is based on:
    1. Direct distance (closer = more similar)
    2. ICA proximity similarity (similar ICA access = more similar)
    3. Company proximity similarity (similar business density = more similar)

    Args:
        lat1, lon1: First location coordinates
        lat2, lon2: Second location coordinates
        ica_stores_df: DataFrame with ICA store locations
        companies_df: DataFrame with company locations
        distance_weight: Weight for direct distance component (default: 0.0)
        ica_weight: Weight for ICA proximity component (default: 0.2)
        company_weight: Weight for company proximity component (default: 0.8)
        max_distance_km: Maximum distance to consider (beyond this, similarity = 0)
        target_ica_score: Optional pre-calculated ICA score for location 1
        target_company_score: Optional pre-calculated company score for location 1

    Returns:
        Dictionary with:
            - similarity: Overall similarity score (0.0-1.0)
            - distance_km: Direct distance between locations
            - distance_similarity: Distance-based similarity component
            - ica_similarity: ICA proximity similarity component
            - company_similarity: Company proximity similarity component
            - breakdown: Weighted contribution of each component
    """

    # CALCULATE DIRECT DISTANCE
    distance = haversine_distance(lat1, lon1, lat2, lon2)

    # DETERMINE DISTANCE SIMILARITY WEIGHT
    distance_similarity = np.exp(-distance / max_distance_km * 3.0)

    # CALCULATE ICA PROXIMITY SIMILARITY
    ica_similarity = compare_ica_proximity(
        lat1, lon1, lat2, lon2,
        ica_stores_df,
        score1=target_ica_score
    )

    # CALCULATE COMPANY PROXIMITY SIMILARITY
    company_similarity = compare_company_proximity(
        lat1, lon1, lat2, lon2,
        companies_df,
        score1=target_company_score
    )

    # NORMALIZE WEIGHTS TO SUM TO 1.0
    total_weight = distance_weight + ica_weight + company_weight
    norm_distance_weight = distance_weight / total_weight
    norm_ica_weight = ica_weight / total_weight
    norm_company_weight = company_weight / total_weight

    # CALCULATE OVERALL SIMILARITY
    overall_similarity = (
            norm_distance_weight * distance_similarity +
            norm_ica_weight * ica_similarity +
            norm_company_weight * company_similarity
    )

    # RETURN
    return {
        'similarity': float(overall_similarity),
        'distance_km': float(distance),
        'distance_similarity': float(distance_similarity),
        'ica_similarity': float(ica_similarity),
        'company_similarity': float(company_similarity),
        'breakdown': {
            'distance_contribution': float(norm_distance_weight * distance_similarity),
            'ica_contribution': float(norm_ica_weight * ica_similarity),
            'company_contribution': float(norm_company_weight * company_similarity)
        }
    }


def calculate_similarity_to_multiple(
        target_lat: float,
        target_lon: float,
        reference_locations_df: pd.DataFrame,
        ica_stores_df: pd.DataFrame,
        companies_df: pd.DataFrame,
        **kwargs
) -> pd.DataFrame:
    """
    Calculate geo-similarity from a target location to multiple reference locations.

    Useful for finding the most similar existing machine locations to a new location.

    Args:
        target_lat, target_lon: Target location coordinates
        reference_locations_df: DataFrame with reference locations (must have 'latitude', 'longitude')
        ica_stores_df: DataFrame with ICA stores
        companies_df: DataFrame with companies
        **kwargs: Additional arguments passed to calculate_geo_similarity

    Returns:
        DataFrame with reference locations and their similarity scores, sorted by similarity (descending)

    Example:
        >>> similarities = calculate_similarity_to_multiple(
        ...     59.3293, 18.0686,  # New location
        ...     existing_machines_df,
        ...     ica_stores_df,
        ...     companies_df
        ... )
        >>> print("Most similar existing locations:")
        >>> print(similarities.head())
    """
    similarities = []

    # PRE-CALCULATE TARGET SCORES ONCE
    # This avoids redundant re-calculation for every reference machine
    target_ica_score = calculate_ica_proximity_score(target_lat, target_lon, ica_stores_df)
    target_company_score = calculate_company_proximity_score(target_lat, target_lon, companies_df)

    # CALCULATE SIMILARITY TO EACH REFERENCE LOCATION
    for idx, location in reference_locations_df.iterrows():

        # CALCULATE
        sim_result = calculate_geo_similarity(
            target_lat, target_lon,
            location['location'].get('latitude'), location['location'].get('longitude'),
            ica_stores_df,
            companies_df,
            target_ica_score=target_ica_score,
            target_company_score=target_company_score,
            **kwargs
        )

        # APPEND RESULTS
        similarities.append({
            'reference_index': idx,
            'similarity': sim_result['similarity'],
            'distance_km': sim_result['distance_km'],
            'distance_similarity': sim_result['distance_similarity'],
            'ica_similarity': sim_result['ica_similarity'],
            'company_similarity': sim_result['company_similarity']
        })

    # UPDATE RETURN
    result_df = pd.DataFrame(similarities)

    # MERGE WITH REFERENCE LOCATIONS
    # Reset index of reference locations to match list order
    ref_reset = reference_locations_df.reset_index(drop=True)
    
    result_df = pd.concat([
        result_df,
        ref_reset
    ], axis=1)

    # CLEAN UP COLUMNS
    # Remove 'machine_id' if present (migration cleanup)
    if 'machine_id' in result_df.columns:
        result_df = result_df.drop(columns=['machine_id'])

    # SORT BY SIMILARITY
    result_df = result_df.sort_values('similarity', ascending=False)

    # RETURN
    return result_df.reset_index(drop=True)


def get_top_similar_locations(
        target_lat: float,
        target_lon: float,
        reference_locations_df: pd.DataFrame,
        ica_stores_df: pd.DataFrame,
        companies_df: pd.DataFrame,
        top_k: int = 10,
        min_similarity: float = 0.3
) -> pd.DataFrame:
    """
    Get top k most similar locations, optionally filtered by minimum similarity.

    Args:
        target_lat, target_lon: Target location coordinates
        reference_locations_df: DataFrame with reference locations
        ica_stores_df: DataFrame with ICA stores
        companies_df: DataFrame with companies
        top_k: Number of top similar locations to return
        min_similarity: Minimum similarity threshold (0.0-1.0)

    Returns:
        DataFrame with top k similar locations
    """

    # CALCULATE SIMILARITY TO ALL REFERENCE LOCATIONS
    all_similarities = calculate_similarity_to_multiple(
        target_lat, target_lon,
        reference_locations_df,
        ica_stores_df,
        companies_df
    )

    # FILTER BY MINIMUM SIMILARITY
    filtered = all_similarities[all_similarities['similarity'] >= min_similarity]

    # RETURN TOP K
    return filtered.head(top_k)
