"""
Geographic Utility Functions

Core distance calculations and geographic utilities for location-based forecasting.
"""

# IMPORTS
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional


def haversine_distance(
        lat1: float | np.ndarray,
        lon1: float | np.ndarray,
        lat2: float | np.ndarray,
        lon2: float | np.ndarray
) -> float | np.ndarray:
    """
    Calculate the great circle distance between two points on Earth.

    Uses the Haversine formula to calculate the distance in kilometers.
    Supports both scalar and vectorized (NumPy array) inputs.

    Args:
        lat1: Latitude of first point(s)
        lon1: Longitude of first point(s)
        lat2: Latitude of second point(s)
        lon2: Longitude of second point(s)

    Returns:
        Distance in kilometers (scalar or array depending on input)
    """
    # EARTHS RADIUS IN KILOMETERS
    R = 6371.0

    # CONVERT DEGREES TO RADIANS
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    # DIFFERENCES
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    # HAVERSINE FORMULA
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance = R * c

    return distance


def find_nearest_points(
        target_lat: float,
        target_lon: float,
        points_df: pd.DataFrame,
        k: int = 5,
        lat_col: str = 'latitude',
        lon_col: str = 'longitude'
) -> pd.DataFrame:
    """
    Find the k nearest points to a target location.

    Args:
        target_lat: Target latitude
        target_lon: Target longitude
        points_df: DataFrame containing points with lat/lon columns
        k: Number of nearest points to return
        lat_col: Name of latitude column in points_df
        lon_col: Name of longitude column in points_df

    Returns:
        DataFrame with k nearest points, sorted by distance, with added 'distance_km' column
    """

    # EDGE CASE
    if points_df.empty:
        return pd.DataFrame()

    # CALCULATE DISTANCES VECTORIZED
    distances = haversine_distance(
        target_lat,
        target_lon,
        points_df[lat_col].values,
        points_df[lon_col].values
    )

    # ADD DISTANCE COLUMN
    result = points_df.copy()
    result['distance_km'] = distances

    # SORT BY DISTANCE AND RETURN TOP K
    result = result.sort_values('distance_km').head(k)

    return result.reset_index(drop=True)


def find_points_within_radius(
        target_lat: float,
        target_lon: float,
        points_df: pd.DataFrame,
        radius_km: float,
        lat_col: str = 'latitude',
        lon_col: str = 'longitude'
) -> pd.DataFrame:
    """
    Find all points within a specified radius of a target location.

    Args:
        target_lat: Target latitude
        target_lon: Target longitude
        points_df: DataFrame containing points with lat/lon coordinates
        radius_km: Search radius in kilometers
        lat_col: Name of latitude column in points_df
        lon_col: Name of longitude column in points_df

    Returns:
        DataFrame with points within radius, with added 'distance_km' column
    """

    # EDGE CASE: NO POINTS
    if points_df.empty:
        return pd.DataFrame()

    # OPTIMIZATION: BOUNDING BOX FILTERING
    # This acts as a coarse filter to reduce the number of haversine calculations
    lat_delta = radius_km / 111.0  # Approx 111km per degree latitude
    # Longitude delta depends on latitude, use max possible delta (at higher lat) or just be generous
    # At 60 deg lat (roughly Stockholm), cos(60) = 0.5, so 1 deg lon = 55km.
    # Be safe and use a slightly larger box
    lon_delta = radius_km / (111.0 * np.cos(np.radians(target_lat))) * 1.5

    min_lat, max_lat = target_lat - lat_delta, target_lat + lat_delta
    min_lon, max_lon = target_lon - lon_delta, target_lon + lon_delta

    # Filter candidates
    candidates = points_df[
        (points_df[lat_col] >= min_lat) &
        (points_df[lat_col] <= max_lat) &
        (points_df[lon_col] >= min_lon) &
        (points_df[lon_col] <= max_lon)
    ].copy()

    if candidates.empty:
        return pd.DataFrame()

    # CALCULATE DISTANCES VECTORIZED ON CANDIDATES
    distances = haversine_distance(
        target_lat,
        target_lon,
        candidates[lat_col].values,
        candidates[lon_col].values
    )

    candidates['distance_km'] = distances

    # FILTER BY PRECISE RADIUS AND SORT
    result = candidates[candidates['distance_km'] <= radius_km].sort_values('distance_km')

    return result.reset_index(drop=True)


def calculate_distance_matrix(
        locations_df: pd.DataFrame,
        lat_col: str = 'latitude',
        lon_col: str = 'longitude'
) -> np.ndarray:
    """
    Calculate pairwise distance matrix for a set of locations.

    Args:
        locations_df: DataFrame with location coordinates
        lat_col: Name of latitude column
        lon_col: Name of longitude column

    Returns:
        NxN numpy array where element [i,j] is distance from location i to j in km
    """

    # CALCULATE DISTANCES
    n = len(locations_df)
    distance_matrix = np.zeros((n, n))

    # CALCULATE DISTANCES BETWEEN ALL PAIRS OF LOCATIONS
    for i in range(n):
        for j in range(i + 1, n):
            dist = haversine_distance(
                locations_df.iloc[i][lat_col],
                locations_df.iloc[i][lon_col],
                locations_df.iloc[j][lat_col],
                locations_df.iloc[j][lon_col]
            )
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist

    # RETURN
    return distance_matrix


def get_bounding_box(
        lat: float,
        lon: float,
        radius_km: float
) -> Tuple[float, float, float, float]:
    """
    Get bounding box coordinates for a circular area.

    Returns approximate lat/lon bounds for filtering before precise distance calculations.
    Useful for performance optimization with large datasets.

    Args:
        lat: Center latitude
        lon: Center longitude
        radius_km: Radius in kilometers

    Returns:
        Tuple of (min_lat, max_lat, min_lon, max_lon)
    """

    # ROUGH APPROXIMATION: 1 DEGREE LATITUDE = 111 KM
    # LONGITUDE VARIES BY LATITUDE
    lat_delta = radius_km / 111.0
    lon_delta = radius_km / (111.0 * np.cos(np.radians(lat)))

    min_lat = lat - lat_delta
    max_lat = lat + lat_delta
    min_lon = lon - lon_delta
    max_lon = lon + lon_delta

    # RETURN
    return min_lat, max_lat, min_lon, max_lon
