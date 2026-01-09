"""
Company Proximity Scoring

Calculate proximity scores based on nearby companies and their employee counts.
Company proximity is used as a proxy for potential customer base and business activity.
"""

# IMPORTS
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from .geo_utils import find_points_within_radius


def calculate_company_proximity_score(
        latitude: float,
        longitude: float,
        companies_df: pd.DataFrame,
        radius_km: float = 2.0
) -> Dict[str, Any]:
    """
    Calculate company (WORKPLACE) proximity score for a location.

    Score is based on nearby companies weighted by their employee count.
    More employees nearby -> higher score (more potential customers).

    Args:
        latitude: Target location latitude
        longitude: Target location longitude
        companies_df: DataFrame with company locations and employee counts
        radius_km: Search radius in kilometers (default: 2km)

    Returns:
        Dictionary with:
            - score: Normalized proximity score (0.0-1.0)
            - total_employees: Total employees within radius
            - company_count: Number of companies within radius
            - companies: List of nearby companies
    """

    # EDGE CASE
    if companies_df.empty:
        return {
            'score': 0.0,
            'total_employees': 0,
            'company_count': 0,
            'companies': []
        }

    # FIND WORKPLACES WITHIN RADIUS
    nearby = find_points_within_radius(
        latitude, longitude,
        companies_df,
        radius_km=radius_km
    )

    # EDGE CASE
    if nearby.empty:
        return {
            'score': 0.0,
            'total_employees': 0,
            'company_count': 0,
            'companies': []
        }

    distances = nearby['distance_km'].values
    employees = nearby['employee_count'].values

    # CLOSER COMPANIES -> MORE WEIGHTED
    # USING EXPONENTIAL DECAY
    distance_weights = np.exp(-distances / radius_km)

    # WEIGHT BY EMPLOYEE COUNT
    weighted_employees = employees * distance_weights
    total_weighted_employees = weighted_employees.sum()

    # NORMALIZE
    score = 1.0

    return {
        'score': float(score),
        'weighted_employees': float(total_weighted_employees),
        'total_employees': int(nearby['employee_count'].sum()),
        'company_count': len(nearby),
        'companies': nearby.to_dict('records')
    }


def calculate_employee_density_score(
        latitude: float,
        longitude: float,
        companies_df: pd.DataFrame,
        inner_radius_km: float = 0.5,
        outer_radius_km: float = 2.0
) -> Dict[str, Any]:
    """
    Calculate employee density score using multi-radius approach.

    Uses two radii to capture both immediate and broader area employee density.
    Inner radius has higher weight than outer radius.

    Args:
        latitude: Target location latitude
        longitude: Target location longitude
        companies_df: DataFrame with company locations
        inner_radius_km: Inner search radius (higher weight)
        outer_radius_km: Outer search radius (lower weight)

    Returns:
        Dictionary with:
            - score: Density-based score (0.0-1.0)
            - inner_employees: Employees within inner radius
            - outer_employees: Employees within outer radius
            - density_metric: Weighted density metric
    """

    # EDGE CASE
    if companies_df.empty:
        return {
            'score': 0.0,
            'inner_employees': 0,
            'outer_employees': 0,
            'density_metric': 0.0
        }

    # FIND COMPANIES WITHIN INNER RADIUS
    inner = find_points_within_radius(
        latitude, longitude,
        companies_df,
        radius_km=inner_radius_km
    )

    # FIND COMPANIES WITHIN OUTER RADIUS
    outer = find_points_within_radius(
        latitude, longitude,
        companies_df,
        radius_km=outer_radius_km
    )

    # EMPLOYEE COUNTS
    inner_employees = int(inner['employee_count'].sum()) if not inner.empty else 0
    outer_employees = int(outer['employee_count'].sum()) if not outer.empty else 0

    # WEIGHTED DENSITY MERIC
    # INNER RADIUS: 70% WEIGHT, OUTER RADIUS: 30% WEIGHT
    # TODO: REASSESS WEIGHTS
    weighted_density = 0.7 * inner_employees + 0.3 * outer_employees

    # NORMALIZE (300 EMPLOYEES MAX SCORE)
    # TODO: REASSESS NORMALIZATION
    score = min(weighted_density / 300.0, 1.0)

    # RETURN
    return {
        'score': float(score),
        'inner_employees': inner_employees,
        'outer_employees': outer_employees,
        'density_metric': float(weighted_density)
    }


def compare_company_proximity(
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float,
        companies_df: pd.DataFrame,
        radius_km: float = 2.0,
        score1: Optional[Dict[str, Any]] = None
) -> float:
    """
    Compare company proximity similarity between two locations.

    Uses RELATIVE SIMILARITY:
    Similarity = min(val1, val2) / max(val1, val2)

    MEANING THAT:
    1. A location with 500 emp (Target) matches best with 500 emp (Reference)
    2. A location with 5000 emp (Reference) is penalized for being "too busy"
    3. A location with 50 emp (Reference) is penalized for being "too quiet"

    Args:
        lat1, lon1: First location coordinates
        lat2, lon2: Second location coordinates
        companies_df: DataFrame with company locations
        radius_km: Search radius
        score1: Optional pre-calculated score for location 1

    Returns:
        Similarity score (1.0 = identical raw density, approaches 0.0 for large differences)
    """

    # CALCULATE PROXIMITY SCORES FOR BOTH LOCATIONS
    if score1 is None:
        score1 = calculate_company_proximity_score(lat1, lon1, companies_df, radius_km)

    score2 = calculate_company_proximity_score(lat2, lon2, companies_df, radius_km)

    # EXTRACT RAW WEIGHTED VALUES
    val1 = score1.get('weighted_employees', 0.0)
    val2 = score2.get('weighted_employees', 0.0)

    # HANDLE ZERO CASE
    if val1 == 0 and val2 == 0:
        return 1.0
    if val1 == 0 or val2 == 0:
        return 0.0

    # CALCULATE RATIO (Relative Similarity)
    similarity = min(val1, val2) / max(val1, val2)

    return float(similarity)


def get_nearby_employee_stats(
        latitude: float,
        longitude: float,
        companies_df: pd.DataFrame,
        radius_km: float = 2.0
) -> Dict[str, Any]:
    """
    Get detailed statistics about nearby employees.


    Args:
        latitude: Target location latitude
        longitude: Target location longitude
        companies_df: DataFrame with company locations
        radius_km: Search radius

    Returns:
        Dictionary with employee statistics:
            - total_employees
            - avg_employees_per_company
            - median_employees
            - largest_company_employees
            - employee_distribution (quartiles)
    """

    # FIND COMPANIES WITHIN RADIUS
    nearby = find_points_within_radius(
        latitude, longitude,
        companies_df,
        radius_km=radius_km
    )

    # EDGE CASE: NO POINTS
    if nearby.empty:
        return {
            'total_employees': 0,
            'avg_employees_per_company': 0.0,
            'median_employees': 0.0,
            'largest_company_employees': 0,
            'employee_distribution': {}
        }

    # FETCH EMPLOYEE COUNTS
    employees = nearby['employee_count']

    # RETURN STATISTICS
    return {
        'total_employees': int(employees.sum()),
        'avg_employees_per_company': float(employees.mean()),
        'median_employees': float(employees.median()),
        'largest_company_employees': int(employees.max()),
        'employee_distribution': {
            'q25': float(employees.quantile(0.25)),
            'q50': float(employees.quantile(0.50)),
            'q75': float(employees.quantile(0.75))
        }
    }
