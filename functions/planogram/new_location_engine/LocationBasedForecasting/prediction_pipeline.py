"""
Prediction Pipeline
===================

High-level API for location-based sales prediction.
Provides convenient one-function interfaces for common use cases.
"""

# IMPORT DEPENDENCIES
import pandas as pd
from typing import Dict, Any, Optional
from pathlib import Path

# IMPORT CORE COMPONENTS
from .location_forecaster import LocationBasedForecaster
from .config import DEFAULT_ICA_STORES_PATH, DEFAULT_COMPANIES_PATH


def predict_sales_at_location(
    product_name: str,
    new_latitude: float,
    new_longitude: float,
    machine_category: str,
    sales_df: pd.DataFrame,
    machines_df: pd.DataFrame,
    ica_stores_df: Optional[pd.DataFrame] = None,
    companies_df: Optional[pd.DataFrame] = None,
    fallback_engine: Optional[Any] = None,
    time_period: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Predict sales for a product at a new location.

    One-function API that handles all setup and prediction.

    Args:
        product_name: Product to predict sales for
        new_latitude: New location latitude
        new_longitude: New location longitude
        machine_category: Machine category (e.g., 'WORK', 'GYM', 'SCHOOLS')
        sales_df: Historical sales data
        machines_df: Machine locations and metadata
        ica_stores_df: ICA store locations (optional, loads default if None)
        companies_df: Company locations (optional, loads default if None)
        fallback_engine: Optional FallbackEngine for data retrieval
        time_period: Optional time period filter (e.g., '2024-03')
        **kwargs: Additional arguments for prediction

    Returns:
        Prediction result dictionary
    """
    # LOAD DEFAULT DATA IF NOT PROVIDED
    if ica_stores_df is None:
        ica_stores_df = load_default_ica_stores()

    if companies_df is None:
        companies_df = load_default_companies()

    # CREATE FORECASTER INSTANCE
    forecaster = LocationBasedForecaster(
        sales_df=sales_df,
        machines_df=machines_df,
        ica_stores_df=ica_stores_df,
        companies_df=companies_df,
        fallback_engine=fallback_engine
    )

    # RUN PREDICTION
    return forecaster.predict_sales(
        product_name=product_name,
        new_latitude=new_latitude,
        new_longitude=new_longitude,
        machine_category=machine_category,
        time_period=time_period,
        **kwargs
    )


def compare_locations(
    product_name: str,
    locations: list,
    machine_category: str,
    sales_df: pd.DataFrame,
    machines_df: pd.DataFrame,
    ica_stores_df: Optional[pd.DataFrame] = None,
    companies_df: Optional[pd.DataFrame] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Compare predicted sales across multiple potential locations.

    Args:
        product_name: Product to predict
        locations: List of dicts with 'name', 'latitude', 'longitude' keys
        machine_category: Machine category
        sales_df: Historical sales data
        machines_df: Machine locations
        ica_stores_df: Optional ICA stores
        companies_df: Optional companies
        **kwargs: Additional prediction arguments

    Returns:
        DataFrame with predictions for each location, sorted by predicted sales
    """
    results = []

    # PREDICT FOR EACH LOCATION
    for location in locations:
        prediction = predict_sales_at_location(
            product_name=product_name,
            new_latitude=location['latitude'],
            new_longitude=location['longitude'],
            machine_category=machine_category,
            sales_df=sales_df,
            machines_df=machines_df,
            ica_stores_df=ica_stores_df,
            companies_df=companies_df,
            **kwargs
        )

        # COLLECT PREDICTION RESULTS
        results.append({
            'name': location.get('name', f"Location ({location['latitude']}, {location['longitude']})"),
            'latitude': location['latitude'],
            'longitude': location['longitude'],
            'predicted_sales': prediction['predicted_sales'],
            'confidence': prediction['confidence'],
            'num_reference_machines': len(prediction['reference_machines']),
            'success': prediction['success']
        })

    # CREATE DATAFRAME AND SORT BY PREDICTED SALES
    df = pd.DataFrame(results)
    df = df.sort_values('predicted_sales', ascending=False)

    return df.reset_index(drop=True)


def get_best_location(
    product_name: str,
    candidate_locations: list,
    machine_category: str,
    sales_df: pd.DataFrame,
    machines_df: pd.DataFrame,
    **kwargs
) -> Dict[str, Any]:
    """
    Find the best location from a list of candidates.

    Returns the location with highest predicted sales (among high-confidence predictions).

    Args:
        product_name: Product to optimize for
        candidate_locations: List of location dicts
        machine_category: Machine category
        sales_df: Sales data
        machines_df: Machine data
        **kwargs: Additional arguments

    Returns:
        Dictionary with:
            - best_location: Location dict with highest predicted sales
            - prediction: Full prediction result for best location
            - all_predictions: DataFrame with all location comparisons
    """
    # GET PREDICTIONS FOR ALL LOCATIONS
    all_predictions = compare_locations(
        product_name,
        candidate_locations,
        machine_category,
        sales_df,
        machines_df,
        **kwargs
    )

    # FILTER FOR HIGH-CONFIDENCE PREDICTIONS
    min_confidence = kwargs.get('min_confidence', 0.5)
    reliable = all_predictions[
        (all_predictions['success'] == True) &
        (all_predictions['confidence'] >= min_confidence)
    ]

    # FALLBACK TO ANY SUCCESSFUL PREDICTIONS
    if reliable.empty:
        reliable = all_predictions[all_predictions['success'] == True]

    # RETURN ERROR IF NO VALID PREDICTIONS
    if reliable.empty:
        return {
            'best_location': None,
            'prediction': None,
            'all_predictions': all_predictions,
            'message': 'No valid predictions available'
        }

    # EXTRACT BEST LOCATION
    best_row = reliable.iloc[0]
    best_location = {
        'name': best_row['name'],
        'latitude': best_row['latitude'],
        'longitude': best_row['longitude']
    }

    # GET DETAILED PREDICTION FOR BEST LOCATION
    best_prediction = predict_sales_at_location(
        product_name=product_name,
        new_latitude=best_location['latitude'],
        new_longitude=best_location['longitude'],
        machine_category=machine_category,
        sales_df=sales_df,
        machines_df=machines_df,
        **kwargs
    )

    return {
        'best_location': best_location,
        'prediction': best_prediction,
        'all_predictions': all_predictions
    }


def load_default_ica_stores() -> pd.DataFrame:
    """
    Load supermarket locations.

    Returns:
        DataFrame with supermarket locations
    """
    try:
        base_path = Path(__file__).parent

        # LOAD SCB SUPERMARKET DATA
        scb_path = base_path / 'data' / 'scb_supermarkets.parquet'
        if scb_path.exists():
            df = pd.read_parquet(scb_path)
            print(f"Loaded {len(df)} supermarkets from SCB (Statistics Sweden)")
            return df

        # RAISE ERROR IF NOT FOUND
        raise FileNotFoundError(f"No supermarket data found at {scb_path}")


    except Exception as e:
        # RETURN EMPTY DATAFRAME ON ERROR
        print(f"Warning: Could not load supermarket data: {e}")
        return pd.DataFrame(columns=['store_id', 'latitude', 'longitude'])


def load_default_companies() -> pd.DataFrame:
    """
    Load company locations with employee counts.

    Returns:
        DataFrame with company locations and employee counts
    """
    try:
        base_path = Path(__file__).parent

        # LOAD SCB COMPANY DATA
        scb_path = base_path / 'data' / 'scb_companies.parquet'
        if scb_path.exists():
            df = pd.read_parquet(scb_path)
            print(f"Loaded {len(df)} companies from SCB (Statistics Sweden)")
            return df

        # RAISE ERROR IF NOT FOUND
        raise FileNotFoundError(f"No company data found at {scb_path}")

    except Exception as e:
        # RETURN EMPTY DATAFRAME ON ERROR
        print(f"Warning: Could not load companies: {e}")
        return pd.DataFrame(columns=['company_id', 'latitude', 'longitude', 'employee_count'])
