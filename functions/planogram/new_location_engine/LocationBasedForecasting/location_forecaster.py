"""
Location-Based Sales Forecaster

Main prediction engine for forecasting sales at new locations based on
geo-weighted sales from similar machine categories.
"""

# IMPORT
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import sys

sys.path.append('..')

from .geo_similarity import calculate_similarity_to_multiple
from .weighting_strategies import get_strategy
from .config import (
    get_category_config,
    get_geo_similarity_weights,
    DEFAULT_WEIGHTING_STRATEGY,
    MIN_CONFIDENCE_THRESHOLD
)


# TODO: THRESHOLDS IN FALLBACK ENGINE MUST BE EXCLUDED
class LocationBasedForecaster:
    """
    Predicts sales for products at new locations using geographic weighting.

    The forecaster:
    1. Finds reference machines in the same category
    2. Gets historical sales for those machines (using FallbackEngine)
    3. Calculates geo-similarity between new location and reference machines
    4. Weights sales predictions by geo-similarity
    5. Returns predicted sales with confidence score
    """

    def __init__(
            self,
            sales_df: pd.DataFrame,
            machines_df: pd.DataFrame,
            ica_stores_df: pd.DataFrame,
            companies_df: pd.DataFrame,
            fallback_engine: Optional[Any] = None
    ):
        """
        Initialize the Location-Based Forecaster.

        Args:
            sales_df: Sales data with columns: machine_key, product_name, etc.
            machines_df: Machine data with columns: machine_key, latitude, longitude,
                        machine_eva_group, machine_sub_group
            ica_stores_df: ICA stores with columns: store_id, latitude, longitude
            companies_df: Companies with columns: company_id, latitude, longitude, employee_count
            fallback_engine: Optional FallbackEngine instance for data retrieval
        """
        self.sales_df = sales_df
        self.machines_df = machines_df
        self.ica_stores_df = ica_stores_df
        self.companies_df = companies_df
        self.fallback_engine = fallback_engine

        # VALIDATE REQUIRED COLUMNS
        self._validate_dataframes()

    def _validate_dataframes(self):
        """Validate that dataframes have required columns."""
        # DEFINE REQUIRED COLUMNS PER DATAFRAME
        required_machine_cols = {'machine_key', 'latitude', 'longitude', 'machine_eva_group'}
        required_ica_cols = {'latitude', 'longitude'}
        required_company_cols = {'latitude', 'longitude', 'employee_count'}

        # CHECK MACHINES DF
        if not required_machine_cols.issubset(self.machines_df.columns):
            missing = required_machine_cols - set(self.machines_df.columns)
            raise ValueError(f"machines_df missing columns: {missing}")

        # CHECK ICA STORES DF
        if not required_ica_cols.issubset(self.ica_stores_df.columns):
            missing = required_ica_cols - set(self.ica_stores_df.columns)
            raise ValueError(f"ica_stores_df missing columns: {missing}")

        # CHECK COMPANIES DF
        if not required_company_cols.issubset(self.companies_df.columns):
            missing = required_company_cols - set(self.companies_df.columns)
            raise ValueError(f"companies_df missing columns: {missing}")

    def _find_reference_machines(
            self,
            machine_category: str
    ) -> pd.DataFrame:
        """
        Find all machines in the specified category.

        Args:
            machine_category: Category to filter by (e.g., 'WORK', 'GYM')

        Returns:
            DataFrame with machines in the category
        """
        # FILTER BY EVA_GROUP (PRIMARY CATEGORY)
        category_machines = self.machines_df[self.machines_df['machine_eva_group'] == machine_category].copy()

        return category_machines

    def _get_reference_sales(
            self,
            machine_keys: List[str],
            product_name: str,
            time_period: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get historical sales for product from reference machines.

        Uses FallbackEngine if available, otherwise calculates directly from sales_df.

        Args:
            machine_keys: List of machine keys
            product_name: Product name to query
            time_period: Optional time period filter

        Returns:
            Dictionary mapping machine_key to sales data:
            {
                machine_key: {
                    'sales': average_sales,
                    'sample_size': number_of_transactions,
                    'fallback_level': level_used (if FallbackEngine)
                }
            }
        """
        reference_sales = {}

        for machine_key in machine_keys:
            if self.fallback_engine:
                # IF DEFINED, USE FALLBACK ENGINE TO GET SALES DATA
                try:

                    # CALL ENGINE
                    result = self.fallback_engine.execute_fallback(
                        machine_key=machine_key,
                        product_name=product_name,
                        time_period=time_period
                    )

                    # IF SUCCESS
                    if result['success'] and len(result['data']) > 0:

                        sales_data = result['data']


                        # CALCULATE DURATION IN MONTHS FOR NORMALIZATION
                        months_factor = 1.0
                        if time_period is None:
                            # FIND TIMESTAMP COLUMN
                            ts_col = next((col for col in ['local_timestamp', 'timestamp'] if col in sales_data.columns), None)

                            if ts_col:
                                try:
                                    # CONVERT TO DATETIME IF NEEDED
                                    if not pd.api.types.is_datetime64_any_dtype(sales_data[ts_col]):
                                        timestamps = pd.to_datetime(sales_data[ts_col], errors='coerce')
                                    else:
                                        timestamps = sales_data[ts_col]

                                    # COUNT UNIQUE MONTHS
                                    if hasattr(timestamps.dt, 'tz') and timestamps.dt.tz is not None:
                                        timestamps = timestamps.dt.tz_localize(None)
                                    unique_months = timestamps.dt.to_period('M').nunique()
                                    months_factor = max(1.0, float(unique_months))
                                except Exception:
                                    # FALLBACK
                                    pass

                        # FIND QUANTITY COLUMN
                        qty_col = None
                        for col in ['quantity', 'qty', 'units', 'amount']:
                            if col in sales_data.columns:
                                qty_col = col
                                break

                        # CALCULATE AVERAGE SALES
                        if qty_col:
                            total_sales_vol = sales_data[qty_col].sum() if 'machine_key' in sales_data.columns else sales_data[qty_col].sum()
                            num_machines = sales_data['machine_key'].nunique() if 'machine_key' in sales_data.columns else 1
                            avg_sales = (total_sales_vol / num_machines) / months_factor
                        else:
                            # COUNT TRANSACTIONS AS FALLBACK
                            num_machines = sales_data['machine_key'].nunique() if 'machine_key' in sales_data.columns else 1
                            total_transactions = len(sales_data)
                            avg_sales = (total_transactions / num_machines) / months_factor

                        # STORE SALES DATA WITH METADATA
                        reference_sales[machine_key] = {
                            'sales': float(avg_sales),
                            'sample_size': result['sample_size'],
                            'fallback_level': result['level'],
                            'fallback_confidence': result['confidence']
                        }
                except Exception as e:
                    # SKIP IF FALLBACK FAILS
                    print(f"Warning: Failed to get sales for machine {machine_key}: {e}")
                    continue
            else:
                # DIRECT CALCULATION FROM SALES_DF
                machine_sales = self.sales_df[
                    (self.sales_df['machine_key'] == machine_key) &
                    (self.sales_df['product_name'] == product_name)
                    ]

                # NORMALIZE BY TIME PERIOD
                months_factor = 1.0
                if time_period:
                    # FILTER BY TIME PERIOD
                    if 'local_timestamp' in machine_sales.columns:
                        machine_sales = machine_sales[
                            machine_sales['local_timestamp'].astype(str).str.startswith(time_period)
                        ]
                else:
                    # COUNT UNIQUE MONTHS IF NO TIME PERIOD
                    ts_col = next((col for col in ['local_timestamp', 'timestamp'] if col in machine_sales.columns), None)
                    if ts_col:
                        try:
                             if not pd.api.types.is_datetime64_any_dtype(machine_sales[ts_col]):
                                timestamps = pd.to_datetime(machine_sales[ts_col], errors='coerce')
                             else:
                                timestamps = machine_sales[ts_col]
                             if hasattr(timestamps.dt, 'tz') and timestamps.dt.tz is not None:
                                timestamps = timestamps.dt.tz_localize(None)
                             unique_months = timestamps.dt.to_period('M').nunique()
                             months_factor = max(1.0, float(unique_months))
                        except Exception:
                            pass

                # CALCULATE SALES IF DATA EXISTS
                if len(machine_sales) > 0:
                    # FIND QUANTITY COLUMN
                    qty_col = None
                    for col in ['quantity', 'qty', 'units', 'amount']:
                        if col in machine_sales.columns:
                            qty_col = col
                            break

                    # CALCULATE AVERAGE SALES
                    if qty_col:
                        total_sales_vol = machine_sales[qty_col].sum()
                        num_machines = machine_sales['machine_key'].nunique() if 'machine_key' in machine_sales.columns else 1
                        avg_sales = (total_sales_vol / num_machines) / months_factor
                    else:
                        # COUNT TRANSACTIONS AS FALLBACK
                        num_machines = machine_sales['machine_key'].nunique() if 'machine_key' in machine_sales.columns else 1
                        total_transactions = len(machine_sales)
                        avg_sales = (total_transactions / num_machines) / months_factor

                    # STORE SALES DATA WITH METADATA
                    reference_sales[machine_key] = {
                        'sales': float(avg_sales),
                        'sample_size': len(machine_sales),
                        'fallback_level': 0,
                        'fallback_confidence': 1.0  # DIRECT DATA = HIGHEST CONFIDENCE
                    }

        return reference_sales

    def predict_sales(
            self,
            product_name: str,
            new_latitude: float,
            new_longitude: float,
            machine_category: str,
            time_period: Optional[str] = None,
            weighting_strategy: str = DEFAULT_WEIGHTING_STRATEGY,
            **kwargs
    ) -> Dict[str, Any]:
        """
        Predict sales for a product at a new location.

        Args:
            product_name: Product to predict sales for
            new_latitude: New location latitude
            new_longitude: New location longitude
            machine_category: Machine category (e.g., 'WORK', 'GYM', 'SCHOOLS')
            time_period: Optional time period for historical sales (e.g., '2024-03')
            weighting_strategy: Strategy for weighting predictions
                               ('inverse_distance', 'gaussian', 'top_k', 'adaptive')
            **kwargs: Additional arguments for weighting strategy

        Returns:
            Dictionary with:
                - predicted_sales: Predicted average sales
                - confidence: Confidence score (0.0-1.0)
                - reference_machines: List of machines used
                - geo_similarities: Similarity score for each reference machine
                - weights: Weight assigned to each reference machine
                - breakdown: Detailed breakdown of prediction
                - success: True if prediction made, False otherwise
                - message: Status/error message
        """
        # GET CATEGORY-SPECIFIC CONFIGURATION
        cat_config = get_category_config(machine_category)
        max_distance = cat_config.get('max_distance', 30.0)
        min_machines = cat_config.get('min_machines', 3)

        # GET GEO-SIMILARITY WEIGHTS FOR CATEGORY
        geo_weights = get_geo_similarity_weights(machine_category)

        # STEP 1: FIND REFERENCE MACHINES IN CATEGORY
        reference_machines = self._find_reference_machines(machine_category)

        # CHECK IF REFERENCE MACHINES FOUND
        if reference_machines.empty:
            return {
                'predicted_sales': 0.0,
                'confidence': 0.0,
                'reference_machines': [],
                'geo_similarities': [],
                'weights': [],
                'breakdown': {},
                'success': False,
                'message': f'No machines found in category: {machine_category}'
            }

        # STEP 2: CALCULATE GEO-SIMILARITY TO ALL REFERENCE MACHINES
        similarities = calculate_similarity_to_multiple(
            new_latitude,
            new_longitude,
            reference_machines,
            self.ica_stores_df,
            self.companies_df,
            distance_weight=geo_weights['distance'],
            ica_weight=geo_weights['ica_profile'],
            company_weight=geo_weights['company_profile'],
            max_distance_km=max_distance
        )

        # FILTER BY DISTANCE AND MINIMUM SIMILARITY
        min_similarity = kwargs.get('min_similarity', 0.2)
        filtered_similarities = similarities[
            (similarities['distance_km'] <= max_distance) &
            (similarities['similarity'] >= min_similarity)
            ]

        # CHECK IF ENOUGH SIMILAR MACHINES
        if len(filtered_similarities) < min_machines:
            return {
                'predicted_sales': 0.0,
                'confidence': 0.0,
                'reference_machines': [],
                'geo_similarities': [],
                'weights': [],
                'breakdown': {},
                'success': False,
                'message': f'Insufficient similar machines ({len(filtered_similarities)} < {min_machines})'
            }

        # STEP 3: GET SALES DATA FOR REFERENCE MACHINES
        machine_keys = filtered_similarities['machine_key'].tolist()
        reference_sales = self._get_reference_sales(
            machine_keys,
            product_name,
            time_period
        )

        # CHECK IF ENOUGH SALES DATA
        if len(reference_sales) < min_machines:
            return {
                'predicted_sales': 0.0,
                'confidence': 0.0,
                'reference_machines': list(reference_sales.keys()),
                'geo_similarities': [],
                'weights': [],
                'breakdown': {},
                'success': False,
                'message': f'Insufficient sales data ({len(reference_sales)} machines with data)'
            }

        # STEP 4: PREPARE DATA FOR WEIGHTING
        valid_machines = []
        geo_sims = []
        sales_vals = []
        fallback_confidences = []

        # COLLECT VALID MACHINE DATA
        for machine_key in reference_sales.keys():
            sim_row = filtered_similarities[filtered_similarities['machine_key'] == machine_key]
            if not sim_row.empty:
                valid_machines.append(machine_key)
                geo_sims.append(sim_row.iloc[0]['similarity'])
                sales_vals.append(reference_sales[machine_key]['sales'])
                fallback_confidences.append(reference_sales[machine_key]['fallback_confidence'])

        # CONVERT TO NUMPY ARRAYS
        geo_sims = np.array(geo_sims)
        sales_vals = np.array(sales_vals)
        fallback_confidences = np.array(fallback_confidences)

        # STEP 5: APPLY WEIGHTING STRATEGY
        strategy_func = get_strategy(weighting_strategy)
        weighting_result = strategy_func(geo_sims, sales_vals, **kwargs)

        predicted_sales = weighting_result['predicted_sales']
        weights = weighting_result['weights']

        # STEP 6: CALCULATE CONFIDENCE
        confidence = self._calculate_confidence(
            geo_sims,
            sales_vals,
            fallback_confidences,
            len(valid_machines),
            min_machines
        )

        # STEP 7: BUILD DETAILED BREAKDOWN
        breakdown = []
        for i, machine_key in enumerate(valid_machines):
            breakdown.append({
                'machine_key': machine_key,
                'geo_similarity': float(geo_sims[i]),
                'sales': float(sales_vals[i]),
                'weight': float(weights[i]),
                'contribution': float(weights[i] * sales_vals[i]),
                'distance_km': float(filtered_similarities[
                                         filtered_similarities['machine_key'] == machine_key
                                         ].iloc[0]['distance_km']),
                'sample_size': reference_sales[machine_key]['sample_size'],
                'fallback_level': reference_sales[machine_key]['fallback_level'],
                'fallback_confidence': float(fallback_confidences[i])
            })

        return {
            'predicted_sales': float(predicted_sales),
            'confidence': float(confidence),
            'reference_machines': valid_machines,
            'geo_similarities': geo_sims.tolist(),
            'weights': weights.tolist(),
            'breakdown': breakdown,
            'success': True,
            'message': f'Successfully predicted using {len(valid_machines)} reference machines'
        }

    def _calculate_confidence(
            self,
            geo_similarities: np.ndarray,
            sales_values: np.ndarray,
            fallback_confidences: np.ndarray,
            num_machines: int,
            min_machines: int
    ) -> float:
        """
        Calculate confidence score for prediction.

        Based on:
        - Average geo-similarity (higher = more confident)
        - Average fallback confidence (higher = better data quality)
        - Number of reference machines (more = more confident)
        - Consistency of sales across machines (less variance = more confident)

        Args:
            geo_similarities: Array of similarity scores
            sales_values: Array of sales values
            fallback_confidences: Array of FallbackEngine confidence scores
            num_machines: Number of machines used
            min_machines: Minimum required machines

        Returns:
            Confidence score (0.0-1.0)
        """
        # COMPONENT 1: AVERAGE SIMILARITY
        avg_similarity = geo_similarities.mean()

        # COMPONENT 2: AVERAGE FALLBACK CONFIDENCE
        avg_fallback_confidence = fallback_confidences.mean()

        # COMPONENT 3: SAMPLE SIZE SCORE
        sample_score = min(num_machines / (2 * min_machines), 1.0)

        # COMPONENT 4: CONSISTENCY SCORE
        if len(sales_values) > 1 and sales_values.mean() > 0:
            cv = sales_values.std() / sales_values.mean()
            consistency_score = np.exp(-cv)
        else:
            consistency_score = 0.5

        # COMBINE COMPONENTS WITH WEIGHTS
        confidence = (
                0.4 * avg_similarity +
                0.3 * avg_fallback_confidence +
                0.15 * sample_score +
                0.15 * consistency_score
        )

        return float(min(confidence, 1.0))

    def compare_strategies(
            self,
            product_name: str,
            new_latitude: float,
            new_longitude: float,
            machine_category: str,
            time_period: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Compare predictions across all fallback strategies.

        Runs the prediction using PRODUCT_FIRST, MACHINE_FIRST, and LOCATION_AFFINITY
        strategies and returns a comparison DataFrame.

        Args:
            product_name: Product to predict sales for
            new_latitude: New location latitude
            new_longitude: New location longitude
            machine_category: Machine category (e.g., 'WORK', 'GYM')
            time_period: Optional time period filter

        Returns:
            DataFrame with columns:
                - strategy: Strategy name
                - predicted_sales: Predicted monthly sales
                - confidence: Overall confidence score
                - num_machines: Number of reference machines with data

        """
        # IMPORT FALLBACK STRATEGIES
        import sys
        sys.path.append('..')
        from FallbackAlgorithm import (
            FallbackEngine,
            PRODUCT_FIRST_STRATEGY,
            MACHINE_FIRST_STRATEGY,
            LOCATION_AFFINITY_STRATEGY
        )

        # DEFINE STRATEGIES TO COMPARE
        strategies = {
            'PRODUCT_FIRST': PRODUCT_FIRST_STRATEGY,
            'MACHINE_FIRST': MACHINE_FIRST_STRATEGY,
            'LOCATION_AFFINITY': LOCATION_AFFINITY_STRATEGY
        }

        results = []

        # TEST EACH STRATEGY
        for strategy_name, strategy in strategies.items():
            # CREATE FALLBACK ENGINE WITH STRATEGY
            fallback_engine = FallbackEngine(self.sales_df, strategy)

            # TEMPORARILY REPLACE FALLBACK ENGINE
            original_engine = self.fallback_engine
            self.fallback_engine = fallback_engine

            # RUN PREDICTION
            prediction = self.predict_sales(
                product_name=product_name,
                new_latitude=new_latitude,
                new_longitude=new_longitude,
                machine_category=machine_category,
                time_period=time_period
            )

            # RESTORE ORIGINAL ENGINE
            self.fallback_engine = original_engine

            # STORE RESULTS
            results.append({
                'strategy': strategy_name,
                'predicted_sales': prediction['predicted_sales'],
                'confidence': prediction['confidence'],
                'num_machines': len(prediction['reference_machines']),
                'success': prediction['success']
            })

        return pd.DataFrame(results)
