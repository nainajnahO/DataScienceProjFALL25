"""
Fallback Integration for SMA Forecasting

This module provides the SMAForecaster class that integrates the FallbackEngine
with Progressive Window SMA forecasting to handle sparse data scenarios.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Any, Union
import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from .progressive_window_sma import ProgressiveWindowSMA


class SMAForecaster:
    """
    High-level forecaster integrating SMA with fallback data retrieval.

    This class handles:
    - Data retrieval for machine+product combinations
    - Automatic fallback to hierarchical aggregations when data is sparse
    - SMA forecasting on retrieved data
    - Comprehensive metadata tracking
    """

    def __init__(
        self,
        sales_df: pd.DataFrame,
        forecast_horizon: int = 4,
        strategy: str = 'product_first',
        use_fallback: bool = True,
        min_samples: Optional[int] = None,
        use_adaptive_thresholds: bool = True
    ):
        """
        Initialize the SMA Forecaster with fallback capabilities.

        Args:
            sales_df: Sales DataFrame with required columns (machine_id, product_name,
                     local_timestamp, and hierarchy columns if using fallback)
            forecast_horizon: Number of weeks to forecast ahead (default: 4)
            strategy: Fallback strategy name - 'product_first', 'machine_first',
                     or 'location_affinity' (default: 'product_first')
            use_fallback: Whether to use fallback engine when data is insufficient
                         (default: True)
            min_samples: Minimum samples override for fallback (if None, uses
                        adaptive thresholds)
            use_adaptive_thresholds: Use different thresholds per fallback level
                                    (default: True)

        Raises:
            ImportError: If fallback is enabled but FallbackAlgorithm not available
            ValueError: If sales_df is empty or missing required columns
        """
        if sales_df is None or len(sales_df) == 0:
            raise ValueError("sales_df cannot be None or empty")

        self.sales_df = sales_df
        self.forecast_horizon = forecast_horizon
        self.strategy_name = strategy
        self.use_fallback = use_fallback
        self.min_samples = min_samples
        self.use_adaptive_thresholds = use_adaptive_thresholds

        # Validate basic columns
        self._validate_dataframe()

        # Initialize fallback engine if enabled
        self.fallback_engine = None
        if use_fallback:
            self._initialize_fallback_engine()

        # Create SMA model
        self.sma_model = ProgressiveWindowSMA(forecast_horizon=forecast_horizon)

    def _validate_dataframe(self) -> None:
        """
        Validate that DataFrame has minimum required columns.

        Raises:
            ValueError: If required columns are missing
        """
        basic_required = {'machine_id', 'product_name'}
        missing = basic_required - set(self.sales_df.columns)

        if missing:
            raise ValueError(f"sales_df missing required columns: {missing}")

    def _initialize_fallback_engine(self) -> None:
        """
        Initialize the fallback engine.

        Raises:
            ImportError: If FallbackAlgorithm package not available
            ValueError: If required hierarchy columns missing
        """
        try:
            from FallbackAlgorithm import FallbackEngine, get_strategy
        except ImportError:
            raise ImportError(
                "FallbackAlgorithm package not found. "
                "Please ensure it's installed or in your Python path."
            )

        # Validate fallback-specific columns
        fallback_required = {
            'machine_id', 'machine_sub_group', 'machine_eva_group',
            'product_name', 'subcategory', 'category'
        }
        missing = fallback_required - set(self.sales_df.columns)
        if missing:
            raise ValueError(
                f"Fallback enabled but sales_df missing required hierarchy columns: {missing}"
            )

        # Get strategy
        try:
            strategy_list = get_strategy(self.strategy_name)
        except (KeyError, ValueError) as e:
            raise ValueError(f"Invalid strategy '{self.strategy_name}': {e}")

        # Initialize engine
        self.fallback_engine = FallbackEngine(
            df=self.sales_df,
            strategy=strategy_list,
            min_samples=self.min_samples,
            use_adaptive_thresholds=self.use_adaptive_thresholds
        )

    def _check_direct_data(
        self,
        machine_id: int,
        product_name: str,
        time_period: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Check if direct query has sufficient data.

        Args:
            machine_id: Machine identifier
            product_name: Product name
            time_period: Optional time period filter

        Returns:
            Dictionary with:
                - data: Filtered DataFrame
                - sufficient: Boolean indicating if data meets threshold
                - sample_size: Number of records
                - threshold: Minimum threshold required
        """
        # Query direct data
        direct_data = self.sales_df[
            (self.sales_df['machine_id'].astype(str) == str(machine_id)) &
            (self.sales_df['product_name'] == product_name)
        ].copy()

        # Apply time period filter if specified
        if time_period and 'local_timestamp' in direct_data.columns:
            direct_data = self._filter_time_period(direct_data, time_period)

        # Get threshold
        if self.fallback_engine:
            threshold = self.fallback_engine._get_threshold('machine_id', 'product_name')
        else:
            threshold = 15  # Default

        return {
            'data': direct_data,
            'sufficient': len(direct_data) >= threshold,
            'sample_size': len(direct_data),
            'threshold': threshold
        }

    def _filter_time_period(
        self,
        data: pd.DataFrame,
        time_period: str
    ) -> pd.DataFrame:
        """
        Filter data by time period.

        Args:
            data: DataFrame to filter
            time_period: Time period string (format: 'YYYY-MM')

        Returns:
            Filtered DataFrame
        """
        if 'local_timestamp' not in data.columns:
            return data

        data = data.copy()
        data['local_timestamp'] = pd.to_datetime(data['local_timestamp'])

        # Parse time period (format: 'YYYY-MM')
        try:
            year, month = map(int, time_period.split('-'))
            filtered = data[
                (data['local_timestamp'].dt.year == year) &
                (data['local_timestamp'].dt.month == month)
            ]
            return filtered
        except (ValueError, AttributeError):
            return data

    def _prepare_weekly_from_transactions(
        self,
        transaction_data: pd.DataFrame,
        machine_level: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Convert transaction data to weekly aggregates.

        Args:
            transaction_data: Raw transaction data
            machine_level: Machine aggregation level from fallback (e.g., 'machine_id',
                          'machine_sub_group', 'machine_eva_group', 'ALL'). If provided
                          and not 'machine_id', normalizes by number of unique machines.

        Returns:
            DataFrame with week_start and quantity columns (normalized per-machine if applicable)
        """
        if len(transaction_data) == 0:
            return pd.DataFrame(columns=['week_start', 'quantity'])

        df = transaction_data.copy()

        # Convert to date
        df['date'] = pd.to_datetime(df['local_timestamp']).dt.date
        df['date'] = pd.to_datetime(df['date'])

        # Calculate week start (Monday)
        df['week_start'] = df['date'] - pd.to_timedelta(df['date'].dt.dayofweek, unit='D')

        # Aggregate by week
        weekly = df.groupby('week_start').size().reset_index(name='quantity')

        # Normalize by machine count if using aggregated fallback data
        if machine_level and machine_level != 'machine_id' and 'machine_id' in df.columns:
            unique_machines = df['machine_id'].nunique()
            if unique_machines > 1:
                # Divide by number of machines to get per-machine average
                weekly['quantity'] = weekly['quantity'] / unique_machines

        weekly = weekly.sort_values('week_start').reset_index(drop=True)

        return weekly

    def _build_metadata(
        self,
        data_source: str,
        weekly_data: pd.DataFrame,
        machine_id: int,
        product_name: str,
        time_period: Optional[str],
        fallback_result: Optional[Dict] = None,
        sma_result: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Build comprehensive metadata dictionary.

        Args:
            data_source: 'direct', 'fallback', or error type
            weekly_data: Weekly aggregated data
            machine_id: Machine identifier
            product_name: Product name
            time_period: Time period filter
            fallback_result: Result from fallback engine (if used)
            sma_result: SMA forecast result (if generated)

        Returns:
            Metadata dictionary
        """
        metadata = {
            'data_source': data_source,
            'machine_id': machine_id,
            'product_name': product_name,
            'time_period': time_period
        }

        # Add fallback info if used
        if fallback_result and data_source == 'fallback':
            metadata['fallback_info'] = {
                'level': fallback_result['level'],
                'confidence': fallback_result['confidence'],
                'pair_used': fallback_result['pair_used'],
                'machine_level': fallback_result['machine_level'],
                'product_level': fallback_result['product_level'],
                'sample_size': fallback_result['sample_size'],
                'threshold': fallback_result['threshold']
            }

        # Add weekly data info
        if len(weekly_data) > 0:
            metadata['weekly_data_points'] = len(weekly_data)
            metadata['total_quantity'] = int(weekly_data['quantity'].sum())
            metadata['date_range'] = (
                str(weekly_data['week_start'].min()),
                str(weekly_data['week_start'].max())
            )
        else:
            metadata['weekly_data_points'] = 0
            metadata['total_quantity'] = 0
            metadata['date_range'] = (None, None)

        # Add SMA info if available
        if sma_result is not None and len(sma_result) > 0:
            metadata['sma_windows'] = sma_result['window_used'].tolist()

        return metadata

    def forecast(
        self,
        machine_id: int,
        product_name: str,
        time_period: Optional[str] = None,
        round_decimals: int = 1
    ) -> Dict[str, Any]:
        """
        Generate forecast for a machine-product combination.

        This method:
        1. Attempts direct query for machine_id + product_name
        2. If use_fallback=True and insufficient data, uses fallback engine
        3. Prepares weekly aggregated data
        4. Runs Progressive Window SMA
        5. Returns forecast with comprehensive metadata

        Args:
            machine_id: Machine identifier
            product_name: Product name
            time_period: Optional time period filter (e.g., '2024-10')
            round_decimals: Decimal places for forecast values (default: 1)

        Returns:
            Dictionary with:
                - forecast: DataFrame with forecasts (or None if failed)
                - historical_data: pandas Series with weekly quantity values used for forecasting
                                  (12 zeros if no data available, None if forecast failed)
                - metadata: Comprehensive metadata dictionary
                - success: Boolean indicating if forecast was generated
                - message: Error message (if failed)

        Example:
            >>> result = forecaster.forecast(machine_id=123, product_name='Twix 50g')
            >>> if result['success']:
            ...     print(result['forecast'])
            ...     print(f"Data source: {result['metadata']['data_source']}")
        """
        # Step 1: Check direct data
        direct_check = self._check_direct_data(machine_id, product_name, time_period)

        data_source = None
        transaction_data = None
        fallback_result = None

        # Step 2: Use direct data if sufficient
        if direct_check['sufficient']:
            data_source = 'direct'
            transaction_data = direct_check['data']

        # Step 3: Try fallback if enabled and direct data insufficient
        elif self.use_fallback and self.fallback_engine:
            try:
                fallback_result = self.fallback_engine.execute_fallback(
                    machine_id=machine_id,
                    product_name=product_name,
                    time_period=time_period
                )

                if fallback_result['success']:
                    data_source = 'fallback'
                    transaction_data = fallback_result['data']
                else:
                    # Fallback failed - no data at any level
                    metadata = self._build_metadata(
                        'fallback_failed',
                        pd.DataFrame(),
                        machine_id,
                        product_name,
                        time_period,
                        fallback_result=fallback_result
                    )
                    return {
                        'forecast': None,
                        'historical_data': None,
                        'metadata': metadata,
                        'success': False,
                        'message': fallback_result.get('message', 'Insufficient data at all fallback levels')
                    }

            except Exception as e:
                # Fallback error
                metadata = self._build_metadata(
                    'fallback_error',
                    pd.DataFrame(),
                    machine_id,
                    product_name,
                    time_period
                )
                return {
                    'forecast': None,
                    'historical_data': None,
                    'metadata': metadata,
                    'success': False,
                    'message': f'Fallback error: {str(e)}'
                }

        else:
            # Fallback disabled and direct data insufficient
            # Return a zero forecast for visualization purposes
            zero_forecast = pd.DataFrame({
                'week_number': range(1, self.forecast_horizon + 1),
                'forecast': [0.0] * self.forecast_horizon,
                'window_used': [0] * self.forecast_horizon
            })

            metadata = self._build_metadata(
                'direct_insufficient',
                pd.DataFrame(),
                machine_id,
                product_name,
                time_period
            )
            metadata['sample_size'] = direct_check['sample_size']
            metadata['threshold'] = direct_check['threshold']

            return {
                'forecast': zero_forecast,
                'historical_data': pd.Series([0.0] * 12),
                'metadata': metadata,
                'success': True,  # Success with zero forecast
                'message': f"Insufficient direct data ({direct_check['sample_size']} samples, "
                          f"need {direct_check['threshold']}). Returning zero forecast (fallback disabled)."
            }

        # Step 4: Prepare weekly data
        # Pass machine_level for normalization if using fallback
        machine_level = fallback_result['machine_level'] if fallback_result else 'machine_id'
        weekly_data = self._prepare_weekly_from_transactions(transaction_data, machine_level)

        # Check if we have enough weeks
        min_weeks = max(2, self.forecast_horizon // 2)  # At least 2 weeks or half forecast horizon
        if len(weekly_data) < min_weeks:
            metadata = self._build_metadata(
                data_source,
                weekly_data,
                machine_id,
                product_name,
                time_period,
                fallback_result=fallback_result
            )
            return {
                'forecast': None,
                'historical_data': None,
                'metadata': metadata,
                'success': False,
                'message': f'Insufficient weekly data points: {len(weekly_data)} weeks '
                          f'(need at least {min_weeks})'
            }

        # Step 5: Generate SMA forecast
        try:
            quantity_series = weekly_data['quantity'].reset_index(drop=True)
            self.sma_model.fit(quantity_series)
            forecast_df = self.sma_model.forecast(round_decimals=round_decimals)

            # Build final metadata
            metadata = self._build_metadata(
                data_source,
                weekly_data,
                machine_id,
                product_name,
                time_period,
                fallback_result=fallback_result,
                sma_result=forecast_df
            )

            return {
                'forecast': forecast_df,
                'historical_data': quantity_series,
                'metadata': metadata,
                'success': True,
                'message': None
            }

        except Exception as e:
            metadata = self._build_metadata(
                data_source,
                weekly_data,
                machine_id,
                product_name,
                time_period,
                fallback_result=fallback_result
            )
            return {
                'forecast': None,
                'historical_data': None,
                'metadata': metadata,
                'success': False,
                'message': f'SMA forecasting error: {str(e)}'
            }

    def backtest(
        self,
        machine_id: int,
        product_name: str,
        test_weeks: int = 4,
        time_period: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform backtesting with fallback-enabled data retrieval.

        Args:
            machine_id: Machine identifier
            product_name: Product name
            test_weeks: Number of weeks to reserve for testing (default: 4)
            time_period: Optional time period filter (e.g., '2024-10')

        Returns:
            Dictionary with:
                - predictions: DataFrame with actual vs forecast
                - metrics: MAE and RMSE
                - metadata: Same as forecast()
                - success: Boolean
                - message: Error message (if failed)

        Example:
            >>> result = forecaster.backtest(machine_id=123, product_name='Twix', test_weeks=4)
            >>> if result['success']:
            ...     print(f"MAE: {result['metrics']['MAE']:.2f}")
            ...     print(f"RMSE: {result['metrics']['RMSE']:.2f}")
        """
        # First get all data using forecast logic
        full_result = self.forecast(machine_id, product_name, time_period, round_decimals=2)

        # If forecast failed, return the same error
        if not full_result['success']:
            return {
                'predictions': None,
                'metrics': None,
                'metadata': full_result['metadata'],
                'success': False,
                'message': full_result['message']
            }

        # If forecast returned zero forecast (direct_insufficient), cannot backtest
        if full_result['metadata']['data_source'] == 'direct_insufficient':
            return {
                'predictions': None,
                'metrics': None,
                'metadata': full_result['metadata'],
                'success': False,
                'message': 'Cannot backtest with zero forecast (no historical data available)'
            }

        # Get the weekly data from metadata to perform backtest
        # We need to re-query and split properly
        machine_level = 'machine_id'  # Default for direct data
        if full_result['metadata']['data_source'] == 'direct':
            direct_check = self._check_direct_data(machine_id, product_name, time_period)
            transaction_data = direct_check['data']
        else:
            # Re-execute fallback
            fallback_result = self.fallback_engine.execute_fallback(
                machine_id=machine_id,
                product_name=product_name,
                time_period=time_period
            )
            transaction_data = fallback_result['data']
            machine_level = fallback_result['machine_level']  # Get level for normalization

        # Prepare full weekly data
        weekly_data = self._prepare_weekly_from_transactions(transaction_data, machine_level)

        # Check if enough data for backtesting
        if len(weekly_data) < test_weeks + 2:
            return {
                'predictions': None,
                'metrics': None,
                'metadata': full_result['metadata'],
                'success': False,
                'message': f'Insufficient data for backtesting: {len(weekly_data)} weeks '
                          f'(need at least {test_weeks + 2})'
            }

        # Perform backtest
        try:
            quantity_series = weekly_data['quantity'].reset_index(drop=True)
            backtest_result = self.sma_model.backtest(quantity_series, test_weeks=test_weeks)

            return {
                'predictions': backtest_result['predictions'],
                'metrics': backtest_result['metrics'],
                'metadata': full_result['metadata'],
                'success': True,
                'message': None
            }

        except Exception as e:
            return {
                'predictions': None,
                'metrics': None,
                'metadata': full_result['metadata'],
                'success': False,
                'message': f'Backtest error: {str(e)}'
            }

    def __repr__(self) -> str:
        """String representation of the forecaster."""
        fallback_status = "enabled" if self.use_fallback else "disabled"
        return (f"SMAForecaster(forecast_horizon={self.forecast_horizon}, "
                f"strategy='{self.strategy_name}', fallback={fallback_status})")
