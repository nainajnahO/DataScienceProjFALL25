"""
Progressive Window Simple Moving Average (SMA) Forecasting

This module implements a Progressive Window SMA forecasting model where:
- Week 1 ahead: Uses SMA of last 1 week
- Week 2 ahead: Uses SMA of last 2 weeks
- Week 3 ahead: Uses SMA of last 3 weeks
- Week N ahead: Uses SMA of last N weeks
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple


class ProgressiveWindowSMA:
    """
    Progressive Window Simple Moving Average Forecaster.

    Uses expanding windows for multi-step ahead forecasting where each forecast
    week N uses the average of the last N weeks of historical data.

    Attributes:
        historical_data: The historical data used for fitting
        forecast_horizon: Number of weeks to forecast ahead
        forecasts: Generated forecasts (available after calling forecast())
    """

    def __init__(self, forecast_horizon: int = 4):
        """
        Initialize the Progressive Window SMA forecaster.

        Args:
            forecast_horizon: Number of weeks to forecast ahead (default: 4)
        """
        if forecast_horizon < 1:
            raise ValueError("forecast_horizon must be at least 1")

        self.forecast_horizon = forecast_horizon
        self.historical_data: Optional[pd.Series] = None
        self.forecasts: Optional[pd.DataFrame] = None

    def fit(self, data: pd.Series) -> 'ProgressiveWindowSMA':
        """
        Fit the model with historical data.

        Args:
            data: Historical time series data (pandas Series with quantity values)

        Returns:
            self: The fitted model instance
        """
        if len(data) == 0:
            raise ValueError("Input data cannot be empty")

        self.historical_data = data.copy()
        return self

    def forecast(self, round_decimals: int = 1) -> pd.DataFrame:
        """
        Generate progressive window SMA forecasts.

        Args:
            round_decimals: Number of decimal places to round forecasts (default: 1)

        Returns:
            DataFrame with columns: week_number, forecast, window_used

        Raises:
            RuntimeError: If model hasn't been fitted yet
        """
        if self.historical_data is None:
            raise RuntimeError("Model must be fitted before forecasting. Call fit() first.")

        forecasts = []
        for week_num in range(1, self.forecast_horizon + 1):
            # Progressive window: week 1 uses last 1 week, week 2 uses last 2 weeks, etc.
            window = min(week_num, len(self.historical_data))
            forecast_value = self.historical_data.iloc[-window:].mean()

            forecasts.append({
                'week_number': week_num,
                'forecast': round(forecast_value, round_decimals),
                'window_used': window
            })

        self.forecasts = pd.DataFrame(forecasts)
        return self.forecasts

    def fit_forecast(self, data: pd.Series, round_decimals: int = 1) -> pd.DataFrame:
        """
        Convenience method to fit and forecast in one step.

        Args:
            data: Historical time series data
            round_decimals: Number of decimal places to round forecasts

        Returns:
            DataFrame with forecast results
        """
        return self.fit(data).forecast(round_decimals)

    def backtest(self,
                 full_data: pd.Series,
                 test_weeks: Optional[int] = None) -> Dict[str, any]:
        """
        Perform backtesting by comparing forecasts with actual values.

        Args:
            full_data: Complete time series including both training and test data
            test_weeks: Number of weeks to use for testing (default: forecast_horizon)

        Returns:
            Dictionary containing:
                - predictions: DataFrame with actual, forecast, and error columns
                - metrics: Dictionary with MAE and RMSE
                - test_data: The actual test values

        Raises:
            ValueError: If test_weeks is larger than available data
        """
        if test_weeks is None:
            test_weeks = self.forecast_horizon

        if test_weeks >= len(full_data):
            raise ValueError(f"test_weeks ({test_weeks}) must be less than data length ({len(full_data)})")

        # Split data
        train_data = full_data.iloc[:-test_weeks]
        test_data = full_data.iloc[-test_weeks:]

        # Generate forecasts
        self.fit(train_data)
        forecast_df = self.forecast()

        # Limit to available test data
        n_compare = min(len(forecast_df), len(test_data))

        # Create comparison dataframe
        results = pd.DataFrame({
            'actual': test_data.iloc[:n_compare].values,
            'forecast': forecast_df['forecast'].iloc[:n_compare].values
        })

        results['error'] = results['actual'] - results['forecast']
        results['abs_error'] = results['error'].abs()
        results['squared_error'] = results['error'] ** 2

        # Calculate metrics
        mae = results['abs_error'].mean()
        rmse = np.sqrt(results['squared_error'].mean())

        return {
            'predictions': results,
            'metrics': {
                'MAE': mae,
                'RMSE': rmse
            },
            'test_data': test_data
        }

    def get_forecast_values(self) -> Optional[List[float]]:
        """
        Get the forecast values as a list.

        Returns:
            List of forecast values, or None if forecast hasn't been generated
        """
        if self.forecasts is None:
            return None
        return self.forecasts['forecast'].tolist()

    def __repr__(self) -> str:
        """String representation of the model."""
        fitted = "fitted" if self.historical_data is not None else "not fitted"
        return f"ProgressiveWindowSMA(forecast_horizon={self.forecast_horizon}, status={fitted})"
