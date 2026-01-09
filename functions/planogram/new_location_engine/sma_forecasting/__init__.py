"""
SMA Forecasting Library

A library for Progressive Window Simple Moving Average (SMA) forecasting with fallback engine integration.

Basic Example (without fallback):
    from sma_forecasting import ProgressiveWindowSMA
    from sma_forecasting.utils import prepare_weekly_data, load_sales_data

    # Load and prepare data
    sales_data = load_sales_data(['sales_2023.parquet', 'sales_2024.parquet'])
    weekly_data = prepare_weekly_data(sales_data, product='Twix 50g', machine=976881334)

    # Create and fit model
    model = ProgressiveWindowSMA(forecast_horizon=4)
    model.fit(weekly_data['quantity'])

    # Generate forecast
    forecast_df = model.forecast()
    print(forecast_df)

    # Backtest
    results = model.backtest(weekly_data['quantity'], test_weeks=4)
    print(f"MAE: {results['metrics']['MAE']:.2f}")
    print(f"RMSE: {results['metrics']['RMSE']:.2f}")

Advanced Example (with fallback engine):
    from sma_forecasting import SMAForecaster

    # Create forecaster with fallback enabled
    forecaster = SMAForecaster(sales_df, use_fallback=True, strategy='product_first')

    # Generate forecast
    result = forecaster.forecast(machine_id=123, product_name='Twix 50g')

    if result['success']:
        print(result['forecast'])
        print(f"Data source: {result['metadata']['data_source']}")
        if result['metadata']['data_source'] == 'fallback':
            print(f"Confidence: {result['metadata']['fallback_info']['confidence']:.1%}")
"""

from .progressive_window_sma import ProgressiveWindowSMA
from .fallback_integration import SMAForecaster
from . import utils

__version__ = "2.0.0"
__all__ = ['ProgressiveWindowSMA', 'SMAForecaster', 'utils']
