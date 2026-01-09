import logging
from pathlib import Path
from typing import Optional, Union
import pandas as pd

from .models.multi_week_forecaster import MultiWeekForecaster
from .features.tier1_features import create_all_features
from .data.processor import process_sales_data

logger = logging.getLogger(__name__)

def moaaz_train(
    sales_df: pd.DataFrame,
    machines_df: pd.DataFrame,
    model_path: Optional[Union[str, Path]] = None
) -> MultiWeekForecaster:
    """
    Train the multi-week forecaster using the full pipeline.
    
    Args:
        sales_df: Historical sales data
        machines_df: Machine metadata
        model_path: Path to save the trained model artifact
        
    Returns:
        Trained MultiWeekForecaster object
    """
    logger.info("Starting moaaz_train pipeline...")
    
    # 1. Data Preparation & Merging
    sales_df = sales_df.copy()
    machines_df = machines_df.copy()

    if 'ean' in sales_df.columns:
        sales_df['ean'] = pd.to_numeric(sales_df['ean'], errors='coerce')

    # 2. Pipeline Processing (Cleaning, Snapshots, etc.)
    # use_cache=False to prevent saving to disk
    logger.info(f"Processing sales data...")
    processed_df = process_sales_data(
        sales_df,
        use_cache=False,
        show_progress=True
    )
    
    # 3. Feature Engineering
    logger.info("Generating features...")
    features_df = create_all_features(processed_df, use_cold_start_fallback=True)
    
    # 4. Training
    logger.info("Training model...")
    forecaster = MultiWeekForecaster(
        horizons=[1, 2, 3, 4],
        strategy='recursive_multi' # Using the advanced strategy
    )

    forecaster.fit(features_df)
    
    # 5. Save Model
    if model_path:
        save_path = Path(model_path)
        forecaster.save(save_path)
        logger.info(f"Model saved to {save_path}")
    
    return forecaster
