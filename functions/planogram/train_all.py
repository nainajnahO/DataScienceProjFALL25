import logging
from typing import Any, Dict, Optional

import pandas as pd

from .config import TRAINING_TASKS
from .product_scoring import train_uniqueness_model
from .location_scoring import train_location_model, predict_location_scores
from .machine_snapshots import train_snapshot_model
from .cousin_scoring import train_cousin_model
from .healthiness_model import train_healthiness_model
from .moaaz_trend.train import moaaz_train

logger = logging.getLogger(__name__)


def _is_valid_df(df: Optional[pd.DataFrame]) -> bool:
    """Check if a DataFrame is valid (not None and not empty)."""
    return df is not None and not df.empty


def train_all(
    sales_df: Optional[pd.DataFrame] = None,
    machines_df: Optional[pd.DataFrame] = None,
    products_df: Optional[pd.DataFrame] = None,
    product_information_df: Optional[pd.DataFrame] = None,
    training_overrides: Optional[Dict[str, bool]] = None,
) -> Dict[str, Any]:
    """
    Train all planogram models and return their artifacts keyed by model name.

    Training can be toggled via `planogram.config.TRAINING_TASKS` or by passing
    `training_overrides` (overrides take precedence). All data inputs are optional;
    if required data is missing for a training task, that task will be skipped with
    a warning log.

    Returns:
        Dict[str, Any]: mapping of model name -> trained artifact.
    """
    training_plan = {**TRAINING_TASKS}
    if training_overrides:
        training_plan.update(training_overrides)

    artifacts: Dict[str, Any] = {}
    
    # Uniqueness model
    if training_plan.get('uniqueness_model'):
        if not _is_valid_df(products_df):
            logger.warning("Skipping uniqueness_model training: missing required data 'products_df'")
        else:
            artifacts['uniqueness_model'] = train_uniqueness_model(products_df=products_df)
            logger.info("Trained uniqueness_model")

    # Location mapping
    if training_plan.get('location_mapping'):
        if not _is_valid_df(sales_df):
            logger.warning("Skipping location_mapping training: missing required data 'sales_df'")
        else:
            location_artifact = train_location_model(
                sales_df=sales_df,
            )
            artifacts['location_mapping'] = predict_location_scores(
                products_df=products_df,
                location_types=list(location_artifact.keys()),
                trained_model=location_artifact,
            )
            logger.info("Trained location_mapping")

    # Machine snapshot model
    if training_plan.get('snapshot_model'):
        if not _is_valid_df(sales_df):
            logger.warning("Skipping snapshot_model training: missing required data 'sales_df'")
        else:
            artifacts['snapshot_model'] = train_snapshot_model(sales_df=sales_df)
            logger.info("Trained snapshot_model")

    # Cousin model
    if training_plan.get('cousin_model'):
        if not _is_valid_df(sales_df):
            logger.warning("Skipping cousin_model training: missing required data 'sales_df'")
        else:
            artifacts['cousin_model'] = train_cousin_model(sales_df=sales_df)
            logger.info("Trained cousin_model")

    # Healthiness mapping
    if training_plan.get('healthiness_mapping'):
        missing = []
        if not _is_valid_df(product_information_df):
            missing.append('product_information_df')
        if not _is_valid_df(products_df):
            missing.append('products_df')
        
        if missing:
            logger.warning(f"Skipping healthiness_mapping training: missing required data: {', '.join(missing)}")
        else:
            artifacts['healthiness_mapping'] = train_healthiness_model(
                product_information_df=product_information_df,
                product_database_df=products_df,
            )
            logger.info("Trained healthiness_mapping")

    # Moaaz trend forecaster
    if training_plan.get('moaaz_trend'):
        missing = []
        if not _is_valid_df(sales_df):
            missing.append('sales_df')
        if not _is_valid_df(machines_df):
            missing.append('machines_df')
        
        if missing:
            logger.warning(f"Skipping moaaz_trend training: missing required data: {', '.join(missing)}")
        else:
            artifacts['moaaz_trend'] = moaaz_train(
                sales_df=sales_df,
                machines_df=machines_df,
                model_path=None,
            )
            logger.info("Trained moaaz_trend")

    return artifacts
