import logging
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from .config import PREDICTION_STATIC_TASKS
from .healthiness_scoring import _normalize_identifier
from .moaaz_trend.predict import moaaz_predict

logger = logging.getLogger(__name__)


def _is_valid_df(df: Optional[pd.DataFrame]) -> bool:
    """Check if a DataFrame is valid (not None and not empty)."""
    return df is not None and not df.empty


def _build_machine_product_grid(
    machines_df: pd.DataFrame,
    product_df: pd.DataFrame,
) -> pd.DataFrame:
    """Create cartesian grid of machine_key x ean."""
    machines = machines_df[['machine_key']].dropna().drop_duplicates()
    products = product_df[['ean']].dropna().drop_duplicates()
    if machines.empty or products.empty:
        return pd.DataFrame()
    machines['tmp'] = 1
    products['tmp'] = 1
    grid = machines.merge(products, on='tmp').drop(columns=['tmp'])
    return grid


def predict_static(
    *,
    sales_df: Optional[pd.DataFrame] = None,
    machines_df: Optional[pd.DataFrame] = None,
    product_df: Optional[pd.DataFrame] = None,
    location_mapping_df: Optional[pd.DataFrame] = None,
    healthiness_mapping: Optional[Dict[str, Any]] = None,
    moaaz_trend_model_path: Optional[str] = None,
    prediction_overrides: Optional[Dict[str, bool]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build a full static score table for all machine_key x ean combinations.

    - Runs moaaz_predict to get forecast columns (if enabled and data present).
    - Adds location scores by matching machine_eva_group to the location_mapping.
    - Adds healthiness letter grades using the complete_mapping artifact.
    - Extracts unique EAN-letter grade mappings into a separate DataFrame.

    Returns:
        Tuple of (base_df, letter_mapping_df):
        - base_df: Static scores table with healthiness_letter column included
        - letter_mapping_df: DataFrame with columns ['ean', 'healthiness_letter'] containing
          unique EAN to letter grade mappings
    """
    plan = {**PREDICTION_STATIC_TASKS}
    if prediction_overrides:
        plan.update(prediction_overrides)

    # Base dataframe: predictions or cartesian grid fallback
    base_df = pd.DataFrame()
    grid_df = pd.DataFrame()

    if plan.get('moaaz_trend'):
        missing = []
        if not _is_valid_df(sales_df):
            missing.append('sales_df')
        if not _is_valid_df(machines_df):
            missing.append('machines_df')
        if not _is_valid_df(product_df):
            missing.append('product_df')
        if missing:
            logger.warning(f"Skipping moaaz_trend prediction: missing required data: {', '.join(missing)}")
        else:
            base_df = moaaz_predict(
                sales_df=sales_df,
                machines_df=machines_df,
                product_df=product_df,
                model_path=moaaz_trend_model_path,
            )

    # If predictions empty, fall back to a machine x product grid
    if base_df.empty and _is_valid_df(machines_df) and _is_valid_df(product_df):
        grid_df = _build_machine_product_grid(machines_df, product_df)
        base_df = grid_df.copy()

    if base_df.empty:
        logger.warning("No base data available for static prediction; returning empty DataFrames.")
        return base_df, pd.DataFrame(columns=['ean', 'healthiness_letter'])

    # Attach product_name via product_df to enable location scoring
    if 'product_name' not in base_df.columns and _is_valid_df(product_df) and 'ean' in base_df.columns:
        name_map = (
            product_df[['ean', 'product_name']]
            .dropna(subset=['ean'])
            .drop_duplicates(subset=['ean'])
            .set_index('ean')['product_name']
        )
        base_df = base_df.copy()
        base_df['product_name'] = base_df['ean'].map(name_map)

    # Add machine_eva_group to base_df
    if _is_valid_df(machines_df) and 'machine_key' in base_df.columns and 'machine_key' in machines_df.columns:
        group_map = (
            machines_df[['machine_key', 'machine_eva_group']]
            .dropna(subset=['machine_key'])
            .drop_duplicates(subset=['machine_key'])
            .set_index('machine_key')['machine_eva_group']
        )
        base_df = base_df.copy()
        base_df['machine_eva_group'] = base_df['machine_key'].map(group_map)

    # Location scores
    if plan.get('location_mapping'):
        missing_loc = []
        if not _is_valid_df(location_mapping_df):
            missing_loc.append('location_mapping_df')
        if 'product_name' not in base_df.columns:
            missing_loc.append('product_name')
        if 'machine_eva_group' not in base_df.columns:
            missing_loc.append('machine_eva_group')

        if missing_loc:
            logger.warning(f"Skipping location_mapping enrichment: missing required data: {', '.join(missing_loc)}")
        else:
            loc_df = location_mapping_df.copy()
            if 'product_name' not in loc_df.columns:
                loc_df = loc_df.reset_index().rename(columns={'index': 'product_name'})

            long_loc = loc_df.melt(
                id_vars=['product_name'],
                var_name='machine_eva_group',
                value_name='location_fit_score'
            )
            base_df = base_df.merge(
                long_loc,
                on=['product_name', 'machine_eva_group'],
                how='left'
            )

    # Healthiness letters
    letter_mapping_df = pd.DataFrame(columns=['ean', 'healthiness_letter'])
    if plan.get('healthiness_mapping'):
        if healthiness_mapping is None:
            logger.warning("Skipping healthiness_mapping enrichment: missing healthiness_mapping")
        elif 'ean' not in base_df.columns:
            logger.warning("Skipping healthiness_mapping enrichment: base data lacks 'ean'")
        else:
            base_df = base_df.copy()
            base_df['healthiness_letter'] = base_df['ean'].apply(
                lambda x: healthiness_mapping.get(_normalize_identifier(x))
            )
            # Extract unique EAN-letter grade mappings into separate DataFrame
            letter_mapping_df = (
                base_df[['ean', 'healthiness_letter']]
                .dropna(subset=['ean'])
                .drop_duplicates(subset=['ean'])
                .reset_index(drop=True)
            )

    return base_df, letter_mapping_df
