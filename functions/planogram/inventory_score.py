"""Inventory synchronization scoring (planogram-compatible, single file).

This module exposes a pure `calculate_inventory_scores` function that scores how
well products in a machine run out at the same time. It follows the
new_feature_implementation guide:
- No data loading or file I/O
- Inputs are DataFrames; prediction artifacts can be passed in
- Copies input DataFrames before mutation
"""

from __future__ import annotations

from typing import Optional, Union
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _extract_machine_products(machines_df: pd.DataFrame, products_df: pd.DataFrame) -> pd.DataFrame:
    """Extract product rows from slots in machines_df, mapping product_name -> ean when needed."""
    if machines_df.empty or "slots" not in machines_df.columns or "machine_key" not in machines_df.columns:
        return pd.DataFrame(columns=["machine_key", "ean", "position"])

    # Build product_name -> ean lookup (case-insensitive), prefer first occurrence.
    name_to_ean: dict[str, float] = {}
    if not products_df.empty and {"product_name", "ean"}.issubset(products_df.columns):
        for _, prow in products_df.dropna(subset=["product_name", "ean"]).iterrows():
            pname = str(prow["product_name"]).strip().lower()
            if pname and pname not in name_to_ean:
                name_to_ean[pname] = prow["ean"]

    records = []
    for _, row in machines_df.iterrows():
        machine_key = row["machine_key"]
        slots = row.get("slots", [])
        if not isinstance(slots, list):
            continue
        for slot in slots:
            if not isinstance(slot, dict):
                continue
            ean_val = slot.get("ean")
            if ean_val is None and "product_name" in slot:
                pname = str(slot.get("product_name", "")).strip().lower()
                if pname:
                    ean_val = name_to_ean.get(pname)
            if ean_val is None:
                continue
            records.append(
                {
                    "machine_key": machine_key,
                    "ean": ean_val,
                    "position": slot.get("position"),
                }
            )

    if not records:
        return pd.DataFrame(columns=["machine_key", "ean", "position"])

    df = pd.DataFrame(records)
    df["ean"] = pd.to_numeric(df["ean"], errors="coerce")
    return df.dropna(subset=["ean"])


def _calculate_runout_times(
    products_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    machine_products: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate days until run-out for each product in a machine."""
    products_df = products_df.copy()
    predictions_df = predictions_df.copy()
    machine_products = machine_products.copy()

    if machine_products.empty or predictions_df.empty:
        return pd.DataFrame()

    result = machine_products.merge(products_df[["ean", "spiral"]], on="ean", how="left")

    if "predicted_sales_score" not in predictions_df.columns:
        pred_cols = [c for c in predictions_df.columns if c.startswith("pred_")]
        if pred_cols:
            predictions_df["predicted_sales_score"] = predictions_df[pred_cols].sum(axis=1)
        else:
            logger.warning("No prediction columns found")
            return pd.DataFrame()

    merge_cols = ["machine_key", "ean"]
    if "position" in predictions_df.columns and "position" in result.columns:
        merge_cols.append("position")

    result = result.merge(
        predictions_df[merge_cols + ["predicted_sales_score"]],
        on=merge_cols,
        how="left",
    ).rename(columns={"predicted_sales_score": "predicted_weekly_sales"})

    result["days_until_runout"] = np.where(
        (result["predicted_weekly_sales"] > 0)
        & (result["spiral"].notna())
        & (result["spiral"] > 0),
        (result["spiral"] / result["predicted_weekly_sales"]) * 7,
        np.where(result["predicted_weekly_sales"] <= 0, 999.0, np.nan),
    )

    return result


def _score_synchronization(runout_times: pd.Series) -> float:
    """Score synchronization based on the coefficient of variation."""
    valid_times = runout_times.dropna()

    if len(valid_times) < 2:
        return 0.0

    std_dev = valid_times.std()
    mean_time = valid_times.mean()

    if std_dev == 0:
        return 1.0
    if mean_time == 0 or np.isnan(mean_time):
        return 0.0

    cv = std_dev / mean_time
    score = 1.0 / (1.0 + cv)
    return max(0.0, min(1.0, score))


def _score_machine_from_products(
    machine_products: pd.DataFrame,
    all_predictions: pd.DataFrame,
    products_df: pd.DataFrame,
    machine_key: str,
) -> Optional[float]:
    """Score a single machine given its products; return None if not possible."""
    predictions_df = all_predictions.copy()
    
    if predictions_df.empty:
        return None

    runout_df = _calculate_runout_times(products_df, predictions_df, machine_products)
    if runout_df.empty:
        return None

    valid_runout_times = runout_df["days_until_runout"].dropna()
    if len(valid_runout_times) < 2:
        return None

    return _score_synchronization(valid_runout_times)


def _score_single_machine(
    machines_df: pd.DataFrame,
    all_predictions: pd.DataFrame,
    products_df: pd.DataFrame,
) -> float:
    """Score one machine and return a float 0-1."""
    machine_products = _extract_machine_products(machines_df, products_df)
    if machine_products.empty:
        logger.warning("No products found in machines_df")
        return 0.0

    machine_key = machine_products["machine_key"].iloc[0]
    if pd.isna(machine_key):
        logger.warning("No machine_key found")
        return 0.0

    score = _score_machine_from_products(machine_products, all_predictions, products_df, machine_key)

    if score is None:
        logger.warning(f"No predictions found for machine {machine_key}")
        return 0.0

    return score


def _score_multiple_machines(
    machines_df: pd.DataFrame,
    all_predictions: pd.DataFrame,
    products_df: pd.DataFrame,
) -> pd.DataFrame:
    """Score multiple machines; return DataFrame with machine_key and inventory_score."""
    all_machine_products = _extract_machine_products(machines_df, products_df)
    if all_machine_products.empty:
        logger.warning("No products found in machine_config")
        return pd.DataFrame(columns=["machine_key", "inventory_score"])

    all_predictions = all_predictions.copy()
    all_machine_products = all_machine_products.copy()

    results = []
    for machine_key in all_machine_products["machine_key"].unique():
        if pd.isna(machine_key):
            continue

        machine_products = all_machine_products[
            all_machine_products["machine_key"] == machine_key
        ].copy()

        if machine_products.empty:
            continue

        score = _score_machine_from_products(machine_products, all_predictions, products_df, machine_key)
        results.append({"machine_key": machine_key, "inventory_score": score})

    return pd.DataFrame(results) if results else pd.DataFrame(columns=["machine_key", "inventory_score"])


def calculate_inventory_scores(
    machines_df: pd.DataFrame,
    products_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
) -> Union[float, pd.DataFrame]:
    """Calculate inventory synchronization scores (0-1) for machines.

    Args:
        machines_df: DataFrame of machines to score (must include machine_key, ean, position).
        products_df: Products DataFrame with 'ean' and 'spiral' columns (required).
        predictions_df: Pre-computed predictions DataFrame (required).

    Returns:
        Float for single machine, or DataFrame with ['machine_key', 'inventory_score'].
    """
    products_df = products_df.copy()

    if products_df.empty:
        logger.warning("products_df is empty; cannot score")
        return pd.DataFrame(columns=["machine_key", "inventory_score"]) if isinstance(
            machines_df, pd.DataFrame
        ) else 0.0

    if "spiral" not in products_df.columns:
        logger.warning("'spiral' column missing in products_df; defaulting to 0")
        products_df["spiral"] = 0

    if predictions_df is None or predictions_df.empty:
        logger.error("predictions_df is required and cannot be empty")
        return pd.DataFrame(columns=["machine_key", "inventory_score"]) if isinstance(
            machines_df, pd.DataFrame
        ) else 0.0

    all_predictions = predictions_df.copy()
    #logger.info(f"Using provided predictions DataFrame ({len(all_predictions)} rows)")

    is_single_machine = len(machines_df) == 1

    if is_single_machine:
        return _score_single_machine(
            machines_df=machines_df,
            all_predictions=all_predictions,
            products_df=products_df,
        )

    return _score_multiple_machines(
        machines_df=machines_df,
        all_predictions=all_predictions,
        products_df=products_df,
    )
