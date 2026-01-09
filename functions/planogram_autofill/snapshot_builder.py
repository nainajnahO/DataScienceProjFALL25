"""Snapshots builder module.

Handles loading raw sales data and aggregating it into weekly machine snapshots
for training the sales prediction model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, List

import pandas as pd

from .config import (
    DATA_YEARS,
    DEFAULT_SNAPSHOT_OUTPUT,
    LOOKBACK_WEEKS,
    MACHINE_SNAPSHOT_DIR,
    MIN_SALES_PER_POSITION,
    REQUIRED_COLUMNS,
    SNAPSHOT_FREQ,
    COLUMN_ALIASES
)

try:
    from planogram.data_loader import load_processed_sales
except ImportError:
    # Fallback or local import if planogram package is not installed as package
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from planogram.data_loader import load_processed_sales


DEFAULT_METADATA_COLS = [
    "product_name",
    "provider",
    "category",
    "subcategory",
    "machine_eva_group",
    "machine_sub_group",
    "purchase_price_kr",
    "price",
]


def load_sales_data_filtered(
    years: Optional[Iterable[int]] = None,
) -> pd.DataFrame:
    """
    Load processed sales data using planogram.data_loader and filter by years.
    """
    print("📥 Loading sales data...")
    df = load_processed_sales()
    print(f"   ✓ Loaded {len(df):,} raw transactions")
    
    if df.empty:
        return df
    
    # Standardize columns immediately to check dates
    print("   🔧 Standardizing columns...")
    df = _standardise_columns(df)
    
    if "event_ts" in df.columns:
        print("   📅 Parsing dates...")
        df["event_ts"] = pd.to_datetime(df["event_ts"], errors="coerce")
        
        if years:
            target_years = list(years)
            print(f"   🔍 Filtering to years: {target_years}...")
            before_filter = len(df)
            df = df[df["event_ts"].dt.year.isin(target_years)]
            print(f"   ✓ Filtered to {len(df):,} rows ({len(df)/before_filter*100:.1f}% of original)")
    
    return df


def build_weekly_snapshots(
    sales_df: pd.DataFrame,
    *,
    snapshot_freq: str = SNAPSHOT_FREQ,
    metadata_cols: Optional[Iterable[str]] = None,
    min_sales: int = MIN_SALES_PER_POSITION,
    lookback_weeks: int = LOOKBACK_WEEKS,
) -> pd.DataFrame:
    """Aggregate transactions into weekly machine/position snapshots."""

    if sales_df.empty:
        return pd.DataFrame()

    df = sales_df.copy()
    # Columns standardized in load_sales_data_filtered, but safeguard here
    df = _standardise_columns(df)
    _validate_columns(df)

    df["event_ts"] = pd.to_datetime(df["event_ts"], errors="coerce")
    df = df.dropna(subset=["event_ts", "machine_id", "position"])

    print("   📆 Creating snapshot dates...")
    df["snapshot_date"] = df["event_ts"].dt.to_period(snapshot_freq).dt.start_time
    df["sales_count"] = 1

    metadata_cols = list(metadata_cols) if metadata_cols else DEFAULT_METADATA_COLS
    available_metadata = [col for col in metadata_cols if col in df.columns]

    agg_map = {"sales_count": "sum"}
    numeric_summary_cols = ["price", "purchase_price_kr"]
    for col in numeric_summary_cols:
        if col in df.columns:
            agg_map[col] = "mean"
    for col in available_metadata:
        if col not in agg_map:
            agg_map[col] = "last"

    group_cols = ["machine_id", "snapshot_date", "position", "ean"]
    print(f"   📊 Aggregating to weekly snapshots (grouping by {len(group_cols)} columns)...")
    print(f"      Input: {len(df):,} transactions")
    print(f"      This may take a moment for large datasets...")
    
    weekly = df.groupby(group_cols, as_index=False).agg(agg_map)
    
    print(f"      ✓ Output: {len(weekly):,} snapshots")
    weekly = weekly.rename(columns={"sales_count": "weekly_sales", "price": "price_mean"})

    weekly["weekly_sales"] = weekly["weekly_sales"].astype("int64")
    if min_sales > 0:
        weekly = weekly[weekly["weekly_sales"] >= min_sales]

    weekly["lookback_weeks"] = lookback_weeks
    weekly = weekly.sort_values(["machine_id", "snapshot_date", "position"])

    return weekly.reset_index(drop=True)


def load_machine_snapshot_parquet(path: Optional[Path] = None) -> pd.DataFrame:
    """Load the canonical machine snapshot parquet produced by machine_snapshots."""
    # This logic was in planogram_autofill/data/loader.py
    default_path = MACHINE_SNAPSHOT_DIR / "machine_snapshots.parquet"
    source_path = Path(path) if path else default_path
    if not source_path.exists():
        # It's okay if it doesn't exist, maybe we just return empty or raise error
        # depending on strictness. Original code raised error.
        raise FileNotFoundError(
            f"Machine snapshot parquet not found at {source_path}. "
        )
    return pd.read_parquet(source_path)


def get_machine_snapshot_date_range(path: Optional[Path] = None) -> tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """Get the date range covered by the machine_snapshots.parquet file."""
    try:
        df = load_machine_snapshot_parquet(path=path)
    except FileNotFoundError:
        return None, None

    if df.empty or "snapshot_date" not in df.columns:
        return None, None
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"], errors="coerce")
    if df["snapshot_date"].isna().all():
        return None, None
    return df["snapshot_date"].min(), df["snapshot_date"].max()


def restrict_to_machine_snapshot_range(
    df: pd.DataFrame,
    *,
    path: Optional[Path] = None,
) -> pd.DataFrame:
    """Restrict a dataframe to the date range covered by machine_snapshots.parquet."""
    start, end = get_machine_snapshot_date_range(path)
    if start is None or end is None or "snapshot_date" not in df.columns:
        return df
    frame = df.copy()
    frame["snapshot_date"] = pd.to_datetime(frame["snapshot_date"], errors="coerce")
    mask = (frame["snapshot_date"] >= start) & (frame["snapshot_date"] <= end)
    return frame.loc[mask].reset_index(drop=True)


def save_snapshots(df: pd.DataFrame, path: Optional[Path] = None) -> Path:
    """Persist weekly snapshots to parquet and return the file path."""
    output_path = Path(path) if path else DEFAULT_SNAPSHOT_OUTPUT
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    return output_path


def _standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for original, new in COLUMN_ALIASES.items():
        if original in df.columns:
             # Special case if both exist (e.g. machine_key and machine_id)
            if original == "machine_key" and "machine_id" in df.columns:
                # machine_key is often string UUID, machine_id int
                # If both exist, usually we want to keep one as primary.
                # Config says: "machine_key": "machine_id"
                # If we map machine_key -> machine_id, we collide.
                # Original logic in loader.py was:
                # if original == "machine_key" and "machine_id" in df.columns:
                #    df["machine_key"] = df["machine_key"].astype(str)
                #    continue
                
                # Adapting that here:
                df["machine_key"] = df["machine_key"].astype(str)
                continue
            
            if new in df.columns:
                 # If target column already exists, drop it or assume it's the same?
                 # Usually safest to drop the target if we are renaming source to target
                df = df.drop(columns=[new])
            
            df = df.rename(columns={original: new})
            
    return df


def _validate_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Sales dataframe missing required columns: {missing}")

