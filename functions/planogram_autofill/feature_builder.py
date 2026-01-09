"""Feature builder module.

Consolidates logic for joining various feature groups (Location, Uniqueness) 
and constructing the final training matrix. Uses planogram package's Train/Predict pattern.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Protocol, Any, Tuple

import pandas as pd
import numpy as np
import pickle

try:
    from tqdm.auto import tqdm
except ImportError:
    # Fallback if tqdm is not installed
    def tqdm(iterable, desc=None, unit=None, **kwargs):
        return iterable

from .config import (
    DEFAULT_FEATURE_DATASET,
    IDENTITY_COLUMNS,
    TARGET_COLUMN,
    UNIQUE_SLOT_SCORES,
    MACHINE_SNAPSHOT_DIR,
    LOCATION_MODEL_ARTIFACT_PATH,
    UNIQUENESS_MODEL_ARTIFACT_PATH
)

# Import planogram functions
try:
    from planogram.location_scoring import predict_location_scores, LocationModel
    from planogram.product_scoring import predict_uniqueness_scores, UniquenessModel
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from planogram.location_scoring import predict_location_scores, LocationModel
    from planogram.product_scoring import predict_uniqueness_scores, UniquenessModel


# Define shared Feature Groups Registry locally or reuse logic
FEATURE_GROUPS: Dict[str, Dict[str, List[str]]] = {
    "CORE": {
        "numeric": ["price_mean", "purchase_price_kr"],
        "categorical": ["category", "subcategory", "provider"],
    },
    "TEMPORAL": {
        "numeric": ["lookback_weeks", "week_of_year", "month", "day_of_week"],
        "categorical": [],
    },
    "UNIQUENESS": {
        "numeric": ["uniqueness_score", "category_diversity_score"],
        "categorical": [],
    },
    "LOCATION": {
        "numeric": ["location_fit_score"],
        "categorical": [],
    },
    "HISTORICAL_SALES": {
        "numeric": [
            "product_sales_lag_1", "product_sales_lag_2", "product_sales_lag_3", "product_sales_lag_4",
            "product_sales_ewma_4", "product_sales_ewma_8",
            "product_sales_rolling_std_4", "product_sales_rolling_std_8",
            "product_machine_sales_lag_1", "product_machine_sales_lag_2",
            "product_machine_sales_lag_3", "product_machine_sales_lag_4",
            "product_machine_sales_ewma_4", "product_machine_sales_ewma_8",
            "product_machine_sales_rolling_std_4", "product_machine_sales_rolling_std_8",
        ],
        "categorical": [],
    },
}


def _add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive temporal columns from snapshot_date without relying on historical feature module."""
    if df.empty or "snapshot_date" not in df.columns:
        return df.copy()

    frame = df.copy()
    frame["snapshot_date"] = pd.to_datetime(frame["snapshot_date"], errors="coerce")

    frame["week_of_year"] = frame["snapshot_date"].dt.isocalendar().week
    frame["month"] = frame["snapshot_date"].dt.month
    frame["day_of_week"] = frame["snapshot_date"].dt.dayofweek
    frame["lookback_weeks"] = frame.get("lookback_weeks", pd.Series(4, index=frame.index))
    return frame


def create_historical_sales_features(
    df: pd.DataFrame, 
    use_cold_start_fallback: bool = True
) -> pd.DataFrame:
    """
    Create historical sales pattern features for products (across all machines) 
    and product-machine combinations.
    
    Features Created (16 total):
    
    Product-Level (8 - aggregated across ALL machines where product appears):
    - product_sales_lag_1, product_sales_lag_2, product_sales_lag_3, product_sales_lag_4
    - product_sales_ewma_4, product_sales_ewma_8
    - product_sales_rolling_std_4, product_sales_rolling_std_8
    
    Product-Machine Level (8 - specific to product in THIS machine):
    - product_machine_sales_lag_1, product_machine_sales_lag_2, 
      product_machine_sales_lag_3, product_machine_sales_lag_4
    - product_machine_sales_ewma_4, product_machine_sales_ewma_8
    - product_machine_sales_rolling_std_4, product_machine_sales_rolling_std_8
    
    If use_cold_start_fallback=True, missing values will be filled using hierarchical
    fallback (category/subcategory averages) for new products.
    
    Args:
        df: DataFrame with columns: ean, machine_id, snapshot_date, weekly_sales, 
            category, subcategory
        use_cold_start_fallback: Whether to apply cold-start fallback logic
        
    Returns:
        DataFrame with historical sales features added
    """
    df = df.copy()
    
    # Ensure proper sorting for time-based operations
    if 'snapshot_date' not in df.columns:
        print("      ⚠️  snapshot_date not found, skipping historical features")
        return df
    
    df = df.sort_values(['ean', 'machine_id', 'snapshot_date']).reset_index(drop=True)
    
    if 'weekly_sales' not in df.columns:
        print("      ⚠️  weekly_sales not found, skipping historical features")
        return df

    # ============================================================================
    # PRODUCT-LEVEL FEATURES (across all machines)
    # ============================================================================
    
    # Product sales aggregated across all machines per week
    product_weekly = df.groupby(['ean', 'snapshot_date'])['weekly_sales'].sum().reset_index()
    product_weekly.columns = ['ean', 'snapshot_date', 'product_total_sales']
    df = df.merge(product_weekly, on=['ean', 'snapshot_date'], how='left')
    
    # Group by product for lag calculations
    product_grouped = df.groupby('ean')['product_total_sales']
    
    # 1-4. Product lags (vectorized)
    for lag in [1, 2, 3, 4]:
        df[f'product_sales_lag_{lag}'] = product_grouped.shift(lag)
    
    # 5-6. Product EWMA (vectorized)
    for span in [4, 8]:
        alpha = 2 / (span + 1)
        df[f'product_sales_ewma_{span}'] = (
            product_grouped.shift(1)
            .transform(lambda x: x.ewm(alpha=alpha, adjust=False).mean())
        )
    
    # 7-8. Product rolling std (vectorized)
    for window in [4, 8]:
        df[f'product_sales_rolling_std_{window}'] = (
            product_grouped.shift(1)
            .transform(lambda x: x.rolling(window=window, min_periods=1).std())
        )
    
    df = df.drop('product_total_sales', axis=1)
    
    # ============================================================================
    # PRODUCT-MACHINE LEVEL FEATURES (specific to this machine)
    # ============================================================================
    
    # Group by machine and product for machine-specific lags
    pm_grouped = df.groupby(['machine_id', 'ean'])['weekly_sales']
    
    # 1-4. Product-machine lags
    for lag in [1, 2, 3, 4]:
        df[f'product_machine_sales_lag_{lag}'] = pm_grouped.shift(lag)
    
    # 5-6. Product-machine EWMA (vectorized)
    for span in [4, 8]:
        alpha = 2 / (span + 1)
        df[f'product_machine_sales_ewma_{span}'] = (
            pm_grouped.shift(1)
            .transform(lambda x: x.ewm(alpha=alpha, adjust=False).mean())
        )
    
    # 7-8. Product-machine rolling std (vectorized)
    for window in [4, 8]:
        df[f'product_machine_sales_rolling_std_{window}'] = (
            pm_grouped.shift(1)
            .transform(lambda x: x.rolling(window=window, min_periods=1).std())
        )
    
    # ============================================================================
    # COLD-START FALLBACK (if enabled)
    # ============================================================================
    
    if use_cold_start_fallback:
        df = _apply_cold_start_fallback(df)
    
    return df


def _apply_cold_start_fallback(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply hierarchical fallback to fill missing sales_lag values for new products.
    
    Fallback hierarchy:
    1. Same product, other machines (product_sales_lag_X)
    2. Subcategory in same machine (rolling mean)
    3. Category in same machine (rolling mean)
    4. Subcategory across all machines (rolling mean)
    5. Category across all machines (rolling mean)
    6. Zero/default
    
    Args:
        df: DataFrame with historical sales features
        
    Returns:
        DataFrame with cold-start fallback applied
    """
    df = df.copy()
    
    if 'weekly_sales' not in df.columns:
        return df

    # List of lag features that need cold-start fallback
    lag_features = (
        [f'product_sales_lag_{i}' for i in [1, 2, 3, 4]] +
        [f'product_machine_sales_lag_{i}' for i in [1, 2, 3, 4]]
    )
    
    for feature in lag_features:
        if feature not in df.columns:
            continue
        
        missing_mask = df[feature].isna()
        
        if not missing_mask.any():
            continue
        
        # For product_machine features, try fallback to product-level first
        if 'product_machine' in feature:
            # Level 1: Same product, other machines (use product_sales_lag_X)
            product_feature = feature.replace('product_machine', 'product')
            if product_feature in df.columns:
                mask_l1 = missing_mask & df[product_feature].notna()
                df.loc[mask_l1, feature] = df.loc[mask_l1, product_feature]
                missing_mask = df[feature].isna()
        
        # Level 2: Subcategory in same machine (if category/subcategory available)
        if 'category' in df.columns and 'subcategory' in df.columns:
            if 'product_machine' in feature:
                # For product-machine features, use subcategory in same machine
                subcat_machine_mean = (
                    df.groupby(['machine_id', 'subcategory', 'snapshot_date'])['weekly_sales']
                    .transform(lambda x: x.shift(1).rolling(4, min_periods=1).mean())
                )
                mask_l2 = missing_mask & subcat_machine_mean.notna()
                df.loc[mask_l2, feature] = subcat_machine_mean[mask_l2]
                missing_mask = df[feature].isna()
            
            # Level 3: Category in same machine
            if 'product_machine' in feature:
                cat_machine_mean = (
                    df.groupby(['machine_id', 'category', 'snapshot_date'])['weekly_sales']
                    .transform(lambda x: x.shift(1).rolling(4, min_periods=1).mean())
                )
                mask_l3 = missing_mask & cat_machine_mean.notna()
                df.loc[mask_l3, feature] = cat_machine_mean[mask_l3]
                missing_mask = df[feature].isna()
            
            # Level 4: Subcategory across all machines
            subcat_global_mean = (
                df.groupby(['subcategory', 'snapshot_date'])['weekly_sales']
                .transform(lambda x: x.shift(1).rolling(4, min_periods=1).mean())
            )
            mask_l4 = missing_mask & subcat_global_mean.notna()
            df.loc[mask_l4, feature] = subcat_global_mean[mask_l4]
            missing_mask = df[feature].isna()
            
            # Level 5: Category across all machines
            cat_global_mean = (
                df.groupby(['category', 'snapshot_date'])['weekly_sales']
                .transform(lambda x: x.shift(1).rolling(4, min_periods=1).mean())
            )
            mask_l5 = missing_mask & cat_global_mean.notna()
            df.loc[mask_l5, feature] = cat_global_mean[mask_l5]
            missing_mask = df[feature].isna()
        
        # Level 6: Zero/default (fill remaining NaN with 0)
        df.loc[missing_mask, feature] = 0.0
    
    # Fill NaN in EWMA and rolling_std features with 0
    for feature in df.columns:
        if ('ewma' in feature or 'rolling_std' in feature) and 'sales' in feature:
            df[feature] = df[feature].fillna(0.0)
    
    return df


def get_feature_columns(groups: Iterable[str]) -> List[str]:
    columns: List[str] = []
    for group in groups:
        group_def = FEATURE_GROUPS.get(group)
        if not group_def:
            raise ValueError(f"Unknown feature group: {group}")
        columns.extend(group_def.get("numeric", []))
        columns.extend(group_def.get("categorical", []))
    return sorted(set(columns))


def get_numeric_features(groups: Iterable[str]) -> List[str]:
    numeric: List[str] = []
    for group in groups:
        numeric.extend(FEATURE_GROUPS.get(group, {}).get("numeric", []))
    return sorted(set(numeric))


def get_categorical_features(groups: Iterable[str]) -> List[str]:
    categorical: List[str] = []
    for group in groups:
        categorical.extend(FEATURE_GROUPS.get(group, {}).get("categorical", []))
    return sorted(set(categorical))


class FeatureJoiner(Protocol):
    """Protocol for classes that add features to a dataframe."""
    def join(self, df: pd.DataFrame) -> pd.DataFrame:
        ...


class UniquenessJoiner:
    """Feature joiner for machine-slot uniqueness scores.
    
    Can work in two modes:
    1. Load pre-computed scores from parquet file (default, for inference)
    2. Compute scores on-the-fly using planogram.product_scoring (requires trained model)
    """

    def __init__(
        self, 
        path: Optional[Path] = None, 
        columns: Optional[List[str]] = None,
        trained_model: Optional[UniquenessModel] = None,
        trained_model_path: Optional[Path] = None
    ):
        self.path = Path(path) if path else UNIQUE_SLOT_SCORES
        self.columns = columns or [
            "machine_id",
            "snapshot_date",
            "position",
            "uniqueness_score",
            "category_diversity_score",
        ]
        self.trained_model = trained_model  # Optionally provide trained model for on-the-fly computation
        self.trained_model_path = Path(trained_model_path) if trained_model_path else UNIQUENESS_MODEL_ARTIFACT_PATH
        self._df: Optional[pd.DataFrame] = None
        if self.trained_model is None and self.trained_model_path and self.trained_model_path.exists():
            self.trained_model = _load_pickle_artifact(self.trained_model_path)
            print(f"✓ Loaded uniqueness model from planogram artifact: {self.trained_model_path}")

    def load_from_file(self) -> pd.DataFrame:
        """Load pre-computed uniqueness scores from parquet file."""
        if not self.path.exists():
            raise FileNotFoundError(
                f"Uniqueness scores not found at {self.path}. "
                "Run parquet_unique_product_score pipeline first, or provide a trained_model."
            )

        df = pd.read_parquet(self.path, columns=list(self.columns))
        if "snapshot_date" in df.columns:
            df["snapshot_date"] = pd.to_datetime(df["snapshot_date"], errors="coerce")
        if "position" in df.columns:
            df["position"] = pd.to_numeric(df["position"], errors="coerce").astype("Int64")
        if "machine_id" in df.columns:
            df["machine_id"] = df["machine_id"].astype(str)
        return df

    def compute_from_model(self, snapshots: pd.DataFrame) -> pd.DataFrame:
        """Compute uniqueness scores on-the-fly using planogram's predict function."""
        if self.trained_model is None:
            raise ValueError("trained_model is required for on-the-fly computation")
        
        # Prepare machine products format expected by planogram.product_scoring
        # Group by machine_id and snapshot_date, then compute scores
        all_scores = []
        
        # Count total groups for progress bar
        groups = list(snapshots.groupby(["machine_id", "snapshot_date"]))
        total_groups = len(groups)
        
        if total_groups == 0:
            return pd.DataFrame(columns=self.columns)
        
        print(f"      Processing {total_groups:,} machine-date combinations...")
        
        for (machine_id, snapshot_date), group in tqdm(groups, desc="      Computing uniqueness scores", unit=" machine-dates"):
            # Convert snapshot group to format expected by predict_uniqueness_scores
            machine_products = group[[
                "product_name", "category", "position", "ean"
            ]].to_dict("records")
            
            # Add machine_id to each record
            for record in machine_products:
                record["machine_id"] = machine_id
            
            machine_df = pd.DataFrame(machine_products)
            
            # Predict scores using planogram
            scores = predict_uniqueness_scores(machine_df, self.trained_model)
            
            # Add snapshot_date back
            if not scores.empty:
                scores["snapshot_date"] = snapshot_date
                all_scores.append(scores)
        
        if not all_scores:
            return pd.DataFrame(columns=self.columns)
        
        print(f"      ✓ Computed scores for {len(all_scores):,} groups")
        print(f"      🔧 Concatenating results...")
        result = pd.concat(all_scores, ignore_index=True)
        
        # Ensure proper column types for merging (especially position)
        print(f"      🔧 Standardizing column types...")
        # Position can be string (e.g., "C6", "D0") or numeric - keep as string for consistency
        if "position" in result.columns:
            result["position"] = result["position"].astype(str)  # Keep as string to match snapshots
        if "machine_id" in result.columns:
            result["machine_id"] = result["machine_id"].astype(str)
        if "snapshot_date" in result.columns:
            result["snapshot_date"] = pd.to_datetime(result["snapshot_date"], errors="coerce")
        
        # Always include product_name for matching (even if not in self.columns)
        columns_to_return = list(self.columns)
        if "product_name" in result.columns and "product_name" not in columns_to_return:
            columns_to_return.append("product_name")
        
        # Return requested columns (plus product_name if available)
        available_columns = [c for c in columns_to_return if c in result.columns]
        return result[available_columns]

    def join(self, snapshots: pd.DataFrame) -> pd.DataFrame:
        if snapshots.empty:
            return snapshots.copy()

        # Determine which method to use
        if self.trained_model is not None:
            # On-the-fly computation using planogram predict function
            print(f"   Computing uniqueness scores using planogram.product_scoring.predict_uniqueness_scores()...")
            features = self.compute_from_model(snapshots)
        else:
            # Load from pre-computed parquet file
            if self._df is None:
                print(f"   Loading uniqueness scores from parquet: {self.path}")
                self._df = self.load_from_file()
            features = self._df
        
        if features.empty:
            # Return snapshots with null scores if we can't compute/load
            for col in ["uniqueness_score", "category_diversity_score"]:
                if col not in snapshots.columns:
                    snapshots[col] = None
            return snapshots
        
        # Prepare join key - ensure consistent types
        # Include product_name if available (needed for merging rows without positions)
        columns_to_keep = list(self.columns)
        if "product_name" in features.columns and "product_name" not in columns_to_keep:
            columns_to_keep.append("product_name")
        
        subset = features[columns_to_keep].copy()
        
        # Standardize column types for merging
        if "snapshot_date" in subset.columns:
            subset["snapshot_date"] = pd.to_datetime(subset["snapshot_date"], errors="coerce")
        # Position can be numeric OR string (e.g., "C6", "D0") - convert to string for matching
        if "position" in subset.columns:
            subset["position"] = subset["position"].astype(str)  # Convert to string
            # Replace "nan" string (from converting NaN to string) with actual None
            subset.loc[subset["position"] == "nan", "position"] = None
        if "machine_id" in subset.columns:
            subset["machine_id"] = subset["machine_id"].astype(str)
        
        subset["snapshot_key"] = subset["snapshot_date"].dt.to_period("W-SUN").dt.end_time

        left = snapshots.copy()
        left["snapshot_date"] = pd.to_datetime(left["snapshot_date"], errors="coerce")
        # Position in snapshots is already a string (e.g., "C6", "D0") - keep as string
        # Convert to string and handle NaN properly
        left["position"] = left["position"].astype(str)  # Convert to string
        # Replace "nan" string (from converting NaN to string) with actual None
        left.loc[left["position"] == "nan", "position"] = None
        left["machine_id"] = left["machine_id"].astype(str)
        left["snapshot_key"] = left["snapshot_date"].dt.to_period("W-SUN").dt.end_time
        
        # Debug: Check what columns are available
        print(f"      📊 Snapshots columns: {list(left.columns)}")
        print(f"      📊 Uniqueness scores columns: {list(subset.columns)}")

        # Handle rows with missing positions separately
        # After converting to string, check for "nan" or empty strings as missing
        left_with_position = left[(left["position"].notna()) & (left["position"] != "nan") & (left["position"] != "")].copy()
        left_without_position = left[(left["position"].isna()) | (left["position"] == "nan") | (left["position"] == "")].copy()
        
        print(f"      📊 Split snapshots: {len(left_with_position):,} with positions, {len(left_without_position):,} without positions")
        print(f"      📊 Uniqueness scores available: {len(subset):,} rows")
        if "position" in subset.columns:
            subset_with_pos = subset[(subset["position"].notna()) & (subset["position"] != "nan") & (subset["position"] != "")]
            subset_without_pos = subset[(subset["position"].isna()) | (subset["position"] == "nan") | (subset["position"] == "")]
            print(f"      📊 Scores split: {len(subset_with_pos):,} with positions, {len(subset_without_pos):,} without positions")
        
        # Merge rows with positions using position as key
        if not left_with_position.empty:
            subset_with_pos_only = subset[(subset["position"].notna()) & (subset["position"] != "nan") & (subset["position"] != "")] if "position" in subset.columns else subset
            
            # Check if we have product_name in both datasets for more precise matching
            # Uniqueness scores are per product, so we should match on product_name + position
            if "product_name" in left_with_position.columns and "product_name" in subset_with_pos_only.columns:
                print(f"      🔧 Matching on machine_id + snapshot_key + position + product_name...")
                print(f"         Snapshots: {len(left_with_position):,} rows, Uniqueness scores: {len(subset_with_pos_only):,} rows")
                
                # Normalize product_name (strip whitespace, convert to string) for exact matching
                left_with_position = left_with_position.copy()
                left_with_position["product_name"] = left_with_position["product_name"].astype(str).str.strip()
                subset_with_pos_only = subset_with_pos_only.copy()
                subset_with_pos_only["product_name"] = subset_with_pos_only["product_name"].astype(str).str.strip()
                
                # Check for duplicates in uniqueness scores before merging
                duplicates = subset_with_pos_only.duplicated(subset=["machine_id", "snapshot_key", "position", "product_name"], keep=False)
                if duplicates.any():
                    dup_count = duplicates.sum()
                    print(f"      ⚠️  Found {dup_count:,} duplicate uniqueness scores - deduplicating...")
                    subset_with_pos_only = subset_with_pos_only.drop_duplicates(
                        subset=["machine_id", "snapshot_key", "position", "product_name"],
                        keep="first"
                    )
                    print(f"         After deduplication: {len(subset_with_pos_only):,} rows")
                
                merged_with_position = left_with_position.merge(
                    subset_with_pos_only,
                    on=["machine_id", "snapshot_key", "position", "product_name"],
                    how="left",
                    suffixes=("", "_uniq"),
                )
                
                # Debug: Check match rate
                matched_count = merged_with_position["uniqueness_score"].notna().sum()
                print(f"      ✓ After merge: {len(merged_with_position):,} rows, {matched_count:,} with scores ({matched_count/len(merged_with_position)*100:.1f}%)")
            else:
                # Fallback: deduplicate and match only on position
                print(f"      🔧 Deduplicating uniqueness scores (no product_name match)...")
                print(f"         Before: {len(subset_with_pos_only):,} rows")
                subset_dedup = subset_with_pos_only.drop_duplicates(
                    subset=["machine_id", "snapshot_key", "position"],
                    keep="first"
                )
                print(f"         After: {len(subset_dedup):,} rows")
                
                merged_with_position = left_with_position.merge(
                    subset_dedup,
                    on=["machine_id", "snapshot_key", "position"],
                    how="left",
                    suffixes=("", "_uniq"),
                )
            matched_with_pos = merged_with_position["uniqueness_score"].notna().sum()
            print(f"      ✓ Matched {matched_with_pos:,} / {len(left_with_position):,} rows with positions")
        else:
            merged_with_position = pd.DataFrame()
        
        # For rows without positions, match by product_name as fallback
        # (uniqueness scores include product_name but not ean)
        if not left_without_position.empty:
            print(f"      📍 Found {len(left_without_position):,} rows with missing positions - using product_name matching")
            
            # Ensure snapshot_key exists in subset for matching
            if "snapshot_key" not in subset.columns:
                subset["snapshot_key"] = subset["snapshot_date"].dt.to_period("W-SUN").dt.end_time
            
            # Check what columns we have for matching
            has_product_name_left = "product_name" in left_without_position.columns
            has_product_name_subset = "product_name" in subset.columns
            
            if has_product_name_left and has_product_name_subset:
                # Use ALL rows from subset (both with and without positions) when matching by product_name
                # Drop position column from subset since we're matching without it
                subset_no_pos = subset.drop(columns=["position"], errors="ignore")
                
                # Deduplicate subset (one score per product per machine-date)
                # Prioritize rows with positions if both exist, otherwise keep first
                subset_dedup = subset_no_pos.drop_duplicates(
                    subset=["machine_id", "snapshot_key", "product_name"],
                    keep="first"
                )
                
                print(f"      📍 Matching {len(left_without_position):,} rows against {len(subset_dedup):,} uniqueness scores by product_name")
                
                # Debug: Check if we have overlapping product names
                left_products = set(left_without_position["product_name"].dropna().unique())
                subset_products = set(subset_dedup["product_name"].dropna().unique())
                overlap = left_products & subset_products
                print(f"      📍 Product name overlap: {len(overlap):,} / {len(left_products):,} unique products in snapshots")
                
                # Match on machine_id, snapshot_key, and product_name
                merged_without_position = left_without_position.merge(
                    subset_dedup,
                    on=["machine_id", "snapshot_key", "product_name"],
                    how="left",
                    suffixes=("", "_uniq"),
                )
                
                # Check how many matched
                matched_count = merged_without_position["uniqueness_score"].notna().sum()
                print(f"      ✓ Matched {matched_count:,} / {len(left_without_position):,} rows without positions ({matched_count/len(left_without_position)*100:.1f}%)")
            else:
                print(f"      ⚠️  Cannot match by product_name: left has product_name={has_product_name_left}, subset has product_name={has_product_name_subset}")
                # No fallback key available, keep original with NaN scores
                merged_without_position = left_without_position.copy()
                for col in ["uniqueness_score", "category_diversity_score"]:
                    if col not in merged_without_position.columns:
                        merged_without_position[col] = None
        else:
            merged_without_position = pd.DataFrame()
        
        # Combine both merged dataframes
        if not merged_with_position.empty and not merged_without_position.empty:
            merged = pd.concat([merged_with_position, merged_without_position], ignore_index=True)
        elif not merged_with_position.empty:
            merged = merged_with_position
        elif not merged_without_position.empty:
            merged = merged_without_position
        else:
            merged = left.copy()

        # Cleanup
        merged = merged.drop(columns=[c for c in merged.columns if c.endswith("_uniq")])
        merged = merged.drop(columns=["snapshot_key"], errors="ignore")
        
        # Final statistics
        if "uniqueness_score" in merged.columns:
            total_rows = len(merged)
            rows_with_scores = merged["uniqueness_score"].notna().sum()
            rows_without_scores = total_rows - rows_with_scores
            coverage = (rows_with_scores / total_rows * 100) if total_rows > 0 else 0
            print(f"      📊 Final coverage: {rows_with_scores:,} / {total_rows:,} rows have uniqueness scores ({coverage:.1f}%)")
            if rows_without_scores > 0:
                print(f"      ⚠️  {rows_without_scores:,} rows without scores (likely new/rare products not in uniqueness model)")
        
        return merged


class LocationFitJoiner:
    """Feature joiner for location fit scores.
    
    Can work in two modes:
    1. Load pre-computed score matrix from parquet file (default, for inference)
    2. Compute scores on-the-fly using planogram.location_scoring (requires trained model)
    """

    def __init__(
        self, 
        score_matrix_path: Optional[Path] = None,
        trained_model: Optional[LocationModel] = None,
        trained_model_path: Optional[Path] = None
    ):
        self.score_matrix_path = Path(score_matrix_path) if score_matrix_path else None
        self.trained_model = trained_model  # Optionally provide trained model for on-the-fly computation
        self.trained_model_path = Path(trained_model_path) if trained_model_path else LOCATION_MODEL_ARTIFACT_PATH
        self._scores: Optional[pd.DataFrame] = None
        self._machine_groups: Optional[pd.DataFrame] = None
        if self.trained_model is None and self.trained_model_path and self.trained_model_path.exists():
            self.trained_model = _load_pickle_artifact(self.trained_model_path)
            print(f"✓ Loaded location model from planogram artifact: {self.trained_model_path}")

    def load_machine_groups(self, snapshots_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Load machine metadata to map machine_id -> location_type.
        
        First tries to load from file, then falls back to extracting from snapshots DataFrame if provided.
        """
        snapshot_path = MACHINE_SNAPSHOT_DIR / "machine_snapshots.parquet"
        if snapshot_path.exists():
            meta = pd.read_parquet(snapshot_path, columns=["machine_id", "machine_eva_group"])
            meta = meta.drop_duplicates(subset=["machine_id"]).copy()
            meta["machine_id"] = meta["machine_id"].astype(str)
            return meta
        
        # Fallback: Extract from snapshots DataFrame if provided
        if snapshots_df is not None and "machine_eva_group" in snapshots_df.columns:
            meta = snapshots_df[["machine_id", "machine_eva_group"]].drop_duplicates(subset=["machine_id"]).copy()
            meta["machine_id"] = meta["machine_id"].astype(str)
            return meta
        
        return pd.DataFrame()

    def load_score_matrix_from_file(self) -> pd.DataFrame:
        """Load pre-computed location score matrix from parquet."""
        if not self.score_matrix_path or not self.score_matrix_path.exists():
            return pd.DataFrame()
        return pd.read_parquet(self.score_matrix_path)

    def compute_score_matrix_from_model(
        self, 
        products_df: pd.DataFrame,
        location_types: List[str]
    ) -> pd.DataFrame:
        """Compute location score matrix on-the-fly using planogram's predict function."""
        if self.trained_model is None:
            raise ValueError("trained_model is required for on-the-fly computation")
        
        return predict_location_scores(products_df, location_types, self.trained_model)

    def join(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        # Load machine groups (needed for both modes)
        # Try loading from file first, then fall back to extracting from snapshots DataFrame
        if self._machine_groups is None:
            self._machine_groups = self.load_machine_groups(snapshots_df=df)
        
        # If still empty, machine_eva_group column doesn't exist in snapshots
        if self._machine_groups.empty:
            print("   ⚠️  Could not determine machine groups from file or snapshots.")
            print("      Adding default location_fit_score=0.5 for all rows")
            df["location_fit_score"] = 0.5  # Default fallback score
            return df
        
        print(f"   ✓ Loaded machine groups: {len(self._machine_groups)} machines")
        location_types = self._machine_groups["machine_eva_group"].dropna().unique().tolist()
        print(f"      Location types: {location_types}")

        # Determine which method to use - wrap in try/except to ensure location_fit_score is always added
        try:
            if self.trained_model is not None:
                # On-the-fly computation using planogram predict function
                print(f"   Computing location fit scores using planogram.location_scoring.predict_location_scores()...")
                # Extract unique products and location types from snapshots
                unique_products = df[["product_name", "category", "subcategory", "provider"]].drop_duplicates()
                location_types = self._machine_groups["machine_eva_group"].dropna().unique().tolist()
                
                if not location_types:
                    print("      ⚠️  No location types found. Adding default location_fit_score=0.5")
                    df["location_fit_score"] = 0.5
                    return df
                
                print(f"      Processing {len(unique_products):,} unique products across {len(location_types)} location types...")
                score_matrix = self.compute_score_matrix_from_model(unique_products, location_types)
                print(f"      ✓ Computed location fit scores")
            else:
                # Load from pre-computed parquet file
                if self._scores is None:
                    if self.score_matrix_path:
                        print(f"   Loading location scores from parquet: {self.score_matrix_path}")
                    self._scores = self.load_score_matrix_from_file()
                
                if self._scores.empty:
                    print("      ⚠️  No location score matrix found. Adding default location_fit_score=0.5")
                    df["location_fit_score"] = 0.5  # Add default score so feature matrix doesn't fail
                    return df
                
                score_matrix = self._scores

            # Attach Location Type to snapshots (if not already present)
            # df may already have machine_eva_group, so we only merge if needed
            if "machine_eva_group" not in df.columns:
                df_with_group = df.merge(
                    self._machine_groups,
                    on="machine_id",
                    how="left"
                )
            else:
                # machine_eva_group already exists in df, use it directly
                df_with_group = df.copy()
                # Ensure machine_id types match for later joins
                df_with_group["machine_id"] = df_with_group["machine_id"].astype(str)

            # Melt Scores (Product x LocationType -> long format)
            location_cols = [c for c in score_matrix.columns if c not in ["product_name", "category", "subcategory", "provider"]]
            
            if not location_cols:
                print("      ⚠️  No location columns in score matrix. Adding default location_fit_score=0.5")
                df["location_fit_score"] = 0.5
                return df
            
            long_scores = score_matrix.melt(
                id_vars=["product_name"], 
                value_vars=location_cols,
                var_name="machine_eva_group", 
                value_name="location_fit_score"
            )

            # Join scores
            merged = df_with_group.merge(
                long_scores,
                on=["product_name", "machine_eva_group"],
                how="left",
                suffixes=("", "_score")
            )
            
            # Drop duplicate column if created
            if "machine_eva_group_score" in merged.columns:
                merged = merged.drop(columns=["machine_eva_group_score"])
            
            # Check if machine_eva_group was originally in df
            original_has_eva_group = "machine_eva_group" in df.columns
            
            # Fill missing location_fit_score with default
            merged["location_fit_score"] = merged["location_fit_score"].fillna(0.5)
            
            # Only drop machine_eva_group if we added it ourselves (it wasn't in original)
            if not original_has_eva_group and "machine_eva_group" in merged.columns:
                merged = merged.drop(columns=["machine_eva_group"])
            
            return merged
            
        except Exception as e:
            print(f"      ⚠️  Error during location scoring: {e}")
            print("      Adding default location_fit_score=0.5 for all rows")
            import traceback
            traceback.print_exc()
            # Always add location_fit_score even if everything fails
            if "location_fit_score" not in df.columns:
                df["location_fit_score"] = 0.5
            return df


def build_feature_matrix(
    snapshots: pd.DataFrame,
    *,
    feature_groups: Iterable[str],
    joiners: Optional[List[FeatureJoiner]] = None,
) -> pd.DataFrame:
    """Select feature groups and apply the provided joiners to enrich the data."""

    enriched = snapshots.copy()
    print(f"📊 Building feature matrix from {len(enriched):,} snapshots...")
    print(f"   Feature groups: {', '.join(feature_groups)}")
    
    # Add temporal features if requested
    if "TEMPORAL" in feature_groups:
        print("\n   Adding temporal features (week_of_year, month, day_of_week)...")
        enriched = _add_temporal_features(enriched)
        print("      ✓ Added temporal features")
    
    # Add historical sales features if requested
    if "HISTORICAL_SALES" in feature_groups:
        print("\n   Adding historical sales features (lags, EWMA, rolling std)...")
        enriched = create_historical_sales_features(enriched, use_cold_start_fallback=True)
        print("      ✓ Added historical sales features")
    
    if joiners:
        print(f"\n   Applying {len(joiners)} feature joiners...")
        for i, joiner in enumerate(joiners, 1):
            joiner_name = joiner.__class__.__name__
            print(f"\n   [{i}/{len(joiners)}] Applying {joiner_name}...")
            enriched = joiner.join(enriched)
            print(f"      ✓ Enriched dataset now has {len(enriched):,} rows, {len(enriched.columns)} columns")

    required_cols = set(IDENTITY_COLUMNS + [TARGET_COLUMN])
    selected_features = get_feature_columns(feature_groups)
    required_cols.update(selected_features)

    missing = [col for col in required_cols if col not in enriched.columns]
    if missing:
        raise ValueError(
            "Feature matrix is missing columns: " + ", ".join(missing) +
            ". Ensure the requested feature groups are compatible with available data/joiners."
        )

    columns = []
    seen = set()
    for col in IDENTITY_COLUMNS + selected_features + [TARGET_COLUMN]:
        if col in seen:
            continue
        seen.add(col)
        columns.append(col)

    matrix = enriched[columns].copy()
    matrix = matrix.sort_values(["machine_id", "snapshot_date", "position"])
    return matrix


def save_feature_matrix(df: pd.DataFrame, path: Optional[Path] = None) -> Path:
    output_path = Path(path) if path else DEFAULT_FEATURE_DATASET
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    return output_path


def _load_pickle_artifact(path: Path) -> Any:
    """Load artifact from disk, supporting both pickle and joblib formats."""
    # Try pickle first (backward compatibility)
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as pickle_error:
        # If pickle fails, try joblib
        try:
            import joblib
            return joblib.load(path)
        except ImportError:
            # joblib not available, re-raise pickle error
            raise pickle_error
        except Exception:
            # joblib also failed, re-raise original pickle error
            raise pickle_error
