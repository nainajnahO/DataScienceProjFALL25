"""Entry point for training XGBoost sales prediction model using planogram features.

This module:
1. TRAINS planogram models using planogram package functions:
   - planogram.product_scoring.train_uniqueness_model() 
   - planogram.location_scoring.train_location_model()
   
2. USES trained planogram models to compute features:
   - planogram.product_scoring.predict_uniqueness_scores() for uniqueness features
   - planogram.location_scoring.predict_location_scores() for location fit features
   
3. TRAINS XGBoost model using planogram features as inputs.

All data loading uses:
- planogram.data_loader for loading sales/products/machines from Firestore/parquet
- planogram.product_filters.apply_all_filters() for product filtering
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Iterable

import pandas as pd

# Allow running as a standalone script
if __name__ == "__main__" and __package__ is None:
    import sys
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.append(str(repo_root))
    __package__ = "projects.planogram.model_training"

from .config import (
    DEFAULT_FEATURE_DATASET,
    DEFAULT_SNAPSHOT_OUTPUT,
    MIN_SALES_PER_POSITION,
    SNAPSHOT_FREQ,
    DATA_YEARS
)
from .snapshot_builder import (
    build_weekly_snapshots,
    load_sales_data_filtered,
    restrict_to_machine_snapshot_range,
    save_snapshots,
    get_machine_snapshot_date_range,
    load_machine_snapshot_parquet,
    LOOKBACK_WEEKS
)
from .feature_builder import (
    build_feature_matrix, 
    save_feature_matrix,
    UniquenessJoiner,
    LocationFitJoiner,
    FeatureJoiner
)
from .sales_modeling import train_sales_model
from .model_trainer import train_all_planogram_models
from .autofill_optimizer import suggest_autofill


def build_snapshot_cli(
    years: Optional[Iterable[int]] = None,
    output: Optional[str] = None,
    snapshot_freq: str = SNAPSHOT_FREQ,
    min_sales: int = MIN_SALES_PER_POSITION,
    source: str = "raw",
    align_to_snapshot_range: bool = True,
) -> Path:
    """Load sales data, generate weekly snapshots, and write parquet."""
    
    print("\n" + "="*70)
    print("📸 Building Weekly Snapshots")
    print("="*70)

    if source == "prebuilt":
        print("📥 Loading prebuilt snapshots...")
        weekly = load_machine_snapshot_parquet()
        weekly = weekly.rename(columns={"n_sales": "weekly_sales"})
        weekly["snapshot_date"] = pd.to_datetime(weekly["snapshot_date"], errors="coerce")
        weekly["lookback_weeks"] = LOOKBACK_WEEKS
        print(f"   ✓ Loaded {len(weekly):,} snapshots")
    else:
        # Auto-detect years if aligning
        if years is None and align_to_snapshot_range:
            start, end = get_machine_snapshot_date_range()
            if start and end:
                years = range(start.year, end.year + 1)
                print(f"ℹ Inferring years from machine_snapshots: {list(years)}")

        sales = load_sales_data_filtered(years=years)
        weekly = build_weekly_snapshots(sales, snapshot_freq=snapshot_freq, min_sales=min_sales)
        print(f"   ✓ Built {len(weekly):,} weekly snapshots")

    if align_to_snapshot_range:
        print(f"\n📅 Restricting to machine_snapshot date range...")
        before = len(weekly)
        weekly = restrict_to_machine_snapshot_range(weekly)
        print(f"   ✓ Restricted: {before:,} → {len(weekly):,} rows")
        print(f"   Date range: {weekly['snapshot_date'].min()} → {weekly['snapshot_date'].max()}")

    print(f"\n💾 Saving snapshots...")
    path = save_snapshots(weekly, output)
    print(f"   ✓ Saved {len(weekly):,} rows to {path}")
    print("="*70)
    return path


def build_feature_cli(
    snapshots_path: Optional[str] = None,
    *,
    feature_groups: Iterable[str],
    uniqueness_path: Optional[str] = None,
    location_model_path: Optional[str] = None,
    uniqueness_model_path: Optional[str] = None,
    output: Optional[str] = None,
) -> Path:
    """Load snapshots, join planogram feature scores (uniqueness, location), and persist a feature matrix.
    
    Uses planogram package's trained models to compute features:
    - Uniqueness scores: Uses planogram.product_scoring.predict_uniqueness_scores()
    - Location scores: Uses planogram.location_scoring.predict_location_scores()
    
    Models are automatically loaded from default paths if available, or from provided paths.
    Falls back to pre-computed parquet files if models aren't available.
    """

    print("\n" + "="*70)
    print("🔧 Building Features")
    print("="*70)

    source = Path(snapshots_path) if snapshots_path else DEFAULT_SNAPSHOT_OUTPUT
    if not source.exists():
        raise FileNotFoundError(
            f"Snapshot parquet not found at {source}. Run `pipeline.py snapshots` first "
            "or provide --snapshots with a custom path."
        )

    joiners: List[FeatureJoiner] = []
    
    # Use planogram features: Uniqueness scores
    if uniqueness_path or "UNIQUENESS" in feature_groups:
        joiner = UniquenessJoiner(path=uniqueness_path, trained_model_path=uniqueness_model_path)
        # Auto-loads trained model from default path if available
        joiners.append(joiner)
    
    # Use planogram features: Location fit scores
    if "LOCATION" in feature_groups:
        joiner = LocationFitJoiner(trained_model_path=location_model_path)
        # Auto-loads trained model from default path if available
        joiners.append(joiner)

    print(f"\n📥 Loading snapshots from: {source}")
    snapshots = pd.read_parquet(source)
    print(f"   ✓ Loaded {len(snapshots):,} snapshots\n")
    
    matrix = build_feature_matrix(
        snapshots,
        feature_groups=feature_groups,
        joiners=joiners,
    )
    
    print(f"\n💾 Saving feature matrix...")
    path = save_feature_matrix(matrix, output)
    print(f"   ✓ Saved feature matrix with {len(matrix):,} rows, {len(matrix.columns)} columns to {path}")
    print("="*70)
    return path

def train_planogram_models_cli(
    firestore_client=None,
    skip_if_exists: bool = True,
) -> dict:
    """Train planogram models using planogram package functions.
    
    Wrapper around model_trainer.train_all_planogram_models() for CLI usage.
    """
    return train_all_planogram_models(
        firestore_client=firestore_client,
        skip_if_exists=skip_if_exists
    )


def run_all_cli(
    years: Optional[Iterable[int]] = None,
    snapshot_freq: str = SNAPSHOT_FREQ,
    feature_groups: Iterable[str] = ["CORE", "TEMPORAL", "UNIQUENESS"],
    use_temp_storage: bool = False,
) -> None:
    """Execute the full pipeline: snapshots -> features (using planogram) -> train XGBoost.
    
    This pipeline:
    1. Builds weekly snapshots from sales data (using planogram.data_loader)
    2. Enriches snapshots with planogram features (uniqueness scores, location fit scores)
       - Uses planogram's trained models to predict scores
    3. Trains a global XGBoost model for sales prediction
    """
    print("=" * 70)
    print("Planogram Autofill Pipeline")
    print("Uses planogram package features for XGBoost sales prediction model")
    print("=" * 70)
    
    print("\n--- Step 1: Building Snapshots (using planogram.data_loader) ---")
    snapshots_path = build_snapshot_cli(
        years=years, 
        snapshot_freq=snapshot_freq
    )
    
    print("\n--- Step 2: Building Features (using planogram scoring models) ---")
    print("   Loading planogram models to compute uniqueness & location scores...")
    features_path = build_feature_cli(
        snapshots_path=str(snapshots_path),
        feature_groups=feature_groups
    )
    
    print("\n--- Step 3: Training Global XGBoost Model ---")
    print("   Training sales prediction model using planogram features...")
    model = train_sales_model(
        feature_path=str(features_path),
        feature_groups=feature_groups,
        use_temp_storage=use_temp_storage
    )
    
    if use_temp_storage:
        print(f"\n✓ Training complete (Temp Mode). Model artifacts cleaned up.")
    else:
        print(f"\n✓ Training complete. Model saved to output/models/")
        print(f"  Model: {model.result.model_path}")
        print(f"  Metrics: MAE={model.result.mae:.4f}, R²={model.result.r2:.4f}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Machine training pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # SNAPSHOTS
    snapshot_parser = subparsers.add_parser("snapshots", help="Build weekly snapshots from raw sales")
    snapshot_parser.add_argument("--years", nargs="*", type=int, help="Years to load (defaults to config.DATA_YEARS)")
    snapshot_parser.add_argument("--output", type=str, help="Custom path for the snapshots parquet")
    snapshot_parser.add_argument("--freq", default=SNAPSHOT_FREQ, help="Pandas frequency alias, default W-MON")
    snapshot_parser.add_argument("--min-sales", type=int, default=MIN_SALES_PER_POSITION)
    snapshot_parser.add_argument(
        "--source",
        choices=["raw", "prebuilt"],
        default="raw",
        help="Use raw sales aggregation or the prebuilt machine_snapshots parquet.",
    )
    snapshot_parser.add_argument(
        "--no-align",
        dest="align_to_snapshot_range",
        action="store_false",
        help="Disable automatic alignment to machine_snapshots.parquet date range (not recommended).",
    )

    # FEATURES
    feature_parser = subparsers.add_parser("features", help="Join uniqueness scores and select feature groups")
    feature_parser.add_argument("--snapshots", type=str, help="Path to snapshot parquet (defaults to latest output)")
    feature_parser.add_argument(
        "--groups",
        nargs="+",
        default=["CORE", "TEMPORAL", "UNIQUENESS"],
        help="Feature groups to include",
    )
    feature_parser.add_argument("--uniqueness", type=str, help="Path to machine_slot_scores.parquet")
    feature_parser.add_argument("--output", type=str, help="Path to save the feature matrix")

    # TRAIN
    train_parser = subparsers.add_parser("train", help="Train a single Global XGBoost model")
    train_parser.add_argument("--features", type=str, help="Path to feature parquet (defaults to latest output)")
    train_parser.add_argument(
        "--groups",
        nargs="+",
        default=["CORE", "TEMPORAL", "UNIQUENESS"],
        help="Feature groups to include during training",
    )
    train_parser.add_argument(
        "--temp",
        action="store_true",
        help="Use temporary storage for model artifacts (no local save)",
    )

    # TRAIN-PLANOGRAM-MODELS
    train_planogram_parser = subparsers.add_parser(
        "train-planogram-models", 
        help="Train planogram models (location & uniqueness) using planogram package functions"
    )
    train_planogram_parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain even if models already exist",
    )

    # RUN-ALL
    run_all_parser = subparsers.add_parser("run-all", help="Run entire pipeline: Snapshots -> Features -> Train")
    run_all_parser.add_argument("--years", nargs="*", type=int, help="Years to load")
    run_all_parser.add_argument(
        "--groups",
        nargs="+",
        default=["CORE", "TEMPORAL", "UNIQUENESS"],
        help="Feature groups to include",
    )
    run_all_parser.add_argument(
        "--temp",
        action="store_true",
        help="Use temporary storage for model artifacts",
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "snapshots":
        build_snapshot_cli(
            years=args.years,
            output=args.output,
            snapshot_freq=args.freq,
            min_sales=args.min_sales,
            source=args.source,
            align_to_snapshot_range=args.align_to_snapshot_range,
        )
    elif args.command == "features":
        build_feature_cli(
            snapshots_path=args.snapshots,
            feature_groups=args.groups,
            uniqueness_path=args.uniqueness,
            output=args.output,
        )
    elif args.command == "train":
        train_sales_model(
            feature_path=args.features,
            feature_groups=args.groups,
            use_temp_storage=args.temp
        )
    elif args.command == "train-planogram-models":
        train_planogram_models_cli(skip_if_exists=not args.force)
    elif args.command == "run-all":
        run_all_cli(
            years=args.years,
            feature_groups=args.groups,
            use_temp_storage=args.temp
        )


if __name__ == "__main__":
    main()
