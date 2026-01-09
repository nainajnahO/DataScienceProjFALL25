"""Sales modeling module.

Implements the Train/Predict pattern for global sales prediction using XGBoost.
Trains a single global model across all machines, rather than one model per machine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Any

import joblib
import pandas as pd
import tempfile
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

try:
    from tqdm.auto import tqdm
except ImportError:
    # Fallback if tqdm is not installed
    def tqdm(iterable, desc=None, unit=None, **kwargs):
        return iterable

from .config import (
    DEFAULT_FEATURE_DATASET,
    DEFAULT_TEST_SIZE,
    METRICS_OUTPUT_DIR,
    MIN_TRAIN_ROWS,
    MODEL_OUTPUT_DIR,
    TARGET_COLUMN,
    XGBOOST_PARAMS,
)
from .feature_builder import get_categorical_features, get_feature_columns, get_numeric_features


@dataclass
class TrainingResult:
    n_train: int
    n_test: int
    mae: float
    r2: float
    model_path: Optional[Path] = None


@dataclass
class GlobalXGBoostRegressor:
    """
    Trainer class that fits a single XGBoost model for ALL machines.
    Matches the 'Train' phase of the architecture.
    """
    feature_groups: Iterable[str]
    params: Dict[str, Any] = field(default_factory=lambda: dict(XGBOOST_PARAMS))
    test_size: float = DEFAULT_TEST_SIZE
    min_train_rows: int = MIN_TRAIN_ROWS
    use_temp_storage: bool = False

    def __post_init__(self) -> None:
        self.feature_groups = list(self.feature_groups)
        self.feature_columns = get_feature_columns(self.feature_groups)
        self.numeric_features = get_numeric_features(self.feature_groups)
        self.categorical_features = get_categorical_features(self.feature_groups)
        
        self.model: Optional[Pipeline] = None
        self.result: Optional[TrainingResult] = None
        self._temp_dir: Optional[tempfile.TemporaryDirectory] = None

    def fit(self, dataset: pd.DataFrame) -> TrainingResult:
        if dataset.empty:
            raise ValueError("Feature dataset is empty. Generate snapshots/features first.")

        print("\n" + "="*70)
        print("🎯 XGBoost Model Training")
        print("="*70)
        print(f"📊 Input dataset: {len(dataset):,} rows, {len(dataset.columns)} columns")
        
        print("\n📋 Step 1: Preparing dataset...")
        dataset = self._prepare_dataset(dataset)
        print(f"   ✓ Dataset prepared")
        
        print("\n📋 Step 2: Splitting into train/test sets...")
        train_df, test_df = self._split_dataset(dataset)
        print(f"   ✓ Training set: {len(train_df):,} rows ({len(train_df)/len(dataset)*100:.1f}%)")
        print(f"   ✓ Test set: {len(test_df):,} rows ({len(test_df)/len(dataset)*100:.1f}%)")
        
        if len(train_df) < self.min_train_rows:
             raise ValueError(f"Insufficient training data: {len(train_df)} rows < {self.min_train_rows}")

        print("\n📋 Step 3: Building preprocessing pipeline...")
        pipeline = self._build_pipeline()
        X_train, y_train = self._split_xy(train_df)
        print(f"   ✓ Features: {len(X_train.columns)} columns")
        print(f"   ✓ Categorical features: {len(self.categorical_features)}")
        print(f"   ✓ Numeric features: {len(self.numeric_features)}")
        
        print("\n📋 Step 4: Fitting preprocessing transformers...")
        print("   ⚠️  This step can take 10-20 minutes for 12M+ rows...")
        print("   - OneHotEncoder: Encoding categorical features (this is the slow part)")
        print("   - StandardScaler: Normalizing numeric features")
        
        # Fit preprocessing separately to show progress and check time
        import time
        import threading
        from datetime import datetime
        
        preprocessing = pipeline.named_steps['preprocess']
        
        print(f"   🕐 Started at: {datetime.now().strftime('%H:%M:%S')}")
        print(f"   Fitting OneHotEncoder on {len(X_train):,} rows...")
        print(f"   Categorical features: {', '.join(self.categorical_features)}")
        print(f"   ⏳ Fitting in progress (this can take 10-20 minutes - OneHotEncoder scans all data)...")
        print(f"   💡 Tip: OneHotEncoder needs to find all unique categories across 12M+ rows")
        
        import sys
        sys.stdout.flush()  # Force output to appear immediately
        
        # Add a heartbeat to show it's still working
        heartbeat_done = threading.Event()
        
        def print_heartbeat():
            elapsed = 0
            while not heartbeat_done.is_set():
                time.sleep(60)  # Print every minute
                if not heartbeat_done.is_set():
                    elapsed += 60
                    mins = elapsed // 60
                    secs = elapsed % 60
                    print(f"   ⏱️  Still fitting OneHotEncoder... ({mins} min {secs} sec elapsed)")
                    sys.stdout.flush()
        
        heartbeat_thread = threading.Thread(target=print_heartbeat, daemon=True)
        heartbeat_thread.start()
        
        start_time = time.time()
        print(f"   🔄 STARTING OneHotEncoder.fit() now - this is the slow step...")
        print(f"   💡 You'll see a progress message every 60 seconds while it's running")
        sys.stdout.flush()
        
        try:
            preprocessing.fit(X_train)
        finally:
            heartbeat_done.set()
            # Give thread a moment to print final message
            time.sleep(1)
        
        elapsed = time.time() - start_time
        print(f"   ✓✓✓ Preprocessing fitted in {elapsed/60:.1f} minutes ({elapsed:.0f} seconds) ✓✓✓")
        sys.stdout.flush()
        
        print("\n📋 Step 5: Transforming training data...")
        print(f"   🕐 Started at: {datetime.now().strftime('%H:%M:%S')}")
        print(f"   Transforming {len(X_train):,} rows through preprocessing pipeline...")
        start_time = time.time()
        X_train_transformed = preprocessing.transform(X_train)
        elapsed = time.time() - start_time
        print(f"   ✓ Transformed in {elapsed/60:.1f} minutes ({elapsed:.0f} seconds)")
        print(f"   Final feature matrix: {X_train_transformed.shape[0]:,} rows × {X_train_transformed.shape[1]:,} columns")
        
        print("\n📋 Step 6: Training XGBoost model...")
        print(f"   🕐 Started at: {datetime.now().strftime('%H:%M:%S')}")
        print(f"   Parameters: {self.params.get('n_estimators')} trees, max_depth={self.params.get('max_depth')}")
        print(f"   Expected time: 15-45 minutes for this dataset size")
        print("   XGBoost tree-by-tree progress:\n")
        
        import sys
        sys.stdout.flush()  # Force output to appear immediately
        
        # Fit XGBoost separately with verbose output
        xgb_model = pipeline.named_steps['model']
        
        # Ensure verbose is enabled (1 = print every tree, which can be verbose for 300 trees)
        # Use verbose=10 to print every 10 trees instead, or verbose=1 for every tree
        if hasattr(xgb_model, 'set_params'):
            xgb_model.set_params(verbose=10)  # Print every 10 trees to reduce spam but show progress
        
        # Add periodic progress messages during training (in case verbose doesn't show in Jupyter)
        training_done = threading.Event()
        
        def print_training_progress():
            elapsed = 0
            tree_count = 0
            while not training_done.is_set():
                time.sleep(60)  # Print every minute
                if not training_done.is_set():
                    elapsed += 60
                    mins = elapsed // 60
                    secs = elapsed % 60
                    # Estimate progress based on expected time (rough estimate)
                    estimated_total_mins = 30  # Rough estimate
                    if elapsed > 0:
                        estimated_pct = min(100, (elapsed / (estimated_total_mins * 60)) * 100)
                        print(f"   ⏱️  XGBoost training in progress... ({mins} min {secs} sec elapsed, ~{estimated_pct:.0f}% complete)")
                    else:
                        print(f"   ⏱️  XGBoost training in progress... ({mins} min {secs} sec elapsed)")
                    sys.stdout.flush()
        
        progress_thread = threading.Thread(target=print_training_progress, daemon=True)
        progress_thread.start()
        
        start_time = time.time()
        print("   🔄 STARTING XGBoost.fit() now - this will take 15-45 minutes...")
        print("   💡 You'll see progress messages every 60 seconds + XGBoost verbose output")
        sys.stdout.flush()
        
        try:
            xgb_model.fit(X_train_transformed, y_train)
        finally:
            training_done.set()
            # Give thread a moment to print final message
            time.sleep(1)
        
        elapsed = time.time() - start_time
        print(f"\n   ✓✓✓ XGBoost training completed in {elapsed/60:.1f} minutes ({elapsed:.0f} seconds) ✓✓✓")
        sys.stdout.flush()
        
        # Update pipeline with fitted components
        pipeline.named_steps['preprocess'] = preprocessing
        pipeline.named_steps['model'] = xgb_model
        
        print("\n📋 Step 7: Evaluating model...")
        mae, r2 = self._evaluate(pipeline, test_df)
        print(f"   ✓ Test MAE: {mae:.4f}")
        print(f"   ✓ Test R²: {r2:.4f}")
        
        print("\n📋 Step 8: Saving model...")
        model_path = self._persist_model(pipeline)
        print(f"   ✓ Model saved to: {model_path}")
        print("="*70)

        self.model = pipeline
        self.result = TrainingResult(
            n_train=len(train_df),
            n_test=len(test_df) if test_df is not None else 0,
            mae=mae,
            r2=r2,
            model_path=model_path,
        )
        
        print(f"Global Model Results - MAE: {mae:.4f}, R2: {r2:.4f}")
        return self.result

    def _prepare_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        frame = df.copy()
        frame["machine_id"] = frame["machine_id"].astype(str)
        # Sort by date to ensure potential time-based splitting works if we used it
        frame = frame.sort_values(["snapshot_date", "machine_id", "position"])
        frame["snapshot_date"] = pd.to_datetime(frame["snapshot_date"], errors="coerce")
        return frame

    def _split_dataset(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        unique_dates = df["snapshot_date"].sort_values().unique()
        split_idx = int(len(unique_dates) * (1 - self.test_size))
        cutoff_date = unique_dates[split_idx]
        
        train_df = df[df["snapshot_date"] < cutoff_date]
        test_df = df[df["snapshot_date"] >= cutoff_date]
        
        return train_df, test_df

    def _split_xy(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        cols_to_use = [c for c in self.feature_columns if c != TARGET_COLUMN]
        X = df[cols_to_use]
        y = df[TARGET_COLUMN]
        return X, y

    def _build_pipeline(self) -> Pipeline:
        transformer = ColumnTransformer(
            transformers=[
                (
                    "categorical",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    self.categorical_features,
                ),
                (
                    "numeric",
                    StandardScaler(),
                    self.numeric_features
                )
            ],
            remainder="drop",
        )
        
        # Ensure XGBoost shows progress
        xgb_params = self.params.copy()
        xgb_params['verbose'] = 1  # Show progress every tree
        xgb_params['n_jobs'] = -1  # Use all CPU cores
        model = XGBRegressor(**xgb_params)
        
        pipeline = Pipeline([
            ("preprocess", transformer),
            ("model", model),
        ])
        return pipeline

    def _evaluate(self, pipeline: Pipeline, test_df: Optional[pd.DataFrame]) -> Tuple[float, float]:
        if test_df is None or test_df.empty:
            return float("nan"), float("nan")

        X_test, y_test = self._split_xy(test_df)
        preds = pipeline.predict(X_test)
        return mean_absolute_error(y_test, preds), r2_score(y_test, preds)

    def _persist_model(self, pipeline: Pipeline) -> Path:
        if self.use_temp_storage:
            if self._temp_dir is None:
                self._temp_dir = tempfile.TemporaryDirectory()
            output_dir = Path(self._temp_dir.name)
        else:
            output_dir = MODEL_OUTPUT_DIR
            output_dir.mkdir(parents=True, exist_ok=True)

        path = output_dir / "global_sales_model.joblib"
        joblib.dump(pipeline, path)
        return path

    def save_metrics(self, path: Optional[Path] = None) -> Path:
        if self.use_temp_storage:
            if self._temp_dir is None:
                 self._temp_dir = tempfile.TemporaryDirectory()
            metrics_path = Path(self._temp_dir.name) / "global_model_metrics.parquet"
        else:
            metrics_path = Path(path) if path else (METRICS_OUTPUT_DIR / "global_model_metrics.parquet")
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.result:
            data = self.result.__dict__.copy()
            if data.get("model_path"):
                data["model_path"] = str(data["model_path"])
            df = pd.DataFrame([data])
            df.to_parquet(metrics_path, index=False)
        
        return metrics_path
    
    def get_temp_model_path(self) -> Optional[Path]:
        """Return path to model if temp storage was used"""
        if self._temp_dir and self.result and self.result.model_path:
            return self.result.model_path
        return None


def train_sales_model(
    feature_path: Optional[str] = None,
    feature_groups: Optional[Iterable[str]] = None,
    *,
    params: Optional[Dict[str, Any]] = None,
    use_temp_storage: bool = False
) -> GlobalXGBoostRegressor:
    """
    Top-level training function.
    """
    source = Path(feature_path) if feature_path else DEFAULT_FEATURE_DATASET
    if not source.exists():
        raise FileNotFoundError(
            f"Feature parquet not found at {source}. Run `pipeline.py features` first or pass --feature-path."
        )

    df = pd.read_parquet(source)
    groups = feature_groups or ["CORE", "TEMPORAL", "UNIQUENESS"]
    
    model = GlobalXGBoostRegressor(
        feature_groups=groups, 
        params=params or XGBOOST_PARAMS,
        use_temp_storage=use_temp_storage
    )
    model.fit(df)
    model.save_metrics()
    return model


def predict_sales_scores(
    feature_df: pd.DataFrame,
    model_path: Optional[Path] = None,
    feature_groups: Optional[Iterable[str]] = None
) -> pd.DataFrame:
    """
    Inference function: Predict sales using the global model.
    
    Args:
        feature_df: DataFrame with features (should match training feature columns)
        model_path: Path to trained model (defaults to output/models/global_sales_model.joblib)
        feature_groups: Feature groups used during training (for column selection)
    
    Returns:
        DataFrame with predicted_sales column added
    """
    path = model_path if model_path else (MODEL_OUTPUT_DIR / "global_sales_model.joblib")
    if not path.exists():
        raise FileNotFoundError(f"Global model not found at {path}")
        
    pipeline = joblib.load(path)
    
    try:
        # Remove target column if present (not needed for prediction)
        if TARGET_COLUMN in feature_df.columns:
            feature_df = feature_df.drop(columns=[TARGET_COLUMN])
        
        # Select only feature columns (exclude identity columns)
        from .feature_builder import get_feature_columns, IDENTITY_COLUMNS
        if feature_groups:
            feature_cols = get_feature_columns(feature_groups)
        else:
            # Try to infer from available columns
            feature_cols = [c for c in feature_df.columns if c not in IDENTITY_COLUMNS + [TARGET_COLUMN]]
        
        # Ensure we have the required feature columns
        missing_cols = [c for c in feature_cols if c not in feature_df.columns]
        if missing_cols:
            print(f"⚠️  Warning: Missing feature columns: {missing_cols[:10]}{'...' if len(missing_cols) > 10 else ''}")
        
        # Use only columns that exist in both feature_df and feature_cols
        available_feature_cols = [c for c in feature_cols if c in feature_df.columns]
        X = feature_df[IDENTITY_COLUMNS + available_feature_cols].copy()
        
        # Predict
        preds = pipeline.predict(X)
        results = feature_df.copy()
        results["predicted_sales"] = preds
        return results
    except Exception as e:
        print(f"Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()
