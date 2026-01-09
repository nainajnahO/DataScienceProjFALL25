"""
Tier 1 Evaluation Pipeline
===========================

Modular evaluation functions for Tier 1 feature engineering with clean API.
Integrates with MultiWeekForecaster and feature_registry.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

from src.features.feature_registry import get_feature_columns, get_feature_groups_summary
from src.models.multi_week_forecaster import MultiWeekForecaster
from src.models.naive_baseline import HierarchicalNaiveForecaster, evaluate_naive_baseline
from src.data.splitter import (
    identify_new_products,
    method_1_all_data_to_new_products,
)
from src.config import SPLITS_DIR
from src.models.save_utils import save_model_bundle, load_model_bundle
from src.features.tier1_features import create_all_features
from src.utils.feature_analysis import correlation_analysis


def validate_feature_group(
    df: pd.DataFrame,
    feature_groups: List[str] = ['BASE'],
    target_col: str = 'weekly_sales'
) -> Dict:
    """
    Comprehensive validation report for feature group(s).
    
    Args:
        df: DataFrame with features and target
        feature_groups: List of feature group names
        target_col: Target column name
        
    Returns:
        Dictionary with validation report:
        - feature_list: List of all features
        - summary_stats: DataFrame with min/max/mean/std
        - null_percentages: Percentage of nulls per feature
        - correlation_matrix: Feature-feature correlations
        - target_correlations: Feature-target correlations
        - summary: Text summary
    """
    print(f"\n{'='*80}")
    print(f"FEATURE GROUP VALIDATION: {get_feature_groups_summary(feature_groups)}")
    print(f"{'='*80}")
    
    # Get feature columns
    feature_cols = get_feature_columns(feature_groups)
    
    # Filter to columns that exist in DataFrame
    existing_features = [f for f in feature_cols if f in df.columns]
    missing_features = set(feature_cols) - set(existing_features)
    
    if missing_features:
        print(f"⚠️  Warning: {len(missing_features)} features not found")
        print(f"   Missing: {list(missing_features)[:5]}...")
    
    if len(existing_features) == 0:
        print("❌ No valid features found")
        return None
    
    print(f"\n✓ Analyzing {len(existing_features)} features")
    
    # Summary statistics
    summary_stats = df[existing_features].describe().T
    summary_stats['null_pct'] = df[existing_features].isnull().sum() / len(df) * 100
    summary_stats['nan_pct'] = df[existing_features].isna().sum() / len(df) * 100
    
    # Correlation analysis
    corr_results = correlation_analysis(df, existing_features, target_col) if len(existing_features) > 0 else None
    
    # Count numeric vs categorical properly
    numeric_count = sum(1 for f in existing_features 
                       if pd.api.types.is_numeric_dtype(df[f].dtype) or pd.api.types.is_bool_dtype(df[f].dtype))
    categorical_count = sum(1 for f in existing_features 
                           if df[f].dtype == 'object' and df[f].nunique() > 2)
    binary_count = sum(1 for f in existing_features 
                      if df[f].dtype == 'object' and df[f].nunique() == 2)
    
    # Summary
    print(f"\n{'='*80}")
    print(f"VALIDATION SUMMARY")
    print(f"{'='*80}")
    print(f"Feature Groups: {get_feature_groups_summary(feature_groups)}")
    print(f"Total Features: {len(existing_features)}")
    print(f"  - Numeric/Binary: {numeric_count + binary_count}")
    print(f"  - Categorical: {categorical_count}")
    print(f"Features with nulls: {sum(summary_stats['null_pct'] > 0)}")
    if corr_results:
        print(f"Avg target correlation: {corr_results['summary']['avg_corr']:.4f}")
        print(f"Max target correlation: {corr_results['feature_target_corr']['target_correlation'].abs().max():.4f}")
        print(f"Highly correlated feature pairs (≥0.95): {corr_results['summary']['high_corr_pairs']}")
    else:
        print("Correlation analysis: N/A")
    
    return {
        'feature_list': existing_features,
        'summary_stats': summary_stats,
        'null_percentages': summary_stats[['null_pct', 'nan_pct']],
        'correlation_matrix': corr_results['feature_feature_pairs'] if corr_results else None,
        'target_correlations': corr_results['feature_target_corr'] if corr_results else None,
        'feature_feature_correlations': corr_results['all_feature_pairs'] if corr_results else None,
        'summary': summary_stats
    }


def train_all_approaches(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_groups: List[str] = None,
    naive_model_path: Path = None,
    random_seed: int = 42
) -> Dict:
    """
    Train all forecasting approaches and return comprehensive metrics.
    
    Approaches:
    1. Direct (Multi-Output)
    2. Recursive
    3. Direct Multi-Model
    4. Recursive Multi-Model
    5. Hierarchical Naive Baseline (loaded from file)
    
    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        feature_groups: List of feature groups to use
        naive_model_path: Path to saved naive baseline model
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary with:
        - metrics_df: MAE, R² per approach per horizon
        - predictions_dict: Full predictions for all approaches
        - importance_dict: Feature importance per approach
        - comparison_plot: Matplotlib figure
        - feature_groups_used: List of groups used
    """
    print(f"\n{'='*80}")
    print(f"TRAINING ALL APPROACHES")
    # Default to ALL engineered groups if not provided
    if feature_groups is None:
        feature_groups = ['BASE', 'TEMPORAL', 'PRODUCT', 'MACHINE', 'HISTORICAL_SALES', 'PRODUCT_LIFECYCLE', 'BRAND']
    print(f"Feature Groups: {get_feature_groups_summary(feature_groups)}")
    print(f"{'='*80}")
    
    # Get feature columns
    feature_cols = get_feature_columns(feature_groups)
    
    # MultiWeekForecaster handles all feature encoding internally
    print(f"✓ Using feature groups: {get_feature_groups_summary(feature_groups)}")
    
    # Create 4-week targets
    from src.models.multi_week_forecaster import MultiWeekForecaster
    
    def create_targets(df):
        df = df.sort_values(['machine_key', 'ean', 'week_start']).copy()
        for horizon in [1, 2, 3, 4]:
            df[f'target_week_{horizon}'] = df.groupby(['machine_key', 'ean'])['weekly_sales'].shift(-horizon)
        return df
    
    train_df_targets = create_targets(train_df)
    test_df_targets = create_targets(test_df)
    
    metrics_list = []
    predictions_dict = {}
    importance_dict = {}
    
    # 1. Train Direct (multi-model) - 4 independent models
    print("\n1️⃣  Training Direct (Multi-Model) approach...")
    forecaster_direct_multi = MultiWeekForecaster(strategy='direct_multi', random_seed=random_seed)
    forecaster_direct_multi.fit(train_df, exclude_cols=[])
    
    preds_direct_multi = forecaster_direct_multi.predict(test_df_targets)
    eval_direct_multi = forecaster_direct_multi.evaluate(preds_direct_multi)
    
    predictions_dict['Direct (Multi)'] = preds_direct_multi
    
    for _, row in eval_direct_multi.iterrows():
        metrics_list.append({
            'approach': 'Direct (Multi)',
            'horizon': row['horizon'],
            'mae': row['mae'],
            'r2': row['r2']
        })
    
    # 2. Train Recursive (single) approach
    print("\n2️⃣  Training Recursive (Single) approach...")
    forecaster_recursive = MultiWeekForecaster(strategy='recursive_single', random_seed=random_seed)
    forecaster_recursive.fit(train_df, exclude_cols=[])
    
    preds_recursive = forecaster_recursive.predict(test_df_targets)
    eval_recursive = forecaster_recursive.evaluate(preds_recursive)
    
    predictions_dict['Recursive (Single)'] = preds_recursive
    
    for _, row in eval_recursive.iterrows():
        metrics_list.append({
            'approach': 'Recursive (Single)',
            'horizon': row['horizon'],
            'mae': row['mae'],
            'r2': row['r2']
        })
    
    # 3. Train Recursive (multi-model) approach
    print("\n3️⃣  Training Recursive (Multi-Model) approach...")
    forecaster_recursive_multi = MultiWeekForecaster(strategy='recursive_multi', random_seed=random_seed)
    forecaster_recursive_multi.fit(train_df, exclude_cols=[])
    
    preds_recursive_multi = forecaster_recursive_multi.predict(test_df_targets)
    eval_recursive_multi = forecaster_recursive_multi.evaluate(preds_recursive_multi)
    
    predictions_dict['Recursive (Multi)'] = preds_recursive_multi
    
    for _, row in eval_recursive_multi.iterrows():
        metrics_list.append({
            'approach': 'Recursive (Multi)',
            'horizon': row['horizon'],
            'mae': row['mae'],
            'r2': row['r2']
        })
    
    # 5. Load and evaluate Naive Baseline
    print("\n5️⃣  Loading Naive Baseline...")
    
    # Check for saved predictions
    naive_preds_path = Path(naive_model_path).parent / 'naive_predictions_BASE.parquet'
    
    if naive_preds_path.exists():
        print(f"Loading naive predictions from file: {naive_preds_path}")
        preds_naive = pd.read_parquet(naive_preds_path)
        eval_naive = evaluate_naive_baseline(test_df_targets, preds_naive)
        print("   (Predictions were pre-computed in separate cell)")
        print("   (Delete file and re-run naive cell if you want fresh predictions)")
    else:
        print("⚠️  Naive predictions file not found")
        print("   Please run the naive baseline cell first")
        preds_naive = None
        eval_naive = None
    
    if preds_naive is not None and eval_naive is not None:
        predictions_dict['Naive'] = preds_naive
        
        for _, row in eval_naive.iterrows():
            metrics_list.append({
                'approach': 'Naive',
                'horizon': row['horizon'],
                'mae': row['mae'],
                'r2': row['r2']
            })
    
    # Create metrics DataFrame
    metrics_df = pd.DataFrame(metrics_list)
    
    # Create comparison plot
    fig = create_comparison_plot(metrics_df, get_feature_groups_summary(feature_groups))
    
    print(f"\n{'='*80}")
    print(f"✓ Training Complete")
    print(f"{'='*80}")
    
    return {
        'metrics_df': metrics_df,
        'predictions_dict': predictions_dict,
        'importance_dict': importance_dict,
        'comparison_plot': fig,
        'feature_groups_used': feature_groups
    }


def create_comparison_plot(metrics_df: pd.DataFrame, feature_groups_str: str):
    """
    Create comparison visualization with dynamic title.
    
    Args:
        metrics_df: DataFrame with metrics per approach per horizon
        feature_groups_str: String of feature groups used
        
    Returns:
        Matplotlib figure
    """
    # Create figure with extra top margin to avoid title overlap
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    plt.subplots_adjust(top=0.88, wspace=0.25)
    fig.suptitle(f"Tier-1 Approaches Comparison (Features: {feature_groups_str})",
                 fontsize=16, fontweight='bold', y=0.98)

    # Distinct styles to avoid line overlap ambiguity
    style_map = {
        'Direct (Multi)': dict(linestyle='-',  marker='o',  linewidth=2.2, alpha=0.95),
        'Recursive (Single)': dict(linestyle='--', marker='s',  linewidth=2.0, alpha=0.95),
        'Recursive (Multi)': dict(linestyle='-.', marker='^', linewidth=2.0, alpha=0.95),
        'Naive': dict(linestyle=':', marker='x', linewidth=2.0, alpha=0.95),
    }
    
    # Extract horizon numbers for plotting
    horizons = metrics_df['horizon'].str.replace('Week +', '').astype(int).values
    
    # MAE plot
    ax1 = axes[0]
    for approach in metrics_df['approach'].unique():
        data = metrics_df[metrics_df['approach'] == approach]
        style = style_map.get(approach, dict(linestyle='-', marker='o', linewidth=2))
        ax1.plot(
            data['horizon'].str.replace('Week +', '').astype(int),
            data['mae'],
            label=approach,
            **style
        )
    
    ax1.set_xlabel('Week Ahead', fontsize=12)
    ax1.set_ylabel('MAE', fontsize=12)
    ax1.set_title('MAE by Horizon', fontsize=14, fontweight='bold', pad=18)
    ax1.set_xticks(horizons)  # Set integer ticks only
    ax1.legend(fontsize=10, loc='upper left', ncol=2, frameon=True, framealpha=0.9)
    ax1.grid(alpha=0.3)
    
    # R² plot
    ax2 = axes[1]
    for approach in metrics_df['approach'].unique():
        data = metrics_df[metrics_df['approach'] == approach]
        style = style_map.get(approach, dict(linestyle='-', marker='o', linewidth=2))
        ax2.plot(
            data['horizon'].str.replace('Week +', '').astype(int),
            data['r2'],
            label=approach,
            **style
        )
    
    ax2.set_xlabel('Week Ahead', fontsize=12)
    ax2.set_ylabel('R²', fontsize=12)
    ax2.set_title('R² by Horizon', fontsize=14, fontweight='bold', pad=18)
    ax2.set_xticks(horizons)  # Set integer ticks only
    ax2.legend(fontsize=10, loc='upper left', ncol=2, frameon=True, framealpha=0.9)
    ax2.grid(alpha=0.3)
    
    # Tight layout with enough padding for titles/legends
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# ============================================================================
# DEMO NOTEBOOK WRAPPERS (ONE-LINERS)
# ============================================================================

def get_or_create_method1_split(
    df: pd.DataFrame,
    target_test_pct: float = 0.15,
    tolerance: float = 0.05,
    min_provider_age_years: float = 1.0,
    output_dir: Path = SPLITS_DIR,
    prefix: str = 'method_1_all_to_new'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    One-line wrapper: returns (train_df, test_df). If split parquet files exist, load them;
    otherwise compute Method 1 split, validate implicitly via splitter logic, save, and return.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / f"{prefix}_train.parquet"
    test_path = output_dir / f"{prefix}_test.parquet"

    if train_path.exists() and test_path.exists():
        train_df = pd.read_parquet(train_path)
        test_df = pd.read_parquet(test_path)
        return train_df, test_df

    new_eans, _ = identify_new_products(df, min_provider_age_years=min_provider_age_years)
    train_df, test_df = method_1_all_data_to_new_products(
        df, new_eans, target_test_pct=target_test_pct, tolerance=tolerance
    )
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)
    return train_df, test_df


def get_or_create_naive_predictions(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_path: Path = Path('../models/hierarchical_naive_baseline.pkl'),
    preds_path: Path = Path('../models/naive_predictions_ALL.parquet'),
    metrics_path: Path = Path('../models/naive_metrics_ALL.json'),
    lookback_weeks: int = 10,
    min_samples: int = 4,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    One-line wrapper: returns (preds_naive, eval_naive). Trains/saves model & predictions if missing.
    """
    model_path.parent.mkdir(parents=True, exist_ok=True)

    if preds_path.exists():
        preds_naive = pd.read_parquet(preds_path)
        eval_naive = evaluate_naive_baseline(test_df, preds_naive)
        return preds_naive, eval_naive

    naive_model = HierarchicalNaiveForecaster(lookback_weeks=lookback_weeks, min_samples=min_samples)
    naive_model.fit(train_df)
    naive_model.save(model_path)

    preds_naive = naive_model.predict(test_df)
    eval_naive = evaluate_naive_baseline(test_df, preds_naive)

    preds_path.parent.mkdir(parents=True, exist_ok=True)
    preds_naive.to_parquet(preds_path, index=False)
    eval_naive.to_json(metrics_path, indent=2)
    return preds_naive, eval_naive


def capped_mape(y_true: np.ndarray, y_pred: np.ndarray, cap: float = 1.0) -> float:
    """
    Capped Mean Absolute Percentage Error (0..cap), default cap=1.0 (100%).
    Avoids exploding percentages and improves interpretability.
    """
    denom = np.abs(y_true) + 1e-8
    pct_err = np.abs((y_true - y_pred) / denom)
    pct_err = np.minimum(pct_err, cap)
    return float(np.mean(pct_err))


def _detect_gpu() -> bool:
    """
    Detect if NVIDIA GPU is available for XGBoost (excludes Intel integrated graphics).
    
    Returns:
        True if NVIDIA GPU is available and XGBoost can use it, False otherwise
    """
    # Check nvidia-smi to ensure it's NVIDIA (not Intel integrated)
    # Note: nvidia-smi only exists for NVIDIA GPUs, Intel integrated graphics won't have it
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            # nvidia-smi working confirms NVIDIA GPU presence
            # Now verify XGBoost can actually use the GPU (requires CUDA-enabled XGBoost)
            try:
                from xgboost import XGBRegressor
                import numpy as np
                # Try creating a small test model with GPU
                test_model = XGBRegressor(
                    n_estimators=1,
                    tree_method='gpu_hist',
                    max_depth=1,
                    random_state=42,
                    verbosity=0
                )
                X_test = np.random.rand(10, 5)
                y_test = np.random.rand(10)
                test_model.fit(X_test, y_test)
                return True
            except Exception:
                # NVIDIA GPU exists but XGBoost can't use it (missing CUDA or GPU support)
                return False
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        # nvidia-smi doesn't exist (Intel integrated graphics) or failed
        pass
    
    return False


def train_single_approach(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    strategy: str = 'recursive_multi',
    feature_groups: List[str] = None,
    random_seed: int = 42,
    top_n_importance: int = 15,
    save_dir: Optional[Path] = None,
    use_gpu: Optional[bool] = None,
    use_optimised: bool = False
) -> Dict:
    """
    Train a single approach and return metrics, predictions, feature importances, and review plots.
    
    Args:
        use_gpu: If True, use GPU (if available). If False, use CPU. If None, auto-detect GPU.
        use_optimised: If True, use optimized hyperparameters from VendTrend experiment.
    """
    # Default to all engineered groups
    if feature_groups is None:
        feature_groups = ['BASE', 'TEMPORAL', 'PRODUCT', 'MACHINE', 'HISTORICAL_SALES', 'PRODUCT_LIFECYCLE', 'BRAND']

    # Detect GPU if not explicitly set
    if use_gpu is None:
        use_gpu = _detect_gpu()
        if use_gpu:
            print("✓ NVIDIA GPU detected and will be used for training")
        else:
            print("✓ GPU not available (or Intel integrated graphics), using CPU")
    elif use_gpu:
        if not _detect_gpu():
            print("⚠ GPU requested but not available, falling back to CPU")
            use_gpu = False

    # Prepare model params - use optimised if requested, otherwise use defaults
    from src.config import XGBOOST_PARAMS, VENDTREND_OPTIMISED_PARAMS
    
    if use_optimised:
        # Start with optimized params, add common settings
        model_params = VENDTREND_OPTIMISED_PARAMS.copy()
        model_params['n_estimators'] = XGBOOST_PARAMS.get('n_estimators', 300)  # Keep n_estimators from default
        model_params['random_state'] = random_seed
        model_params['n_jobs'] = XGBOOST_PARAMS.get('n_jobs', -1)
        model_params['verbosity'] = 0
        print("✓ Using optimized hyperparameters from VendTrend experiment")
    else:
        model_params = XGBOOST_PARAMS.copy()
        model_params['random_state'] = random_seed
    
    # Set GPU or CPU tree method
    if use_gpu:
        model_params['tree_method'] = 'gpu_hist'
        model_params['gpu_id'] = 0
    else:
        model_params['tree_method'] = 'hist'

    # Prepare targets (same as in train_all_approaches)
    forecaster = MultiWeekForecaster(strategy=strategy, random_seed=random_seed, model_params=model_params)
    def create_targets(df):
        df = df.sort_values(['machine_key', 'ean', 'week_start']).copy()
        for h in [1, 2, 3, 4]:
            df[f'target_week_{h}'] = df.groupby(['machine_key', 'ean'])['weekly_sales'].shift(-h)
        return df

    train_df_targets = create_targets(train_df)
    test_df_targets = create_targets(test_df)

    # Fit & predict
    forecaster.fit(train_df, exclude_cols=[])
    preds = forecaster.predict(test_df_targets)
    eval_df = forecaster.evaluate(preds)

    # Capped MAPE per horizon
    capped_mapes = []
    for h in [1, 2, 3, 4]:
        y_true = preds[f'actual_week_{h}'].values
        y_pred = preds[f'pred_week_{h}'].values
        capped_mapes.append(capped_mape(y_true, y_pred))

    # Feature importance (average across models if multi, or single)
    importance_items = []
    for h, model in forecaster.models.items():
        if hasattr(model, 'feature_importances_'):
            imp = model.feature_importances_
            for name, val in zip(forecaster.feature_names, imp):
                importance_items.append({'horizon': h, 'feature': name, 'importance': float(val)})
    importance_df = pd.DataFrame(importance_items)
    if not importance_df.empty:
        importance_top = (
            importance_df.groupby('feature')['importance']
            .mean()
            .sort_values(ascending=False)
            .head(top_n_importance)
            .reset_index()
        )
    else:
        importance_top = pd.DataFrame(columns=['feature', 'importance'])

    # Review plots (MAE/R2 lines, capped MAPE bar, feature importance bar)
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    horizons = [int(h.split('+')[-1]) for h in eval_df['horizon'].astype(str)]

    # Plot 1: MAE/R2
    ax1 = axes[0]
    ax1.plot(horizons, eval_df['mae'], marker='o', linewidth=2)
    ax1_t = ax1.twinx()
    ax1_t.plot(horizons, eval_df['r2'], marker='s', color='tab:orange', linewidth=2)
    ax1.set_title('MAE and R² by Horizon', fontweight='bold', pad=12)
    ax1.set_xlabel('Week Ahead')
    ax1.set_ylabel('MAE')
    ax1_t.set_ylabel('R²', color='tab:orange')
    ax1.grid(alpha=0.3)

    # Plot 2: Capped MAPE
    ax2 = axes[1]
    ax2.bar(horizons, capped_mapes, color='tab:green', alpha=0.8)
    ax2.set_title('Capped MAPE (cap=100%)', fontweight='bold', pad=12)
    ax2.set_xlabel('Week Ahead')
    ax2.set_ylabel('Capped MAPE')
    ax2.set_ylim(0, 1.05)
    ax2.grid(alpha=0.3)

    # Plot 3: Top-N Feature Importance
    ax3 = axes[2]
    if not importance_top.empty:
        ax3.barh(importance_top['feature'][::-1], importance_top['importance'][::-1], color='tab:blue', alpha=0.8)
        ax3.set_title(f'Top {top_n_importance} Feature Importances (avg across horizons)', fontweight='bold', pad=12)
        ax3.set_xlabel('Importance')
    else:
        ax3.text(0.5, 0.5, 'No importance available', ha='center', va='center')
        ax3.set_axis_off()

    plt.tight_layout()

    saved_path = None
    if save_dir is not None:
        saved_path = save_model_bundle(
            forecaster=forecaster,
            strategy=strategy,
            feature_groups=feature_groups,
            base_dir=save_dir,
            extra_metadata={
                'random_seed': random_seed,
                'top_n_importance': top_n_importance,
            }
        )

    return {
        'metrics_df': eval_df,
        'predictions': preds,
        'capped_mapes': capped_mapes,
        'importance_df': importance_df,
        'importance_top': importance_top,
        'review_figure': fig,
        'feature_groups_used': feature_groups,
        'strategy': strategy,
        'forecaster': forecaster,
        'saved_path': saved_path,
    }


def evaluate_loaded_forecaster(
    forecaster,
    test_df: pd.DataFrame,
    feature_groups: List[str] = None,
    top_n_importance: int = 15
) -> Dict:
    """
    Evaluate an already-trained forecaster on test data and produce the same review outputs.
    """
    # Recreate features using the same groups used in training
    if feature_groups is None:
        feature_groups = ['BASE', 'TEMPORAL', 'PRODUCT', 'MACHINE', 'HISTORICAL_SALES', 'PRODUCT_LIFECYCLE', 'BRAND']
    test_df = create_all_features(test_df.copy(), feature_groups=None)  # None => all; safe superset
    # Create targets
    def create_targets(df):
        df = df.sort_values(['machine_key', 'ean', 'week_start']).copy()
        for h in [1, 2, 3, 4]:
            df[f'target_week_{h}'] = df.groupby(['machine_key', 'ean'])['weekly_sales'].shift(-h)
        return df

    test_df_targets = create_targets(test_df)

    preds = forecaster.predict(test_df_targets)
    eval_df = forecaster.evaluate(preds)

    # Importance aggregation (if available)
    importance_items = []
    for h, model in getattr(forecaster, 'models', {}).items():
        if hasattr(model, 'feature_importances_'):
            for name, val in zip(getattr(forecaster, 'feature_names', []), model.feature_importances_):
                importance_items.append({'horizon': h, 'feature': name, 'importance': float(val)})
    importance_df = pd.DataFrame(importance_items)
    if not importance_df.empty:
        importance_top = (
            importance_df.groupby('feature')['importance']
            .mean()
            .sort_values(ascending=False)
            .head(top_n_importance)
            .reset_index()
        )
    else:
        importance_top = pd.DataFrame(columns=['feature', 'importance'])

    # Simple review plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    horizons = [int(h.split('+')[-1]) for h in eval_df['horizon'].astype(str)]
    ax1 = axes[0]
    ax1.plot(horizons, eval_df['mae'], marker='o', linewidth=2)
    ax1_t = ax1.twinx()
    ax1_t.plot(horizons, eval_df['r2'], marker='s', color='tab:orange', linewidth=2)
    ax1.set_title('MAE and R² by Horizon', fontweight='bold', pad=12)
    ax1.set_xlabel('Week Ahead')
    ax1.set_ylabel('MAE')
    ax1_t.set_ylabel('R²', color='tab:orange')
    ax1.grid(alpha=0.3)

    ax2 = axes[1]
    if not importance_top.empty:
        ax2.barh(importance_top['feature'][::-1], importance_top['importance'][::-1], color='tab:blue', alpha=0.8)
        ax2.set_title(f'Top {top_n_importance} Feature Importances', fontweight='bold', pad=12)
        ax2.set_xlabel('Importance')
    else:
        ax2.text(0.5, 0.5, 'No importance available', ha='center', va='center')
        ax2.set_axis_off()

    plt.tight_layout()

    return {
        'metrics_df': eval_df,
        'predictions': preds,
        'importance_df': importance_df,
        'importance_top': importance_top,
        'review_figure': fig,
    }

