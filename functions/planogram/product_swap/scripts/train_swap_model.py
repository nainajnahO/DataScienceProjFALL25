"""
Train Swap Prediction Model
===========================

Train XGBoost models to predict swap outcomes:
1. Revenue change (regression)
2. Profit change (regression)
3. Success probability (classification)
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, classification_report, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import joblib


def create_training_features(df):
    """
    Create features for training swap prediction models.
    
    Args:
        df: DataFrame with enriched swap data
        
    Returns:
        DataFrame with training features
    """
    print("Creating training features...")
    
    df = df.copy()
    
    # 1. Product similarity features
    df['same_category'] = (df['subcategory_before'] == df['subcategory_after']).astype(int)
    df['same_provider'] = (df['provider_before'] == df['provider_after']).astype(int)
    
    # 2. Historical performance features
    df['revenue_per_day_before'] = df['revenue_before_4w'] / df['days_observed_before_4w'].replace(0, 1)
    df['profit_per_day_before'] = df['profit_before_4w'] / df['days_observed_before_4w'].replace(0, 1)
    df['sales_per_day_before'] = df['sales_count_before_4w'] / df['days_observed_before_4w'].replace(0, 1)
    
    # 3. Profit margin before swap
    df['profit_margin_before'] = (df['profit_before_4w'] / df['revenue_before_4w'].replace(0, np.nan) * 100).fillna(0)
    
    # 4. Log transforms for skewed features (add small value to avoid log(0))
    df['log_revenue_per_day_before'] = np.log1p(df['revenue_per_day_before'].clip(lower=0))
    df['log_profit_per_day_before'] = np.log1p(df['profit_per_day_before'].clip(lower=0))
    df['log_sales_per_day_before'] = np.log1p(df['sales_per_day_before'].clip(lower=0))
    df['log_revenue_before_4w'] = np.log1p(df['revenue_before_4w'].clip(lower=0))
    df['log_profit_before_4w'] = np.log1p(df['profit_before_4w'].clip(lower=0))
    
    # 5. Interaction features
    df['revenue_profit_interaction'] = df['revenue_per_day_before'] * df['profit_per_day_before']
    df['category_provider_interaction'] = df['same_category'] * df['same_provider']
    df['sales_revenue_interaction'] = df['sales_per_day_before'] * df['revenue_per_day_before']
    
    # NOTE: We do NOT include revenue_ratio or profit_ratio because they use
    # future information (revenue_after_4w, profit_after_4w) which would cause data leakage
    
    # 6. Temporal features (if swap_date available)
    if 'swap_date' in df.columns:
        df['swap_date'] = pd.to_datetime(df['swap_date'])
        df['swap_month'] = df['swap_date'].dt.month
        df['swap_quarter'] = df['swap_date'].dt.quarter
        df['is_summer'] = df['swap_date'].dt.month.isin([6, 7, 8]).astype(int)
        df['is_winter'] = df['swap_date'].dt.month.isin([12, 1, 2]).astype(int)
        df['day_of_year'] = df['swap_date'].dt.dayofyear
    
    # 7. Position features (if available)
    if 'position' in df.columns:
        df['is_top_half'] = (df['position'] <= df['position'].median()).astype(int) if df['position'].notna().any() else 0
        df['position_normalized'] = (df['position'] - df['position'].mean()) / (df['position'].std() + 1e-10) if df['position'].notna().any() else 0
    
    # 8. Target variables
    df['revenue_increased'] = (df['revenue_change_4w'] > 0).astype(int)
    df['profit_increased'] = (df['profit_change_4w'] > 0).astype(int)
    df['significant_improvement'] = (df['revenue_change_4w'] > 50).astype(int)  # >50 SEK increase
    
    # 9. Encode categorical features with EAN-first fallback hierarchy
    # Hierarchy: EAN → subcategory → category
    
    # Count occurrences for fallback logic (minimum 3 occurrences to use EAN)
    MIN_EAN_COUNT = 3
    
    # Before product encoding
    if 'ean_before' in df.columns or 'product_before' in df.columns:
        # Try to get EAN from product_before if ean_before doesn't exist
        if 'ean_before' not in df.columns and 'product_before' in df.columns:
            # Assume product_before might contain EAN info - check if it's numeric
            df['ean_before'] = df['product_before'].apply(
                lambda x: str(x) if pd.notna(x) and str(x).isdigit() and len(str(x)) >= 8 else None
            )
        
        ean_before_counts = df['ean_before'].value_counts() if 'ean_before' in df.columns else pd.Series()
        ean_before_counts_dict = ean_before_counts.to_dict()
        df['ean_before_available'] = df['ean_before'].apply(
            lambda x: x if pd.notna(x) and ean_before_counts_dict.get(x, 0) >= MIN_EAN_COUNT else None
        )
    else:
        df['ean_before_available'] = None
    
    # After product encoding
    if 'ean_after' in df.columns or 'product_after' in df.columns:
        if 'ean_after' not in df.columns and 'product_after' in df.columns:
            df['ean_after'] = df['product_after'].apply(
                lambda x: str(x) if pd.notna(x) and str(x).isdigit() and len(str(x)) >= 8 else None
            )
        
        ean_after_counts = df['ean_after'].value_counts() if 'ean_after' in df.columns else pd.Series()
        ean_after_counts_dict = ean_after_counts.to_dict()
        df['ean_after_available'] = df['ean_after'].apply(
            lambda x: x if pd.notna(x) and ean_after_counts_dict.get(x, 0) >= MIN_EAN_COUNT else None
        )
    else:
        df['ean_after_available'] = None
    
    # Create fallback identifiers
    # Check if category columns exist
    has_category_before = 'category_before' in df.columns
    has_category_after = 'category_after' in df.columns
    
    df['product_id_before'] = df.apply(
        lambda row: (
            f"EAN_{row['ean_before_available']}" if pd.notna(row.get('ean_before_available')) 
            else f"SUBCAT_{row.get('subcategory_before', 'Unknown')}" if pd.notna(row.get('subcategory_before'))
            else f"CAT_{row.get('category_before', 'Unknown')}" if has_category_before and pd.notna(row.get('category_before'))
            else "Unknown"
        ), axis=1
    )
    
    df['product_id_after'] = df.apply(
        lambda row: (
            f"EAN_{row['ean_after_available']}" if pd.notna(row.get('ean_after_available'))
            else f"SUBCAT_{row.get('subcategory_after', 'Unknown')}" if pd.notna(row.get('subcategory_after'))
            else f"CAT_{row.get('category_after', 'Unknown')}" if has_category_after and pd.notna(row.get('category_after'))
            else "Unknown"
        ), axis=1
    )
    
    # Encode with LabelEncoders
    le_product_before = LabelEncoder()
    le_product_after = LabelEncoder()
    le_subcat_before = LabelEncoder()
    le_subcat_after = LabelEncoder()
    le_provider_before = LabelEncoder()
    le_provider_after = LabelEncoder()
    
    # Encode product IDs (with fallback hierarchy built in)
    df['product_before_encoded'] = le_product_before.fit_transform(df['product_id_before'].fillna('Unknown'))
    df['product_after_encoded'] = le_product_after.fit_transform(df['product_id_after'].fillna('Unknown'))
    
    # Also keep subcategory encoders for backward compatibility and additional features
    df['subcategory_before_encoded'] = le_subcat_before.fit_transform(df['subcategory_before'].fillna('Unknown'))
    df['subcategory_after_encoded'] = le_subcat_after.fit_transform(df['subcategory_after'].fillna('Unknown'))
    df['provider_before_encoded'] = le_provider_before.fit_transform(df['provider_before'].fillna('Unknown'))
    df['provider_after_encoded'] = le_provider_after.fit_transform(df['provider_after'].fillna('Unknown'))
    
    # Track which level was used for statistics
    df['product_before_level'] = df['product_id_before'].str.split('_').str[0]
    df['product_after_level'] = df['product_id_after'].str.split('_').str[0]
    
    # Save encoders
    encoders = {
        'product_before': le_product_before,
        'product_after': le_product_after,
        'subcategory_before': le_subcat_before,
        'subcategory_after': le_subcat_after,
        'provider_before': le_provider_before,
        'provider_after': le_provider_after,
        'ean_before_counts': ean_before_counts_dict if 'ean_before' in df.columns else {},
        'ean_after_counts': ean_after_counts_dict if 'ean_after' in df.columns else {},
        'min_ean_count': MIN_EAN_COUNT
    }
    
    # Print statistics
    print(f"\n  Product identification statistics:")
    print(f"    Before: EAN={df['product_before_level'].value_counts().get('EAN', 0)}, "
          f"Subcategory={df['product_before_level'].value_counts().get('SUBCAT', 0)}, "
          f"Category={df['product_before_level'].value_counts().get('CAT', 0)}")
    print(f"    After:  EAN={df['product_after_level'].value_counts().get('EAN', 0)}, "
          f"Subcategory={df['product_after_level'].value_counts().get('SUBCAT', 0)}, "
          f"Category={df['product_after_level'].value_counts().get('CAT', 0)}")
    
    print(f"✓ Created features for {len(df):,} swaps")
    
    return df, encoders


def prepare_data(df, min_sales_before=3):
    """
    Prepare data for training by filtering and splitting.
    
    Args:
        df: DataFrame with features
        min_sales_before: Minimum sales required before swap
        
    Returns:
        Train/test splits
    """
    # Filter for quality data
    df = df[df['sales_count_before_4w'] >= min_sales_before].copy()
    df = df[df['revenue_before_4w'] > 0].copy()  # Must have some revenue
    df = df[df['days_observed_before_4w'] > 0].copy()  # Must have valid observation period
    
    print(f"\nFiltered to {len(df):,} swaps with sufficient data")
    print(f"  (Min {min_sales_before} sales before swap)")
    
    # Define feature columns (with all new features)
    feature_cols = [
        # Categorical features (encoded) - using product IDs with fallback hierarchy
        'product_before_encoded',
        'product_after_encoded',
        # Keep subcategory and provider for additional signal
        'subcategory_before_encoded',
        'subcategory_after_encoded',
        'provider_before_encoded',
        'provider_after_encoded',
        
        # Product similarity
        'same_category',
        'same_provider',
        
        # Historical performance (original)
        'revenue_per_day_before',
        'profit_per_day_before',
        'sales_per_day_before',
        'profit_margin_before',
        'sales_count_before_4w',
        
        # Log transforms
        'log_revenue_per_day_before',
        'log_profit_per_day_before',
        'log_sales_per_day_before',
        'log_revenue_before_4w',
        'log_profit_before_4w',
        
        # Interaction features
        'revenue_profit_interaction',
        'category_provider_interaction',
        'sales_revenue_interaction',
    ]
    
    # Add optional features if they exist (EXCLUDING revenue_ratio and profit_ratio - data leakage!)
    optional_features = [
        'swap_month',
        'swap_quarter',
        'is_summer',
        'is_winter',
        'day_of_year',
        'is_top_half',
        'position_normalized'
    ]
    
    for feat in optional_features:
        if feat in df.columns:
            feature_cols.append(feat)
    
    # Filter to only existing columns
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    # Safety check: Remove any features that contain future information (data leakage prevention)
    # These patterns indicate features using data AFTER the swap, which wouldn't be available at prediction time
    forbidden_patterns = ['_after_', 'revenue_ratio', 'profit_ratio']
    safe_features = []
    for col in feature_cols:
        if not any(pattern in col.lower() for pattern in forbidden_patterns):
            safe_features.append(col)
        else:
            print(f"  WARNING: Excluding '{col}' - contains future information (data leakage)")
    
    feature_cols = safe_features
    
    print(f"  Using {len(feature_cols)} features (after leakage check)")
    
    # Prepare X and y
    X = df[feature_cols].copy()
    
    # Handle any missing values
    X = X.fillna(0)
    
    # Clip extreme values to prevent outliers from dominating
    for col in X.select_dtypes(include=[np.number]).columns:
        if X[col].abs().max() > 1e6:  # Very large values
            q99 = X[col].quantile(0.99)
            q01 = X[col].quantile(0.01)
            X[col] = X[col].clip(lower=q01, upper=q99)
    
    return X, df, feature_cols


def get_default_parameters(model_type='revenue'):
    """
    Get default parameters for model training.
    
    Args:
        model_type: 'revenue' or 'classification'
        
    Returns:
        Dictionary of default parameters
    """
    # Basic default parameters for XGBoost models
    default_params = {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0
    }
    
    return default_params


def train_revenue_model(X, y, feature_cols, best_params=None, model_name='revenue'):
    """Train model to predict revenue/profit change using best parameters."""
    print("\n" + "="*70)
    print(f"TRAINING {model_name.upper()} PREDICTION MODEL (Regression)")
    print("="*70)
    
    # Diagnostic: Analyze target variable
    metric_name = model_name.capitalize()
    print(f"\n--- Target Variable Diagnostics ---")
    print(f"Mean {model_name} change: {y.mean():.2f} SEK")
    print(f"Std {model_name} change:  {y.std():.2f} SEK")
    print(f"Min {model_name} change:  {y.min():.2f} SEK")
    print(f"Max {model_name} change:  {y.max():.2f} SEK")
    print(f"Median {model_name} change: {y.median():.2f} SEK")
    print(f"Positive changes: {(y > 0).sum():,} ({(y > 0).mean()*100:.1f}%)")
    print(f"Negative changes: {(y < 0).sum():,} ({(y < 0).mean()*100:.1f}%)")
    
    # Handle extreme outliers in target variable
    q99 = y.quantile(0.99)
    q01 = y.quantile(0.01)
    print(f"\nClipping target variable to [{q01:.2f}, {q99:.2f}] (1st-99th percentile)")
    y_clipped = y.clip(lower=q01, upper=q99)
    outliers_removed = (y != y_clipped).sum()
    if outliers_removed > 0:
        print(f"  Adjusted {outliers_removed} extreme values")
        y = y_clipped
    
    # Use default parameters if not provided
    if best_params is None:
        best_params = get_default_parameters(model_type='revenue')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nTraining set: {len(X_train):,} swaps")
    print(f"Test set:     {len(X_test):,} swaps")
    
    print(f"\nUsing model parameters:")
    print(f"  n_estimators: {best_params['n_estimators']}")
    print(f"  max_depth: {best_params['max_depth']}")
    print(f"  learning_rate: {best_params['learning_rate']}")
    print(f"  subsample: {best_params['subsample']}")
    print(f"  colsample_bytree: {best_params['colsample_bytree']}")
    print(f"  reg_alpha: {best_params['reg_alpha']}")
    print(f"  reg_lambda: {best_params['reg_lambda']}")
    
    # Train model with best parameters
    print("\nTraining model...")
    model = XGBRegressor(
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        learning_rate=best_params["learning_rate"],
        subsample=best_params["subsample"],
        colsample_bytree=best_params["colsample_bytree"],
        reg_alpha=best_params["reg_alpha"],
        reg_lambda=best_params["reg_lambda"],
        random_state=42,
        n_jobs=-1,
        eval_metric='rmse'  # Explicit evaluation metric
    )
    model.fit(X_train, y_train)
    
    # Calculate metrics
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    train_mae = mean_absolute_error(y_train, train_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    
    print(f"\n--- Final Performance Metrics ---")
    print(f"Train MAE: {train_mae:.2f} SEK")
    print(f"Test MAE:  {test_mae:.2f} SEK")
    print(f"Train R²:  {train_r2:.3f}")
    print(f"Test R²:   {test_r2:.3f}")
    
    # Feature importance
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n--- Top 10 Most Important Features ---")
    print(importance_df.head(10).to_string(index=False))
    
    # Save results to file
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"{model_name}_model_results_{timestamp}.json"
    
    results = {
        'timestamp': timestamp,
        'best_parameters': best_params,
        'performance': {
            'train_mae': float(train_mae),
            'test_mae': float(test_mae),
            'train_r2': float(train_r2),
            'test_r2': float(test_r2)
        },
        'top_features': importance_df.head(10).to_dict('records')
    }
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {results_file}")
    
    return model, test_pred, y_test, importance_df


def train_classification_model(X, y, feature_cols, best_params=None):
    """Train model to predict swap success (binary) using best parameters."""
    print("\n" + "="*70)
    print("TRAINING SUCCESS PREDICTION MODEL (Classification)")
    print("="*70)
    
    # Use default parameters if not provided
    if best_params is None:
        best_params = get_default_parameters(model_type='classification')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {len(X_train):,} swaps")
    print(f"Test set:     {len(X_test):,} swaps")
    print(f"Success rate in training: {y_train.mean()*100:.1f}%")
    print(f"Success rate in test:     {y_test.mean()*100:.1f}%")
    
    # Calculate class weights for imbalance
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    
    print(f"\nClass distribution: {neg_count} negative, {pos_count} positive")
    print(f"Using scale_pos_weight: {scale_pos_weight:.3f} (to handle class imbalance)")
    
    print(f"\nUsing model parameters:")
    print(f"  n_estimators: {best_params['n_estimators']}")
    print(f"  max_depth: {best_params['max_depth']}")
    print(f"  learning_rate: {best_params['learning_rate']}")
    print(f"  subsample: {best_params['subsample']}")
    print(f"  colsample_bytree: {best_params['colsample_bytree']}")
    print(f"  reg_alpha: {best_params['reg_alpha']}")
    print(f"  reg_lambda: {best_params['reg_lambda']}")
    
    # Train model with best parameters
    print("\nTraining model...")
    model = XGBClassifier(
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        learning_rate=best_params["learning_rate"],
        subsample=best_params["subsample"],
        colsample_bytree=best_params["colsample_bytree"],
        reg_alpha=best_params["reg_alpha"],
        reg_lambda=best_params["reg_lambda"],
        scale_pos_weight=scale_pos_weight,  # Handle class imbalance
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'  # Explicit evaluation metric
    )
    model.fit(X_train, y_train)
    
    # Calculate metrics
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    test_proba = model.predict_proba(X_test)[:, 1]
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    
    print(f"\n--- Final Performance Metrics ---")
    print(f"Train Accuracy: {train_acc:.3f}")
    print(f"Test Accuracy:  {test_acc:.3f}")
    
    print(f"\n--- Classification Report ---")
    print(classification_report(y_test, test_pred, target_names=['Failure', 'Success']))
    
    # Feature importance
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n--- Top 10 Most Important Features ---")
    print(importance_df.head(10).to_string(index=False))
    
    # Save results to file
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"classification_model_results_{timestamp}.json"
    
    # Get classification report as dict for saving
    results = {
        'timestamp': timestamp,
        'best_parameters': best_params,
        'performance': {
            'train_accuracy': float(train_acc),
            'test_accuracy': float(test_acc),
            'precision': float(precision_score(y_test, test_pred)),
            'recall': float(recall_score(y_test, test_pred)),
            'f1_score': float(f1_score(y_test, test_pred))
        },
        'top_features': importance_df.head(10).to_dict('records')
    }
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {results_file}")
    
    return model, test_pred, test_proba, y_test, importance_df


def save_models(revenue_model, profit_model, success_model, encoders, feature_cols, output_dir=None):
    """Save trained models and encoders."""
    if output_dir is None:
        # Default to project root models directory
        script_dir = Path(__file__).parent  # scripts/
        output_dir = script_dir.parent / "models"  # product_swap/models/
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save models
    joblib.dump(revenue_model, output_dir / 'revenue_model.pkl')
    joblib.dump(profit_model, output_dir / 'profit_model.pkl')
    joblib.dump(success_model, output_dir / 'success_model.pkl')
    joblib.dump(encoders, output_dir / 'encoders.pkl')
    joblib.dump(feature_cols, output_dir / 'feature_columns.pkl')
    
    print(f"\n✓ Saved models to {output_dir}/")
    print(f"  - revenue_model.pkl (secondary metric)")
    print(f"  - profit_model.pkl (PRIMARY METRIC ⭐)")
    print(f"  - success_model.pkl (profit-based)")
    print(f"  - encoders.pkl")
    print(f"  - feature_columns.pkl")


def main():
    """Run complete training pipeline using best parameters from results."""
    print("="*70)
    print("SWAP PREDICTION MODEL TRAINING")
    print("Using Best Parameters from Previous Results")
    print("="*70)
    
    # Paths
    script_dir = Path(__file__).parent  # scripts/
    project_root = script_dir.parent     # product_swap/
    
    # Load data
    print("\n1. Loading enriched swap data...")
    df = pd.read_parquet(project_root / 'data/swaps/product_swaps_enriched.parquet')
    print(f"   Loaded {len(df):,} swaps")
    
    # Create features
    print("\n2. Creating training features...")
    df, encoders = create_training_features(df)
    
    # Prepare data
    print("\n3. Preparing training data...")
    X, df_clean, feature_cols = prepare_data(df, min_sales_before=3)
    
    # Target variables
    y_revenue = df_clean['revenue_change_4w']
    y_profit = df_clean['profit_change_4w']
    y_success = df_clean['profit_increased']  # Changed to profit-based success
    
    # Data quality check
    print(f"\n--- Data Quality Check ---")
    print(f"Total swaps: {len(df_clean):,}")
    print(f"Swaps with profit increase: {y_success.sum():,} ({y_success.mean()*100:.1f}%)")
    print(f"Average revenue change: {y_revenue.mean():.2f} SEK")
    print(f"Average profit change: {y_profit.mean():.2f} SEK")
    print(f"Median revenue change: {y_revenue.median():.2f} SEK")
    print(f"Median profit change: {y_profit.median():.2f} SEK")
    
    # Get default parameters for training
    print("\n4. Using default model parameters...")
    revenue_params = get_default_parameters(model_type='revenue')
    classification_params = get_default_parameters(model_type='classification')
    print(f"   Revenue/Profit model parameters: {revenue_params}")
    print(f"   Classification model parameters: {classification_params}")
    
    # Train revenue prediction model
    print("\n5. Training revenue prediction model...")
    revenue_model, rev_pred, y_rev_test, rev_importance = train_revenue_model(
        X, y_revenue, feature_cols, best_params=revenue_params
    )
    
    # Train profit prediction model (PRIMARY METRIC)
    print("\n6. Training profit prediction model (PRIMARY)...")
    profit_model, prof_pred, y_prof_test, prof_importance = train_revenue_model(
        X, y_profit, feature_cols, best_params=revenue_params, model_name='profit'
    )
    
    # Train success classification model (based on profit)
    print("\n7. Training success classification model (profit-based)...")
    success_model, succ_pred, succ_proba, y_succ_test, succ_importance = train_classification_model(
        X, y_success, feature_cols, best_params=classification_params
    )
    
    # Save models
    print("\n8. Saving models...")
    save_models(revenue_model, profit_model, success_model, encoders, feature_cols)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print("\nModels trained:")
    print("  1. Revenue Change Predictor (Regression) - Secondary metric")
    print("  2. Profit Change Predictor (Regression) - PRIMARY METRIC ⭐")
    print("  3. Swap Success Predictor (Classification) - Based on profit")
    print("\nYou can now use these models to predict swap outcomes!")
    print("\nNote: Profit is the primary metric for swap recommendations.")
    

if __name__ == "__main__":
    main()




