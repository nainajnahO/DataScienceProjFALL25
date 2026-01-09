"""
Feature Analysis Utilities
===========================

Functions for analyzing feature quality and importance during incremental feature engineering.

Includes:
- Correlation analysis (feature-target, feature-feature)
- Feature importance analysis via XGBoost
- Automatic redundant feature removal
- Comprehensive feature group evaluation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')


def correlation_analysis(
    df: pd.DataFrame, 
    feature_cols: List[str], 
    target_col: str = 'weekly_sales',
    correlation_threshold: float = 0.95
) -> Dict:
    """
    Analyze correlations between features and target, and between features.
    
    Handles categorical features by encoding them:
    - Binary features (0/1, bool): Treated as numeric or converted to int
    - Multi-category categoricals: Label encoded
    
    Args:
        df: DataFrame with features and target
        feature_cols: List of feature column names
        target_col: Target column name
        correlation_threshold: Threshold for identifying highly correlated features
        
    Returns:
        Dictionary with:
        - feature_target_corr: DataFrame with feature-target correlations
        - feature_feature_pairs: DataFrame with highly correlated feature pairs
        - summary: Summary statistics
    """
    print(f"\n{'='*60}")
    print(f"CORRELATION ANALYSIS")
    print(f"{'='*60}")
    
    # Create a copy for encoding without modifying original
    df_encoded = df.copy()
    encoding_map = {}
    
    # Identify numeric vs categorical features
    existing_features = [col for col in feature_cols if col in df.columns]
    
    numeric_features = []
    categorical_features = []
    
    for col in existing_features:
        dtype = df[col].dtype
        n_unique = df[col].nunique()
        
        # Check if numeric (any numeric dtype including int8, int32, bool, etc.)
        if pd.api.types.is_numeric_dtype(dtype):
            numeric_features.append(col)
        elif pd.api.types.is_bool_dtype(dtype):
            # Convert bool to int for correlation
            df_encoded[col] = df_encoded[col].astype(int)
            numeric_features.append(col)
        elif n_unique == 2:
            # Binary categorical - label encode
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df[col].astype(str))
            encoding_map[col] = 'label_encoded_binary'
            numeric_features.append(col)
        else:
            # Multi-category categorical - label encode
            categorical_features.append(col)
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df[col].astype(str))
            encoding_map[col] = 'label_encoded'
            numeric_features.append(col)
    
    if len(numeric_features) == 0:
        print("❌ No numeric features to analyze")
        return None
    
    if encoding_map:
        encoded_cols = list(encoding_map.keys())
        print(f"✓ Encoded {len(encoded_cols)} categorical features:")
        for col, enc_type in encoding_map.items():
            print(f"  - {col}: {enc_type}")
    
    print(f"Analyzing {len(numeric_features)} features...")
    
    # Feature-target correlations (use encoded dataframe)
    feature_target_corr = pd.DataFrame({
        'feature': numeric_features,
        'target_correlation': [df_encoded[col].corr(df_encoded[target_col]) for col in numeric_features]
    }).sort_values('target_correlation', key=abs, ascending=False)
    
    # Feature-feature correlations (use encoded dataframe)
    corr_matrix = df_encoded[numeric_features].corr().abs()
    
    # Find highly correlated pairs
    highly_correlated = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] >= correlation_threshold:
                highly_correlated.append({
                    'feature_1': corr_matrix.columns[i],
                    'feature_2': corr_matrix.columns[j],
                    'correlation': corr_matrix.iloc[i, j]
                })
    
    # Handle empty list case
    if len(highly_correlated) > 0:
        feature_feature_pairs = pd.DataFrame(highly_correlated).sort_values('correlation', ascending=False)
    else:
        feature_feature_pairs = pd.DataFrame(columns=['feature_1', 'feature_2', 'correlation'])
    
    # Find moderately correlated pairs (0.5-0.95) for additional insight
    moderate_corr_threshold = 0.5
    moderately_correlated = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if moderate_corr_threshold <= corr_val < correlation_threshold:
                moderately_correlated.append({
                    'feature_1': corr_matrix.columns[i],
                    'feature_2': corr_matrix.columns[j],
                    'correlation': corr_val
                })
    
    # Summary
    print(f"\n📊 Results:")
    print(f"  Features analyzed: {len(numeric_features)}")
    print(f"  Avg target correlation: {feature_target_corr['target_correlation'].abs().mean():.4f}")
    print(f"  Max target correlation: {feature_target_corr['target_correlation'].abs().max():.4f}")
    print(f"\n  Feature-Feature Correlations:")
    print(f"    - Highly correlated pairs (≥{correlation_threshold}): {len(feature_feature_pairs)}")
    print(f"    - Moderately correlated pairs (≥{moderate_corr_threshold}): {len(moderately_correlated)}")
    
    if len(feature_target_corr) > 0:
        print(f"\n  Top 5 Features by Target Correlation:")
        for idx, row in feature_target_corr.head(5).iterrows():
            direction = "📈" if row['target_correlation'] > 0 else "📉"
            print(f"    {direction} {row['feature']:35s}: {row['target_correlation']:7.4f}")
    
    # Show top moderately/highly correlated pairs if any
    if len(moderately_correlated) > 0:
        moderate_df = pd.DataFrame(moderately_correlated).sort_values('correlation', ascending=False)
        print(f"\n  Top 5 Moderately Correlated Feature Pairs:")
        for idx, row in moderate_df.head(5).iterrows():
            shorter_1 = row['feature_1'][:20] + "..." if len(row['feature_1']) > 23 else row['feature_1']
            shorter_2 = row['feature_2'][:20] + "..." if len(row['feature_2']) > 23 else row['feature_2']
            print(f"    {shorter_1:23s} ↔ {shorter_2:23s}: {row['correlation']:.4f}")
    
    # Combine all feature-feature pairs (moderate + highly correlated)
    all_feature_pairs = feature_feature_pairs.copy()
    if len(moderately_correlated) > 0:
        moderate_df = pd.DataFrame(moderately_correlated).sort_values('correlation', ascending=False)
        all_feature_pairs = pd.concat([feature_feature_pairs, moderate_df], ignore_index=True) if len(feature_feature_pairs) > 0 else moderate_df
    
    return {
        'feature_target_corr': feature_target_corr,
        'feature_feature_pairs': feature_feature_pairs,
        'all_feature_pairs': all_feature_pairs.sort_values('correlation', ascending=False) if len(all_feature_pairs) > 0 else pd.DataFrame(columns=['feature_1', 'feature_2', 'correlation']),
        'summary': {
            'n_features': len(numeric_features),
            'avg_corr': feature_target_corr['target_correlation'].abs().mean(),
            'high_corr_pairs': len(feature_feature_pairs)
        }
    }


def feature_importance_analysis(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = 'weekly_sales',
    random_seed: int = 42
) -> Tuple[pd.DataFrame, float, float]:
    """
    Train XGBoost and analyze feature importance.
    
    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        feature_cols: List of feature column names
        target_col: Target column name
        random_seed: Random seed for reproducibility
        
    Returns:
        (importance_df, test_mae, test_r2)
    """
    print(f"\n{'='*60}")
    print(f"FEATURE IMPORTANCE ANALYSIS")
    print(f"{'='*60}")
    
    # Prepare data
    X_train = train_df[feature_cols].copy()
    y_train = train_df[target_col].copy()
    X_test = test_df[feature_cols].copy()
    y_test = test_df[target_col].copy()
    
    # Handle missing values
    X_train = X_train.fillna(X_train.median())
    X_test = X_test.fillna(X_train.median())
    
    # Encode categorical features
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns
    if len(categorical_features) > 0:
        from sklearn.preprocessing import LabelEncoder
        le_dict = {}
        for col in categorical_features:
            le = LabelEncoder()
            X_train[col] = le.fit_transform(X_train[col].astype(str))
            X_test[col] = le.transform(X_test[col].astype(str))
            le_dict[col] = le
    
    print(f"Training XGBoost on {len(X_train):,} samples with {len(feature_cols)} features...")
    
    # Train model
    model = XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=random_seed,
        n_jobs=-1,
        verbosity=0
    )
    
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"\n📊 Test Set Performance:")
    print(f"  MAE: {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²: {r2:.4f}")
    
    # Feature importance
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 5 features by importance:")
    for idx, row in importance_df.head(5).iterrows():
        print(f"  {row['feature']}: {row['importance']:.6f}")
    
    return importance_df, mae, r2


def remove_redundant_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    importance_dict: Optional[Dict[str, float]] = None,
    correlation_threshold: float = 0.95,
    min_importance: float = 0.001
) -> List[str]:
    """
    Remove redundant features based on correlation and importance.
    
    Args:
        df: DataFrame with features
        feature_cols: List of feature column names
        importance_dict: Dict mapping feature names to importance scores
        correlation_threshold: Threshold for removing highly correlated features
        min_importance: Minimum importance to keep a feature
        
    Returns:
        List of features to keep
    """
    print(f"\n{'='*60}")
    print(f"REMOVING REDUNDANT FEATURES")
    print(f"{'='*60}")
    
    features_to_keep = set(feature_cols)
    
    # Run correlation analysis
    corr_results = correlation_analysis(df, list(features_to_keep), correlation_threshold=correlation_threshold)
    
    if corr_results is None or len(corr_results['feature_feature_pairs']) == 0:
        print("✓ No highly correlated features found")
    else:
        # Remove one feature from each highly correlated pair
        for _, row in corr_results['feature_feature_pairs'].iterrows():
            feat1, feat2 = row['feature_1'], row['feature_2']
            
            # Keep the one with higher importance
            if importance_dict and feat1 in importance_dict and feat2 in importance_dict:
                if importance_dict[feat1] >= importance_dict[feat2]:
                    features_to_keep.discard(feat2)
                    print(f"❌ Removed {feat2} (corr={row['correlation']:.3f} with {feat1})")
                else:
                    features_to_keep.discard(feat1)
                    print(f"❌ Removed {feat1} (corr={row['correlation']:.3f} with {feat2})")
            else:
                # If no importance data, remove the second one
                features_to_keep.discard(feat2)
                print(f"❌ Removed {feat2} (corr={row['correlation']:.3f} with {feat1})")
    
    # Remove low-importance features
    if importance_dict:
        removed_low_importance = []
        for feat in list(features_to_keep):
            if feat in importance_dict and importance_dict[feat] < min_importance:
                features_to_keep.discard(feat)
                removed_low_importance.append(feat)
        
        if removed_low_importance:
            print(f"\n❌ Removed {len(removed_low_importance)} low-importance features (<{min_importance}):")
            for feat in removed_low_importance:
                print(f"  - {feat}: {importance_dict[feat]:.6f}")
    
    print(f"\n✓ Kept {len(features_to_keep)} features (removed {len(feature_cols) - len(features_to_keep)})")
    
    return sorted(list(features_to_keep))


def evaluate_feature_group(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    base_features: List[str],
    new_features: List[str],
    group_name: str,
    target_col: str = 'weekly_sales',
    random_seed: int = 42
) -> Dict:
    """
    Comprehensive evaluation of a new feature group.
    
    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        base_features: List of existing features
        new_features: List of new features to evaluate
        group_name: Name of the feature group
        target_col: Target column name
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary with evaluation results
    """
    print(f"\n{'='*80}")
    print(f"EVALUATING FEATURE GROUP: {group_name}")
    print(f"{'='*80}")
    
    # Check which features actually exist
    existing_new_features = [f for f in new_features if f in train_df.columns]
    missing_features = set(new_features) - set(existing_new_features)
    
    if missing_features:
        print(f"⚠️  Warning: {len(missing_features)} features not found in data")
        print(f"   Missing: {list(missing_features)[:5]}...")
    
    if len(existing_new_features) == 0:
        print("❌ No valid features to evaluate")
        return None
    
    # 1. Correlation analysis for new features
    print(f"\n1️⃣  Correlation Analysis")
    corr_results = correlation_analysis(train_df, existing_new_features, target_col)
    
    # 2. XGBoost with only new features
    print(f"\n2️⃣  Model Performance: Only {group_name} Features")
    new_only_importance, new_only_mae, new_only_r2 = feature_importance_analysis(
        train_df, test_df, existing_new_features, target_col, random_seed
    )
    
    # 3. XGBoost with base + new features
    combined_features = base_features + existing_new_features
    existing_combined = [f for f in combined_features if f in train_df.columns]
    
    print(f"\n3️⃣  Model Performance: Base + {group_name} Features")
    combined_importance, combined_mae, combined_r2 = feature_importance_analysis(
        train_df, test_df, existing_combined, target_col, random_seed
    )
    
    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY: {group_name}")
    print(f"{'='*80}")
    print(f"New features added: {len(existing_new_features)}")
    print(f"MAE (new only):      {new_only_mae:.4f}")
    print(f"MAE (combined):      {combined_mae:.4f}")
    print(f"R² (new only):       {new_only_r2:.4f}")
    print(f"R² (combined):       {combined_r2:.4f}")
    
    return {
        'group_name': group_name,
        'new_features': existing_new_features,
        'correlation': corr_results,
        'importance_new_only': new_only_importance,
        'mae_new_only': new_only_mae,
        'r2_new_only': new_only_r2,
        'importance_combined': combined_importance,
        'mae_combined': combined_mae,
        'r2_combined': combined_r2,
        'improvement': None  # To be calculated when comparing to baseline
    }

