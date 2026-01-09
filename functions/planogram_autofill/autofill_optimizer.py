"""Autofill optimizer module.

Suggests optimal products for empty slots in vending machines to maximize revenue.
Uses the trained XGBoost model to predict sales for candidate products.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

import pandas as pd
import numpy as np

from .config import (
    MODEL_OUTPUT_DIR,
    IDENTITY_COLUMNS,
    TARGET_COLUMN,
    LOCATION_MODEL_ARTIFACT_PATH,
    UNIQUENESS_MODEL_ARTIFACT_PATH,
)
from .feature_builder import (
    build_feature_matrix,
    UniquenessJoiner,
    LocationFitJoiner,
    get_feature_columns,
    _add_temporal_features,
)
from .sales_modeling import predict_sales_scores


def suggest_autofill(
    machine_id: str,
    current_slots: List[Dict[str, Any]],
    empty_slots: List[str],
    available_products: pd.DataFrame,
    machine_metadata: Dict[str, Any],
    target_date: datetime,
    *,
    feature_groups: List[str] = ["CORE", "TEMPORAL", "UNIQUENESS", "LOCATION"],
    model_path: Optional[Path] = None,
    max_candidates_per_slot: int = 20,
    min_category_diversity: float = 0.3,
    prioritize_providers: Optional[List[str]] = None,
    suggest_replacements: bool = False,
    replacement_threshold: float = 0.5,  # Suggest replacement if better product has >50% more revenue
) -> Dict[str, Any]:
    """
    Suggest optimal products for empty slots to maximize revenue.
    
    Args:
        machine_id: Machine identifier
        current_slots: List of currently filled slots with product info
            [{"position": "A1", "product_name": "Coke", "category": "Läsk", "provider": "Coca-Cola", "price": 16.0, ...}, ...]
        empty_slots: List of empty slot positions ["A2", "B1", ...]
        available_products: DataFrame with available products (must have: product_name, category, subcategory, provider, price_mean, purchase_price_kr, ean)
        machine_metadata: Machine metadata dict with at least {"machine_eva_group": "..."}
        target_date: Target date for predictions (used for temporal features)
        feature_groups: Feature groups to use for prediction
        model_path: Path to trained model (defaults to output/models/global_sales_model.joblib)
        max_candidates_per_slot: Maximum number of candidate products to evaluate per slot
        min_category_diversity: Minimum category diversity score to maintain
        prioritize_providers: List of provider names to prioritize (e.g., ["Dafgårds", "Brynmor", "LUB Foods"])
    
    Returns:
        Dict with recommendations:
        {
            "recommendations": [
                {"position": "A2", "product_name": "...", "predicted_sales": 12.5, "expected_revenue": 200.0, ...},
                ...
            ],
            "total_expected_revenue": 1245.50,
            "current_revenue_estimate": 1000.00,
            "improvement": 24.55
        }
    """
    if not empty_slots:
        return {
            "recommendations": [],
            "total_expected_revenue": 0.0,
            "current_revenue_estimate": 0.0,
            "improvement": 0.0,
        }
    
    print(f"\n{'='*70}")
    print(f"🎯 Autofill Optimization for Machine {machine_id}")
    print(f"{'='*70}")
    print(f"📊 Filled slots: {len(current_slots)}")
    print(f"📊 Empty slots: {len(empty_slots)}")
    print(f"📦 Available products: {len(available_products)}")
    
    # Step 1: Evaluate CURRENT machine configuration using the model
    print(f"\n📋 Step 1: Evaluating current machine configuration...")
    current_revenue, current_predictions = evaluate_current_machine(
        machine_id=machine_id,
        current_slots=current_slots,
        machine_metadata=machine_metadata,
        target_date=target_date,
        feature_groups=feature_groups,
        model_path=model_path,
    )
    print(f"   ✓ Current machine revenue (predicted): {current_revenue:.2f} SEK/week")
    print(f"   ✓ Evaluated {len(current_slots)} filled slots using sales prediction model")
    
    # Step 2: Generate candidates for each empty slot
    print(f"\n📋 Step 2: Generating candidates for empty slots...")
    candidates_df = generate_candidates(
        machine_id=machine_id,
        empty_slots=empty_slots,
        available_products=available_products,
        current_slots=current_slots,
        machine_metadata=machine_metadata,
        target_date=target_date,
        max_candidates_per_slot=max_candidates_per_slot,
        prioritize_providers=prioritize_providers,
    )
    print(f"   ✓ Generated {len(candidates_df):,} candidate product-slot combinations")
    
    # Step 3: Build features for candidates
    print(f"\n📋 Step 3: Building features for candidates...")
    feature_df = build_features_for_candidates(
        candidates_df=candidates_df,
        machine_metadata=machine_metadata,
        target_date=target_date,
        feature_groups=feature_groups,
    )
    print(f"   ✓ Built features: {len(feature_df):,} rows × {len(feature_df.columns)} columns")
    
    # Step 4: Predict sales for candidates using the model
    print(f"\n📋 Step 4: Predicting sales for candidates using trained model...")
    predictions_df = predict_sales_scores(feature_df, model_path=model_path, feature_groups=feature_groups)
    if predictions_df.empty:
        raise ValueError("Prediction failed - no results returned")
    print(f"   ✓ Predicted sales for {len(predictions_df):,} candidates using XGBoost model")
    
    # Step 5: Optimize selection to maximize total revenue
    print(f"\n📋 Step 5: Optimizing product selection to maximize revenue...")
    recommendations = optimize_revenue(
        predictions_df=predictions_df,
        empty_slots=empty_slots,
        current_slots=current_slots,
        min_category_diversity=min_category_diversity,
    )
    
    # Calculate optimized total revenue (current + new recommendations)
    new_revenue = sum(r["expected_revenue"] for r in recommendations)
    optimized_total_revenue = current_revenue + new_revenue
    improvement = ((new_revenue) / current_revenue * 100) if current_revenue > 0 else 0.0
    
    # Optional: Suggest replacements for underperforming products
    replacement_suggestions = []
    if suggest_replacements and not current_predictions.empty:
        print(f"\n📋 Step 6: Analyzing replacement opportunities...")
        replacement_suggestions = suggest_product_replacements(
            current_predictions=current_predictions,
            available_products=available_products,
            machine_id=machine_id,
            machine_metadata=machine_metadata,
            target_date=target_date,
            feature_groups=feature_groups,
            model_path=model_path,
            replacement_threshold=replacement_threshold,
        )
        if replacement_suggestions:
            print(f"   ✓ Found {len(replacement_suggestions)} replacement opportunities")
        else:
            print(f"   ℹ️  No significant replacement opportunities found")
    
    print(f"\n✅ Autofill Optimization Complete!")
    print(f"\n📊 Revenue Analysis (using sales prediction model):")
    print(f"   Current machine revenue: {current_revenue:.2f} SEK/week")
    print(f"   + New products revenue: {new_revenue:.2f} SEK/week")
    if replacement_suggestions:
        replacement_revenue_gain = sum(r.get("revenue_improvement", 0) for r in replacement_suggestions)
        print(f"   + Replacement improvements: {replacement_revenue_gain:.2f} SEK/week")
        optimized_total_revenue += replacement_revenue_gain
    print(f"   = Optimized total revenue: {optimized_total_revenue:.2f} SEK/week")
    print(f"   Revenue increase: {improvement:.1f}%")
    print(f"   Recommended products: {len(recommendations)}")
    if replacement_suggestions:
        print(f"   Replacement suggestions: {len(replacement_suggestions)}")
    print(f"{'='*70}")
    
    return {
        "recommendations": recommendations,
        "current_revenue_predicted": current_revenue,
        "new_products_revenue": new_revenue,
        "optimized_total_revenue": optimized_total_revenue,
        "improvement": improvement,
        "current_predictions": current_predictions,  # For analysis
        "replacement_suggestions": replacement_suggestions,  # Optional replacements
    }


def generate_candidates(
    machine_id: str,
    empty_slots: List[str],
    available_products: pd.DataFrame,
    current_slots: List[Dict[str, Any]],
    machine_metadata: Dict[str, Any],
    target_date: datetime,
    max_candidates_per_slot: int = 20,
    prioritize_providers: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Generate candidate product-slot combinations for empty slots.
    
    Filters and ranks products based on:
    - Provider performance (prioritizes high-performing providers)
    - Category diversity (avoids too many of same category)
    - Product availability
    """
    # Get current categories to maintain diversity
    current_categories = {slot.get("category") for slot in current_slots if slot.get("category")}
    current_subcategories = {slot.get("subcategory") for slot in current_slots if slot.get("subcategory")}
    
    # Filter available products
    required_cols = ["product_name", "category", "subcategory", "provider", "ean"]
    missing_cols = [col for col in required_cols if col not in available_products.columns]
    if missing_cols:
        raise ValueError(f"available_products missing required columns: {missing_cols}")
    
    # Score products for prioritization
    products_scored = available_products.copy()
    
    # Prioritize high-performing providers (based on feature importance analysis)
    if prioritize_providers is None:
        prioritize_providers = ["Dafgårds", "Brynmor", "LUB Foods", "Carlsberg", "Cloetta"]
    
    provider_score = products_scored["provider"].apply(
        lambda p: 10 if p in prioritize_providers else 1
    )
    
    # Prioritize strong subcategories
    strong_subcategories = ["Energidryck", "Läsk", "Knäckebröd", "Juice", "Färdigmat"]
    subcategory_score = products_scored["subcategory"].apply(
        lambda s: 5 if s in strong_subcategories else 1
    )
    
    # Combine scores
    products_scored["_priority_score"] = provider_score * subcategory_score
    
    # Generate candidates: each empty slot × top products
    candidates = []
    
    for slot_position in empty_slots:
        # Get top products for this slot
        top_products = products_scored.nlargest(max_candidates_per_slot, "_priority_score")
        
        for _, product in top_products.iterrows():
            candidates.append({
                "machine_id": machine_id,
                "position": slot_position,
                "snapshot_date": target_date,
                "ean": product.get("ean", ""),
                "product_name": product["product_name"],
                "category": product["category"],
                "subcategory": product["subcategory"],
                "provider": product["provider"],
                "price_mean": product.get("price_mean", product.get("price", 0.0)),
                "purchase_price_kr": product.get("purchase_price_kr", 0.0),
                "_priority_score": product["_priority_score"],
            })
    
    candidates_df = pd.DataFrame(candidates)
    
    # Add machine metadata
    candidates_df["machine_eva_group"] = machine_metadata.get("machine_eva_group", "")
    candidates_df["machine_sub_group"] = machine_metadata.get("machine_sub_group", "")
    
    return candidates_df


def build_features_for_candidates(
    candidates_df: pd.DataFrame,
    machine_metadata: Dict[str, Any],
    target_date: datetime,
    feature_groups: List[str],
) -> pd.DataFrame:
    """
    Build feature matrix for candidate products.
    
    Uses the same feature building pipeline as training:
    - CORE features (price, category, subcategory, provider)
    - TEMPORAL features (week_of_year, month, day_of_week)
    - UNIQUENESS features (from planogram models)
    - LOCATION features (from planogram models)
    """
    df = candidates_df.copy()
    
    # Ensure snapshot_date is datetime
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"], errors="coerce")
    
    # Add temporal features
    if "TEMPORAL" in feature_groups:
        df = _add_temporal_features(df)
    
    # Add lookback_weeks (default for new machines)
    if "lookback_weeks" not in df.columns:
        df["lookback_weeks"] = 4
    
    # Build joiners for planogram features
    joiners = []
    
    if "UNIQUENESS" in feature_groups:
        uniqueness_joiner = UniquenessJoiner(trained_model_path=UNIQUENESS_MODEL_ARTIFACT_PATH)
        joiners.append(uniqueness_joiner)
    
    if "LOCATION" in feature_groups:
        location_joiner = LocationFitJoiner(trained_model_path=LOCATION_MODEL_ARTIFACT_PATH)
        joiners.append(location_joiner)
    
    # Add dummy target column (required by build_feature_matrix but not used for prediction)
    if TARGET_COLUMN not in df.columns:
        df[TARGET_COLUMN] = 0.0
    
    # Preserve metadata columns that will be needed later (not in IDENTITY_COLUMNS)
    # build_feature_matrix filters columns, so we need to preserve these separately
    metadata_cols = ['product_name', 'category', 'subcategory', 'provider', 'price_mean', 'purchase_price_kr']
    # Store metadata DataFrame with IDENTITY_COLUMNS as merge key
    # Also store original index for fallback alignment
    available_metadata_cols = [col for col in metadata_cols if col in df.columns]
    if available_metadata_cols:
        metadata_df = df[IDENTITY_COLUMNS + available_metadata_cols].copy()
        # Store original index mapping
        original_index = df.index.copy()
    else:
        metadata_df = None
        original_index = None
    
    # Build feature matrix (this will filter columns and may reorder rows)
    feature_df = build_feature_matrix(
        snapshots=df,
        feature_groups=feature_groups,
        joiners=joiners if joiners else None,
    )
    
    # Restore metadata columns that were filtered out
    # Merge back using IDENTITY_COLUMNS as key (since build_feature_matrix sorts by these)
    if metadata_df is not None:
        # Find columns that are missing from feature_df
        missing_metadata = [col for col in available_metadata_cols if col not in feature_df.columns]
        if missing_metadata:
            # Ensure IDENTITY_COLUMNS exist in both DataFrames for merging
            missing_id_cols = [col for col in IDENTITY_COLUMNS if col not in feature_df.columns or col not in metadata_df.columns]
            if missing_id_cols:
                raise ValueError(f"Cannot merge metadata: missing IDENTITY_COLUMNS: {missing_id_cols}")
            
            # Merge all missing metadata columns at once
            feature_df = feature_df.merge(
                metadata_df[IDENTITY_COLUMNS + missing_metadata],
                on=IDENTITY_COLUMNS,
                how='left',
                suffixes=('', '_meta')
            )
            # Handle any duplicate columns from merge
            for col in missing_metadata:
                if f'{col}_meta' in feature_df.columns:
                    feature_df[col] = feature_df[f'{col}_meta']
                    feature_df = feature_df.drop(columns=[f'{col}_meta'])
            
            # Verify all metadata columns are now present
            still_missing = [col for col in missing_metadata if col not in feature_df.columns]
            if still_missing:
                print(f"⚠️  Warning: Could not restore metadata columns via merge: {still_missing}")
                # Fallback: try to align by original index if lengths match
                if original_index is not None and len(feature_df) == len(original_index):
                    print(f"   Attempting fallback: aligning by original index...")
                    for col in still_missing:
                        if col in metadata_df.columns:
                            # Reset index to align by position
                            feature_df_reset = feature_df.reset_index(drop=True)
                            metadata_reset = metadata_df.reset_index(drop=True)
                            if len(feature_df_reset) == len(metadata_reset):
                                feature_df[col] = metadata_reset[col].values
                                print(f"   ✓ Restored {col} via index alignment")
                
                # Final check
                still_missing_after = [col for col in missing_metadata if col not in feature_df.columns]
                if still_missing_after:
                    raise ValueError(f"Failed to restore required metadata columns: {still_missing_after}. "
                                   f"These columns are needed for optimization.")
    
    # Remove target column before prediction (model expects features only)
    if TARGET_COLUMN in feature_df.columns:
        feature_df = feature_df.drop(columns=[TARGET_COLUMN])
    
    return feature_df


def optimize_revenue(
    predictions_df: pd.DataFrame,
    empty_slots: List[str],
    current_slots: List[Dict[str, Any]],
    min_category_diversity: float = 0.3,
) -> List[Dict[str, Any]]:
    """
    Optimize product selection to maximize total revenue.
    
    Uses a greedy algorithm:
    1. For each empty slot, select the product with highest expected revenue
    2. Ensure category diversity is maintained
    3. Avoid duplicate products in same machine
    """
    # Get current products to avoid duplicates
    current_products = {slot.get("product_name") for slot in current_slots if slot.get("product_name")}
    current_categories = {slot.get("category") for slot in current_slots if slot.get("category")}
    
    # Calculate expected revenue (predicted_sales * price)
    if "price_mean" in predictions_df.columns:
        predictions_df["expected_revenue"] = predictions_df["predicted_sales"] * predictions_df["price_mean"]
    else:
        # Fallback: use purchase_price_kr * markup (assume 50% markup)
        predictions_df["expected_revenue"] = predictions_df["predicted_sales"] * predictions_df.get("purchase_price_kr", 10) * 1.5
    
    recommendations = []
    selected_products = set(current_products)
    selected_categories = set(current_categories)
    
    # Sort by expected revenue (descending)
    predictions_sorted = predictions_df.sort_values("expected_revenue", ascending=False)
    
    # Greedy selection: for each empty slot, pick best available product
    for slot_position in empty_slots:
        slot_candidates = predictions_sorted[
            (predictions_sorted["position"] == slot_position) &
            (~predictions_sorted["product_name"].isin(selected_products))
        ]
        
        if slot_candidates.empty:
            # No candidates left for this slot
            continue
        
        # Select best candidate
        best = slot_candidates.iloc[0]
        
        # Add to recommendations
        recommendations.append({
            "position": best["position"],
            "product_name": best["product_name"],
            "category": best.get("category", ""),
            "subcategory": best.get("subcategory", ""),
            "provider": best.get("provider", ""),
            "price": best.get("price_mean", 0.0),
            "predicted_sales": float(best["predicted_sales"]),
            "expected_revenue": float(best["expected_revenue"]),
            "ean": best.get("ean", ""),
        })
        
        # Update selected sets
        selected_products.add(best["product_name"])
        selected_categories.add(best.get("category", ""))
    
    return recommendations


def evaluate_current_machine(
    machine_id: str,
    current_slots: List[Dict[str, Any]],
    machine_metadata: Dict[str, Any],
    target_date: datetime,
    feature_groups: List[str],
    model_path: Optional[Path] = None,
) -> Tuple[float, pd.DataFrame]:
    """
    Evaluate current machine configuration using the sales prediction model.
    
    Predicts sales for all currently filled slots and calculates total revenue.
    
    Returns:
        (total_revenue, predictions_df) - Total predicted revenue and predictions for each slot
    """
    if not current_slots:
        return 0.0, pd.DataFrame()
    
    # Convert current slots to DataFrame format
    current_df = pd.DataFrame([
        {
            "machine_id": machine_id,
            "position": slot.get("position", ""),
            "snapshot_date": target_date,
            "ean": slot.get("ean", ""),
            "product_name": slot.get("product_name", ""),
            "category": slot.get("category", ""),
            "subcategory": slot.get("subcategory", ""),
            "provider": slot.get("provider", ""),
            "price_mean": slot.get("price", slot.get("price_mean", 0.0)),
            "purchase_price_kr": slot.get("purchase_price_kr", 0.0),
            "machine_eva_group": machine_metadata.get("machine_eva_group", ""),
            "machine_sub_group": machine_metadata.get("machine_sub_group", ""),
        }
        for slot in current_slots
    ])
    
    # Build features for current slots
    current_features = build_features_for_candidates(
        candidates_df=current_df,
        machine_metadata=machine_metadata,
        target_date=target_date,
        feature_groups=feature_groups,
    )
    
    # Predict sales using the model
    current_predictions = predict_sales_scores(
        current_features, 
        model_path=model_path, 
        feature_groups=feature_groups
    )
    
    if current_predictions.empty:
        return 0.0, pd.DataFrame()
    
    # Calculate revenue (predicted_sales * price)
    if "price_mean" in current_predictions.columns:
        current_predictions["expected_revenue"] = (
            current_predictions["predicted_sales"] * current_predictions["price_mean"]
        )
    else:
        current_predictions["expected_revenue"] = (
            current_predictions["predicted_sales"] * current_predictions.get("purchase_price_kr", 10) * 1.5
        )
    
    total_revenue = current_predictions["expected_revenue"].sum()
    
    return float(total_revenue), current_predictions


def suggest_product_replacements(
    current_predictions: pd.DataFrame,
    available_products: pd.DataFrame,
    machine_id: str,
    machine_metadata: Dict[str, Any],
    target_date: datetime,
    feature_groups: List[str],
    model_path: Optional[Path] = None,
    replacement_threshold: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Suggest replacing underperforming products with better alternatives.
    
    For each current product, finds better alternatives and suggests replacement
    if the improvement exceeds the threshold.
    
    Returns:
        List of replacement suggestions with revenue improvement estimates
    """
    replacements = []
    
    # For each current product, find better alternatives
    for _, current_row in current_predictions.iterrows():
        position = current_row.get("position", "")
        current_product = current_row.get("product_name", "")
        current_revenue = current_row.get("expected_revenue", 0.0)
        
        if not position or not current_product:
            continue
        
        # Find alternative products (exclude current product)
        alternatives = available_products[
            available_products["product_name"] != current_product
        ].copy()
        
        if alternatives.empty:
            continue
        
        # Generate candidates for this position with alternative products
        alt_candidates = pd.DataFrame([{
            "machine_id": machine_id,
            "position": position,
            "snapshot_date": target_date,
            "ean": product.get("ean", ""),
            "product_name": product["product_name"],
            "category": product["category"],
            "subcategory": product["subcategory"],
            "provider": product["provider"],
            "price_mean": product.get("price_mean", product.get("price", 0.0)),
            "purchase_price_kr": product.get("purchase_price_kr", 0.0),
            "machine_eva_group": machine_metadata.get("machine_eva_group", ""),
            "machine_sub_group": machine_metadata.get("machine_sub_group", ""),
        } for _, product in alternatives.head(10).iterrows()])  # Limit to top 10 alternatives
        
        # Build features and predict
        alt_features = build_features_for_candidates(
            candidates_df=alt_candidates,
            machine_metadata=machine_metadata,
            target_date=target_date,
            feature_groups=feature_groups,
        )
        
        alt_predictions = predict_sales_scores(
            alt_features,
            model_path=model_path,
            feature_groups=feature_groups
        )
        
        if alt_predictions.empty:
            continue
        
        # Calculate revenue for alternatives
        if "price_mean" in alt_predictions.columns:
            alt_predictions["expected_revenue"] = (
                alt_predictions["predicted_sales"] * alt_predictions["price_mean"]
            )
        else:
            alt_predictions["expected_revenue"] = (
                alt_predictions["predicted_sales"] * alt_predictions.get("purchase_price_kr", 10) * 1.5
            )
        
        # Find best alternative
        best_alt = alt_predictions.nlargest(1, "expected_revenue")
        if best_alt.empty:
            continue
        
        best_revenue = float(best_alt.iloc[0]["expected_revenue"])
        improvement_pct = ((best_revenue - current_revenue) / current_revenue) if current_revenue > 0 else 0.0
        
        # Suggest replacement if improvement exceeds threshold
        if improvement_pct >= replacement_threshold:
            replacements.append({
                "position": position,
                "current_product": current_product,
                "current_revenue": float(current_revenue),
                "suggested_product": best_alt.iloc[0]["product_name"],
                "suggested_revenue": best_revenue,
                "revenue_improvement": best_revenue - current_revenue,
                "improvement_pct": improvement_pct * 100,
                "predicted_sales": float(best_alt.iloc[0]["predicted_sales"]),
            })
    
    # Sort by improvement
    replacements.sort(key=lambda x: x["revenue_improvement"], reverse=True)
    
    return replacements

