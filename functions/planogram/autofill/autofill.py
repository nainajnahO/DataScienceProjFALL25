"""Autofill module for product suggestions based on predicted revenue.

This module provides two core workflows:
1. run_autofill_workflow(): Fills empty slots in a machine config.
2. run_swap_workflow(): Optimizes a machine config by swapping existing products for better ones.
"""

from __future__ import annotations
from typing import Optional, Dict, Any, Tuple
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def _setup_workflow_data(
    machine_config: dict, 
    predictions_df: pd.DataFrame, 
    products_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, dict, dict, dict]:
    """Shared helper to prepare DataFrames and metadata maps for both workflows."""
    slots_df = pd.DataFrame(machine_config['slots'])
    if 'ean' not in slots_df.columns:
        slots_df['ean'] = None
    slots_df['ean_int'] = pd.to_numeric(slots_df['ean'], errors='coerce').fillna(0).astype(int)
    
    # Metadata lookup - Ensure unique EANs and prefer rows with data
    prod_meta = products_df.dropna(subset=['ean']).copy()
    prod_meta['ean_int'] = pd.to_numeric(prod_meta['ean'], errors='coerce').fillna(0).astype(int)
    prod_meta = prod_meta.sort_values(['subcategory', 'provider'], na_position='last')
    prod_meta = prod_meta.drop_duplicates(subset=['ean_int'], keep='first')
    
    meta_map = prod_meta.set_index('ean_int').to_dict('index')
    width_map = prod_meta.set_index('ean_int')['width'].fillna(1.0).to_dict()
    
    # Global Enrichment
    for idx, row in slots_df.iterrows():
        ean = row['ean_int']
        if ean in meta_map:
            info = meta_map[ean]
            for field in ['product_name', 'category', 'subcategory', 'provider']:
                val = info.get(field)
                if pd.notna(val) and val != "" and val != "None":
                    slots_df.at[idx, field] = val
            m_width = float(info.get('width', 0))
            slots_df.at[idx, 'width'] = m_width if m_width > 0 else float(row.get('width', 1.0))
        if pd.isna(slots_df.at[idx, 'width']) or slots_df.at[idx, 'width'] <= 0:
            slots_df.at[idx, 'width'] = 1.0

    machine_key = str(machine_config['machine_key'])
    machine_predictions = predictions_df[predictions_df['machine_key'] == machine_key].copy()
    
    # Pre-compute for speed
    machine_predictions['ean_int'] = pd.to_numeric(machine_predictions['ean'], errors='coerce').fillna(0).astype(int)
    machine_predictions['ean_str'] = machine_predictions['ean_int'].astype(str)
    
    # Get width by merging with products_df instead of using width_map (EANs may not match)
    products_ean_int = pd.to_numeric(products_df['ean'], errors='coerce').fillna(0).astype(int)
    products_width_lookup = pd.Series(products_df['width'].fillna(1.0).values, index=products_ean_int.values).to_dict()
    machine_predictions['cand_width'] = machine_predictions['ean_int'].map(products_width_lookup).fillna(1.0)

    lookup_maps = {
        'names': prod_meta.set_index('ean_int')['product_name'].to_dict(),
        'categories': prod_meta.set_index('ean_int')['category'].to_dict()
    }
    
    return slots_df, machine_predictions, meta_map, width_map, lookup_maps

def run_autofill_workflow(
    machine_config: dict,
    predictions_df: pd.DataFrame,
    products_df: pd.DataFrame,
    static_weights: Dict[str, float],
    dynamic_weights: Dict[str, float],
    artifacts: Dict[str, Any]
) -> dict:
    """Phase 1: Fill all empty slots in the machine config with best candidates."""
    slots_df, machine_predictions, meta_map, width_map, lookup_maps = _setup_workflow_data(machine_config, predictions_df, products_df)
    if machine_predictions.empty:
        return machine_config

    has_dynamic = any(w > 0 for w in dynamic_weights.values())
    slots_df['ean_str'] = slots_df['ean_int'].astype(str).replace('0', None)

    def get_current_cfg():
        cfg = machine_config.copy()
        cfg['slots'] = slots_df.drop(columns=['ean_int', 'ean_str', 'cand_width'], errors='ignore').to_dict('records')
        return cfg

    # Initial ranking
    machine_predictions = recalculate_ranking_with_scores(
        machine_predictions, products_df, get_current_cfg(), static_weights, dynamic_weights, artifacts, lookup_maps
    )

    empty_mask = slots_df['ean_str'].isna()
    if empty_mask.any():
        for idx, slot in slots_df[empty_mask].iterrows():
            if has_dynamic:
                machine_predictions = recalculate_ranking_with_scores(
                    machine_predictions, products_df, get_current_cfg(), static_weights, dynamic_weights, artifacts, lookup_maps
                )

            slot_width = float(slot['width'])
            existing_eans = slots_df['ean_str'].dropna().tolist()
            candidates = machine_predictions[
                (~machine_predictions['ean_str'].isin(existing_eans)) & 
                (machine_predictions['cand_width'] == slot_width)
            ]

            if not candidates.empty:
                selected = candidates.iloc[0]
                ean_val = int(selected['ean_int'])
                info = meta_map.get(ean_val, {})
                
                slots_df.at[idx, 'ean'] = ean_val
                slots_df.at[idx, 'ean_str'] = str(ean_val)
                slots_df.at[idx, 'product_name'] = info.get('product_name', 'Unknown')
                slots_df.at[idx, 'category'] = info.get('category', 'Unknown')
                slots_df.at[idx, 'subcategory'] = info.get('subcategory', '')
                slots_df.at[idx, 'provider'] = info.get('provider', '')
                slots_df.at[idx, 'price'] = float(selected.get('price_mean') if pd.notna(selected.get('price_mean')) else (info.get('purchase_price_kr') or 0))

    updated_config = machine_config.copy()
    updated_config['slots'] = slots_df.drop(columns=['ean_str', 'ean_int'], errors='ignore').to_dict('records')
    return updated_config

def run_swap_workflow(
    machine_config: dict,
    predictions_df: pd.DataFrame,
    products_df: pd.DataFrame,
    static_weights: Dict[str, float],
    dynamic_weights: Dict[str, float],
    artifacts: Dict[str, Any]
) -> dict:
    """Phase 2: Optimize a machine by swapping existing products for better alternatives."""
    slots_df, machine_predictions, meta_map, width_map, lookup_maps = _setup_workflow_data(machine_config, predictions_df, products_df)
    if machine_predictions.empty:
        return machine_config

    slots_df['ean_str'] = slots_df['ean_int'].astype(str)

    # Refresh rankings for full machine state
    machine_predictions = recalculate_ranking_with_scores(
        machine_predictions, products_df, machine_config, static_weights, dynamic_weights, artifacts, lookup_maps
    )

    current_scores = slots_df.merge(
        machine_predictions[['ean_str', 'final_score']], on='ean_str', how='left'
    ).sort_values('final_score')

    for _, current_row in current_scores.dropna(subset=['ean_str']).iterrows():
        pos, cur_score, width = current_row['position'], current_row['final_score'], float(current_row['width'])
        
        better = machine_predictions[
            (machine_predictions['final_score'] > cur_score) & 
            (~machine_predictions['ean_str'].isin(slots_df['ean_str'].dropna())) &
            (machine_predictions['cand_width'] == width)
        ]
        
        if not better.empty:
            selected = better.iloc[0]
            ean_val = int(selected['ean_int'])
            info = meta_map.get(ean_val, {})
            
            mask = slots_df['position'] == pos
            slots_df.loc[mask, 'ean'] = ean_val
            slots_df.loc[mask, 'ean_str'] = str(ean_val)
            slots_df.loc[mask, 'product_name'] = info.get('product_name', 'Unknown')
            slots_df.loc[mask, 'category'] = info.get('category', 'Unknown')
            slots_df.loc[mask, 'subcategory'] = info.get('subcategory', '')
            slots_df.loc[mask, 'provider'] = info.get('provider', '')
            slots_df.loc[mask, 'price'] = float(selected.get('price_mean') if pd.notna(selected.get('price_mean')) else (info.get('purchase_price_kr') or 0))

    updated_config = machine_config.copy()
    updated_config['slots'] = slots_df.drop(columns=['ean_str', 'ean_int'], errors='ignore').to_dict('records')
    return updated_config

def recalculate_ranking_with_scores(
    predictions_df: pd.DataFrame,
    products_df: pd.DataFrame,
    machine_config: dict,
    static_weights: Dict[str, float],
    dynamic_weights: Dict[str, float],
    artifacts: Dict[str, Any],
    lookup_maps: Optional[dict] = None
) -> pd.DataFrame:
    """Recalculate product ranking using dynamic score functions."""
    result = predictions_df.copy()
    result = _calculate_dynamic_scores(result, machine_config, products_df, dynamic_weights, artifacts, result, lookup_maps)
    result['final_score'] = _calculate_final_score(result, static_weights, dynamic_weights)
    return result.sort_values('final_score', ascending=False).reset_index(drop=True)

def _calculate_dynamic_scores(
    predictions_df: pd.DataFrame,
    machine_config: dict,
    products_df: pd.DataFrame,
    weights: Dict[str, float],
    artifacts: Dict[str, Any],
    static_predictions_df: pd.DataFrame,
    lookup_maps: Optional[dict] = None
) -> pd.DataFrame:
    """Calculate dynamic product scores using predict_dynamic_autofill."""
    from planogram.predict_dynamic_autofill import predict_dynamic_autofill
    
    active_needs = [k for k in ['uniqueness', 'cousin', 'inventory'] if weights.get(k, 0) > 0]
    if not active_needs:
        return predictions_df
    
    product_df = predictions_df[['ean']].drop_duplicates().copy()
    if lookup_maps:
        product_df['ean_int'] = pd.to_numeric(product_df['ean'], errors='coerce').fillna(0).astype(int)
        product_df['product_name'] = product_df['ean_int'].map(lookup_maps['names'])
        product_df['category'] = product_df['ean_int'].map(lookup_maps['categories'])
    else:
        name_map = products_df.dropna(subset=['ean']).groupby('ean')['product_name'].first().to_dict()
        cat_map = products_df.dropna(subset=['ean']).groupby('ean')['category'].first().to_dict()
        product_df['product_name'] = product_df['ean'].map(name_map)
        product_df['category'] = product_df['ean'].map(cat_map)
    
    overrides = {f'{k}_model' if k != 'inventory' else 'inventory_score': True for k in active_needs}
    dynamic_df = predict_dynamic_autofill(
        machine_df=pd.DataFrame([machine_config]), product_df=product_df,
        uniqueness_model=artifacts.get('uniqueness_model'), cousin_model=artifacts.get('cousin_model'),
        static_predictions_df=static_predictions_df, products_df=products_df, prediction_overrides=overrides
    )

    if not dynamic_df.empty:
        cols = [f'{k}_{suff}' for k in ['uniqueness', 'cousin', 'inventory'] for suff in ['score', 'improvement', 'baseline_score']]
        cols_to_add = [c for c in cols if c in dynamic_df.columns]
        if cols_to_add:
            predictions_df = predictions_df.drop(columns=[c for c in cols_to_add if c in predictions_df.columns])
            predictions_df = predictions_df.merge(dynamic_df[['ean'] + cols_to_add], on='ean', how='left')
            
    return predictions_df

def _calculate_final_score(
    predictions_df: pd.DataFrame,
    static_weights: Dict[str, float],
    dynamic_weights: Dict[str, float]
) -> pd.Series:
    """Calculate final ranking score by combining revenue with static and dynamic scores."""
    final_score = predictions_df['predicted_weekly_revenue'].copy()
    STATIC_CAP, DYNAMIC_CAP = 0.60, 0.20

    # Static Penalties
    score_cols = {'healthiness': 'healthiness_score', 'location': 'location_fit_score', 'confidence': 'confidence_score'}
    for name, weight in static_weights.items():
        col = score_cols.get(name, f'{name}_score')
        if weight > 0 and col in predictions_df.columns:
            scores = predictions_df[col].fillna(0.0 if name == 'confidence' else 0.5)
            final_score *= (1 - (1 - scores**2) * weight * STATIC_CAP)

    # Dynamic Adjustments
    for name, weight in dynamic_weights.items():
        if weight > 0:
            imp_col, score_col = f'{name}_improvement', f'{name}_score'
            if imp_col in predictions_df.columns:
                imps = predictions_df[imp_col].fillna(0.0)
                max_abs = imps.abs().max()
                if max_abs > 0:
                    final_score *= (1 + (imps / max_abs) * weight * DYNAMIC_CAP)
            elif score_col in predictions_df.columns:
                scores = predictions_df[score_col].fillna(0.5)
                final_score *= (1 - (1 - scores**2) * weight * STATIC_CAP)
    
    return final_score.clip(0)
