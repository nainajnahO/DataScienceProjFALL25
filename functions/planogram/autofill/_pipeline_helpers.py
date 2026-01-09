"""Helper functions for the autofill pipeline notebook.

These functions are used by run_autofill_pipeline.ipynb but are not part of
the core run_autofill_workflow() function.
"""

from __future__ import annotations

import pandas as pd


def _get_price_mean(
    sales_df: pd.DataFrame,
    machines_df: pd.DataFrame,
    predictions_df: pd.DataFrame
) -> pd.DataFrame:
    """Extract price_mean from sales history using multi-tier fallback."""
    sales_df = sales_df.copy()
    predictions_df = predictions_df.copy()
    
    for df in [sales_df, predictions_df]:
        if 'machine_key' in df.columns:
            df['machine_key'] = df['machine_key'].astype(str)
    
    price_results = predictions_df[['machine_key', 'ean']].drop_duplicates().copy()
    price_col = 'price_mean' if 'price_mean' in sales_df.columns else 'price'
    has_week_start = 'week_start' in sales_df.columns
    
    # Tier 1: Latest/average per (machine_key, ean)
    if price_col in sales_df.columns:
        sales_price = sales_df[sales_df[price_col].notna()].copy()
        
        if has_week_start:
            if not pd.api.types.is_datetime64_any_dtype(sales_price['week_start']):
                sales_price['week_start'] = pd.to_datetime(sales_price['week_start'], errors='coerce')
            tier1 = (
                sales_price.sort_values('week_start')
                .groupby(['machine_key', 'ean'])
                .tail(1)[['machine_key', 'ean', price_col]]
                .drop_duplicates(['machine_key', 'ean'])
            )
        else:
            tier1 = (
                sales_price.groupby(['machine_key', 'ean'])[price_col]
                .mean()
                .reset_index()
            )
        
        tier1 = tier1.rename(columns={price_col: 'price_mean'})
        price_results = price_results.merge(tier1, on=['machine_key', 'ean'], how='left')
    
    # Tier 2: Average by (machine_sub_group, ean)
    missing_mask = price_results['price_mean'].isna()
    if missing_mask.any() and 'machine_sub_group' in machines_df.columns and 'machine_sub_group' in sales_df.columns:
        machines_subgroup = machines_df[['machine_key', 'machine_sub_group']].copy()
        machines_subgroup['machine_key'] = machines_subgroup['machine_key'].astype(str)
        price_results = price_results.merge(machines_subgroup, on='machine_key', how='left')
        
        if price_col in sales_df.columns:
            tier2 = (
                sales_df[sales_df[price_col].notna()]
                .groupby(['machine_sub_group', 'ean'])[price_col]
                .mean()
                .reset_index()
                .rename(columns={price_col: 'price_mean'})
            )
            price_results = price_results.merge(tier2, on=['machine_sub_group', 'ean'], how='left', suffixes=('', '_tier2'))
            price_results.loc[missing_mask, 'price_mean'] = price_results.loc[missing_mask, 'price_mean_tier2']
            price_results = price_results.drop(columns=['price_mean_tier2'], errors='ignore')
    
    # Tier 3: Average by ean
    missing_mask = price_results['price_mean'].isna()
    if missing_mask.any() and price_col in sales_df.columns:
        tier3 = (
            sales_df[sales_df[price_col].notna()]
            .groupby('ean')[price_col]
            .mean()
            .reset_index()
            .rename(columns={price_col: 'price_mean'})
        )
        price_results = price_results.merge(tier3, on='ean', how='left', suffixes=('', '_tier3'))
        price_results.loc[missing_mask, 'price_mean'] = price_results.loc[missing_mask, 'price_mean_tier3']
        price_results = price_results.drop(columns=['price_mean_tier3'], errors='ignore')
    
    price_results = price_results.drop(columns=['machine_sub_group'], errors='ignore')
    return price_results[['machine_key', 'ean', 'price_mean']]


def _calculate_revenue(
    predictions_df: pd.DataFrame,
    sales_df: pd.DataFrame,
    machines_df: pd.DataFrame
) -> pd.DataFrame:
    """Calculate predicted revenue from predictions."""
    results = predictions_df.copy()
    price_df = _get_price_mean(sales_df, machines_df, predictions_df)
    
    results = results.merge(price_df, on=['machine_key', 'ean'], how='left')
    
    if 'pred_week_1' in results.columns:
        results['predicted_weekly_revenue'] = (
            results['pred_week_1'].fillna(0) * results['price_mean'].fillna(0)
        )
    else:
        results['predicted_weekly_revenue'] = 0.0
    
    return results


def enrich_slots_with_eans(slots: list, products_df: pd.DataFrame) -> list:
    """Enrich slots with EANs using product_name lookup."""
    product_lookup = {}
    valid_mask = products_df['product_name'].notna() & products_df['ean'].notna()
    valid_df = products_df[valid_mask]
    
    for idx in valid_df.index:
        row = valid_df.loc[idx]
        name = str(row['product_name']).strip()
        ean_val = int(row['ean'])
        product_lookup[name] = ean_val
        product_lookup[name.lower()] = ean_val
            
    enriched = []
    for slot in slots:
        if not isinstance(slot, dict):
            enriched.append(slot)
            continue
            
        new_slot = slot.copy()
        if 'ean' not in new_slot or pd.isna(new_slot['ean']):
            pname = str(new_slot.get('product_name', '')).strip()
            if pname:
                if pname in product_lookup:
                    new_slot['ean'] = product_lookup[pname]
                elif pname.lower() in product_lookup:
                    new_slot['ean'] = product_lookup[pname.lower()]
                else:
                    for k, v in product_lookup.items():
                        if pname.lower() in k.lower() or k.lower() in pname.lower():
                            new_slot['ean'] = v
                            break
        enriched.append(new_slot)
    return enriched

