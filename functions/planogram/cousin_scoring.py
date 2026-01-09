"""
Cousin Products Scoring Module
==============================

This module provides functions for training cousin product association models
and predicting cousin scores for products in machines.

The cousin scoring system identifies products that are frequently purchased together
within a time window, using Association Rule Mining (ARM) to compute confidence
scores between product pairs. This module only processes subgroups (not full dataset).

Product-level scores:
- mean_cousin_score: Mean confidence based on subgroup matrix

Machine-level scores:
- avg_mean_cousin_score: Average mean confidence (mean of means in machine)
- cousin_fraction: Fraction of products with non-trivial confidence scores (> 0.005)
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, Tuple, Set, TypeAlias
from collections import Counter
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Type alias for the trained model artifact
# Contains confidence matrices for each machine subgroup
CousinModel: TypeAlias = Dict[str, pd.DataFrame]


def _normalize_product_name(name: str) -> str:
    """
    Normalize product name for case-insensitive matching.
    
    Args:
        name: Product name string
        
    Returns:
        Normalized product name (lowercase, stripped)
    """
    if pd.isna(name):
        return ""
    return str(name).lower().strip()


def _association_rule_mining(sales_data_df: pd.DataFrame, time_interval_minutes: int = 5) -> pd.DataFrame:
    """
    Association Rule Mining - identifies products sold together within a time window.
    
    Args: 
        sales_data_df: DataFrame with columns customer_id, local_timestamp, product_name
        time_interval_minutes: Time interval in minutes for co-occurrence window
        
    Returns:
        DataFrame with columns customer_id, local_timestamp, Product_Sold_Before, 
        Product_Sold_After, products_sold_together
    """
    # Use numpy timedelta for consistent comparisons
    delta_ns = np.timedelta64(time_interval_minutes, 'm')

    # Work with minimal columns to reduce memory footprint
    df = sales_data_df[['customer_id', 'local_timestamp', 'product_name']].copy()
    naive_ts = pd.to_datetime(df['local_timestamp'], utc=True).dt.tz_convert('UTC').dt.tz_localize(None).astype('datetime64[ns]')
    df['local_timestamp'] = naive_ts
    df.sort_values(['customer_id', 'local_timestamp'], inplace=True)

    # Define a per-customer function with optimized sliding window approach
    def process_customer(group):
        # Force numpy datetime64[ns] to avoid pandas Timestamp in iteration
        t = group['local_timestamp'].values.astype('datetime64[ns]')
        p = group['product_name'].to_numpy()
        n = len(t)
        
        if n == 0:
            return pd.DataFrame(columns=['customer_id', 'local_timestamp', 'Product_Sold_Before', 'Product_Sold_After', 'products_sold_together'])

        before_list, after_list = [], []

        # Optimized: use sliding window instead of full array comparison
        for i in range(n):
            ti = t[i]
            before, after = [], []
            
            # Look backward (before)
            j = i - 1
            while j >= 0:
                dt = ti - t[j]
                if dt > delta_ns:
                    break  # Too far back, can stop since sorted
                before.append(p[j])
                j -= 1
            
            # Look forward (after)
            j = i + 1
            while j < n:
                dt = t[j] - ti
                if dt > delta_ns:
                    break  # Too far forward, can stop since sorted
                after.append(p[j])
                j += 1

            before_list.append(before)
            after_list.append(after)

        return pd.DataFrame({
            'customer_id': group['customer_id'].iloc[0],
            'local_timestamp': t,
            'Product_Sold_Before': before_list,
            'Product_Sold_After': after_list,
            'products_sold_together': [b + a for b, a in zip(before_list, after_list)]
        })

    # Process customers sequentially with batching to reduce memory usage
    customer_groups = df.groupby('customer_id')
    n_customers = df['customer_id'].nunique()
    batch_size = 1000  # Process in batches to avoid memory issues
    all_results = []
    
    batch = []
    for idx, (_, g) in enumerate(tqdm(customer_groups, desc="Processing customers (ARM)", total=n_customers)):
        batch.append(process_customer(g))
        
        # Concatenate and clear batch periodically to save memory
        if len(batch) >= batch_size:
            batch_df = pd.concat(batch, ignore_index=True)
            all_results.append(batch_df)
            batch = []
    
    # Add remaining batch
    if batch:
        batch_df = pd.concat(batch, ignore_index=True)
        all_results.append(batch_df)

    # Check if results is empty (no customer groups to process)
    if not all_results:
        # Return empty DataFrame with expected columns
        arm_df = pd.DataFrame(columns=['customer_id', 'local_timestamp', 'Product_Sold_Before', 'Product_Sold_After', 'products_sold_together'])
    else:
        arm_df = pd.concat(all_results, ignore_index=True)

    # Merge results back to original dataframe (only merge necessary columns)
    result_cols = ['customer_id', 'local_timestamp', 'Product_Sold_Before', 'Product_Sold_After', 'products_sold_together']
    sales_df_for_merge = sales_data_df.copy()
    
    # Convert timestamp to datetime64[ns]
    sales_df_for_merge['local_timestamp'] = pd.to_datetime(sales_df_for_merge['local_timestamp'], utc=True).dt.tz_convert('UTC').dt.tz_localize(None).astype('datetime64[ns]')
    
    # Add dummy row to ensure correct dtypes if arm_df is empty
    if arm_df.empty:
        dummy_timestamp = pd.Timestamp('1970-01-01').to_datetime64()
        arm_df = pd.DataFrame({
            'customer_id': [None],
            'local_timestamp': [dummy_timestamp],
            'Product_Sold_Before': [None],
            'Product_Sold_After': [None],
            'products_sold_together': [None]
        })
    else:
        arm_df['local_timestamp'] = pd.to_datetime(arm_df['local_timestamp']).astype('datetime64[ns]')
    
    return pd.merge(sales_df_for_merge, arm_df[result_cols], on=['customer_id', 'local_timestamp'], how='left')


def _initialize_cousin_products(df: pd.DataFrame) -> pd.DataFrame:
    """
    Initialize cousin products by exploding co-occurrence lists and counting pairs.
    
    Args:
        df: DataFrame with products_sold_together column (from association_rule_mining)
        
    Returns:
        DataFrame with columns product_1, product_2, count
    """
    mask = df['products_sold_together'].notna()
    filtered_df = df.loc[mask, ['customer_id', 'product_name', 'products_sold_together']]
    
    # Explode the lists
    exploded_df = filtered_df.explode('products_sold_together')
    
    # Rename and filter self-pairs
    cousin_df = (
        exploded_df
        .rename(columns={'product_name': 'product_1', 'products_sold_together': 'product_2'})
        .loc[lambda d: d['product_1'] != d['product_2']]
    )
    
    # Groupby and count
    cousin_df = (
        cousin_df
        .groupby(['product_1', 'product_2'], as_index=False)
        .size()
        .rename(columns={'size': 'count'})
    )
    
    return cousin_df


def _build_baskets(df_arm: pd.DataFrame, time_interval_minutes: int = 5) -> Tuple[pd.DataFrame, int]:
    """
    Build baskets from ARM results, grouping products within time window.
    
    Args:
        df_arm: DataFrame with columns customer_id, local_timestamp, product_name,
                Product_Sold_Before, Product_Sold_After, products_sold_together
        time_interval_minutes: Time window for grouping products into same basket
    
    Returns:
        Tuple of (baskets DataFrame with columns customer_id, local_timestamp, products, number of baskets)
    """
    # Extract transaction data from df_arm
    df_work = df_arm[['customer_id', 'local_timestamp', 'product_name']].copy()
    df_work['product_name'] = df_work['product_name'].astype(str).str.strip()
    df_work = df_work.drop_duplicates()
    
    naive_ts = pd.to_datetime(df_work['local_timestamp'], utc=True).dt.tz_convert('UTC').dt.tz_localize(None).astype('datetime64[ns]')
    df_work['local_timestamp'] = naive_ts
    df_work.sort_values(['customer_id', 'local_timestamp'], inplace=True)
    
    # Group products within time window into same basket
    delta_ns = np.timedelta64(time_interval_minutes, 'm')
    
    # Use list of tuples instead of dicts for better memory efficiency
    basket_data = []
    
    # Process each customer's transactions
    for customer_id, group in tqdm(df_work.groupby('customer_id'), desc="Building baskets"):
        t = group['local_timestamp'].values.astype('datetime64[ns]')
        p = group['product_name'].to_numpy()
        
        if len(t) == 0:
            continue
        
        # Create basket groups: transactions within time_interval_minutes are in same basket
        basket_start_idx = 0
        basket_end_idx = 0
        basket_products = {p[0]}
        
        for i in range(1, len(t)):
            # Check if current transaction is within time window of the most recent transaction in basket
            time_diff = t[i] - t[basket_end_idx]
            
            if time_diff > delta_ns:
                # Start new basket: save current basket
                basket_data.append((customer_id, t[basket_start_idx], basket_products))
                # Start new basket
                basket_start_idx = i
                basket_end_idx = i
                basket_products = {p[i]}
            else:
                # Add to current basket and update end index
                basket_products.add(p[i])
                basket_end_idx = i
        
        # Add final basket for this customer
        basket_data.append((customer_id, t[basket_start_idx], basket_products))
    
    # Create DataFrame from list of tuples
    baskets = pd.DataFrame(basket_data, columns=['customer_id', 'local_timestamp', 'products'])
    
    return baskets, int(baskets.shape[0])


def _product_basket_counts(baskets: pd.DataFrame) -> pd.Series:
    """
    Count in how many baskets each product appears (deduped per basket).
    
    Args:
        baskets: DataFrame with 'products' column containing sets of products
        
    Returns:
        Series with product names as index and basket counts as values
    """
    counts = Counter(p for prods in baskets['products'] for p in prods)
    return pd.Series(counts, name='basket_count')


def _score_cousin_pairs(cousin_pairs: pd.DataFrame, product_counts: pd.Series, n_baskets: int) -> pd.DataFrame:
    """
    Compute support and confidence for directed pairs i->j.
    
    Args:
        cousin_pairs: DataFrame with columns product_1, product_2, count
        product_counts: Series with product basket counts
        n_baskets: Total number of baskets
        
    Returns:
        DataFrame with additional columns: support, conf
    """
    scored = cousin_pairs.copy()
    
    # Map basket-level counts for antecedent (i)
    count_i = scored['product_1'].map(product_counts).fillna(0).astype(int)

    # Probabilities
    n_baskets_float = float(n_baskets)
    scored['support'] = scored['count'] / n_baskets_float  # P(i,j)

    # confidence P(j|i) = P(i,j) / P(i); guard divide-by-zero
    scored['conf'] = np.where(count_i > 0, scored['count'] / count_i, np.nan)

    return scored


def _create_matrices(product_counts: pd.Series, scored: pd.DataFrame) -> pd.DataFrame:
    """
    Create full matrix of all product pairs with confidence scores.
    
    Args:
        product_counts: Series with product basket counts
        scored: DataFrame with scored pairs (product_1, product_2, count, support, conf)
        
    Returns:
        DataFrame with all product pairs and their confidence scores
    """
    # Create full matrix of all product pairs
    all_products = product_counts.index.tolist()

    full_pairs = pd.DataFrame(
        [(p1, p2) for p1 in tqdm(all_products, desc="Creating product pairs") for p2 in all_products],
        columns=['product_1', 'product_2']
    )

    # Merge with scored data
    scored_full = full_pairs.merge(scored, on=['product_1', 'product_2'], how='left')
    
    # Fill missing values
    scored_full['count'] = scored_full['count'].fillna(0).astype(int)
    scored_full['support'] = scored_full['support'].fillna(0.0)
    scored_full['conf'] = scored_full['conf'].fillna(0.0)

    return scored_full


def _create_confidence_matrix(scored_full: pd.DataFrame) -> pd.DataFrame:
    """
    Create confidence matrix from scored pairs.
    
    Args:
        scored_full: DataFrame with all product pairs and confidence scores
        
    Returns:
        Pivot table (DataFrame) with products as index/columns and confidence as values
    """
    matrix_conf = scored_full.pivot_table(
        index='product_1', 
        columns='product_2', 
        values='conf', 
        aggfunc='max'
    ).fillna(0)
    
    return matrix_conf


def _extract_relevant_submatrix(
    conf_matrix: pd.DataFrame,
    machine_slots_df: pd.DataFrame
) -> Tuple[Set[str], pd.DataFrame]:
    """
    Extract submatrix from confidence matrix for products in machine.
    
    Args:
        conf_matrix: Full confidence matrix (products x products)
        machine_slots_df: DataFrame with 'product_name' column for products in machine
        
    Returns:
        Tuple of (products_in_machine_set, conf_submatrix)
    """
    # Get unique products in machine
    products_in_machine = set(machine_slots_df['product_name'].dropna().unique())
    
    # Normalize product names for matching
    machine_products_normalized = {_normalize_product_name(p): p for p in products_in_machine}
    matrix_products_normalized = {_normalize_product_name(p): p for p in conf_matrix.index}
    
    # Find matching products (case-insensitive)
    matching_products = []
    for norm_name, orig_name in machine_products_normalized.items():
        if norm_name in matrix_products_normalized:
            matching_products.append(matrix_products_normalized[norm_name])
    
    if len(matching_products) == 0:
        return set(), pd.DataFrame()
    
    # Extract submatrix for matching products
    conf_submatrix = conf_matrix.loc[matching_products, matching_products]
    
    return set(matching_products), conf_submatrix


def _align_matrices(
    conf_matrix_1: pd.DataFrame,
    conf_matrix_2: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align two confidence matrices to common products.
    
    Args:
        conf_matrix_1: First confidence matrix
        conf_matrix_2: Second confidence matrix
        
    Returns:
        Tuple of (aligned_matrix_1, aligned_matrix_2) with common products only
    """
    # Get common products (case-insensitive matching)
    products_1_normalized = {_normalize_product_name(p): p for p in conf_matrix_1.index}
    products_2_normalized = {_normalize_product_name(p): p for p in conf_matrix_2.index}
    
    # Find common normalized names
    common_normalized = set(products_1_normalized.keys()) & set(products_2_normalized.keys())
    
    if len(common_normalized) == 0:
        return pd.DataFrame(), pd.DataFrame()
    
    # Get original product names for common products
    common_products_1 = [products_1_normalized[n] for n in common_normalized]
    common_products_2 = [products_2_normalized[n] for n in common_normalized]
    
    # Extract aligned submatrices
    aligned_1 = conf_matrix_1.loc[common_products_1, common_products_1]
    aligned_2 = conf_matrix_2.loc[common_products_2, common_products_2]
    
    # Reindex to ensure same order (use products from matrix 1 as reference)
    norm_to_orig_1 = {_normalize_product_name(p): p for p in common_products_1}
    
    # Reindex matrix 2 to match matrix 1's product order
    aligned_2_reindexed = aligned_2.copy()
    aligned_2_reindexed.index = [norm_to_orig_1[_normalize_product_name(p)] for p in aligned_2.index]
    aligned_2_reindexed.columns = [norm_to_orig_1[_normalize_product_name(p)] for p in aligned_2.columns]
    
    return aligned_1, aligned_2_reindexed


def _compute_product_scores_from_conf(
    conf_matrix: pd.DataFrame,
    context_name: str
) -> pd.DataFrame:
    """
    Compute product-level scores from confidence matrix.
    
    For each product, computes mean confidence across all pairings with other products.
    
    Args:
        conf_matrix: Confidence matrix (products x products)
        context_name: Name of context (e.g., 'full' or 'subgroup') for column naming
        
    Returns:
        DataFrame with columns: product, mean_conf_{context_name}
    """
    scores = []
    for product in conf_matrix.index:
        # Exclude diagonal (self-associations)
        conf_values = conf_matrix.loc[product].drop(product)
        
        scores.append({
            'product': product,
            f'mean_conf_{context_name}': conf_values.mean(),
        })
    return pd.DataFrame(scores)


def train_cousin_model(
    sales_df: pd.DataFrame,
    time_interval_minutes: int = 5,
) -> CousinModel:
    """
    Trains the cousin product association model by computing confidence matrices for each subgroup.
    
    This function performs heavy computation:
    - Association Rule Mining to identify co-occurring products
    - Building product baskets
    - Scoring product pairs
    - Creating confidence matrices for each machine_sub_group
    
    Args:
        sales_df: Sales DataFrame with columns: customer_id, local_timestamp, product_name,
                  machine_sub_group (required).
        time_interval_minutes: Time window in minutes for product co-occurrence (default: 5)
        
    Returns:
        Dictionary mapping subgroup names to confidence matrices:
        - '{subgroup_name}': Subgroup-specific confidence matrices
    """
    sales_df = sales_df.copy()
    
    # Check if machine_sub_group column exists
    if 'machine_sub_group' not in sales_df.columns:
        raise ValueError("'machine_sub_group' column is required in sales_df")
    
    # Get available subgroups
    available_subgroups = sales_df['machine_sub_group'].dropna().unique()
    
    if len(available_subgroups) == 0:
        logger.warning("No subgroups found in sales_df. Returning empty model.")
        return {}
    
    logger.info(f"Starting cousin model training for {len(available_subgroups)} subgroups")
    
    # Initialize result dictionary
    model = {}
    
    # Process each subgroup
    for subgroup in available_subgroups:
        logger.info(f"Processing subgroup: {subgroup}")
        
        # Filter by subgroup
        subgroup_df = sales_df[sales_df['machine_sub_group'] == subgroup].copy()
        
        if len(subgroup_df) == 0:
            logger.warning(f"Subgroup '{subgroup}' has no data. Skipping.")
            continue

        # Compute associations for this subgroup (data cleaning is done in data_loader)
        df_arm_subgroup = _association_rule_mining(subgroup_df, time_interval_minutes=time_interval_minutes)

        # Compute cousin products for this subgroup
        cousin_df_subgroup = _initialize_cousin_products(df_arm_subgroup)

        # Build baskets for this subgroup
        baskets_subgroup, n_baskets_subgroup = _build_baskets(df_arm_subgroup, time_interval_minutes=time_interval_minutes)

        # Count product baskets for this subgroup
        product_counts_subgroup = _product_basket_counts(baskets_subgroup)

        # Score cousin pairs for this subgroup
        scored_subgroup = _score_cousin_pairs(cousin_df_subgroup, product_counts_subgroup, n_baskets_subgroup)

        # Create full matrix for this subgroup
        scored_full_subgroup = _create_matrices(product_counts_subgroup, scored_subgroup)

        # Create confidence matrix for this subgroup
        conf_matrix_subgroup = _create_confidence_matrix(scored_full_subgroup)
        
        # Store in model dictionary
        model[subgroup] = conf_matrix_subgroup
        logger.info(f"Created confidence matrix for subgroup '{subgroup}': {conf_matrix_subgroup.shape}")
    
    logger.info(f"Training complete. Created {len(model)} subgroup matrices.")
    return model


def predict_machine_cousin_score(
    machines_df: pd.DataFrame,
    trained_model: CousinModel,
) -> Optional[float]:
    """
    Lightweight function that calculates only machine-level avg_mean_cousin_score.
    
    Optimized for performance when only aggregate score is needed (e.g., autofill).
    Returns single float instead of full DataFrame with updated slots.
    
    Args:
        machines_df: DataFrame with single machine (1 row) or multiple machines.
                    Must have 'slots' column (list of dicts) and 'machine_sub_group' column.
        trained_model: Dictionary mapping subgroup names to confidence matrices.
    
    Returns:
        Float: avg_mean_cousin_score for the machine, or None if unavailable.
               For multiple machines, returns score for first machine.
    """
    if machines_df.empty:
        return None
    
    if 'slots' not in machines_df.columns or 'machine_sub_group' not in machines_df.columns:
        return None
    
    if len(trained_model) == 0:
        return None
    
    # Get first machine (for single machine use case)
    row = machines_df.iloc[0]
    slots = row.get('slots', [])
    machine_sub_group = row.get('machine_sub_group')
    
    if not isinstance(slots, list) or len(slots) == 0:
        return None
    
    if pd.isna(machine_sub_group) or machine_sub_group is None:
        return None
    
    if machine_sub_group not in trained_model:
        return None
    
    conf_matrix_subgroup = trained_model[machine_sub_group]
    
    # Extract product names from slots
    product_names = []
    for slot in slots:
        if isinstance(slot, dict):
            product_name = slot.get('product_name')
            if product_name:
                product_names.append(product_name)
    
    if len(product_names) == 0:
        return None
    
    # Create temporary DataFrame for extracting submatrix
    machine_products_df = pd.DataFrame({'product_name': product_names})
    
    # Extract relevant submatrix for subgroup
    products_in_machine, conf_submatrix = _extract_relevant_submatrix(
        conf_matrix_subgroup, machine_products_df
    )
    
    if len(conf_submatrix) == 0:
        return None
    
    # Compute product scores for subgroup
    product_scores = _compute_product_scores_from_conf(conf_submatrix, 'subgroup')
    
    # Calculate machine-level avg_mean_cousin_score
    if not product_scores.empty and 'mean_conf_subgroup' in product_scores.columns:
        avg_mean_cousin_score = product_scores['mean_conf_subgroup'].mean()
        return float(avg_mean_cousin_score)
    
    return None


def predict_cousin_scores(
    machines_df: pd.DataFrame,
    trained_model: CousinModel,
) -> pd.DataFrame:
    """
    Predicts cousin scores for products in machines using trained subgroup models.
    
    This function performs fast lookups using pre-computed confidence matrices:
    - Extracts relevant submatrices for products in each machine based on subgroup
    - Computes product-level mean confidence scores
    - Updates slot dictionaries with scores
    - Computes machine-level aggregate scores
    
    Args:
        machines_df: DataFrame containing machines with 'slots' column.
                     Each slot is a dict with 'product_name' field.
                     Must also have 'machine_sub_group' column.
        trained_model: Dictionary mapping subgroup names to confidence matrices.
        
    Returns:
        DataFrame: Input DataFrame with:
            - 'slots' column updated: Each slot dict now has 'mean_cousin_score' field
            - 'avg_mean_cousin_score' column added: Average mean confidence for machine
            - 'cousin_fraction' column added: Fraction of products with non-trivial confidence (> 0.005)
    """
    machines_df = machines_df.copy()  # Always copy first!
    
    if 'slots' not in machines_df.columns:
        logger.warning("No 'slots' column found in machines_df.")
        machines_df['avg_mean_cousin_score'] = np.nan
        machines_df['cousin_fraction'] = np.nan
        return machines_df
    
    if 'machine_sub_group' not in machines_df.columns:
        logger.error("machines_df must have 'machine_sub_group' column")
        machines_df['avg_mean_cousin_score'] = np.nan
        machines_df['cousin_fraction'] = np.nan
        return machines_df
    
    if len(trained_model) == 0:
        logger.warning("trained_model is empty. Returning machines_df with NaN scores.")
        machines_df['avg_mean_cousin_score'] = np.nan
        machines_df['cousin_fraction'] = np.nan
        return machines_df
    
    machine_avg_scores = []
    machine_cousin_fractions = []
    total_slots = 0
    slots_matched = 0
    slots_no_product_name = 0
    machines_no_subgroup = 0
    machines_no_matrix = 0
    machines_no_products = 0
    
    for idx, row in machines_df.iterrows():
        slots = row.get('slots', [])
        machine_sub_group = row.get('machine_sub_group')
        
        if not isinstance(slots, list) or len(slots) == 0:
            machine_avg_scores.append(np.nan)
            machine_cousin_fractions.append(0.0)
            continue
        
        # Check if machine has subgroup
        if pd.isna(machine_sub_group) or machine_sub_group is None:
            machines_no_subgroup += 1
            machine_avg_scores.append(np.nan)
            machine_cousin_fractions.append(0.0)
            # Still need to set scores to None for slots
            for slot in slots:
                if isinstance(slot, dict):
                    slot['mean_cousin_score'] = None
            machines_df.at[idx, 'slots'] = slots
            continue
        
        # Get subgroup matrix
        if machine_sub_group not in trained_model:
            machines_no_matrix += 1
            logger.debug(f"Machine {row.get('machine_id', idx)}: No matrix found for subgroup '{machine_sub_group}'")
            machine_avg_scores.append(np.nan)
            machine_cousin_fractions.append(0.0)
            # Still need to set scores to None for slots
            for slot in slots:
                if isinstance(slot, dict):
                    slot['mean_cousin_score'] = None
            machines_df.at[idx, 'slots'] = slots
            continue
        
        conf_matrix_subgroup = trained_model[machine_sub_group]
        
        # Extract product names from slots
        product_names = []
        for slot in slots:
            if isinstance(slot, dict):
                product_name = slot.get('product_name')
                if product_name:
                    product_names.append(product_name)
        
        if len(product_names) == 0:
            machines_no_products += 1
            machine_avg_scores.append(np.nan)
            machine_cousin_fractions.append(0.0)
            # Still need to set scores to None for slots
            for slot in slots:
                if isinstance(slot, dict):
                    slot['mean_cousin_score'] = None
            machines_df.at[idx, 'slots'] = slots
            continue
        
        # Create temporary DataFrame for extracting submatrix
        machine_products_df = pd.DataFrame({'product_name': product_names})
        
        # Extract relevant submatrix for subgroup
        products_in_machine, conf_submatrix = _extract_relevant_submatrix(
            conf_matrix_subgroup, machine_products_df
        )
        
        # Validate that we found matching products
        if len(conf_submatrix) == 0:
            machines_no_products += 1
            machine_avg_scores.append(np.nan)
            machine_cousin_fractions.append(0.0)
            # Still need to set scores to None for slots
            for slot in slots:
                if isinstance(slot, dict):
                    slot['mean_cousin_score'] = None
            machines_df.at[idx, 'slots'] = slots
            continue
        
        # Compute product scores for subgroup
        product_scores = _compute_product_scores_from_conf(conf_submatrix, 'subgroup')
        
        # Create mapping from product name to score (case-insensitive)
        product_score_map = {}
        for _, score_row in product_scores.iterrows():
            product = score_row['product']
            score = score_row['mean_conf_subgroup']
            # Store both normalized and original name
            normalized_name = _normalize_product_name(product)
            product_score_map[normalized_name] = score
            product_score_map[product] = score
        
        # Process each slot and assign scores
        slot_scores = []
        for slot in slots:
            if not isinstance(slot, dict):
                slot_scores.append(None)
                continue
            
            total_slots += 1
            
            product_name = slot.get('product_name')
            if product_name is None:
                slot['mean_cousin_score'] = None
                slot_scores.append(None)
                slots_no_product_name += 1
                continue
            
            # Try to find score (case-insensitive)
            normalized_name = _normalize_product_name(product_name)
            score = product_score_map.get(normalized_name) or product_score_map.get(product_name)
            
            slot['mean_cousin_score'] = score
            slot_scores.append(score)
            
            if score is not None and not pd.isna(score):
                slots_matched += 1
        
        # Calculate machine-level metrics
        valid_scores = [s for s in slot_scores if s is not None and not pd.isna(s)]
        if len(valid_scores) > 0:
            # Average mean confidence (mean of means in machine)
            avg_mean_cousin_score = np.mean(valid_scores)
            
            # Fraction of products with non-trivial confidence scores (> 0.005)
            total_products = len(slot_scores)
            products_with_confidence = sum(1 for s in valid_scores if s > 0.005)
            cousin_fraction = products_with_confidence / total_products if total_products > 0 else 0.0
        else:
            avg_mean_cousin_score = np.nan
            cousin_fraction = 0.0
        
        machine_avg_scores.append(avg_mean_cousin_score)
        machine_cousin_fractions.append(cousin_fraction)
        
        # Update slots in the row
        machines_df.at[idx, 'slots'] = slots
    
    machines_df['avg_mean_cousin_score'] = machine_avg_scores
    machines_df['cousin_fraction'] = machine_cousin_fractions
    
    return machines_df
