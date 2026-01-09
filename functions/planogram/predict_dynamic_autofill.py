import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances

from .config import PREDICTION_DYNAMIC_TASKS
from .product_scoring import get_product_embeddings

logger = logging.getLogger(__name__)


def _is_valid_df(df: Optional[pd.DataFrame]) -> bool:
    """Check if a DataFrame is valid (not None and not empty)."""
    return df is not None and not df.empty


def _extract_machine_products(machine_row: pd.Series, products_df: pd.DataFrame) -> List[Dict]:
    """Extract product records from machine slots for uniqueness scoring."""
    slots = machine_row.get('slots', [])
    if not isinstance(slots, list):
        return []

    # Build product_name (lower) -> category lookup (prefer first occurrence)
    product_category: Dict[str, Any] = {}
    if not products_df.empty and 'product_name' in products_df.columns:
        # Avoid iterrows(); this can be called on big products_df in notebooks.
        cols = ['product_name']
        if 'category' in products_df.columns:
            cols.append('category')
        meta = products_df[cols].dropna(subset=['product_name'])
        if not meta.empty:
            # Keep first occurrence to match previous behavior ("first wins")
            meta = meta.drop_duplicates(subset=['product_name'], keep='first')
            names = meta['product_name'].astype(str).str.strip()
            if 'category' in meta.columns:
                cats = meta['category']
            else:
                cats = pd.Series(['Unknown'] * len(meta), index=meta.index)
            for name, cat in zip(names, cats):
                if name:
                    key = name.lower()
                    if key not in product_category:
                        product_category[key] = cat if pd.notna(cat) else 'Unknown'
    
    products = []
    for slot in slots:
        if not isinstance(slot, dict):
            continue
        product_name = slot.get('product_name')
        if not product_name:
            continue
        
        pname = str(product_name).strip()
        if not pname:
            continue
        pname_lower = pname.lower()
        products.append(
            {
                'product_name': pname,
                'category': slot.get('category', product_category.get(pname_lower, 'Unknown')),
                'position': slot.get('position'),
            }
        )
    
    return products


def _calculate_baseline_uniqueness(
    machine_row: pd.Series,
    products_df: pd.DataFrame,
    uniqueness_model: Tuple[np.ndarray, Dict[str, int]],
) -> Tuple[Optional[float], np.ndarray, np.ndarray, List[Dict]]:
    """
    Calculate baseline uniqueness for current machine state.
    Returns: (machine_mean_uniqueness, embeddings, distances, products_list)
    """
    embeddings, product_to_index = uniqueness_model
    
    # Extract current products
    current_products = _extract_machine_products(machine_row, products_df)
    
    if len(current_products) < 2:
        return None, np.array([]), np.array([]), current_products
    
    # Get embeddings for current products
    current_embeddings, valid_indices, _, _ = get_product_embeddings(
        current_products, embeddings, product_to_index, case_sensitive=False
    )
    
    if len(current_embeddings) < 2:
        return None, current_embeddings, np.array([]), current_products
    
    # Calculate pairwise distances
    distances = cosine_distances(current_embeddings)
    np.fill_diagonal(distances, 0)
    
    # Calculate uniqueness scores
    n = len(current_embeddings)
    uniqueness_scores = distances.sum(axis=1) / (n - 1)
    
    # Machine-level mean uniqueness
    machine_mean = uniqueness_scores.mean()
    
    return machine_mean, current_embeddings, distances, current_products


def _normalize_name(value: Any) -> Optional[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    s = str(value).strip()
    return s.casefold() if s else None


def _safe_numeric_series(s: pd.Series) -> pd.Series:
    """Coerce to numeric, leaving non-parsable values as NaN."""
    return pd.to_numeric(s, errors='coerce')


def _compute_inventory_scores_fast(
    *,
    machine_row: pd.Series,
    products_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    candidate_eans: pd.Series,
) -> Tuple[Optional[float], List[Optional[float]]]:
    """
    Fast-path inventory scoring for one machine and many candidate products.

    Matches the logic in `inventory_score.calculate_inventory_scores` but avoids repeated
    DataFrame construction/merges per candidate by precomputing runout-time stats.

    Returns:
        (baseline_inventory_score, inventory_scores_per_candidate)
    """
    # Guardrails: if we can't score, return no candidate scores.
    if products_df is None or products_df.empty:
        return 0.0, [None for _ in range(len(candidate_eans))]
    if predictions_df is None or predictions_df.empty:
        return 0.0, [None for _ in range(len(candidate_eans))]

    def _ean_num(x: Any) -> Any:
        """
        Normalize EAN to a stable, hashable numeric key.

        IMPORTANT: Avoid converting to float for 13-digit EANs (precision loss).
        Prefer Python int whenever value is integral.
        """
        if x is None:
            return None
        if isinstance(x, (np.integer, int)):
            return int(x)
        if isinstance(x, (np.floating, float)):
            if np.isnan(x):
                return None
            # Preserve integers as ints; otherwise keep float
            return int(x) if float(x).is_integer() else float(x)
        if isinstance(x, str):
            s = x.strip()
            if not s:
                return None
            # Try numeric parse; if it fails, keep original string key
            try:
                v = pd.to_numeric(s, errors='raise')
            except Exception:
                return s
            if isinstance(v, (np.integer, int)):
                return int(v)
            if isinstance(v, (np.floating, float)):
                if np.isnan(v):
                    return None
                return int(v) if float(v).is_integer() else float(v)
            return v

        # Fallback: try pandas conversion, but don't force float
        try:
            v = pd.to_numeric(x, errors='coerce')
        except Exception:
            return x
        if isinstance(v, (np.integer, int)):
            return int(v)
        if isinstance(v, (np.floating, float)):
            if np.isnan(v):
                return None
            return int(v) if float(v).is_integer() else float(v)
        return x

    all_pred_eans: set[Any] = set()
    if 'ean' in predictions_df.columns:
        all_pred_eans = set(predictions_df['ean'].dropna().map(_ean_num).tolist())

    if 'spiral' not in products_df.columns:
        products_df = products_df.copy()
        products_df['spiral'] = 0

    # Extract baseline machine products from slots (ean + position)
    slots = machine_row.get('slots', [])
    machine_key = machine_row.get('machine_key')
    if not isinstance(slots, list) or machine_key is None or (isinstance(machine_key, float) and np.isnan(machine_key)):
        return 0.0, [None for _ in range(len(candidate_eans))]

    records: List[Tuple[Any, Any]] = []
    # Build product_name -> ean lookup (case-insensitive) for slots with missing ean
    name_to_ean: Dict[str, Any] = {}
    if {'product_name', 'ean'}.issubset(products_df.columns):
        meta = products_df[['product_name', 'ean']].dropna(subset=['product_name', 'ean'])
        if not meta.empty:
            meta = meta.drop_duplicates(subset=['product_name'], keep='first')
            for name, ean in zip(meta['product_name'].astype(str), meta['ean']):
                key = _normalize_name(name)
                if key and key not in name_to_ean:
                    name_to_ean[key] = _ean_num(ean)

    for slot in slots:
        if not isinstance(slot, dict):
            continue
        ean_val = slot.get('ean')
        if ean_val is None and slot.get('product_name'):
            key = _normalize_name(slot.get('product_name'))
            if key:
                ean_val = name_to_ean.get(key)
        ean_val = _ean_num(ean_val)
        if ean_val is None or (isinstance(ean_val, float) and np.isnan(ean_val)):
            continue
        records.append((ean_val, slot.get('position')))

    if not records:
        return 0.0, [None for _ in range(len(candidate_eans))]

    # Spiral lookup
    spiral_by_ean = (
        products_df[['ean', 'spiral']]
        .assign(_ean_num=products_df['ean'].map(_ean_num))
        .dropna(subset=['_ean_num'])
        .drop_duplicates(subset=['_ean_num'], keep='first')
        .set_index('_ean_num')['spiral']
    )

    # Predicted weekly sales lookup.
    # In `inventory_score`, if no predicted_sales_score exists it sums pred_* columns.
    pred = predictions_df.copy()
    if 'predicted_sales_score' in pred.columns:
        pred['predicted_weekly_sales'] = _safe_numeric_series(pred['predicted_sales_score'])
    else:
        pred_cols = [c for c in pred.columns if c.startswith('pred_')]
        if not pred_cols:
            return 0.0, [None for _ in range(len(candidate_eans))]
        pred['predicted_weekly_sales'] = _safe_numeric_series(pred[pred_cols].sum(axis=1))

    # The inventory scorer merges on ['machine_key','ean'] and optionally 'position'
    has_position = 'position' in pred.columns
    if 'machine_key' in pred.columns:
        pred = pred[pred['machine_key'] == machine_key]
    if pred.empty or 'ean' not in pred.columns:
        return 0.0, [None for _ in range(len(candidate_eans))]

    if has_position:
        pred_key = pred[['ean', 'position', 'predicted_weekly_sales']].copy()
        pred_key['ean'] = pred_key['ean'].map(_ean_num)
        pred_key = pred_key.dropna(subset=['ean'])
        # Prefer first match (merge would effectively do this depending on duplicates)
        pred_key = pred_key.drop_duplicates(subset=['ean', 'position'], keep='first')
        pred_map = {(r.ean, r.position): r.predicted_weekly_sales for r in pred_key.itertuples(index=False)}
    else:
        pred_key = pred[['ean', 'predicted_weekly_sales']].copy()
        pred_key['ean'] = pred_key['ean'].map(_ean_num)
        pred_key = pred_key.dropna(subset=['ean'])
        pred_key = pred_key.drop_duplicates(subset=['ean'], keep='first')
        pred_map = {r.ean: r.predicted_weekly_sales for r in pred_key.itertuples(index=False)}

    def runout_time(ean_val: Any, position: Any) -> float:
        ean_val = _ean_num(ean_val)
        spiral = spiral_by_ean.get(ean_val)
        weekly = pred_map.get((ean_val, position)) if has_position else pred_map.get(ean_val)
        weekly = float(weekly) if weekly is not None and not (isinstance(weekly, float) and np.isnan(weekly)) else np.nan
        spiral = float(spiral) if spiral is not None and not (isinstance(spiral, float) and np.isnan(spiral)) else np.nan
        if weekly > 0 and spiral > 0:
            return (spiral / weekly) * 7.0
        if weekly <= 0:
            return 999.0
        return np.nan

    baseline_times = np.array([runout_time(e, p) for e, p in records], dtype=float)
    baseline_valid = baseline_times[~np.isnan(baseline_times)]

    def score_from_stats(n: int, s: float, ss: float) -> float:
        # Mirrors _score_synchronization (pandas std with ddof=1)
        if n < 2:
            return 0.0
        mean = s / n
        if mean == 0 or np.isnan(mean):
            return 0.0
        if n == 2:
            # General formula also works, but keep explicit for numerical stability.
            var = (ss - (s * s) / n) / (n - 1)
        else:
            var = (ss - (s * s) / n) / (n - 1)
        if var < 0 and var > -1e-12:
            var = 0.0
        std = float(np.sqrt(var)) if var > 0 else 0.0
        if std == 0:
            return 1.0
        cv = std / mean
        score = 1.0 / (1.0 + cv)
        return float(max(0.0, min(1.0, score)))

    n0 = int(baseline_valid.size)
    s0 = float(baseline_valid.sum()) if n0 else 0.0
    ss0 = float((baseline_valid * baseline_valid).sum()) if n0 else 0.0
    baseline_score = score_from_stats(n0, s0, ss0)

    # Candidate scores: only score candidates that have prediction entries (mimics old behavior)
    scores: List[Optional[float]] = []
    for ean_val in candidate_eans.tolist():
        ean_val = _ean_num(ean_val)
        if ean_val is None or (isinstance(ean_val, float) and np.isnan(ean_val)):
            scores.append(None)
            continue
        # Old implementation also required the EAN to exist in predictions (via groupby lookup)
        if all_pred_eans and ean_val not in all_pred_eans:
            scores.append(None)
            continue

        t = runout_time(ean_val, None)
        if np.isnan(t):
            scores.append(score_from_stats(n0, s0, ss0))
            continue
        scores.append(score_from_stats(n0 + 1, s0 + float(t), ss0 + float(t) * float(t)))

    return baseline_score, scores


def _compute_cousin_scores_fast(
    *,
    machine_row: pd.Series,
    product_df: pd.DataFrame,
    trained_model: Dict[str, Any],
) -> Tuple[Optional[float], List[Optional[float]]]:
    """
    Fast-path cousin scoring for one machine and many candidate products.

    It reproduces `predict_machine_cousin_score` for each candidate, but avoids:
    - building a full normalized-name index map on every iteration
    - creating a new DataFrame of slots on every iteration
    """
    if product_df.empty:
        return None, []
    if 'slots' not in machine_row.index or 'machine_sub_group' not in machine_row.index:
        return None, [None for _ in range(len(product_df))]
    if not trained_model:
        return None, [None for _ in range(len(product_df))]

    slots = machine_row.get('slots', [])
    if not isinstance(slots, list) or len(slots) == 0:
        return None, [None for _ in range(len(product_df))]

    machine_sub_group = machine_row.get('machine_sub_group')
    if machine_sub_group is None or (isinstance(machine_sub_group, float) and np.isnan(machine_sub_group)):
        return None, [None for _ in range(len(product_df))]
    if machine_sub_group not in trained_model:
        return None, [None for _ in range(len(product_df))]

    conf_matrix = trained_model[machine_sub_group]
    if conf_matrix is None or getattr(conf_matrix, 'empty', False):
        return None, [None for _ in range(len(product_df))]

    # Build normalized -> canonical label mapping once (this is the huge speed win).
    norm_to_label: Dict[str, str] = {}
    for label in conf_matrix.index:
        key = _normalize_name(label)
        if key and key not in norm_to_label:
            norm_to_label[key] = label

    # Baseline machine unique matching products (set semantics, like the original).
    baseline_norms: set[str] = set()
    for slot in slots:
        if isinstance(slot, dict):
            k = _normalize_name(slot.get('product_name'))
            if k:
                baseline_norms.add(k)

    baseline_labels = []
    for k in baseline_norms:
        label = norm_to_label.get(k)
        if label is not None:
            baseline_labels.append(label)

    k0 = len(baseline_labels)
    if k0 == 0:
        baseline_score: Optional[float] = None
    elif k0 == 1:
        baseline_score = float('nan')
    else:
        sub = conf_matrix.loc[baseline_labels, baseline_labels].to_numpy(dtype=float, copy=False)
        total = float(sub.sum())
        trace = float(np.trace(sub))
        baseline_score = (total - trace) / (k0 * (k0 - 1))

    # Candidate mapping
    if 'product_name' not in product_df.columns:
        return baseline_score, [baseline_score for _ in range(len(product_df))]

    candidate_norms = product_df['product_name'].apply(_normalize_name)
    candidate_labels = candidate_norms.map(lambda k: norm_to_label.get(k) if k else None)

    # If baseline has 0 matching products, we can still score candidates that exist in matrix:
    # - if candidate exists: 1x1 matrix -> NaN (same as original)
    if k0 == 0:
        scores: List[Optional[float]] = []
        for lbl in candidate_labels.tolist():
            if lbl is None:
                scores.append(None)
            else:
                scores.append(float('nan'))
        return baseline_score, scores

    baseline_set = set(baseline_labels)
    # Build a small submatrix containing baseline + all unique candidate labels we might add.
    add_labels = [lbl for lbl in pd.unique(candidate_labels) if lbl is not None and lbl not in baseline_set]
    all_labels = baseline_labels + add_labels
    sub_all = conf_matrix.loc[all_labels, all_labels].to_numpy(dtype=float, copy=False)
    label_to_idx = {lbl: i for i, lbl in enumerate(all_labels)}

    baseline_sub = sub_all[:k0, :k0]
    baseline_total = float(baseline_sub.sum())
    baseline_trace = float(np.trace(baseline_sub))

    scores = []
    for lbl in candidate_labels.tolist():
        if lbl is None:
            scores.append(baseline_score)
            continue
        if lbl in baseline_set:
            scores.append(baseline_score)
            continue

        p_idx = label_to_idx.get(lbl)
        if p_idx is None:
            scores.append(baseline_score)
            continue

        row_sum = float(sub_all[p_idx, :k0].sum())
        col_sum = float(sub_all[:k0, p_idx].sum())
        diag = float(sub_all[p_idx, p_idx])

        k1 = k0 + 1
        total = baseline_total + row_sum + col_sum + diag
        trace = baseline_trace + diag
        if k1 < 2:
            scores.append(float('nan'))
        else:
            scores.append((total - trace) / (k1 * (k1 - 1)))

    return baseline_score, scores


def predict_dynamic_autofill(
    machine_df: pd.DataFrame,
    product_df: pd.DataFrame,
    *,
    uniqueness_model: Optional[Tuple[np.ndarray, Dict[str, int]]] = None,
    cousin_model: Optional[Dict[str, Any]] = None,
    static_predictions_df: Optional[pd.DataFrame] = None,
    products_df: Optional[pd.DataFrame] = None,
    prediction_overrides: Optional[Dict[str, bool]] = None,
) -> pd.DataFrame:
    """
    Calculate dynamic scores for each product when added to a machine.
    
    For each product in product_df, creates a pseudo-machine by adding that product
    to the current machine's slots, then calculates uniqueness, cousin, and inventory
    scores. Compares against baseline (current machine state) and returns improvement metrics.
    
    Args:
        machine_df: Single-row DataFrame with machine (machine_key, slots, machine_sub_group, etc.)
        product_df: DataFrame of products to test (must have product_name, optionally ean, category)
        uniqueness_model: Trained uniqueness model artifact (embeddings, product_to_index)
        cousin_model: Trained cousin model artifact (dict of subgroup -> confidence matrices)
        static_predictions_df: Static predictions from predict_static (for inventory scoring)
        products_df: Full products DataFrame with spiral column (for inventory scoring). 
                    If None, falls back to product_df.
        prediction_overrides: Dict to override PREDICTION_DYNAMIC_TASKS config
    
    Returns:
        DataFrame with one row per product tested, containing:
        - ean, product_name
        - baseline_uniqueness_score, baseline_cousin_score, baseline_inventory_score
        - uniqueness_score, cousin_score, inventory_score (with product added)
        - uniqueness_improvement, cousin_improvement, inventory_improvement
    """
    if machine_df.empty or len(machine_df) != 1:
        logger.error("machine_df must contain exactly one row")
        return pd.DataFrame()
    
    if not _is_valid_df(product_df):
        logger.error("product_df is required and cannot be empty")
        return pd.DataFrame()
    
    plan = {**PREDICTION_DYNAMIC_TASKS}
    if prediction_overrides:
        plan.update(prediction_overrides)
    
    machine_row = machine_df.iloc[0]

    # Initialize results DataFrame with product info
    results_df = product_df.copy()
    if 'ean' not in results_df.columns:
        results_df['ean'] = None
    if 'product_name' not in results_df.columns:
        results_df['product_name'] = ''

    # Baseline scores (defaults)
    baseline_uniqueness: Optional[float] = None
    baseline_cousin: Optional[float] = None
    baseline_inventory: Optional[float] = None

    # -------------------------
    # Uniqueness (vectorized)
    # -------------------------
    results_df['uniqueness_score'] = None
    if plan.get('uniqueness_model'):
        if uniqueness_model is None:
            logger.warning("Skipping uniqueness_model: missing uniqueness_model artifact")
        else:
            baseline_uniqueness, baseline_embeddings, baseline_distances, _ = _calculate_baseline_uniqueness(
                machine_row, product_df, uniqueness_model
            )
            n0 = int(baseline_embeddings.shape[0]) if baseline_embeddings is not None else 0

            if n0 == 0:
                # Machine has no recognizable products; adding a single product yields uniqueness 1.0
                results_df['uniqueness_score'] = 1.0
            else:
                # Candidate embeddings in one shot
                new_products_list = [
                    {
                        'product_name': row.get('product_name', ''),
                        'category': row.get('category', 'Unknown'),
                        'position': None,
                    }
                    for _, row in product_df.iterrows()
                ]
                new_embeddings, valid_indices, _, _ = get_product_embeddings(
                    new_products_list, uniqueness_model[0], uniqueness_model[1], case_sensitive=False
                )

                uniqueness_out = np.full(len(product_df), None, dtype=object)
                if len(new_embeddings) > 0:
                    dists = cosine_distances(new_embeddings, baseline_embeddings)
                    sum_d = dists.sum(axis=1)
                    if n0 == 1:
                        # Adding to a 1-product machine -> both products uniqueness = distance
                        means = sum_d  # since n0 == 1
                    else:
                        # Derivation:
                        # new_mean = ((n0-1)*sum(baseline_uniq) + 2*sum(dist_to_baseline)) / (n0*(n0+1))
                        baseline_uniq = baseline_distances.sum(axis=1) / (n0 - 1)
                        sum_baseline_uniq = float(baseline_uniq.sum())
                        means = ((n0 - 1) * sum_baseline_uniq + 2.0 * sum_d) / (n0 * (n0 + 1))

                    for score_idx, orig_idx in enumerate(valid_indices):
                        uniqueness_out[orig_idx] = float(means[score_idx])

                results_df['uniqueness_score'] = uniqueness_out

    # -------------------------
    # Cousin (fast-path)
    # -------------------------
    results_df['cousin_score'] = None
    if plan.get('cousin_model'):
        if cousin_model is None:
            logger.warning("Skipping cousin_model: missing cousin_model artifact")
        else:
            baseline_cousin, cousin_scores = _compute_cousin_scores_fast(
                machine_row=machine_row,
                product_df=product_df,
                trained_model=cousin_model,
            )
            results_df['cousin_score'] = cousin_scores

    # -------------------------
    # Inventory (fast-path)
    # -------------------------
    results_df['inventory_score'] = None
    inventory_products_df = products_df if _is_valid_df(products_df) else product_df
    if plan.get('inventory_score'):
        if not _is_valid_df(static_predictions_df) or not _is_valid_df(inventory_products_df):
            logger.warning("Skipping inventory_score: missing required data (static_predictions_df and/or products_df)")
        else:
            baseline_inventory, inventory_scores = _compute_inventory_scores_fast(
                machine_row=machine_row,
                products_df=inventory_products_df,
                predictions_df=static_predictions_df,
                candidate_eans=results_df['ean'],
            )
            results_df['inventory_score'] = inventory_scores

    # Attach baseline columns (scalar repeats)
    results_df['baseline_uniqueness_score'] = baseline_uniqueness
    results_df['baseline_cousin_score'] = baseline_cousin
    results_df['baseline_inventory_score'] = baseline_inventory

    # Improvements: do arithmetic only on numeric values; leave missing as None
    u = _safe_numeric_series(results_df['uniqueness_score'])
    ub = _safe_numeric_series(results_df['baseline_uniqueness_score'])
    c = _safe_numeric_series(results_df['cousin_score'])
    cb = _safe_numeric_series(results_df['baseline_cousin_score'])
    inv = _safe_numeric_series(results_df['inventory_score'])
    invb = _safe_numeric_series(results_df['baseline_inventory_score'])

    results_df['uniqueness_improvement'] = (u - ub).replace([np.inf, -np.inf, np.nan], None)
    results_df['cousin_improvement'] = (c - cb).replace([np.inf, -np.inf, np.nan], None)
    results_df['inventory_improvement'] = (inv - invb).replace([np.inf, -np.inf, np.nan], None)

    # Keep a stable, human-friendly column order (matches prior behavior closely)
    preferred = []
    for col in list(product_df.columns) + ['ean', 'product_name']:
        if col in results_df.columns and col not in preferred:
            preferred.append(col)
    preferred += [
        'baseline_uniqueness_score',
        'baseline_cousin_score',
        'baseline_inventory_score',
        'uniqueness_score',
        'cousin_score',
        'inventory_score',
        'uniqueness_improvement',
        'cousin_improvement',
        'inventory_improvement',
    ]
    remaining = [c for c in results_df.columns if c not in preferred]
    results_df = results_df[preferred + remaining]

    return results_df
