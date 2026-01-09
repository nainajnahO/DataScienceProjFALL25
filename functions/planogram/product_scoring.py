import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

from . import embed_products

# Constants for category diversity
OVERREPRESENTED_THRESHOLD = 0.3
UNDERREPRESENTED_THRESHOLD = 0.1

# Type alias for the trained model artifact
# It contains the embeddings matrix and the product-to-index map
UniquenessModel = Tuple[np.ndarray, Dict[str, int]]

def train_uniqueness_model(products_df: pd.DataFrame) -> UniquenessModel:
    """
    Trains the uniqueness model by generating embeddings for all known products.
    Returns the (embeddings, product_to_index) tuple which acts as the model state.
    """
    # 1. Initialize/Train the underlying embedding model (e.g., SentenceTransformer)
    embedding_model = embed_products.train_embedding_model(products_df)
    
    # 2. Generate reference embeddings for all products
    embeddings, product_to_index = embed_products.generate_embeddings(products_df, embedding_model)
    
    return embeddings, product_to_index

def get_product_embeddings(
    products: List[Dict],
    embeddings: np.ndarray,
    product_to_index: Dict[str, int],
    case_sensitive: bool = False,
    handle_missing: str = 'skip'
) -> tuple[np.ndarray, list[int], list[str], list[int]]:
    """
    Get embeddings for products in the machine using the pre-calculated embedding matrix.
    """
    product_embeddings = []
    valid_indices = []
    valid_products = []
    missing_indices = []
    
    existing_products_set = set(product_to_index.keys())
    
    normalized_mapping = {}
    if not case_sensitive:
        for product_name, index in product_to_index.items():
            normalized = product_name.strip().lower()
            if normalized not in normalized_mapping:
                normalized_mapping[normalized] = (product_name, index)
    
    normalized_set = set(normalized_mapping.keys()) if not case_sensitive else set()
    
    for idx, product in enumerate(products):
        product_name = product['product_name']
        found = False
        
        if product_name in existing_products_set:
            embedding_idx = product_to_index[product_name]
            product_embeddings.append(embeddings[embedding_idx])
            valid_indices.append(idx)
            valid_products.append(product_name)
            found = True
        elif not case_sensitive:
            normalized = product_name.strip().lower()
            if normalized in normalized_set:
                original_name, embedding_idx = normalized_mapping[normalized]
                product_embeddings.append(embeddings[embedding_idx])
                valid_indices.append(idx)
                valid_products.append(original_name)
                found = True
        
        if not found:
            missing_indices.append(idx)
            if handle_missing == 'error':
                raise ValueError(f"Product '{product_name}' not found in embeddings")
    
    if not product_embeddings:
        return np.array([]), [], [], missing_indices

    return np.array(product_embeddings), valid_indices, valid_products, missing_indices


def calculate_uniqueness_scores(product_embeddings: np.ndarray) -> np.ndarray:
    """
    Calculate uniqueness scores for each product based on cosine distance.
    """
    from sklearn.metrics.pairwise import cosine_distances
    
    n_products = len(product_embeddings)
    if n_products <= 1:
        return np.array([1.0] * n_products)
    
    distances = cosine_distances(product_embeddings)
    np.fill_diagonal(distances, 0)
    
    uniqueness_scores = distances.sum(axis=1) / (n_products - 1)
    return uniqueness_scores


def calculate_category_diversity_scores(product_categories: List[str]) -> np.ndarray:
    """
    Calculate category diversity scores. Products in underrepresented categories get higher scores.
    """
    if not product_categories:
        return np.array([])
    
    from collections import Counter
    category_counts = Counter(product_categories)
    total_products = len(product_categories)
    
    category_proportions = {cat: count / total_products for cat, count in category_counts.items()}
    
    proportions_array = np.array([category_proportions.get(cat, 0.0) for cat in product_categories])
    diversity_scores = 1.0 - proportions_array
    
    return diversity_scores


def predict_machine_uniqueness_score(
    machine_products_df: pd.DataFrame,
    trained_model: UniquenessModel
) -> Optional[float]:
    """
    Lightweight function that calculates only machine-level mean uniqueness score.
    
    Optimized for performance when only aggregate score is needed (e.g., autofill).
    Returns single float instead of per-product DataFrame.
    
    Args:
        machine_products_df: DataFrame with products for a single machine.
                           Must have 'machine_id' column and product records.
        trained_model: Trained uniqueness model (embeddings, product_to_index)
    
    Returns:
        Float: Mean uniqueness score for the machine, or None if insufficient products.
    """
    if machine_products_df.empty:
        return None
    
    embeddings, product_to_index = trained_model
    
    # Get products for the machine (assuming single machine or first group)
    products_in_machine = machine_products_df.to_dict('records')
    
    if not products_in_machine:
        return None

    # Get embeddings for products in the machine
    machine_product_embeddings, valid_indices, _, _ = get_product_embeddings(
        products_in_machine, embeddings, product_to_index, case_sensitive=False
    )
    
    if machine_product_embeddings.shape[0] < 2:
        return None  # Not enough products to calculate meaningful scores

    # Calculate uniqueness scores
    uniqueness = calculate_uniqueness_scores(machine_product_embeddings)
    
    # Return machine-level mean
    return float(uniqueness.mean())


def predict_uniqueness_scores(
    machine_products_df: pd.DataFrame,
    trained_model: UniquenessModel
) -> pd.DataFrame:
    """
    Calculates uniqueness and diversity scores for all machines using the trained model.
    """
    machine_products_df = machine_products_df.copy()
    embeddings, product_to_index = trained_model
    
    all_scores = []

    for machine_id, machine_group in machine_products_df.groupby('machine_id'):
        products_in_machine = machine_group.to_dict('records')

        if not products_in_machine:
            continue

        # Get embeddings for products in the current machine using the reference matrix
        machine_product_embeddings, valid_indices, _, _ = get_product_embeddings(
            products_in_machine, embeddings, product_to_index
        )
        
        if machine_product_embeddings.shape[0] < 2:
            continue # Not enough products to calculate meaningful scores

        # Calculate scores
        uniqueness = calculate_uniqueness_scores(machine_product_embeddings)
        
        valid_products = [products_in_machine[i] for i in valid_indices]
        categories = [p.get('category', 'Unknown') for p in valid_products]
        category_diversity = calculate_category_diversity_scores(categories)

        # Store results
        for i, original_idx in enumerate(valid_indices):
            product_info = products_in_machine[original_idx]
            all_scores.append({
                'machine_id': machine_id,
                'product_name': product_info['product_name'],
                'position': product_info.get('position'),
                'uniqueness_score': uniqueness[i],
                'category_diversity_score': category_diversity[i]
            })

    return pd.DataFrame(all_scores)


def calculate_machine_uniqueness_scores(
    machines_df: pd.DataFrame,
    products_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Legacy wrapper. Trains model on-the-fly and predicts scores.
    """
    machines_df = machines_df.copy()
    products_df = products_df.copy()
    # 1. Train model (create global embeddings)
    model = train_uniqueness_model(products_df)
    
    # 2. Predict scores
    return predict_uniqueness_scores(machines_df, model)
