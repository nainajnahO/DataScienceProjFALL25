import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, Tuple, List, Union, Any
from planogram.config import BASE_DIR
import os

# Try to import sentence-transformers, fall back to TF-IDF if not available
try:
    from sentence_transformers import SentenceTransformer
    USE_SENTENCE_TRANSFORMERS = True
except ImportError:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    USE_SENTENCE_TRANSFORMERS = False

ModelArtifact = Union[SentenceTransformer, Tuple[Any, Any]]

def create_product_texts_vectorized(df: pd.DataFrame) -> List[str]:
    """
    Create text representations for all products using vectorized operations.
    """
    name = df['product_name'].fillna('').astype(str)
    cat = df['category'].fillna('').astype(str)
    subcat = df['subcategory'].fillna('').astype(str)
    prov = df['provider'].fillna('').astype(str)
    
    texts = name.copy()
    
    mask_cat = cat != ''
    if mask_cat.any():
        texts[mask_cat] = texts[mask_cat] + ' | Category: ' + cat[mask_cat]
    
    mask_subcat = subcat != ''
    if mask_subcat.any():
        texts[mask_subcat] = texts[mask_subcat] + ' | Subcategory: ' + subcat[mask_subcat]
    
    mask_prov = prov.str.strip() != ''
    if mask_prov.any():
        texts[mask_prov] = texts[mask_prov] + ' | Brand: ' + prov[mask_prov]
    
    return texts.tolist()

def train_embedding_model(products_df: pd.DataFrame = None) -> ModelArtifact:
    """
    Prepares/Trains the embedding model.
    
    If using SentenceTransformers, it loads the model (heavy operation).
    If using TF-IDF, it fits the Vectorizer and SVD on the provided products_df.
    
    Args:
        products_df: DataFrame containing products. Required for TF-IDF training.
                     Optional for SentenceTransformers (pre-trained).
    """
    if USE_SENTENCE_TRANSFORMERS:
        # This path might need adjustment depending on the final structure
        local_model_path = os.path.join(BASE_DIR, "ragnar", "models", "all-MiniLM-L6-v2")
        
        if os.path.exists(local_model_path):
            print(f"Loading local SentenceTransformer model from {local_model_path}")
            model = SentenceTransformer(str(local_model_path))
        else:
            print("Loading default SentenceTransformer model (downloading if needed)...")
            model = SentenceTransformer('all-MiniLM-L6-v2')
        return model
    else:
        if products_df is None or products_df.empty:
             raise ValueError("products_df is required to train TF-IDF model")
             
        print("Training TF-IDF + SVD model...")
        product_texts = create_product_texts_vectorized(products_df)
        
        vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform(product_texts)
        
        svd = TruncatedSVD(n_components=128, random_state=42)
        svd.fit(tfidf_matrix)
        
        return (vectorizer, svd)

def generate_embeddings(
    products_df: pd.DataFrame, 
    model_artifact: ModelArtifact
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Generates embeddings for a DataFrame of products using the provided model artifact.
    Returns the embeddings matrix and a mapping of product_name -> index.
    """
    if products_df.empty or 'product_name' not in products_df.columns:
        return np.array([]), {}

    unique_products_df = products_df.drop_duplicates(subset=['product_name']).reset_index(drop=True)
    product_texts = create_product_texts_vectorized(unique_products_df)
    
    if USE_SENTENCE_TRANSFORMERS:
        # model_artifact is SentenceTransformer
        model = model_artifact
        embeddings = model.encode(product_texts, show_progress_bar=True)
    else:
        # model_artifact is (vectorizer, svd)
        vectorizer, svd = model_artifact
        tfidf_matrix = vectorizer.transform(product_texts)
        embeddings = svd.transform(tfidf_matrix)
        
    product_to_index = {name: i for i, name in enumerate(unique_products_df['product_name'])}
    
    return embeddings, product_to_index

def create_embeddings_for_products(products_df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Legacy wrapper for backward compatibility.
    Trains/Loads model and generates embeddings in one go.
    """
    model = train_embedding_model(products_df)
    return generate_embeddings(products_df, model)
