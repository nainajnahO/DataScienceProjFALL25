"""Shared utilities for training planogram models using planogram package functions.

This module centralizes Firebase setup and model training logic to avoid duplication
across pipeline.py, notebooks, and standalone scripts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any
import pickle
import gc

import pandas as pd

from .config import LOCATION_MODEL_ARTIFACT_PATH, UNIQUENESS_MODEL_ARTIFACT_PATH

# Import planogram package functions
try:
    from planogram import data_loader, location_scoring, product_scoring, product_filters
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from planogram import data_loader, location_scoring, product_scoring, product_filters


def setup_firestore_client(service_account_path: Optional[Path] = None) -> Optional[Any]:
    """Set up and return a Firestore client.
    
    Args:
        service_account_path: Path to service account JSON. If None, uses default location.
        
    Returns:
        Firestore client or None if setup fails
    """
    try:
        from firebase_admin import credentials, initialize_app, get_app
        from google.cloud import firestore as google_firestore
        
        if service_account_path is None:
            FUNCTIONS_DIR = Path(__file__).resolve().parents[1]
            service_account_path = FUNCTIONS_DIR / "serviceAccountKey.json"
        
        if not service_account_path.exists():
            print(f"⚠️  Warning: Firebase service account key not found at {service_account_path}")
            return None
        
        try:
            firebase_app = get_app()
        except ValueError:
            cred = credentials.Certificate(str(service_account_path))
            firebase_app = initialize_app(cred)
        
        firestore_client = google_firestore.Client.from_service_account_json(str(service_account_path))
        return firestore_client
        
    except Exception as e:
        print(f"⚠️  Error setting up Firestore client: {e}")
        return None


def load_training_data(firestore_client: Any) -> Dict[str, pd.DataFrame]:
    """Load products, machines, and sales data using planogram.data_loader.
    
    Args:
        firestore_client: Firestore client
        
    Returns:
        Dict with 'products', 'machines', 'sales' DataFrames
    """
    print("\n📥 Loading data using planogram.data_loader...")
    
    # Load Firebase collections
    firebase_data = data_loader.load_firebase_collections(firestore_client)
    products_df = firebase_data.get("products")
    machines_df = firebase_data.get("machines")
    
    # Apply product filters
    if products_df is not None and not products_df.empty:
        print("🔍 Applying product filters using planogram.product_filters.apply_all_filters()...")
        products_before = len(products_df)
        products_df = product_filters.apply_all_filters(products_df.copy())
        print(f"   Products before filtering: {products_before}")
        print(f"   Products after filtering: {len(products_df)}")
    
    # Load sales data
    sales_df = pd.DataFrame()
    try:
        sales_df = data_loader.load_all_sales_data(firestore_client)
        print(f"   ✓ Sales rows loaded: {len(sales_df)}")
    except Exception as e:
        print(f"   ⚠️  Could not load sales data: {e}")
    
    return {
        'products': products_df if products_df is not None else pd.DataFrame(),
        'machines': machines_df if machines_df is not None else pd.DataFrame(),
        'sales': sales_df
    }


def train_location_model(
    sales_df: pd.DataFrame,
    machines_df: pd.DataFrame,
    products_df: pd.DataFrame,
    model_path: Optional[Path] = None,
    skip_if_exists: bool = True
) -> Optional[Path]:
    """Train location fit model using planogram.location_scoring.train_location_model().
    
    Args:
        sales_df: Sales data DataFrame
        machines_df: Machines data DataFrame
        products_df: Products data DataFrame
        model_path: Where to save the model (defaults to config path)
        skip_if_exists: If True, skip if model already exists
        
    Returns:
        Path to saved model or None if training failed/skipped
    """
    if model_path is None:
        model_path = LOCATION_MODEL_ARTIFACT_PATH
    
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    if skip_if_exists and model_path.exists():
        print(f"\n⏭️  Location model already exists at {model_path}")
        return model_path
    
    if sales_df.empty or machines_df.empty or products_df.empty:
        print("\n⚠️  Cannot train location model: missing required data")
        return None
    
    print("\n📍 Training location fit model using planogram.location_scoring.train_location_model()...")
    try:
        location_model = location_scoring.train_location_model(
            sales_df, machines_df, products_df
        )
        
        with open(model_path, "wb") as f:
            pickle.dump(location_model, f)
        
        size_mb = model_path.stat().st_size / 1024 / 1024
        print(f"   ✓ Location model saved to {model_path}")
        print(f"   📦 Model size: {size_mb:.2f} MB")
        
        del location_model
        gc.collect()
        return model_path
        
    except Exception as e:
        print(f"   ⚠️  Failed to train location model: {e}")
        import traceback
        traceback.print_exc()
        return None


def train_uniqueness_model(
    products_df: pd.DataFrame,
    model_path: Optional[Path] = None,
    skip_if_exists: bool = True
) -> Optional[Path]:
    """Train uniqueness model using planogram.product_scoring.train_uniqueness_model().
    
    Args:
        products_df: Products data DataFrame
        model_path: Where to save the model (defaults to config path)
        skip_if_exists: If True, skip if model already exists
        
    Returns:
        Path to saved model or None if training failed/skipped
    """
    if model_path is None:
        model_path = UNIQUENESS_MODEL_ARTIFACT_PATH
    
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    if skip_if_exists and model_path.exists():
        print(f"\n⏭️  Uniqueness model already exists at {model_path}")
        return model_path
    
    if products_df.empty:
        print("\n⚠️  Cannot train uniqueness model: no products data")
        return None
    
    print("\n🎯 Training uniqueness model using planogram.product_scoring.train_uniqueness_model()...")
    print(f"   📊 Processing {len(products_df)} products (this may take a few minutes)...")
    print("   ⚠️  WARNING: This is memory-intensive.")
    
    try:
        # Override embedding model loading to use local model path if available
        # This avoids downloading and reduces memory usage (change only in planogram_autofill)
        from planogram import embed_products
        local_model_path = Path(__file__).resolve().parent.parent / "ragnar" / "models" / "all-MiniLM-L6-v2"
        
        if local_model_path.exists():
            print(f"   ✓ Found local SentenceTransformer model at: {local_model_path}")
            print("     (Using local model avoids download and reduces memory usage)")
            
            # Monkey-patch the train_embedding_model function to use local path
            original_train_embedding_model = embed_products.train_embedding_model
            
            def train_embedding_model_with_local_path(products_df=None):
                """Wrapper that forces use of local model path."""
                from sentence_transformers import SentenceTransformer
                print(f"   Loading local SentenceTransformer model from:")
                print(f"     {local_model_path}")
                model = SentenceTransformer(str(local_model_path), device='cpu')  # Use CPU to reduce memory
                return model
            
            # Temporarily replace the function
            embed_products.train_embedding_model = train_embedding_model_with_local_path
        
        # Force garbage collection before training to free up memory
        gc.collect()
        
        # Train the model step-by-step to control memory usage
        print("   🔄 Starting embedding generation...")
        
        # Step 1: Load embedding model (monkey-patched to use local path)
        embedding_model = embed_products.train_embedding_model(products_df)
        print("      ✓ Embedding model loaded")
        
        # Force garbage collection after model loading
        gc.collect()
        
        # Ensure model is on CPU and reduce torch threads to save memory
        try:
            import torch
            if hasattr(torch, 'set_num_threads'):
                torch.set_num_threads(1)  # Use single thread to reduce memory
            print("      ✓ Configured torch for low memory usage")
        except:
            pass
        
        # Step 2: Generate embeddings with very small batch size to avoid memory issues
        print("      🔄 Generating embeddings (this may take a few minutes)...")
        print("      ⚠️  Using very small batch size (8) to prevent memory crashes")
        
        # Generate embeddings with explicit batch processing
        unique_products_df = products_df.drop_duplicates(subset=['product_name']).reset_index(drop=True)
        print(f"      Preparing {len(unique_products_df)} unique products...")
        
        product_texts = embed_products.create_product_texts_vectorized(unique_products_df)
        
        # Clean up products_df before encoding (no longer needed)
        del unique_products_df
        gc.collect()
        
        # Check available memory before starting
        try:
            import psutil
            mem = psutil.virtual_memory()
            available_gb = mem.available / (1024**3)
            print(f"      📊 Available RAM: {available_gb:.2f} GB")
            if available_gb < 2.0:
                print(f"      ⚠️  WARNING: Low available memory ({available_gb:.2f} GB)")
                print(f"      💡 Close other applications or use standalone script")
        except:
            pass
        
        # Final memory cleanup before encoding
        gc.collect()
        import time
        time.sleep(0.5)  # Brief pause to let GC complete
        
        # Encode with very small batch size and CPU device
        # Process in chunks to avoid loading all texts into memory at once
        print(f"      Processing {len(product_texts)} product texts in batches of 4...")
        print("      ⚠️  Using tiny batch size (4) to prevent crashes - this will be slow but safer")
        
        import numpy as np
        
        # Process in chunks ourselves to have more control
        chunk_size = 4  # Process 4 texts at a time
        num_chunks = (len(product_texts) + chunk_size - 1) // chunk_size
        all_embeddings = []
        
        try:
            print(f"      Starting encoding (will process {num_chunks} chunks)...")
            
            for i in range(0, len(product_texts), chunk_size):
                chunk = product_texts[i:i+chunk_size]
                chunk_num = (i // chunk_size) + 1
                
                if chunk_num % 50 == 0 or chunk_num == 1:  # Print every 50 chunks or first
                    print(f"         Processing chunk {chunk_num}/{num_chunks} ({i+len(chunk)}/{len(product_texts)} texts)...")
                
                # Encode tiny chunk (size=1 to minimize memory per call)
                chunk_embeddings = embedding_model.encode(
                    chunk,
                    batch_size=1,  # Process one at a time to minimize memory
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    device='cpu',
                    normalize_embeddings=False
                )
                
                all_embeddings.append(chunk_embeddings)
                
                # Clean up chunk from memory immediately
                del chunk, chunk_embeddings
                
                # Force GC every 10 chunks to keep memory low
                if chunk_num % 10 == 0:
                    gc.collect()
            
            # Concatenate all embeddings
            print("      Concatenating embeddings...")
            embeddings = np.vstack(all_embeddings)
            del all_embeddings
            gc.collect()
            
            print(f"      ✓ Embeddings generated: shape={embeddings.shape}")
            
        except MemoryError as e:
            print(f"\n      ❌❌❌ OUT OF MEMORY during encoding! ❌❌❌")
            print(f"      Error: {e}")
            print("\n      💡 Solutions:")
            print("         1. Restart kernel and use standalone script:")
            print("            python train_uniqueness_model.py")
            print("         2. Close other applications to free RAM")
            print("         3. Reduce number of products being processed")
            raise
        except Exception as e:
            print(f"\n      ❌ Error during encoding: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # Create product index
        unique_products_df = products_df.drop_duplicates(subset=['product_name']).reset_index(drop=True)
        product_to_index = {name: i for i, name in enumerate(unique_products_df['product_name'])}
        del unique_products_df
        gc.collect()
        print(f"      ✓ Created product index: {len(product_to_index)} products")
        
        # Step 3: Immediately delete the embedding model to free memory
        # The SentenceTransformer model can be very large and is no longer needed
        print("      🧹 Deleting embedding model from memory...")
        del embedding_model
        
        # Restore original function
        if local_model_path.exists():
            embed_products.train_embedding_model = original_train_embedding_model
            del original_train_embedding_model
        
        # Force garbage collection to actually free the memory
        gc.collect()
        print("      ✓ Memory freed")
        
        print("   ✓ Embeddings generated successfully")
        
        # Step 4: Create model tuple and save
        print("   💾 Preparing to save model...")
        import time
        time.sleep(1)  # Give garbage collection time to complete
        gc.collect()
        
        # Create the tuple just before saving (minimize time in memory)
        uniqueness_model = (embeddings, product_to_index)
        
        # Use joblib if available - more memory-efficient for numpy arrays
        try:
            import joblib
            use_joblib = True
        except ImportError:
            use_joblib = False
        
        # Immediately save to disk to free memory
        print("   💾 Saving uniqueness model to disk...")
        print("      (This step may take a moment - saving large numpy arrays)")
        
        try:
            if use_joblib:
                print("      Using joblib (more memory-efficient for numpy arrays)...")
                try:
                    # joblib is more memory-efficient for numpy arrays
                    # compress=0 means no compression (faster, less memory during save)
                    joblib.dump(uniqueness_model, model_path, compress=0)
                    print("      ✓ Saved using joblib")
                except MemoryError as e:
                    print(f"      ❌ Memory error during joblib save: {e}")
                    raise
                except Exception as e:
                    print(f"      ⚠️  joblib save failed: {e}")
                    print("      Falling back to pickle...")
                    with open(model_path, "wb") as f:
                        pickle.dump(uniqueness_model, f, protocol=4)
                    print("      ✓ Saved using pickle (fallback)")
            else:
                print("      Using pickle (joblib not available)...")
                with open(model_path, "wb") as f:
                    pickle.dump(uniqueness_model, f, protocol=4)
                print("      ✓ Saved using pickle")
            
            size_mb = model_path.stat().st_size / 1024 / 1024
            print(f"   ✓ Uniqueness model saved to {model_path}")
            print(f"   📦 Model size: {size_mb:.2f} MB")
            
        except MemoryError as e:
            print(f"\n   ❌❌❌ OUT OF MEMORY during model save! ❌❌❌")
            print(f"   Error: {e}")
            print("\n   💡 Solutions:")
            print("      1. Restart kernel and use standalone script:")
            print("         python train_uniqueness_model.py")
            print("      2. Close other applications to free RAM")
            print("      3. The embeddings are generated - you can try saving manually")
            raise
        
        # Aggressive cleanup immediately after saving
        del uniqueness_model
        del embeddings, product_to_index
        gc.collect()
        print("      ✓ Memory cleaned up after save")
        
        return model_path
        
    except MemoryError as e:
        print(f"   ❌ Out of memory during uniqueness model training: {e}")
        print("   💡 Suggestions:")
        print("      - Close other applications to free RAM")
        print("      - Restart the kernel and run just the uniqueness model training")
        import traceback
        traceback.print_exc()
        return None
    except Exception as e:
        print(f"   ⚠️  Failed to train uniqueness model: {e}")
        import traceback
        traceback.print_exc()
        return None


def train_all_planogram_models(
    firestore_client: Optional[Any] = None,
    skip_if_exists: bool = True,
    train_location: bool = True,
    train_uniqueness: bool = True
) -> Dict[str, Optional[Path]]:
    """Train both planogram models using planogram package functions.
    
    This is the main entry point that orchestrates training both models.
    
    Args:
        firestore_client: Firestore client (if None, will try to create one)
        skip_if_exists: If True, skip training if models already exist
        train_location: Whether to train location model
        train_uniqueness: Whether to train uniqueness model
        
    Returns:
        Dict with 'location_model_path' and 'uniqueness_model_path' (may be None)
    """
    print("=" * 70)
    print("Training Planogram Models")
    print("Using planogram package functions:")
    print("  - planogram.location_scoring.train_location_model()")
    print("  - planogram.product_scoring.train_uniqueness_model()")
    print("=" * 70)
    
    results = {'location_model_path': None, 'uniqueness_model_path': None}
    
    # Setup Firestore client if needed
    if firestore_client is None:
        firestore_client = setup_firestore_client()
        if firestore_client is None:
            print("\n⚠️  Cannot train models without Firestore access.")
            return results
    
    # Load training data
    data = load_training_data(firestore_client)
    products_df = data['products']
    machines_df = data['machines']
    sales_df = data['sales']
    
    if products_df.empty:
        print("\n⚠️  No products data available. Cannot train models.")
        return results
    
    # Train location model
    if train_location:
        location_path = train_location_model(
            sales_df, machines_df, products_df,
            skip_if_exists=skip_if_exists
        )
        results['location_model_path'] = location_path
        
        # Cleanup before uniqueness training (memory-intensive)
        del sales_df
        del machines_df
        gc.collect()
    
    # Train uniqueness model
    if train_uniqueness:
        uniqueness_path = train_uniqueness_model(
            products_df,
            skip_if_exists=skip_if_exists
        )
        results['uniqueness_model_path'] = uniqueness_path
    
    print("\n✅ Planogram model training complete!")
    return results

