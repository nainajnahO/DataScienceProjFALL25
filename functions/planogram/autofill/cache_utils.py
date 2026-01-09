"""Cache utilities for testing mode - speeds up iteration during development.

⚠️ TESTING ONLY - This module should NOT be used in production!
"""

import pickle
import json
from pathlib import Path
from typing import Any, Optional
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent / 'test_cache'


def setup_cache_dir():
    """Create cache directory if it doesn't exist."""
    CACHE_DIR.mkdir(exist_ok=True)
    # Create .gitignore to prevent committing cache
    gitignore_path = CACHE_DIR / '.gitignore'
    if not gitignore_path.exists():
        gitignore_path.write_text('*\n!.gitignore\n')


def save_data(name: str, data: Any, testing_mode: bool = True) -> bool:
    """Save data to cache.
    
    Args:
        name: Cache key name
        data: Data to save (DataFrame, dict, etc.)
        testing_mode: Only save if True
    
    Returns:
        True if saved, False otherwise
    """
    if not testing_mode:
        return False
    
    try:
        setup_cache_dir()
        cache_path = CACHE_DIR / f'{name}.pkl'
        
        if isinstance(data, pd.DataFrame):
            # Use parquet for DataFrames (smaller, faster)
            cache_path = CACHE_DIR / f'{name}.parquet'
            
            # Make a copy to avoid modifying original data
            df_to_save = data.copy()
            
            # Check for complex types (lists, numpy arrays) that parquet can't handle
            has_complex_types = False
            for col in df_to_save.columns:
                if df_to_save[col].dtype == 'object':
                    # Sample a few non-null values to check for complex types
                    sample_vals = df_to_save[col].dropna().head(5)
                    if len(sample_vals) > 0:
                        first_val = sample_vals.iloc[0]
                        if isinstance(first_val, (list, np.ndarray)):
                            has_complex_types = True
                            break
            
            # If we have complex types, use pickle directly (preserves types)
            if has_complex_types:
                cache_path = CACHE_DIR / f'{name}.pkl'
                with open(cache_path, 'wb') as f:
                    pickle.dump(data, f)
            else:
                # Try parquet with type conversions
                # Fix type conversion issues before saving
                for col in df_to_save.columns:
                    col_dtype = df_to_save[col].dtype
                    
                    # Handle string booleans (e.g., 'True', 'False')
                    if col_dtype == 'object':
                        # Check if column contains string booleans
                        unique_vals = df_to_save[col].dropna().unique()
                        if len(unique_vals) > 0:
                            str_vals = set(str(v).lower() for v in unique_vals)
                            if str_vals.issubset({'true', 'false', '1', '0', 'yes', 'no', ''}):
                                # Convert string booleans to actual booleans
                                def str_to_bool(val):
                                    if pd.isna(val):
                                        return pd.NA
                                    val_str = str(val).lower().strip()
                                    if val_str in ('true', '1', 'yes'):
                                        return True
                                    elif val_str in ('false', '0', 'no', ''):
                                        return False
                                    return pd.NA
                                
                                df_to_save[col] = df_to_save[col].apply(str_to_bool).astype('boolean')
                                continue
                        
                        # Try to convert string numbers to numeric
                        # Only attempt if all non-null values look numeric
                        non_null = df_to_save[col].dropna()
                        if len(non_null) > 0:
                            try:
                                # Try converting to numeric
                                numeric_series = pd.to_numeric(non_null, errors='raise')
                                # If successful, convert the whole column
                                df_to_save[col] = pd.to_numeric(df_to_save[col], errors='coerce')
                            except (ValueError, TypeError):
                                # Not numeric, keep as object
                                pass
                
                # Try to save as parquet, fall back to pickle if it fails
                try:
                    df_to_save.to_parquet(cache_path, index=False)
                except Exception as parquet_error:
                    # If parquet fails (e.g., due to complex types), fall back to pickle
                    logger.warning(f"Parquet save failed for {name}, falling back to pickle: {parquet_error}")
                    cache_path = CACHE_DIR / f'{name}.pkl'
                    with open(cache_path, 'wb') as f:
                        pickle.dump(data, f)  # Use original data, not converted version
        else:
            # Use pickle for other objects
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
        
        logger.info(f"✓ Cached {name} to {cache_path}")
        return True
    except Exception as e:
        logger.warning(f"Failed to cache {name}: {e}")
        return False


def load_data(name: str, testing_mode: bool = True) -> Optional[Any]:
    """Load data from cache.
    
    Args:
        name: Cache key name
        testing_mode: Only load if True
    
    Returns:
        Cached data or None if not found/not in testing mode
    """
    if not testing_mode:
        return None
    
    try:
        # Try parquet first (for DataFrames)
        parquet_path = CACHE_DIR / f'{name}.parquet'
        if parquet_path.exists():
            data = pd.read_parquet(parquet_path)
            logger.info(f"✓ Loaded {name} from cache ({len(data):,} rows)")
            return data
        
        # Try pickle
        pickle_path = CACHE_DIR / f'{name}.pkl'
        if pickle_path.exists():
            with open(pickle_path, 'rb') as f:
                data = pickle.load(f)
            logger.info(f"✓ Loaded {name} from cache")
            return data
        
        return None
    except Exception as e:
        logger.warning(f"Failed to load {name} from cache: {e}")
        return None


def clear_cache():
    """Clear all cached files (for testing)."""
    if CACHE_DIR.exists():
        for file in CACHE_DIR.glob('*'):
            if file.name != '.gitignore':
                file.unlink()
        logger.info("✓ Cache cleared")


def cache_exists(name: str) -> bool:
    """Check if cache exists for given name."""
    parquet_path = CACHE_DIR / f'{name}.parquet'
    pickle_path = CACHE_DIR / f'{name}.pkl'
    return parquet_path.exists() or pickle_path.exists()

