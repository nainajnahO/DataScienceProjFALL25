"""
Utility functions for saving and loading trained model artifacts.
Artifacts are saved locally so they can be accessed by frontend code.
"""
import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from planogram.config import ARTIFACTS_DIR


def ensure_artifacts_dir():
    """Create artifacts directory if it doesn't exist."""
    Path(ARTIFACTS_DIR).mkdir(parents=True, exist_ok=True)
    return ARTIFACTS_DIR


def save_artifacts(artifacts: Dict[str, Any], base_path: Optional[str] = None) -> Dict[str, str]:
    """
    Save all artifacts from train_all() to local files.
    
    Args:
        artifacts: Dictionary of artifacts from train_all()
        base_path: Optional base path for artifacts (defaults to ARTIFACTS_DIR)
        
    Returns:
        Dictionary mapping artifact names to their saved file paths
    """
    if base_path is None:
        base_path = ensure_artifacts_dir()
    
    saved_paths = {}
    
    for artifact_name, artifact_data in artifacts.items():
        if artifact_data is None:
            print(f"Skipping {artifact_name}: None value")
            continue
            
        try:
            file_path = _save_single_artifact(artifact_name, artifact_data, base_path)
            saved_paths[artifact_name] = file_path
            print(f"✓ Saved {artifact_name} to {file_path}")
        except Exception as e:
            print(f"✗ Failed to save {artifact_name}: {e}")
            saved_paths[artifact_name] = None
    
    return saved_paths


def _save_single_artifact(name: str, data: Any, base_path: str) -> str:
    """Save a single artifact based on its type."""
    
    # 1. DataFrame (location_mapping, snapshot_model)
    if isinstance(data, pd.DataFrame):
        file_path = os.path.join(base_path, f"{name}.parquet")
        data.to_parquet(file_path, index=False, engine='pyarrow')
        return file_path
    
    # 2. Tuple of (numpy array, dict) - uniqueness_model
    if isinstance(data, tuple) and len(data) == 2:
        embeddings, product_to_index = data
        if isinstance(embeddings, np.ndarray) and isinstance(product_to_index, dict):
            # Save as NPZ (numpy archive) which can store arrays + metadata
            file_path = os.path.join(base_path, f"{name}.npz")
            np.savez_compressed(
                file_path,
                embeddings=embeddings,
                product_to_index=json.dumps(product_to_index)  # Store dict as JSON string
            )
            return file_path
    
    # 3. Dict[str, pd.DataFrame] - cousin_model, location_model (raw)
    if isinstance(data, dict):
        # Check if all values are DataFrames
        if all(isinstance(v, pd.DataFrame) for v in data.values()):
            # Save each DataFrame in a subdirectory
            dir_path = os.path.join(base_path, name)
            os.makedirs(dir_path, exist_ok=True)
            
            for key, df in data.items():
                # Sanitize key for filename
                safe_key = str(key).replace('/', '_').replace('\\', '_')
                file_path = os.path.join(dir_path, f"{safe_key}.parquet")
                # Preserve index for dict of DataFrames (e.g., cousin_model confidence matrices)
                # These matrices have product names as index/columns which must be preserved
                df.to_parquet(file_path, index=True, engine='pyarrow')
            
            # Save a manifest file with keys
            manifest_path = os.path.join(dir_path, "_manifest.json")
            with open(manifest_path, 'w') as f:
                json.dump(list(data.keys()), f)
            
            return dir_path
        
        # 4. Dict[str, Optional[str]] - healthiness_mapping
        elif all(isinstance(v, (str, type(None))) for v in data.values()):
            file_path = os.path.join(base_path, f"{name}.json")
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            return file_path
    
    # 5. Fallback: pickle for complex objects (moaaz_trend, etc.)
    file_path = os.path.join(base_path, f"{name}.pkl")
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    return file_path


def load_artifacts(artifact_names: Optional[list[str]] = None, base_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load artifacts from local files.
    
    Args:
        artifact_names: List of artifact names to load. If None, loads all found artifacts.
        base_path: Optional base path for artifacts (defaults to ARTIFACTS_DIR)
        
    Returns:
        Dictionary of loaded artifacts
    """
    if base_path is None:
        base_path = ARTIFACTS_DIR
    
    if not os.path.exists(base_path):
        print(f"Artifacts directory not found: {base_path}")
        return {}
    
    artifacts = {}
    
    # If no specific names provided, discover all artifacts
    if artifact_names is None:
        artifact_names = _discover_artifacts(base_path)
    
    for name in artifact_names:
        try:
            artifact_data = _load_single_artifact(name, base_path)
            if artifact_data is not None:
                artifacts[name] = artifact_data
                print(f"✓ Loaded {name}")
            else:
                print(f"✗ Failed to load {name}: file not found")
        except Exception as e:
            print(f"✗ Failed to load {name}: {e}")
    
    return artifacts


def _load_single_artifact(name: str, base_path: str) -> Any:
    """Load a single artifact based on file extension."""
    
    # Try different file extensions
    extensions = [
        ('.parquet', _load_dataframe),
        ('.npz', _load_npz_tuple),
        ('.json', _load_json),
        ('.pkl', _load_pickle),
    ]
    
    # Check if it's a directory (for dict of DataFrames)
    dir_path = os.path.join(base_path, name)
    if os.path.isdir(dir_path):
        return _load_dict_of_dataframes(dir_path)
    
    # Try file extensions
    for ext, loader_func in extensions:
        file_path = os.path.join(base_path, f"{name}{ext}")
        if os.path.exists(file_path):
            return loader_func(file_path)
    
    return None


def _load_dataframe(file_path: str) -> pd.DataFrame:
    """Load a DataFrame from parquet."""
    return pd.read_parquet(file_path, engine='pyarrow')


def _load_npz_tuple(file_path: str) -> tuple:
    """Load tuple of (embeddings, product_to_index) from NPZ."""
    data = np.load(file_path, allow_pickle=True)
    embeddings = data['embeddings']
    product_to_index = json.loads(str(data['product_to_index']))
    return embeddings, product_to_index


def _load_json(file_path: str) -> dict:
    """Load a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def _load_pickle(file_path: str) -> Any:
    """Load a pickled object."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def _load_dict_of_dataframes(dir_path: str) -> Dict[str, pd.DataFrame]:
    """Load a dictionary of DataFrames from a directory."""
    manifest_path = os.path.join(dir_path, "_manifest.json")
    
    if not os.path.exists(manifest_path):
        # Try to infer keys from parquet files
        parquet_files = [f for f in os.listdir(dir_path) if f.endswith('.parquet')]
        keys = [f.replace('.parquet', '') for f in parquet_files]
    else:
        with open(manifest_path, 'r') as f:
            keys = json.load(f)
    
    result = {}
    for key in keys:
        # Try both original key and sanitized key
        safe_key = str(key).replace('/', '_').replace('\\', '_')
        file_path = os.path.join(dir_path, f"{safe_key}.parquet")
        
        if os.path.exists(file_path):
            result[key] = pd.read_parquet(file_path, engine='pyarrow')
    
    return result


def _discover_artifacts(base_path: str) -> list[str]:
    """Discover all artifact names in the base path."""
    artifacts = set()
    
    if not os.path.exists(base_path):
        return []
    
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        
        # Skip hidden files and directories that are artifacts themselves
        if item.startswith('_'):
            continue
        
        # If it's a directory, it's likely a dict-of-DataFrames artifact
        if os.path.isdir(item_path):
            artifacts.add(item)
        else:
            # Remove extension to get artifact name
            name = os.path.splitext(item)[0]
            artifacts.add(name)
    
    return sorted(list(artifacts))


def list_saved_artifacts(base_path: Optional[str] = None) -> Dict[str, str]:
    """
    List all saved artifacts and their file paths.
    
    Returns:
        Dictionary mapping artifact names to their file paths
    """
    if base_path is None:
        base_path = ARTIFACTS_DIR
    
    if not os.path.exists(base_path):
        return {}
    
    artifacts = {}
    artifact_names = _discover_artifacts(base_path)
    
    for name in artifact_names:
        dir_path = os.path.join(base_path, name)
        if os.path.isdir(dir_path):
            artifacts[name] = dir_path
        else:
            # Find the actual file
            for ext in ['.parquet', '.npz', '.json', '.pkl']:
                file_path = os.path.join(base_path, f"{name}{ext}")
                if os.path.exists(file_path):
                    artifacts[name] = file_path
                    break
    
    return artifacts


def compare_artifacts(original: Dict[str, Any], loaded: Dict[str, Any], verbose: bool = True) -> Dict[str, bool]:
    """
    Compare original artifacts with loaded artifacts to verify they match.
    
    Args:
        original: Dictionary of original artifacts (from train_all)
        loaded: Dictionary of loaded artifacts (from load_artifacts)
        verbose: Whether to print detailed comparison results
        
    Returns:
        Dictionary mapping artifact names to True/False (True if they match)
    """
    results = {}
    
    # Get all unique artifact names
    all_names = set(original.keys()) | set(loaded.keys())
    
    for name in all_names:
        if name not in original:
            if verbose:
                print(f"✗ {name}: Missing in original artifacts")
            results[name] = False
            continue
            
        if name not in loaded:
            if verbose:
                print(f"✗ {name}: Missing in loaded artifacts")
            results[name] = False
            continue
        
        orig_data = original[name]
        loaded_data = loaded[name]
        
        try:
            is_match = _compare_single_artifact(name, orig_data, loaded_data, verbose)
            results[name] = is_match
        except Exception as e:
            if verbose:
                print(f"✗ {name}: Comparison failed with error: {e}")
            results[name] = False
    
    if verbose:
        total = len(results)
        matched = sum(results.values())
        print(f"\n{'='*50}")
        print(f"Comparison Summary: {matched}/{total} artifacts match")
        print(f"{'='*50}")
    
    return results


def _compare_single_artifact(name: str, original: Any, loaded: Any, verbose: bool) -> bool:
    """Compare a single artifact."""
    
    # 1. DataFrame comparison
    if isinstance(original, pd.DataFrame):
        if not isinstance(loaded, pd.DataFrame):
            if verbose:
                print(f"✗ {name}: Type mismatch (DataFrame vs {type(loaded).__name__})")
            return False
        
        # Compare shape
        if original.shape != loaded.shape:
            if verbose:
                print(f"✗ {name}: Shape mismatch {original.shape} vs {loaded.shape}")
            return False
        
        # Compare columns
        if not original.columns.equals(loaded.columns):
            if verbose:
                print(f"✗ {name}: Column mismatch")
                print(f"  Original: {list(original.columns)}")
                print(f"  Loaded: {list(loaded.columns)}")
            return False
        
        # Compare values (handle NaN properly)
        try:
            # Reset index for comparison
            orig_sorted = original.sort_index().reset_index(drop=True)
            loaded_sorted = loaded.sort_index().reset_index(drop=True)
            
            # Compare with pandas testing
            pd.testing.assert_frame_equal(
                orig_sorted, 
                loaded_sorted, 
                check_dtype=False,  # Allow dtype differences (e.g., int64 vs Int64)
                check_exact=False,  # Allow small floating point differences
                atol=1e-5
            )
            if verbose:
                print(f"✓ {name}: DataFrames match")
            return True
        except AssertionError as e:
            if verbose:
                print(f"✗ {name}: DataFrame values differ")
                _show_dataframe_diff(original, loaded, name)
            return False
    
    # 2. Tuple comparison (uniqueness_model: numpy array + dict)
    if isinstance(original, tuple) and isinstance(loaded, tuple):
        if len(original) != len(loaded):
            if verbose:
                print(f"✗ {name}: Tuple length mismatch")
            return False
        
        # Compare first element (numpy array)
        if isinstance(original[0], np.ndarray) and isinstance(loaded[0], np.ndarray):
            if not np.allclose(original[0], loaded[0], atol=1e-5, equal_nan=True):
                if verbose:
                    print(f"✗ {name}: Numpy arrays differ")
                    print(f"  Shape: {original[0].shape} vs {loaded[0].shape}")
                    print(f"  Max diff: {np.nanmax(np.abs(original[0] - loaded[0]))}")
                return False
        elif original[0] != loaded[0]:
            if verbose:
                print(f"✗ {name}: First tuple element differs")
            return False
        
        # Compare second element (dict)
        if isinstance(original[1], dict) and isinstance(loaded[1], dict):
            if original[1] != loaded[1]:
                if verbose:
                    print(f"✗ {name}: Dicts differ")
                    orig_keys = set(original[1].keys())
                    loaded_keys = set(loaded[1].keys())
                    if orig_keys != loaded_keys:
                        print(f"  Keys differ: {orig_keys - loaded_keys} vs {loaded_keys - orig_keys}")
                    else:
                        diff_vals = {k: (original[1][k], loaded[1][k]) for k in orig_keys if original[1][k] != loaded[1][k]}
                        if diff_vals:
                            print(f"  {len(diff_vals)} values differ")
                return False
        elif original[1] != loaded[1]:
            if verbose:
                print(f"✗ {name}: Second tuple element differs")
            return False
        
        if verbose:
            print(f"✓ {name}: Tuples match")
        return True
    
    # 3. Dict of DataFrames (cousin_model)
    if isinstance(original, dict) and isinstance(loaded, dict):
        if set(original.keys()) != set(loaded.keys()):
            if verbose:
                print(f"✗ {name}: Dict keys differ: {set(original.keys())} vs {set(loaded.keys())}")
            return False
        
        # Check if all values are DataFrames
        if all(isinstance(v, pd.DataFrame) for v in original.values()):
            all_match = True
            for key in original.keys():
                if not _compare_single_artifact(f"{name}[{key}]", original[key], loaded[key], False):
                    all_match = False
                    if verbose:
                        print(f"✗ {name}: DataFrame at key '{key}' differs")
                        _show_dataframe_diff(original[key], loaded[key], f"{name}[{key}]")
            if all_match:
                if verbose:
                    print(f"✓ {name}: Dict of DataFrames matches")
            return all_match
        
        # Simple dict comparison (healthiness_mapping)
        if original == loaded:
            if verbose:
                print(f"✓ {name}: Dicts match")
            return True
        else:
            if verbose:
                # Find differences
                orig_keys = set(original.keys())
                loaded_keys = set(loaded.keys())
                only_orig = orig_keys - loaded_keys
                only_loaded = loaded_keys - orig_keys
                diff_values = {k: (original[k], loaded[k]) for k in orig_keys & loaded_keys if original[k] != loaded[k]}
                
                if only_orig:
                    print(f"✗ {name}: Keys only in original: {only_orig}")
                if only_loaded:
                    print(f"✗ {name}: Keys only in loaded: {only_loaded}")
                if diff_values:
                    print(f"✗ {name}: {len(diff_values)} values differ")
            return False
    
    # 4. Direct comparison for simple types
    try:
        if original == loaded:
            if verbose:
                print(f"✓ {name}: Values match")
            return True
        else:
            if verbose:
                print(f"✗ {name}: Values differ")
            return False
    except Exception:
        # For complex objects, try pickle comparison
        try:
            orig_pickle = pickle.dumps(original)
            loaded_pickle = pickle.dumps(loaded)
            if orig_pickle == loaded_pickle:
                if verbose:
                    print(f"✓ {name}: Pickled objects match")
                return True
            else:
                if verbose:
                    print(f"✗ {name}: Pickled objects differ")
                return False
        except Exception as e:
            if verbose:
                print(f"✗ {name}: Cannot compare - {e}")
            return False


def _show_dataframe_diff(original: pd.DataFrame, loaded: pd.DataFrame, name: str):
    """Show detailed differences between two DataFrames."""
    print(f"  Detailed diff for {name}:")
    
    # Shape
    if original.shape != loaded.shape:
        print(f"    Shape: {original.shape} vs {loaded.shape}")
    
    # Columns
    orig_cols = set(original.columns)
    loaded_cols = set(loaded.columns)
    if orig_cols != loaded_cols:
        print(f"    Columns only in original: {orig_cols - loaded_cols}")
        print(f"    Columns only in loaded: {loaded_cols - orig_cols}")
    
    # Compare common columns
    common_cols = orig_cols & loaded_cols
    if common_cols:
        # Reset index for comparison
        orig_sorted = original.sort_index().reset_index(drop=True)
        loaded_sorted = loaded.sort_index().reset_index(drop=True)
        
        # Find rows that differ
        for col in common_cols:
            try:
                if not orig_sorted[col].equals(loaded_sorted[col]):
                    # Find first differing value
                    diff_mask = orig_sorted[col] != loaded_sorted[col]
                    # Handle NaN
                    if diff_mask.any():
                        diff_mask = diff_mask | (orig_sorted[col].isna() != loaded_sorted[col].isna())
                    
                    if diff_mask.any():
                        first_diff_idx = diff_mask.idxmax() if hasattr(diff_mask, 'idxmax') else diff_mask.argmax()
                        print(f"    Column '{col}' differs at index {first_diff_idx}")
                        print(f"      Original: {orig_sorted[col].iloc[first_diff_idx]}")
                        print(f"      Loaded: {loaded_sorted[col].iloc[first_diff_idx]}")
                        break
            except Exception:
                pass
        
        # Show sample of differences
        try:
            diff_count = 0
            for col in common_cols:
                if not orig_sorted[col].equals(loaded_sorted[col]):
                    diff_mask = (orig_sorted[col] != loaded_sorted[col]) | (
                        orig_sorted[col].isna() != loaded_sorted[col].isna()
                    )
                    diff_count += diff_mask.sum()
            
            if diff_count > 0:
                print(f"    Total differing values: {diff_count}")
        except Exception:
            pass


def verify_artifacts_integrity(artifacts: Dict[str, Any], base_path: Optional[str] = None) -> Dict[str, bool]:
    """
    Verify that artifacts in memory match what's saved on disk.
    
    This loads artifacts from disk and compares them with the provided artifacts.
    
    Args:
        artifacts: Dictionary of artifacts to verify
        base_path: Optional base path for artifacts (defaults to ARTIFACTS_DIR)
        
    Returns:
        Dictionary mapping artifact names to True/False (True if they match)
    """
    loaded = load_artifacts(list(artifacts.keys()), base_path=base_path)
    return compare_artifacts(artifacts, loaded, verbose=True)
