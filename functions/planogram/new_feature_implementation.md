# New Feature Implementation Guide

This document provides instructions for converting a finished codebase (with a full pipeline approach) into the planogram folder structure. The goal is to extract reusable functions and make them easily callable, following the established patterns in this repository.

## Overview

You are given a folder containing finished code that implements a feature using a full pipeline approach. Your task is to:

1. **Extract the core logic** from the pipeline into reusable functions
2. **Restructure the code** to follow the planogram folder's architecture
3. **Ensure compatibility** with existing utilities (data loaders, cleaners, etc.)
4. **Create 1-2 Python files** in the planogram folder containing the extracted functions

## Critical Requirements

### 1. **No Data Loading or File I/O**
- ❌ **NEVER** load DataFrames from files, databases, or APIs within your functions
- ❌ **NEVER** read CSV, Parquet, JSON, or any other file formats
- ❌ **NEVER** make database queries or API calls
- ✅ **ALWAYS** accept all required DataFrames as function parameters
- ✅ Data loading will be handled later in a larger pipeline

**Example:**
```python
# ❌ WRONG - Loading data inside function
def calculate_scores():
    df = pd.read_csv('data.csv')  # NO!
    # ... calculations ...

# ✅ CORRECT - Accept DataFrame as input
def calculate_scores(input_df: pd.DataFrame) -> pd.DataFrame:
    df = input_df.copy()  # Always copy first!
    # ... calculations ...
    return df
```

### 2. **Always Copy Input DataFrames**
- ⚠️ **CRITICAL:** All `predict_*` and `calculate_*` functions **MUST** copy input DataFrames at the start
- This prevents mutating original data and ensures data integrity
- Use `df = input_df.copy()` as the first line in your function

**Example:**
```python
def predict_something(df: pd.DataFrame, model) -> pd.DataFrame:
    df = df.copy()  # Always copy first!
    # ... rest of function logic ...
    return df
```

### 3. **Train/Predict Separation Pattern**
All scoring and modeling modules must follow a strict **Train/Predict** separation:

#### Training Phase (`train_...`)
- **Purpose:** Heavy computation, model fitting, statistical aggregation
- **Input:** Raw data (full sales history, product catalog, etc.) as DataFrames
- **Output:** A serializable "Model Artifact" (e.g., `dict`, `tuple`, trained sklearn model, or DataFrame)
- **Naming:** `train_<feature_name>_model(...)`
- **When to use:** Called once to generate artifacts that can be cached/saved

#### Inference Phase (`predict_...` or `calculate_...`)
- **Purpose:** Fast lookups, dot products, lightweight transformations
- **Input:** Target data to score (e.g., specific machine's products) AND the `Model Artifact`
- **Output:** Scores, recommendations, or modified DataFrames
- **Naming:** `predict_<feature_name>_scores(...)` or `calculate_<feature_name>(..., trained_model)`
- **When to use:** Called on-demand using pre-calculated artifacts

**Example:**
```python
# Training: Heavy computation, returns artifact
def train_location_model(
    sales_df: pd.DataFrame,
    machines_df: pd.DataFrame,
    products_df: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """Returns dictionary mapping location types to performance DataFrames."""
    # Heavy aggregation and computation
    results = {}
    for location_type in location_types:
        # ... calculate metrics ...
        results[location_type] = metrics_df
    return results

# Prediction: Fast lookup using artifact
def predict_location_scores(
    products_df: pd.DataFrame,
    location_types: list[str],
    trained_model: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Uses trained model to predict scores."""
    products_df = products_df.copy()  # Always copy!
    # Fast lookups using trained_model
    # ... return scores ...
    return products_df
```

### 4. **Output Handling**
- For outputs, **modify a relevant input DataFrame** and return it
- Always return **model artifacts** (for training) or **DataFrames** (for prediction)
- Never write to files or databases directly

**Example:**
```python
def predict_scores(
    products_df: pd.DataFrame,
    trained_model: dict
) -> pd.DataFrame:
    products_df = products_df.copy()
    # Add score columns to the DataFrame
    products_df['score'] = calculate_scores(products_df, trained_model)
    return products_df
```

### 5. **Do Not Rewrite Existing Functions**
- ❌ **DO NOT** create new data loaders, data cleaners, or file I/O utilities
- ❌ **DO NOT** duplicate functionality that exists in:
  - `data_loader.py` (loading from Firestore, Parquet files)
  - `utils.py` (Firestore commit utilities)
  - `product_filters.py` (product filtering logic)
  - Any other existing modules
- ✅ **USE** existing functions by importing them
- ✅ **FOCUS** on the core feature logic only

**Check existing modules before implementing:**
- `data_loader.py` - Data loading from Firestore and Parquet
- `utils.py` - Firestore utilities
- `config.py` - Configuration constants

### 6. **Code Structure**
- Create **1-2 Python files** in the planogram folder
- Use clear, descriptive function names following naming conventions
- Include type hints for all function parameters and return types
- Add docstrings explaining what each function does
- Keep functions focused and single-purpose

**File Organization:**
- If the feature is simple: One file (e.g., `feature_name.py`)
- If the feature has clear train/predict separation: One file is usually sufficient
- If the feature is complex with multiple sub-modules: Consider splitting into 2 files (e.g., `feature_name.py` and `feature_name_utils.py`)

## Implementation Steps

### Step 1: Analyze the Finished Code
1. Identify the **core logic** that performs the actual feature computation
2. Identify **data loading/cleaning** code (to be removed - use existing utilities)
3. Identify **training/learning** operations (heavy computation)
4. Identify **prediction/scoring** operations (lightweight lookups)
5. Identify **pipeline orchestration** code (to be removed - handled externally)

### Step 2: Extract Functions
1. Create `train_<feature_name>_model()` function:
   - Takes DataFrames as inputs (no file I/O)
   - Performs heavy computation
   - Returns a serializable artifact (dict, tuple, model, or DataFrame)

2. Create `predict_<feature_name>_scores()` or `calculate_<feature_name>()` function:
   - Takes DataFrames AND the trained model artifact as inputs
   - Copies input DataFrames first
   - Performs fast lookups/calculations
   - Returns modified DataFrame or scores

3. If needed, create helper functions (prefixed with `_` for private functions)

### Step 3: Remove Pipeline Code
- Remove all file I/O operations
- Remove data loading code (use `data_loader.py` functions instead)
- Remove data cleaning code (use existing utilities)
- Remove orchestration/sequencing code
- Keep only the pure computation logic

### Step 4: Integrate with Existing Code
- Import and use existing utilities where appropriate
- Follow existing patterns (see examples in `product_scoring.py`, `location_scoring.py`)
- Ensure function signatures match the expected patterns

### Step 5: Test Structure
- Verify functions accept DataFrames as inputs
- Verify functions copy input DataFrames (for predict/calculate functions)
- Verify train functions return artifacts
- Verify predict functions use artifacts
- Verify no file I/O or data loading exists

## Examples from Existing Code

### Example 1: Product Scoring (`product_scoring.py`)
```python
# Training: Creates embeddings for all products
def train_uniqueness_model(products_df: pd.DataFrame) -> tuple[np.ndarray, dict[str, int]]:
    embedding_model = embed_products.train_embedding_model(products_df)
    embeddings, product_to_index = embed_products.generate_embeddings(products_df, embedding_model)
    return embeddings, product_to_index

# Prediction: Uses embeddings to score machines
def predict_uniqueness_scores(
    machine_products_df: pd.DataFrame,
    trained_model: tuple[np.ndarray, dict[str, int]]
) -> pd.DataFrame:
    machine_products_df = machine_products_df.copy()  # Always copy!
    embeddings, product_to_index = trained_model
    # ... fast lookups using embeddings ...
    return scores_df
```

### Example 2: Location Scoring (`location_scoring.py`)
```python
# Training: Aggregates performance metrics by location
def train_location_model(
    sales_df: pd.DataFrame,
    machines_df: pd.DataFrame,
    products_df: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    merged_sales = location_preprocessing.prepare_data_for_scoring(
        sales_df, machines_df, products_df
    )
    # ... heavy aggregation ...
    return results  # dict mapping location_type -> metrics DataFrame

# Prediction: Fast lookup using trained metrics
def predict_location_scores(
    products_df: pd.DataFrame,
    location_types: list[str],
    trained_model: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    products_df = products_df.copy()  # Always copy!
    # ... fast lookups using trained_model ...
    return products_df
```

## Common Patterns

### Pattern 1: Simple Scoring (No Training Needed)
If your feature doesn't require training, use `calculate_*` naming:
```python
def calculate_feature_scores(input_df: pd.DataFrame) -> pd.DataFrame:
    input_df = input_df.copy()
    # ... calculations ...
    return input_df
```

### Pattern 2: Model-Based Feature
If your feature uses a trained model:
```python
def train_feature_model(data_df: pd.DataFrame) -> ModelArtifact:
    # Heavy computation
    return artifact

def predict_feature_scores(
    target_df: pd.DataFrame,
    trained_model: ModelArtifact
) -> pd.DataFrame:
    target_df = target_df.copy()
    # Fast lookups
    return target_df
```

### Pattern 3: Feature with Helper Functions
```python
def _helper_function(data: pd.DataFrame) -> pd.DataFrame:
    """Private helper function."""
    return data

def train_feature_model(data_df: pd.DataFrame) -> ModelArtifact:
    processed = _helper_function(data_df)
    # ... training logic ...
    return artifact
```

## Checklist

Before finalizing your implementation, verify:

- [ ] All functions accept DataFrames as parameters (no file I/O)
- [ ] All `predict_*` and `calculate_*` functions copy input DataFrames first
- [ ] Training functions return serializable artifacts
- [ ] Prediction functions accept and use trained artifacts
- [ ] No data loading code (use existing `data_loader.py`)
- [ ] No data cleaning code (use existing utilities)
- [ ] No file I/O operations
- [ ] Functions follow naming conventions (`train_*`, `predict_*`, `calculate_*`)
- [ ] Code is in 1-2 Python files in the planogram folder
- [ ] Type hints are included for all functions
- [ ] Docstrings explain function purpose and parameters
- [ ] Existing utilities are imported and used (not reimplemented)

## Final Notes

- The finished code you're given will have a **full pipeline approach** that is not needed in this repo
- Your job is to **extract the core functions** and make them **easily callable**
- The pipeline orchestration will be handled externally (in notebooks or larger scripts)
- Focus on **pure, reusable functions** that can be composed together
- Follow the existing code patterns and structure as closely as possible

## Questions to Ask Yourself

1. Does this function load data from files? → Remove it, accept DataFrame as parameter
2. Does this function mutate input data? → Add `.copy()` at the start
3. Is this heavy computation that should be cached? → Make it a `train_*` function
4. Is this a fast lookup/calculation? → Make it a `predict_*` or `calculate_*` function
5. Does this duplicate existing functionality? → Use existing utilities instead
6. Can this be called independently? → Good! This is what we want

Good luck! Follow these guidelines and your code will integrate seamlessly with the planogram folder structure.
