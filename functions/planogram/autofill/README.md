# Autofill Module

AI-powered automatic product optimization for vending machines. This module automatically fills empty slots and performs beneficial product swaps based on predicted revenue and configurable product scores.

## Quick Start: Running the Notebook

The easiest way to use the autofill module is through the interactive Jupyter notebook:

### Running `run_autofill_pipeline.ipynb`

1. **Open the notebook** in Jupyter Lab/Notebook
2. **Run cells in order:**
   - **Cell 2**: Setup and Imports (configures paths, enables caching)
   - **Cell 3**: Load Data (loads machines, products, sales from Firestore or cache)
   - **Cell 4**: Generate Base Predictions (creates revenue predictions for all products)
   - **Cell 5**: Configuration (select machine, configure score weights)
   - **Cell 6**: Train Models and Add Static Scores (trains models, adds healthiness/location scores)
   - **Cell 7**: Run Autofill Workflow (fills empty slots)
   - **Cell 8**: Run Swap Workflow (optimizes full machine with swaps)
   - **Cell 10**: Review Results (displays final machine configuration)

### Testing Mode (Recommended for Development)

Set `TESTING_MODE = True` in Cell 2 to enable local caching:
- **First run**: Loads data from Firestore, generates predictions, trains models (~5-10 minutes)
- **Subsequent runs**: Loads from cache (~10 seconds)
- **Cache location**: `test_cache/` directory (gitignored)

**Quick iteration**: After changing score weights in Cell 5, just re-run Cell 7 - it reuses existing scores and recalculates rankings.

### Configuration

In **Cell 5**, configure:

```python
# Select machine by index
MACHINE_TO_SCORE = 12

# Configure score weights (0.0 = no impact, 1.0 = maximum impact)
STATIC_WEIGHTS = {
    'healthiness': 0.6,  # Product healthiness (Nutri-Score)
    'location': 1.0,    # Location fit (historical performance)
    'confidence': 0.8   # Prediction confidence
}

DYNAMIC_WEIGHTS = {
    'uniqueness': 1.0,  # Product diversity within machine
    'cousin': 0.0,      # Product complementarity
    'inventory': 0.8    # Inventory synchronization efficiency
}
```

## Core Module: `autofill.py`

The `autofill.py` module provides the core optimization logic. It's designed to be called programmatically or from the notebook.

### Architecture Overview

The module implements a two-phase optimization strategy:

1. **Phase 1: Fill Empty Slots** (`run_autofill_workflow`)
   - Identifies empty positions in the machine
   - Ranks all candidate products by final score
   - Fills each empty slot with the best-matching product (width must match)
   - Recalculates dynamic scores after each fill (if dynamic weights enabled)

2. **Phase 2: Optimize Full Machine** (`run_swap_workflow`)
   - Evaluates current products in the machine
   - Identifies low-scoring products
   - Swaps them with better alternatives (same width constraint)
   - Recalculates dynamic scores after each swap

### Key Functions

#### `run_autofill_workflow()`

**Purpose**: Fills all empty slots in a machine configuration with top-ranked products.

**Signature**:
```python
def run_autofill_workflow(
    machine_config: dict,
    predictions_df: pd.DataFrame,
    products_df: pd.DataFrame,
    static_weights: Dict[str, float],
    dynamic_weights: Dict[str, float],
    artifacts: Dict[str, Any]
) -> dict
```

**Parameters**:
- `machine_config` (dict): Machine configuration with `machine_key` and `slots` list
  - Each slot is a dict with `position`, `ean` (optional), `width`, etc.
- `predictions_df` (DataFrame): Revenue predictions for all products
  - Required columns: `machine_key`, `ean`, `predicted_weekly_revenue`
  - Optional columns: `healthiness_score`, `location_fit_score`, `confidence_score`, `price_mean`
- `products_df` (DataFrame): Product metadata
  - Required columns: `ean`, `product_name`, `category`, `subcategory`, `provider`, `width`
- `static_weights` (dict): Weights for static scores
  - Keys: `'healthiness'`, `'location'`, `'confidence'`
  - Values: `0.0` (no impact) to `1.0` (maximum impact)
- `dynamic_weights` (dict): Weights for dynamic scores
  - Keys: `'uniqueness'`, `'cousin'`, `'inventory'`
  - Values: `0.0` (no impact) to `1.0` (maximum impact)
- `artifacts` (dict): Trained models and mappings
  - Required keys: `'uniqueness_model'`, `'cousin_model'` (if dynamic weights > 0)
  - Optional: `'healthiness_mapping'`, `'location_mapping'`

**Returns**: Updated machine configuration dict (same structure as input, with filled slots)

**How It Works**:

1. **Setup**: Converts EANs to integers, filters predictions for the target machine, builds metadata maps
2. **Initial Ranking**: Calculates final scores for all candidate products
3. **Fill Loop**: For each empty slot:
   - Filters candidates by width (must match slot width)
   - Excludes products already in machine
   - If dynamic weights enabled, recalculates scores (to account for newly filled slots)
   - Selects top-ranked candidate (highest `final_score`)
   - Updates slot with product information (EAN, name, category, price, etc.)

**Example Usage**:
```python
from planogram.autofill import run_autofill_workflow

updated_machine = run_autofill_workflow(
    machine_config={
        'machine_key': 'My Machine_12345',
        'slots': [
            {'position': 'A0', 'ean': None, 'width': 1.0},
            {'position': 'A1', 'ean': 7340001805871, 'width': 1.0},
            # ... more slots
        ]
    },
    predictions_df=predictions_df,  # DataFrame with predictions
    products_df=products_df,        # DataFrame with product metadata
    static_weights={'healthiness': 0.6, 'location': 1.0, 'confidence': 0.8},
    dynamic_weights={'uniqueness': 1.0, 'cousin': 0.0, 'inventory': 0.8},
    artifacts=artifacts  # Trained models from train_all()
)
```

#### `run_swap_workflow()`

**Purpose**: Optimizes a fully-filled machine by swapping low-scoring products with better alternatives.

**Signature**:
```python
def run_swap_workflow(
    machine_config: dict,
    predictions_df: pd.DataFrame,
    products_df: pd.DataFrame,
    static_weights: Dict[str, float],
    dynamic_weights: Dict[str, float],
    artifacts: Dict[str, Any]
) -> dict
```

**Parameters**: Same as `run_autofill_workflow()`

**Returns**: Updated machine configuration with swapped products

**How It Works**:

1. **Setup**: Same as autofill workflow
2. **Score Current Products**: Calculates final scores for all products currently in machine
3. **Sort by Score**: Orders current products from lowest to highest score
4. **Swap Loop**: For each low-scoring product:
   - Finds candidates with higher `final_score`
   - Filters by width (must match slot width)
   - Excludes products already in machine
   - Swaps with best alternative
   - Updates slot metadata

**Example Usage**:
```python
from planogram.autofill import run_swap_workflow

# After filling empty slots, optimize the full machine
optimized_machine = run_swap_workflow(
    machine_config=updated_machine,  # From run_autofill_workflow()
    predictions_df=predictions_df,
    products_df=products_df,
    static_weights=STATIC_WEIGHTS,
    dynamic_weights=DYNAMIC_WEIGHTS,
    artifacts=artifacts
)
```

#### `recalculate_ranking_with_scores()`

**Purpose**: Calculates final ranking scores by combining revenue predictions with static and dynamic scores.

**Signature**:
```python
def recalculate_ranking_with_scores(
    predictions_df: pd.DataFrame,
    products_df: pd.DataFrame,
    machine_config: dict,
    static_weights: Dict[str, float],
    dynamic_weights: Dict[str, float],
    artifacts: Dict[str, Any],
    lookup_maps: Optional[dict] = None
) -> pd.DataFrame
```

**Parameters**:
- `predictions_df`: DataFrame with base predictions (must include `predicted_weekly_revenue`)
- `products_df`: Product metadata DataFrame
- `machine_config`: Current machine state (used for dynamic score calculation)
- `static_weights`: Weights for static scores
- `dynamic_weights`: Weights for dynamic scores
- `artifacts`: Trained models
- `lookup_maps`: Optional pre-computed lookup maps (for performance)

**Returns**: DataFrame with added `final_score` column, sorted by score (descending)

**How It Works**:

1. **Calculate Dynamic Scores**: Calls `predict_dynamic_autofill` to compute uniqueness, cousin, and inventory scores based on current machine state
2. **Calculate Final Score**: Combines revenue with score penalties/bonuses using the formula:
   ```
   final_score = predicted_weekly_revenue 
               × (1 - healthiness_penalty × weight × 0.60)
               × (1 - location_penalty × weight × 0.60)
               × (1 - confidence_penalty × weight × 0.60)
               × (1 + uniqueness_improvement × weight × 0.20)
               × (1 + inventory_improvement × weight × 0.20)
   ```
3. **Sort**: Returns DataFrame sorted by `final_score` (highest first)

**Example Usage**:
```python
from planogram.autofill import recalculate_ranking_with_scores

# Get ranked products for a machine
ranked_products = recalculate_ranking_with_scores(
    predictions_df=predictions_df[predictions_df['machine_key'] == 'My Machine_12345'],
    products_df=products_df,
    machine_config=current_machine_config,
    static_weights=STATIC_WEIGHTS,
    dynamic_weights=DYNAMIC_WEIGHTS,
    artifacts=artifacts
)

# Top 10 products by score
top_10 = ranked_products.head(10)
```

### Internal Helper Functions

#### `_setup_workflow_data()`

**Purpose**: Shared helper that prepares DataFrames and metadata maps for both workflows.

**What It Does**:
- Converts EANs to integers (handles precision for large EANs)
- Filters predictions for the target machine
- Builds metadata maps (EAN → product info)
- Maps product widths from `products_df`
- Enriches slots with product metadata

**Key Feature**: Uses `pd.to_numeric()` to preserve precision for 13-digit EANs (avoids float conversion issues).

#### `_calculate_dynamic_scores()`

**Purpose**: Calculates machine-dependent scores (uniqueness, cousin, inventory).

**What It Does**:
- Extracts unique products from predictions
- Maps EANs to product names/categories
- Calls `predict_dynamic_autofill` with current machine state
- Merges dynamic scores back into predictions DataFrame

**Note**: Only calculates scores for weights > 0 (skips if all dynamic weights are 0).

#### `_calculate_final_score()`

**Purpose**: Combines revenue with score penalties/bonuses to compute final ranking.

**Scoring Formula**:

**Static Scores** (healthiness, location, confidence):
- Penalty = `(1 - score²) × weight × 0.60`
- Applied multiplicatively: `final_score *= (1 - penalty)`
- Higher score = lower penalty = higher final score

**Dynamic Scores** (uniqueness, inventory):
- Uses `improvement` value (relative to baseline machine score)
- Normalized by maximum absolute improvement
- Bonus/Penalty = `(improvement / max_abs) × weight × 0.20`
- Applied multiplicatively: `final_score *= (1 + bonus)`

**Example**:
- Product with revenue 100 SEK
- Healthiness score 0.9, weight 0.6 → penalty = (1 - 0.9²) × 0.6 × 0.60 = 0.054
- Final = 100 × (1 - 0.054) = 94.6 SEK
- Uniqueness improvement +0.01, weight 1.0, max_abs = 0.05 → bonus = (0.01/0.05) × 1.0 × 0.20 = 0.04
- Final = 94.6 × (1 + 0.04) = 98.4 SEK

## Score Types Explained

### Static Scores

**Healthiness Score** (`healthiness_score`):
- Based on Nutri-Score (A=1.0, B=0.8, C=0.6, D=0.4, E=0.2)
- Same for all machines (product-level attribute)
- Higher score = healthier product

**Location Fit Score** (`location_fit_score`):
- Based on historical sales performance by location type
- Same for all machines of the same location type
- Higher score = better historical performance

**Confidence Score** (`confidence_score`):
- Prediction confidence from revenue model
- Higher score = more reliable prediction

### Dynamic Scores

**Uniqueness Score** (`uniqueness_score`, `uniqueness_improvement`):
- Measures product diversity within the machine
- Calculated based on current machine contents
- `improvement` = how much adding this product improves machine diversity
- Positive improvement = better diversity

**Cousin Score** (`cousin_score`, `cousin_improvement`):
- Measures product complementarity
- Products that sell well together get higher scores
- `improvement` = how much adding this product improves complementarity

**Inventory Score** (`inventory_score`, `inventory_improvement`):
- Measures inventory synchronization efficiency
- Products with similar inventory patterns get higher scores
- `improvement` = how much adding this product improves inventory alignment

**Note**: Dynamic scores use `improvement` values (relative to baseline) rather than absolute scores. This ensures the optimization considers how each product affects the overall machine balance.

## Data Requirements

### Machine Configuration Format

```python
machine_config = {
    'machine_key': 'My Machine_12345',  # String identifier
    'slots': [
        {
            'position': 'A0',           # Slot position (string)
            'ean': 7340001805871,       # EAN code (int/float, optional)
            'width': 1.0,              # Slot width (float, required)
            'product_name': '...',      # Optional, auto-filled
            'category': '...',         # Optional, auto-filled
            'price': 30.0               # Optional, auto-filled
        },
        # ... more slots
    ]
}
```

### Predictions DataFrame Format

Required columns:
- `machine_key` (str): Machine identifier
- `ean` (float/int): Product EAN code
- `predicted_weekly_revenue` (float): Predicted revenue in SEK

Optional columns (for scoring):
- `healthiness_score` (float): 0.0-1.0
- `location_fit_score` (float): 0.0-1.0
- `confidence_score` (float): 0.0-1.0
- `price_mean` (float): Average price in SEK

### Products DataFrame Format

Required columns:
- `ean` (float/int): Product EAN code
- `product_name` (str): Product name
- `category` (str): Product category
- `subcategory` (str): Product subcategory
- `provider` (str): Product provider
- `width` (float): Product width (for slot matching)

Optional columns:
- `purchase_price_kr` (float): Purchase price (used as price fallback)

## Complete Workflow Example

```python
from planogram.autofill import run_autofill_workflow, run_swap_workflow
from planogram.train_all import train_all
from planogram.predict_static import predict_static

# 1. Load data
sales_data = load_sales_data()
products_df = load_products()
machines_df = load_machines()

# 2. Generate base predictions
predictions_df, _ = predict_static(
    sales_df=sales_data,
    machines_df=machines_df,
    product_df=products_df,
    moaaz_trend_model_path=model_path,
    prediction_overrides={'moaaz_trend': True}
)

# 3. Train models and get artifacts
artifacts = train_all(
    sales_df=sales_data,
    machines_df=machines_df,
    products_df=products_df,
    product_information_df=product_info
)

# 4. Add static scores to predictions
static_df, _ = predict_static(
    sales_df=sales_data,
    machines_df=machines_df,
    product_df=products_df,
    location_mapping_df=artifacts.get('location_mapping'),
    healthiness_mapping=artifacts.get('healthiness_mapping'),
    prediction_overrides={'healthiness_mapping': True, 'location_mapping': True}
)
predictions_df = predictions_df.merge(static_df[['machine_key', 'ean', 'healthiness_score', 'location_fit_score']], 
                                     on=['machine_key', 'ean'], how='left')

# 5. Configure weights
STATIC_WEIGHTS = {'healthiness': 0.6, 'location': 1.0, 'confidence': 0.8}
DYNAMIC_WEIGHTS = {'uniqueness': 1.0, 'cousin': 0.0, 'inventory': 0.8}

# 6. Get target machine
target_machine = machines_df[machines_df['machine_key'] == 'My Machine_12345'].iloc[0].to_dict()

# 7. Fill empty slots
filled_machine = run_autofill_workflow(
    machine_config=target_machine,
    predictions_df=predictions_df,
    products_df=products_df,
    static_weights=STATIC_WEIGHTS,
    dynamic_weights=DYNAMIC_WEIGHTS,
    artifacts=artifacts
)

# 8. Optimize with swaps
optimized_machine = run_swap_workflow(
    machine_config=filled_machine,
    predictions_df=predictions_df,
    products_df=products_df,
    static_weights=STATIC_WEIGHTS,
    dynamic_weights=DYNAMIC_WEIGHTS,
    artifacts=artifacts
)

# 9. Use optimized_machine (in-memory only, no database writes)
```

## Performance Considerations

- **Prediction generation**: ~1-2 minutes for all machines (cache after first run)
- **Model training**: ~2-5 minutes (cache after first run)
- **Autofill execution**: < 1 second per machine
- **Score recalculation**: ~0.1-0.5 seconds per recalculation (when dynamic weights enabled)

**Optimization Tips**:
- Enable `TESTING_MODE = True` to cache expensive operations
- Re-run only the autofill cell after changing weights (reuses existing scores)
- Cache predictions and artifacts for faster iteration

## Safety & Data Integrity

- **Read-only by default**: All changes stored in memory only
- **Explicit save required**: No automatic database writes
- **Input immutability**: Functions copy input DataFrames to prevent mutations
- **Safe testing**: Cache utilities are testing-only and won't affect production

## Module Structure

```
autofill/
├── __init__.py              # Module exports
├── autofill.py              # Core autofill logic (this module)
├── cache_utils.py           # Testing utilities for caching
├── run_autofill_pipeline.ipynb  # Interactive notebook
├── README.md                # This file
└── test_cache/              # Local cache directory (gitignored)
```

## Dependencies

The autofill module integrates with other planogram modules:

- **`predict_static`**: Base predictions (moaaz_trend) and static scores
- **`predict_dynamic_autofill`**: Dynamic scores (uniqueness, cousin, inventory)
- **`train_all`**: Model training (uniqueness, cousin, healthiness, location)
- **`data_loader`**: Data loading utilities

## Troubleshooting

### No candidates found for empty slots

**Possible causes**:
- No predictions for the target machine (check `machine_key` matches)
- Width mismatch (candidates don't match slot width)
- All products already in machine

**Solution**: Check that `predictions_df` contains rows for your `machine_key`, and that products with matching widths exist.

### Dynamic scores not updating

**Possible causes**:
- Dynamic weights are all 0 (no dynamic scoring enabled)
- `artifacts` missing required models (`uniqueness_model`, `cousin_model`)
- `predict_dynamic_autofill` not returning improvement columns

**Solution**: Ensure `DYNAMIC_WEIGHTS` has at least one weight > 0, and that `artifacts` contains trained models.

### EAN mismatch errors

**Possible causes**:
- EANs stored as strings vs integers
- Precision loss from float conversion

**Solution**: The module uses `pd.to_numeric()` to handle EAN conversion safely. Ensure EANs are numeric (int/float) in your DataFrames.

## Advanced Usage

### Custom Score Weights

Experiment with different weight combinations:

```python
# Revenue-focused (minimal score impact)
STATIC_WEIGHTS = {'healthiness': 0.0, 'location': 0.2, 'confidence': 0.0}
DYNAMIC_WEIGHTS = {'uniqueness': 0.0, 'cousin': 0.0, 'inventory': 0.0}

# Health-focused
STATIC_WEIGHTS = {'healthiness': 1.0, 'location': 0.5, 'confidence': 0.3}
DYNAMIC_WEIGHTS = {'uniqueness': 0.5, 'cousin': 0.0, 'inventory': 0.2}

# Diversity-focused
STATIC_WEIGHTS = {'healthiness': 0.4, 'location': 0.6, 'confidence': 0.5}
DYNAMIC_WEIGHTS = {'uniqueness': 1.0, 'cousin': 0.8, 'inventory': 0.6}
```

### Batch Processing

Process multiple machines:

```python
machines_to_optimize = machines_df[machines_df['slots'].notna()].head(10)

for idx, machine_row in machines_to_optimize.iterrows():
    machine_config = machine_row.to_dict()
    
    filled = run_autofill_workflow(
        machine_config=machine_config,
        predictions_df=predictions_df,
        products_df=products_df,
        static_weights=STATIC_WEIGHTS,
        dynamic_weights=DYNAMIC_WEIGHTS,
        artifacts=artifacts
    )
    
    optimized = run_swap_workflow(
        machine_config=filled,
        predictions_df=predictions_df,
        products_df=products_df,
        static_weights=STATIC_WEIGHTS,
        dynamic_weights=DYNAMIC_WEIGHTS,
        artifacts=artifacts
    )
    
    # Save or process optimized machine
    print(f"Optimized {machine_config['machine_key']}")
```

## Future Improvements

- **EAN Fallback**: Support product_name as fallback identifier
- **Batch Processing**: Process multiple machines in parallel
- **Incremental Updates**: Only recalculate scores for affected products
- **Validation**: Add validation for slot width compatibility and product availability
