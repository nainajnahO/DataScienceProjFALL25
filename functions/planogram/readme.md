# Planogram Functions

This directory contains logic for planogram optimization, including product scoring, location fit analysis, and data loading.

## Architecture: Train vs. Use (Inference)

All scoring and modeling modules in this package follow a strict **Train/Predict** separation pattern. This allows expensive training operations (calculating embeddings, aggregating historical statistics) to be performed independently and their results (artifacts) to be saved/cached. The "Predict" (or "Use") functions are then lightweight and can be called on-demand using the pre-calculated artifacts.

When implementing new features, adhere to this pattern:

### 1. Training Phase (`train_...`)
- **Input:** Raw data (full sales history, product catalog, etc.).
- **Operation:** Heavy computation, model fitting, statistical aggregation.
- **Output:** A serializable "Model Artifact" (e.g., `dict`, `tuple`, or trained sklearn model).
- **Naming Convention:** `train_<feature_name>_model(...)`

### 2. Inference Phase (`predict_...`)
- **Input:** Target data to score (e.g., a specific machine's current products) AND the `Model Artifact`.
- **Operation:** Fast lookups, dot products, or lightweight transformations using the artifact.
- **Output:** Scores or recommendations.
- **Naming Convention:** `predict_<feature_name>_scores(...)` or `calculate_<feature_name>(..., trained_model)`
- **⚠️ CRITICAL:** All `predict_*` and `calculate_*` functions **MUST** copy input DataFrames at the start (e.g., `df = input_df.copy()`) to avoid mutating the original data. This prevents side effects and ensures data integrity.

---

## Modules

### `location_scoring.py`
Calculates how well a product fits a specific location type based on historical performance.
- **Train:** `train_location_model(...)` -> Returns a dictionary of performance DataFrames per location type.
- **Predict:** `predict_location_scores(...)` -> Generates a score matrix for products given the trained model.

### `product_scoring.py`
Calculates uniqueness and diversity of products within a machine.
- **Train:** `train_uniqueness_model(...)` -> Returns global product embeddings and index map.
- **Predict:** `predict_uniqueness_scores(...)` -> Calculates scores for a machine's current contents using pre-calculated embeddings.

### `embed_products.py`
Handles text vectorization of products.
- **Train:** `train_embedding_model(...)` -> Returns trained SentenceTransformer or TF-IDF pipeline.
- **Use:** `generate_embeddings(...)` -> Transforms text to vectors using the artifact.

### `config.py`
Central configuration module containing:
- Firebase collection names (`FIREBASE_COLLECTIONS`)
- Data paths (processed sales data, Swedish holidays)
- Artifacts directory path (`ARTIFACTS_DIR`) for saving/loading trained model artifacts
- Output collection names (`OUTPUT_COLLECTION_STATIC`, `OUTPUT_COLLECTION_LETTER_GRADE_MAPPING`)
- Snapshot configuration parameters (`SNAPSHOT_N_DAYS`, `SNAPSHOT_DATE_INTERVAL_DAYS`, `SNAPSHOT_MIN_SALES`)
- Training and prediction task toggles (`TRAINING_TASKS`, `PREDICTION_STATIC_TASKS`, `PREDICTION_DYNAMIC_TASKS`)

### `data_loader.py`
Utilities for loading data from Firestore and Parquet files.
- **`load_firebase_collections()`** -> Loads machines, products, purchase prices, and categories from Firestore
- **`load_processed_sales()`** -> Loads historical sales data from Parquet files
- **`load_all_sales_data()`** -> Combines Firestore and Parquet data sources
- **`enrich_with_purchase_prices()`** -> Enriches product data with purchase price information
- Includes data cleaning utilities (removing unspecific rows, cleaning machine slots, category mapping)

### `artifact_utils.py`
Utility functions for saving and loading trained model artifacts to/from local files. Artifacts are saved locally so they can be accessed by frontend code.
- **`save_artifacts(artifacts, base_path)`** -> Saves all artifacts from `train_all()` to local files (supports DataFrames, numpy arrays, dictionaries, JSON, pickle)
- **`load_artifacts(artifact_names, base_path)`** -> Loads artifacts from local files (can discover all artifacts if names not specified)
- **`ensure_artifacts_dir()`** -> Creates artifacts directory if it doesn't exist
- Handles various artifact types: DataFrames (parquet), numpy arrays (NPZ), dictionaries of DataFrames (subdirectories with manifests), JSON mappings, and complex objects (pickle)
- Supports both single-file artifacts and directory-based artifacts (e.g., `cousin_model` saved as subdirectory with multiple parquet files)

### `machine_snapshots.py`
Creates point-in-time snapshots of machine configurations showing which products are in which positions and their sales performance.
- **`get_available_machines()`** -> Retrieves list of available machines
- **`generate_snapshot(...)`** -> Generates a single point-in-time snapshot for a specific machine
- **Train:** `train_snapshot_model(...)` -> Returns a DataFrame of snapshots for multiple machines and dates
- **Predict:** `predict_snapshot_metrics(...)` -> Analyzes snapshot artifacts to produce summary metrics (e.g., n_unique_products, avg_sales_per_position)

### `product_filters.py`
Product filtering and tagging system for dietary and product attributes.
- **Filter Functions:** `filter_gluten_free()`, `filter_sugar_free()`, `filter_lactose_free()`, `filter_vegan()`, `filter_vegetarian()`, `filter_no_nuts()`, `filter_no_energy_drinks()`, `filter_no_fresh_food()`
- **`evaluate_filters_batch()`** -> Evaluates filters for multiple products and returns accumulated statistics
- **`apply_all_filters()`** -> Applies all defined filters to a products DataFrame, enriching with data from products.json
- Uses keyword matching and nutrient analysis to determine product attributes
- Returns a list of applicable filter tags for each product

### `inventory_score.py` (single-file module)
Calculates inventory synchronization scores for machines using **precomputed predictions**.
- **Input:** `machines_df` with `slots` (product_name/position/optional ean), `products_df` (must include `spiral`, optional `product_name`), and `predictions_df` (from `moaaz_predict`).
- **Mapping:** Falls back to `product_name -> ean` using `map_ean_to_product_name` logic when slots lack EANs.
- **Output:** Float for a single machine or DataFrame with `machine_key` and `inventory_score`.
- **Note:** No training phase or I/O; assumes predictions are already provided.

### `cousin_scoring.py`
Association-rule-based “cousin product” scoring.
- **Train:** `train_cousin_model(...)` -> Builds confidence matrices per subgroup from co-purchased products.
- **Predict:** `predict_cousin_scores(...)` -> Assigns product/machine-level cousin scores using the trained matrices.
- Outputs mean confidence per product and machine-level aggregates (`avg_mean_cousin_score`, `cousin_fraction`).

### `healthiness_model.py`
Nutri-Score training and application.
- **Train:** `train_healthiness_model(...)` -> Extracts nutrition, calculates Nutri-Score letters, and maps normalized EAN → letter grade.
- **Predict:** `predict_healthiness_scores(...)` -> Applies the mapping to machines/products, producing product-level and machine-level healthiness scores.
- Uses helpers from `healthiness_scoring.py` for nutrition extraction and Nutri-Score calculation.

### `utils.py`
Firestore utility functions for committing data.
- **`commit_to_firestore()`** -> Commits a DataFrame to a Firestore collection in batches using shared utility functions

### `prediction_aggregator.py`
Aggregates sales predictions from multiple reference locations using similarity-weighted averaging.
- **Function:** `aggregate_predictions_weighted(...)` -> Returns aggregated predictions with one row per product.
- **Usage:** Used to combine predictions from multiple similar machines, giving more weight to machines that are geographically and structurally similar to the target location.

### `LocationBasedForecasting/`
Advanced location analytics for finding similar machines.
- **`geo_similarity.py`**: Calculates similarity between a target coordinates and existing machines based on:
    - Distance (haversine)
    - Proximity to ICA stores
    - Proximity to companies (using employee counts)
    - Machine type matching

### Example Usage: Location-Based Prediction Pipeline
This pipeline finds similar machines, filters predictions, and aggregates them using weighted averages.

```python
# 1. IMPORTS
from LocationBasedForecasting.geo_similarity import calculate_similarity_to_multiple
from planogram import aggregate_predictions_weighted, load_ica_stores, load_companies

# Load standardized data
ica_stores_df = load_ica_stores() 
companies_df = load_companies()

# 2. RUN SIMILARITY
all_similarities = calculate_similarity_to_multiple(
    target_lat=59.3293,
    target_lon=18.0686,
    reference_locations_df=machines_df,
    ica_stores_df=ica_stores_df,
    companies_df=companies_df,
    distance_weight=0.1,
    ica_weight=0.45,
    company_weight=0.45
)

# 3. FILTER
# Select only the 'machine_key' column for filtering
top_10_keys = all_similarities.head(10)['machine_key'] 
filtered_static_predictions = static_predictions[
    static_predictions['machine_key'].isin(top_10_keys)
]

# 4. AGGREGATE (Weighted Average)
aggregated_predictions = aggregate_predictions_weighted(
    predictions_df=filtered_static_predictions,
    similarity_df=all_similarities,
    product_col='product_name',
    machine_key_col='machine_key',
    similarity_col='similarity'
)

aggregated_predictions.head()
```

### `moaaz_trend/`
Multi-week sales forecasting using machine learning models. This module provides a complete pipeline for training and predicting sales forecasts across multiple weeks.

**Main Entry Points:**
- **Train:** `moaaz_train()` in `train.py` -> Trains forecasting models and saves artifacts
- **Predict:** `moaaz_predict()` in `predict.py` -> Generates sales predictions for ALL (machine_key, ean) combinations for future weeks using trained models. Requires `product_df` to specify which products to predict for

**Core Functions:**
- **`prepare_prediction_dataframe()`** in `predict.py` -> Prepares prediction dataframe by creating rows for ALL (machine_key, ean) combinations. Handles metadata filling, price imputation, and sets `confidence_score` values
- **`map_ean_to_product_name()`** in `predict.py` -> Maps EAN codes to product names using a product dataframe. Handles duplicate EANs gracefully

**Key Features:**
- **Universal Predictions:** `moaaz_predict()` generates predictions for ALL (machine_key, ean) combinations from `product_df`, not just products with historical sales. This enables recommendations for new products
- **Cold-Start Handling:** Uses hierarchical fallback logic for products without history (subcategory → category → zero) to generate valid features
- **Price Imputation:** Multi-tier price imputation strategy:
  1. Uses machine-specific prices from historical data
  2. Falls back to average price by (machine_sub_group, ean) combination
  3. Prices vary by machine subcategory, ensuring accurate imputation
- **Confidence Score:** `confidence_score` column is a normalized score (0.0-1.0) representing sales history representativeness. Combines:
  - **Spatial Coverage (50%):** Number of machines that have sold the product (10+ machines = 1.0)
  - **Temporal Span (50%):** Duration over which product has been sold, from first sale to last sale (52+ weeks = 1.0)
  - Uses full history (no time restriction) to determine overall representativeness. Higher scores indicate more reliable predictions based on established sales patterns across multiple machines over longer periods

**Submodules:**
- **`config.py`**: Configuration for moaaz_trend module
  - Data paths (raw, processed, splits, models)
  - Date ranges and essential columns
  - Model parameters (XGBoost, prediction settings)
  - Feature engineering flags

- **`models/`**: Core forecasting models
  - `multi_week_forecaster.py`: Main ensemble model that combines multiple week-specific models. Supports direct_multi, recursive_single, and recursive_multi strategies
  - `naive_baseline.py`: Hierarchical naive baseline model for comparison
  - `save_utils.py`: Utilities for saving/loading model artifacts with versioning
  - `utils.py`: Model utility functions (weekly metrics, strategy comparison, error analysis, prediction intervals, business recommendations)

- **`features/`**: Feature engineering
  - `tier1_features.py`: Primary feature creation functions (lag features, rolling statistics, price positioning, machine context, historical sales, product lifecycle, brand features, etc.). Creates 20+ features derived from `price_mean`, making it critical for model performance
  - `feature_registry.py`: Registry for managing feature definitions and grouping (BASE, TEMPORAL, PRODUCT, MACHINE, HISTORICAL_SALES, etc.)

- **`data/`**: Data processing and preparation
  - `processor.py`: Main data processing pipeline (cleaning, weekly aggregation, product swap detection, snapshot creation, stale product/machine removal, working days calculation)
  - `splitter.py`: Time-series data splitting utilities with multiple methods (method_1_all_data_to_new_products, method_2_new_products_only, method_3_first_weeks_only, method_4_vendtrend_comparison, method_5_august_2024_temporal)

- **`utils/`**: Supporting utilities
  - `calendar.py`: Swedish holiday calendar for working days calculation
  - `feature_analysis.py`: Feature analysis tools (correlation analysis, feature importance, redundant feature removal, feature group evaluation)
  - `tier1_evaluation.py`: Evaluation metrics and comparison tools for tier1 features, model training utilities, and performance visualization

---

## Important Notes

### DataFrame Immutability
⚠️ **CRITICAL RULE:** All `predict_*` and `calculate_*` functions **MUST** copy input DataFrames at the start of the function to avoid mutating the original data. This prevents unexpected side effects and ensures data integrity.

**Example:**
```python
def predict_something(df: pd.DataFrame, model) -> pd.DataFrame:
    df = df.copy()  # Always copy first!
    # ... rest of function logic ...
    return df
```

**Why this matters:**
- Prevents accidental modification of input data that may be used elsewhere
- Ensures functions are idempotent and safe to call multiple times
- Makes debugging easier by isolating function behavior
- Follows functional programming best practices

### EAN Fallback Requirement
⚠️ **Future Improvement Needed:** Currently, the `moaaz_trend` forecaster and related functions require the `ean` column to exist and will filter out or fail on items without `ean`. When implementing future fixes or new features, **ensure all functions (including `moaaz_trend`) have fallback logic to handle missing `ean` values** (e.g., using `product_name` as an alternative identifier). This will improve robustness and allow prediction on items that don't have EAN codes. Most importantly giving bigger training data.
