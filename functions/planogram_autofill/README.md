# Planogram Autofill Pipeline

This module trains an **XGBoost sales prediction model** using **planogram features** (uniqueness scores and location fit scores) as inputs. It leverages the `planogram` package's trained models and utilities following the Train/Predict pattern.

## Folder Structure

```
planogram_autofill/
├── __init__.py              # Package marker
├── config.py                # Configuration (paths, parameters)
├── snapshot_builder.py      # Builds weekly snapshots from sales data
├── feature_builder.py       # Enriches snapshots with planogram scores
├── sales_modeling.py        # Trains XGBoost model for sales prediction
├── model_trainer.py         # Trains planogram models (location/uniqueness)
├── pipeline.py              # CLI entry point (orchestrates all steps)
├── run_pipeline.ipynb       # Interactive Jupyter notebook
└── README.md                # This file
```

**Core Components:**
- `pipeline.py` - Main entry point with CLI commands
- `snapshot_builder.py` - Aggregates sales into weekly snapshots
- `feature_builder.py` - Adds planogram scores as features
- `sales_modeling.py` - Trains global XGBoost model
- `model_trainer.py` - Trains planogram models using planogram package

## Architecture

**Key Principle:** `planogram_autofill` **calls planogram package functions** to train models and use them as features.

### Complete Pipeline Flow

1. **Train Planogram Models** (using planogram package functions):
   - Call `planogram.product_scoring.train_uniqueness_model()` → Creates uniqueness embeddings
   - Call `planogram.location_scoring.train_location_model()` → Creates location fit lookup tables
   - Models saved to `output/models/`

2. **Snapshots** → Aggregate sales data into weekly machine/position snapshots (using `planogram.data_loader`)

3. **Features** → Enrich snapshots with planogram scores:
   - **Uniqueness scores**: Uses `planogram.product_scoring.predict_uniqueness_scores()` with trained model
   - **Location fit scores**: Uses `planogram.location_scoring.predict_location_scores()` with trained model

4. **Train XGBoost** → Train a global XGBoost model for sales prediction using planogram features

## What This Module Uses from `planogram`

✅ **Model Training Functions:**
- `planogram.product_scoring.train_uniqueness_model()` - Trains uniqueness embeddings model
- `planogram.location_scoring.train_location_model()` - Trains location fit lookup tables

✅ **Data Loading:**
- `planogram.data_loader.load_processed_sales()` - Loads sales from parquet files
- `planogram.data_loader.load_all_sales_data()` - Loads sales from Firestore
- `planogram.data_loader.load_firebase_collections()` - Loads products/machines from Firestore

✅ **Feature Generation (Predict Phase):**
- `planogram.product_scoring.predict_uniqueness_scores()` - Computes uniqueness scores using trained embeddings
- `planogram.location_scoring.predict_location_scores()` - Computes location fit scores using trained models
- `planogram.product_filters.apply_all_filters()` - Filters products before feature generation

✅ **Configuration:**
- `planogram.config` - Shared paths and settings

## What This Module Does NOT Do

❌ Does NOT write to Firestore - all outputs are local  
❌ Does NOT modify planogram package code

## Quickstart

### Option 1: Run Everything at Once

```bash
cd uno/functions
python -m planogram_autofill.pipeline run-all --years 2023 2024 --groups CORE TEMPORAL UNIQUENESS
```

**Note:** This assumes planogram models already exist. If not, train them first (see Option 2, Step 0).

### Option 2: Step-by-Step

0. **Train planogram models** (using planogram package functions):
   ```bash
   python -m planogram_autofill.pipeline train-planogram-models
   ```
   This calls:
   - `planogram.location_scoring.train_location_model()` 
   - `planogram.product_scoring.train_uniqueness_model()`
   
   Models are saved to `output/models/` and will be automatically used by feature builders.

1. **Build snapshots** (aggregates sales into weekly snapshots):
   ```bash
   python -m planogram_autofill.pipeline snapshots --years 2023 2024
   ```

2. **Build features** (enriches with planogram scores):
   ```bash
   python -m planogram_autofill.pipeline features --groups CORE TEMPORAL UNIQUENESS
   ```
   This automatically:
   - Loads trained planogram models from `output/models/` (if available)
   - Uses `planogram.product_scoring.predict_uniqueness_scores()` to compute scores
   - Uses `planogram.location_scoring.predict_location_scores()` to compute scores
   - Falls back to pre-computed parquet files if models aren't available

3. **Train XGBoost model**:
   ```bash
   python -m planogram_autofill.pipeline train --groups CORE TEMPORAL UNIQUENESS
   ```

### Option 3: Interactive Notebook

Use `run_pipeline.ipynb` for interactive execution and inspection.

## How Planogram Features Are Used

### Uniqueness Scores
- **Source**: `planogram.product_scoring.predict_uniqueness_scores()`
- **Computed using**: Trained embedding model (SentenceTransformer or TF-IDF)
- **Features added**: `uniqueness_score`, `category_diversity_score`

### Location Fit Scores  
- **Source**: `planogram.location_scoring.predict_location_scores()`
- **Computed using**: Trained location performance model (lookup tables)
- **Features added**: `location_fit_score`

### Auto-Loading Models

The feature builders automatically:
1. Check for trained model artifacts in `output/models/`
2. Load them if available and use planogram's `predict_*` functions:
   - `planogram.product_scoring.predict_uniqueness_scores()`
   - `planogram.location_scoring.predict_location_scores()`
3. Fall back to pre-computed parquet files if models don't exist

## Prerequisites

### Required (for full pipeline):
- Sales data: `uno/functions/data/processed/Sales_*.parquet`
- Firebase credentials: `uno/functions/serviceAccountKey.json` (for loading products/machines)
- Planogram trained models (optional): `planogram_autofill/output/models/location_model.pkl` and `uniqueness_model.pkl`

### Training Planogram Models:
- Use `python -m planogram_autofill.pipeline train-planogram-models` 
- Or use `run_pipeline.ipynb` (Section 0)

Both call planogram package functions:
- `planogram.product_scoring.train_uniqueness_model()`
- `planogram.location_scoring.train_location_model()`

## Outputs

All outputs are saved locally in `planogram_autofill/output/`:

- **Snapshots**: `output/snapshots/machine_weekly_snapshots.parquet`
- **Features**: `output/features/machine_weekly_features.parquet`  
- **XGBoost Model**: `output/models/global_sales_model.joblib`
- **Metrics**: `output/metrics/global_model_metrics.parquet`

## Feature Groups

- **CORE**: Product metadata (price, category, subcategory, provider)
- **TEMPORAL**: Time-based features (lookback_weeks)
- **UNIQUENESS**: Product uniqueness and diversity scores (from planogram)
- **LOCATION**: Location fit scores (from planogram)

## Dependencies

This module depends on the `planogram` package for:
- Data loading utilities
- Feature scoring functions (predict phase)
- Product filtering
- Shared configuration
