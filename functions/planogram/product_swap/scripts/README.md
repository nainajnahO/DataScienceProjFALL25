# Setup Scripts - One-Time Data Preparation

This folder contains scripts that need to be run **once** to prepare data and train models. These are not needed for frontend runtime.

## Scripts Overview

### Data Preparation (Run in order):

1. **`aggregate_sales_with_profit.py`**
   - Processes sales data from 2020-2025
   - Adds profit and revenue calculations
   - Output: `../data/sales/Sales_{year}_with_profit.parquet` and `../data/aggregates/product_profit_revenue.parquet`

2. **`detect_swaps.py`**
   - Detects product swaps from machine snapshots
   - Output: `../data/swaps/product_swaps.parquet`

3. **`enrich_swaps_with_profit_revenue.py`**
   - Enriches swap data with historical profit/revenue metrics
   - Output: `../data/swaps/product_swaps_enriched.parquet`

### Model Training:

4. **`train_swap_model.py`**
   - Trains XGBoost models for swap prediction
   - Output: `../models/` (revenue_model.pkl, success_model.pkl, encoders.pkl, feature_columns.pkl)

5. **`train_swap_model_lgbm.py`**
   - Alternative: Trains LightGBM models
   - Output: `../models_lgbm/` (if you prefer LightGBM)

### Analysis & Validation:

6. **`analyze_profit_revenue.py`**
   - Analyzes product profit/revenue data
   - For exploration and validation only

7. **`eda_swap_analysis.py`**
   - Exploratory data analysis of swaps
   - For research/analysis only

8. **`verify_data.py`**
   - Validates data quality
   - Use before training models

## Running Setup

### Option 1: Run All Scripts at Once (Recommended)

Run the complete pipeline from the project root:

```bash
# Run all steps in order
python scripts/run_pipeline.py

# Skip steps if output files already exist
python scripts/run_pipeline.py --skip-existing

# Skip model training (only prepare data)
python scripts/run_pipeline.py --skip-training

# Only train models (assumes data is already prepared)
python scripts/run_pipeline.py --only-training
```

### Option 2: Run Scripts Individually

Run from the project root (`product_swap/`):

```bash
# 1. Aggregate sales data
python scripts/aggregate_sales_with_profit.py

# 2. Detect swaps
python scripts/detect_swaps.py

# 3. Enrich swaps
python scripts/enrich_swaps_with_profit_revenue.py

# 4. Train models
python scripts/train_swap_model.py
```

**Note:** All paths are automatically resolved relative to the project root, so these scripts can be run from any location or the project root.

