# IMPORTANT TO KNOW

This version currently have completely mocked profit and revenue. This is due to missing data of purchase prices of products. This needs to be adressed in the future if this is going to be used.

This is a very basic version of the product swap prediction system. It is currently only used for testing and development, it was not used in the final project.

# Product Swap Prediction

Predicts product swap outcomes and provides recommendations for vending machine planograms. Uses **profit as the primary metric** to ensure recommendations reflect actual business value.

---

## Step 1: Setup (One-Time)

### Prerequisites

Make sure you have:
- Python 3.9+ installed
- Sales data files in `../moaaz-prod/data/raw/` (Sales_2020.parquet through Sales_2025.parquet)
- Machine snapshots file at `../machine_snapshots/data/machine_snapshots.parquet`

### Install Dependencies

```bash
# Navigate to project
cd projects/planogram/product_swap

# Activate virtual environment (or create one)
source ../venv/bin/activate  # Linux/Mac
# OR
..\venv\Scripts\activate     # Windows

# Install required packages
pip install pandas pyarrow xgboost scikit-learn joblib numpy tqdm
```

### Run Setup Pipeline

Run all setup scripts in one command:

```bash
python scripts/run_pipeline.py
```

This will automatically:
1. Process sales data (add profit/revenue columns)
2. Detect product swaps from snapshots
3. Enrich swaps with financial metrics
4. Train prediction models

**Time:** This may take 10-30 minutes depending on your system.

**Options:**
- `--skip-existing`: Skip steps if output files already exist
- `--skip-training`: Only prepare data, skip model training
- `--only-training`: Only train models (if data is already prepared)

---

## Step 2: Use the Prediction System

### Quick Example

```python
from runtime.predict_swap_outcome import SwapPredictor

# Initialize predictor
predictor = SwapPredictor()

# Predict swap outcome
result = predictor.predict(
    product_before="7310070961234",  # EAN code of current product
    product_after="7310070961241"    # EAN code of replacement
)

# Check results
print(f"Profit Change: {result['predicted_profit_change']:+.2f} SEK")
print(f"Revenue Change: {result['predicted_revenue_change']:+.2f} SEK")
print(f"Success Probability: {result['success_probability']:.1%}")
print(f"Recommendation: {result['recommendation']}")
```

### Integration Example

```python
from runtime.predict_swap_outcome import SwapPredictor

def get_recommendations(current_product_ean, candidate_products):
    """Get top swap recommendations for a product."""
    predictor = SwapPredictor()
    recommendations = []
    
    for candidate_ean in candidate_products:
        if candidate_ean == current_product_ean:
            continue
        
        result = predictor.predict(current_product_ean, candidate_ean)
        recommendations.append({
            'product_ean': candidate_ean,
            'profit_change': result['predicted_profit_change'],
            'revenue_change': result['predicted_revenue_change'],
            'success_probability': result['success_probability'],
            'recommendation': result['recommendation']
        })
    
    # Sort by profit change (primary metric)
    recommendations.sort(key=lambda x: x['profit_change'], reverse=True)
    return recommendations

# Usage
candidates = ["7310070961234", "7310070961241", "7310070961258"]
recommendations = get_recommendations("7310070961200", candidates)
print(recommendations[0])  # Best recommendation
```

---

## What You Get

### Predictions Include:

- **`predicted_profit_change`** (PRIMARY) - Expected profit change in SEK
- **`predicted_revenue_change`** (Secondary) - Expected revenue change in SEK  
- **`success_probability`** - Likelihood of success (0.0 to 1.0)
- **`success_prediction`** - Binary recommendation (True/False)
- **`confidence`** - Prediction confidence (Low/Medium/High)
- **`recommendation`** - Human-readable recommendation text

### Key Files Created:

- **`data/sales/Sales_{year}_with_profit.parquet`** - Transaction-level sales with profit
- **`data/swaps/product_swaps_enriched.parquet`** - Historical swap data
- **`models/profit_model.pkl`** - Profit prediction model (PRIMARY)
- **`models/revenue_model.pkl`** - Revenue prediction model
- **`models/success_model.pkl`** - Success classification model

---

## Project Structure

```
product_swap/
├── runtime/          # Frontend integration code
│   └── predict_swap_outcome.py
├── scripts/          # Setup scripts (run once)
│   └── run_pipeline.py  # Run all setup scripts
├── data/             # Generated data files
│   ├── sales/        # Sales with profit/revenue
│   ├── swaps/        # Swap detection data
│   └── aggregates/   # Product aggregates
└── models/           # Trained ML models
```

---

## More Information

- **Frontend Integration:** See [`runtime/README.md`](runtime/README.md)
- **Setup Details:** See [`scripts/README.md`](scripts/README.md)
- **Model Usage:** See [`MODEL_USAGE.md`](MODEL_USAGE.md)

---

## Troubleshooting

**Models not found?**  
Run the setup pipeline: `python scripts/run_pipeline.py`

**Missing data files?**  
Ensure sales data files exist in `../moaaz-prod/data/raw/` and snapshots in `../machine_snapshots/data/`

**Import errors?**  
Install dependencies: `pip install pandas pyarrow xgboost scikit-learn joblib numpy tqdm`
