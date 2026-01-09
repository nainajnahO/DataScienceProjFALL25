# How to Use the Swap Prediction Models

This guide explains how to train and use the swap prediction models to predict the outcome of product swaps.

## Step 1: Train the Models

First, train the models using the best parameters from previous results:

```bash
cd projects/planogram/product_swap
python train_swap_model.py
```

This will:
1. Load the enriched swap data
2. Create training features
3. Load best parameters from `results/` directory
4. Train three models (revenue, profit, and success classification)
5. Save models to `models/` directory:
   - `revenue_model.pkl` - Predicts revenue change (SEK) - Secondary metric
   - `profit_model.pkl` - Predicts profit change (SEK) - **PRIMARY METRIC** ⭐
   - `success_model.pkl` - Predicts success probability (0-1) - Based on profit
   - `encoders.pkl` - Label encoders for categorical features
   - `feature_columns.pkl` - Feature column names

**Note:** Profit is the primary metric for swap recommendations as it reflects actual business value.

## Step 2: Use the Models for Predictions

### Quick Example

```python
from predict_swap_outcome import SwapPredictor

# Initialize predictor (loads models from 'models/' directory)
predictor = SwapPredictor(model_dir='models')

# Define a potential swap
swap = {
    'subcategory_before': 'Läsk & Saft',
    'subcategory_after': 'Läsk & Saft',
    'provider_before': 'Coca-Cola',
    'provider_after': 'PepsiCo',
    'revenue_before_4w': 150.0,      # Revenue in SEK over 4 weeks
    'profit_before_4w': 60.0,        # Profit in SEK over 4 weeks
    'sales_count_before_4w': 20,     # Number of sales in 4 weeks
    'days_observed_before_4w': 28    # Days of observation (default: 28)
}

# Make prediction
result = predictor.predict(swap)

print(f"Predicted Profit Change:  {result['predicted_profit_change']:+.2f} SEK (PRIMARY ⭐)")
print(f"Predicted Revenue Change: {result['predicted_revenue_change']:+.2f} SEK (secondary)")
print(f"Success Probability:      {result['success_probability']:.1%}")
print(f"Success Prediction:       {'✓ Yes' if result['success_prediction'] else '✗ No'}")
print(f"Confidence:               {result['confidence']}")
print(f"Recommendation:           {result['recommendation']}")
```

### Run the Example Script

```bash
python predict_swap_outcome.py
```

## Input Data Format

The `swap` dictionary must contain:

### Required Fields:
- **`subcategory_before`** (str): Subcategory of the product being removed
  - Examples: `'Läsk & Saft'`, `'Snacks'`, `'Kaffe'`, etc.
- **`subcategory_after`** (str): Subcategory of the product being added
- **`provider_before`** (str): Provider/brand of the product being removed
  - Examples: `'Coca-Cola'`, `'PepsiCo'`, `'Red Bull'`, etc.
- **`provider_after`** (str): Provider/brand of the product being added
- **`revenue_before_4w`** (float): Total revenue (SEK) from the current product over 4 weeks
- **`profit_before_4w`** (float): Total profit (SEK) from the current product over 4 weeks
- **`sales_count_before_4w`** (int): Number of sales transactions in the 4-week period

### Optional Fields:
- **`days_observed_before_4w`** (int): Number of days observed (default: 28)

## Output Format

The `predict()` method returns a dictionary with:

- **`predicted_profit_change`** (float): Predicted change in profit in SEK - **PRIMARY METRIC** ⭐
  - Positive = profit increase expected (recommended)
  - Negative = profit decrease expected (not recommended)
- **`predicted_revenue_change`** (float): Predicted change in revenue in SEK - Secondary metric
  - Useful for understanding sales volume impact
- **`success_probability`** (float): Probability of success (0.0 to 1.0) - Based on profit
  - > 0.5 = likely to succeed (profit increase)
  - < 0.5 = likely to fail (profit decrease)
- **`success_prediction`** (bool): Binary prediction (True = success, False = failure)
- **`confidence`** (str): Confidence level
  - `"High"`: Probability > 0.7 or < 0.3
  - `"Medium"`: Probability between 0.6-0.7 or 0.3-0.4
  - `"Low"`: Probability between 0.4-0.6
- **`recommendation`** (str): Human-readable recommendation based on profit and revenue
  - Examples: "✅ Recommended - Increases both profit and revenue"
  - "⚠️ Marginal - Increases profit but revenue down significantly"
  - "❌ Not recommended - Decreases profit"

## Batch Predictions

To predict outcomes for multiple swaps at once:

```python
import pandas as pd
from predict_swap_outcome import SwapPredictor

# Load swap data from a DataFrame
swaps_df = pd.read_parquet('data/swaps/product_swaps_enriched.parquet')

# Initialize predictor
predictor = SwapPredictor(model_dir='models')

# Make predictions for all swaps
predictions = predictor.predict_multiple(swaps_df)

# Combine with original data
results = pd.concat([swaps_df, predictions], axis=1)

# Filter for high-confidence successful swaps (profit-based)
high_confidence_success = results[
    (results['success_prediction'] == True) & 
    (results['confidence'] == 'High') &
    (results['predicted_profit_change'] > 0)  # Profit must increase
]

print(f"Found {len(high_confidence_success)} high-confidence profitable swaps")

# Sort by profit change (primary metric)
top_profitable = results.nlargest(10, 'predicted_profit_change')
print("\nTop 10 swaps by predicted profit change:")
print(top_profitable[['product_before', 'product_after', 'predicted_profit_change', 'predicted_revenue_change', 'recommendation']])
```

## Real-World Usage Example

```python
from predict_swap_outcome import SwapPredictor
import pandas as pd

# Initialize predictor
predictor = SwapPredictor()

# Load historical sales data for a specific product
sales_data = pd.read_parquet('data/sales/Sales_2024_with_profit.parquet')

# Calculate 4-week metrics for a product at a specific machine
machine_id = 'MACHINE_123'
product_name = 'Coca-Cola Zero'
position = 15

# Filter sales for this product/machine
product_sales = sales_data[
    (sales_data['machine_id'] == machine_id) &
    (sales_data['product_name'] == product_name)
]

# Calculate 4-week metrics
revenue_4w = product_sales['revenue'].sum()
profit_4w = product_sales['profit'].sum()
sales_count_4w = len(product_sales)
days_observed = (product_sales['local_timestamp'].max() - 
                 product_sales['local_timestamp'].min()).days

# Get product metadata
subcategory_before = product_sales['subcategory'].iloc[0]
provider_before = product_sales['provider'].iloc[0]

# Define potential swap
swap = {
    'subcategory_before': subcategory_before,
    'subcategory_after': 'Läsk & Saft',  # New product subcategory
    'provider_before': provider_before,
    'provider_after': 'PepsiCo',  # New product provider
    'revenue_before_4w': revenue_4w,
    'profit_before_4w': profit_4w,
    'sales_count_before_4w': sales_count_4w,
    'days_observed_before_4w': max(days_observed, 28)
}

# Predict outcome
prediction = predictor.predict(swap)

# Make decision based on profit (primary metric)
if prediction['predicted_profit_change'] > 0 and prediction['confidence'] in ['High', 'Medium']:
    print(f"✓ Recommended swap: Expected profit change of {prediction['predicted_profit_change']:+.2f} SEK")
    print(f"  Revenue change: {prediction['predicted_revenue_change']:+.2f} SEK")
    print(f"  {prediction['recommendation']}")
elif prediction['predicted_profit_change'] > 0:
    print(f"⚠️ Marginal swap: Expected profit change of {prediction['predicted_profit_change']:+.2f} SEK")
    print(f"  But low confidence ({prediction['confidence']})")
else:
    print(f"✗ Not recommended: Expected profit decrease of {prediction['predicted_profit_change']:+.2f} SEK")
    print(f"  {prediction['recommendation']}")
```

## Model Details

### Profit Model (Regression) - **PRIMARY** ⭐
- **Purpose**: Predicts the expected change in profit (SEK) after a swap
- **Output**: Continuous value (can be positive or negative)
- **Use case**: Primary metric for swap recommendations - reflects actual business value
- **Why profit?**: Profit accounts for costs and shows real financial impact, not just sales volume

### Revenue Model (Regression) - Secondary
- **Purpose**: Predicts the expected change in revenue (SEK) after a swap
- **Output**: Continuous value (can be positive or negative)
- **Use case**: Secondary metric to understand sales volume impact

### Success Model (Classification)
- **Purpose**: Predicts whether a swap will be successful (profit increase)
- **Output**: Probability (0-1) and binary prediction
- **Use case**: Decide whether to proceed with a swap
- **Note**: Success is defined as profit increase, not revenue increase

## Troubleshooting

### Models Not Found
If you get `FileNotFoundError`, make sure you've trained the models first:
```bash
python train_swap_model.py
```

### Unknown Categories/Providers
If you use a category or provider that wasn't in the training data, the model will encode it as `-1` (unknown). This may reduce prediction accuracy.

### Missing Required Fields
Make sure all required fields are provided. Missing fields will cause errors.

## Best Practices

1. **Use recent data**: Calculate 4-week metrics from recent sales data
2. **Minimum sales**: Ensure at least 3 sales in the 4-week period for reliable predictions
3. **Check confidence**: Only act on predictions with "High" or "Medium" confidence
4. **Consider context**: Models predict based on historical patterns - consider external factors (seasonality, promotions, etc.)
5. **Validate predictions**: Compare predictions with actual outcomes to improve trust

