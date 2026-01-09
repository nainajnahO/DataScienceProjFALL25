# Runtime Code - Frontend Integration

This folder contains the code that can be used by your frontend planogram application.

## Files

### `predict_swap_outcome.py`
The main prediction engine for product swap recommendations.

**Usage:**
```python
from runtime.predict_swap_outcome import SwapPredictor

# Initialize predictor (automatically uses models/ and data/ from project root)
predictor = SwapPredictor()

# Get swap recommendations for a product
result = predictor.predict(
    product_before="7310070961234",  # EAN code of current product
    product_after="7310070961241"    # EAN code of candidate replacement
)

# Result contains:
# - predicted_profit_change: float (SEK) - PRIMARY METRIC ⭐
# - predicted_revenue_change: float (SEK) - Secondary metric
# - success_probability: float (0-1) - Based on profit
# - success_prediction: bool
# - confidence: str ("Low"/"Medium"/"High")
# - recommendation: str - Human-readable recommendation
```

## For Frontend Integration

### Use Case 1: Click on Product → Show Swap Recommendations

```python
def get_recommendations_for_product(
    product_ean: str,
    candidate_products: List[str],  # Products available in catalog
    top_n: int = 10
) -> List[dict]:
    """Get top N swap recommendations for a product."""
    predictor = SwapPredictor()
    recommendations = []
    
    for candidate_ean in candidate_products:
        if candidate_ean == product_ean:
            continue  # Skip same product
        
        result = predictor.predict(product_ean, candidate_ean)
        recommendations.append({
            'product_ean': candidate_ean,
            'predicted_profit_change': result['predicted_profit_change'],  # PRIMARY
            'predicted_revenue_change': result['predicted_revenue_change'],  # Secondary
            'success_probability': result['success_probability'],
            'confidence': result['confidence'],
            'recommendation': result['recommendation']
        })
    
    # Sort by profit change (primary metric) and return top N
    recommendations.sort(key=lambda x: x['predicted_profit_change'], reverse=True)
    return recommendations[:top_n]
```

### Use Case 2: View Entire Machine → Show Recommendations

```python
def get_recommendations_for_machine(
    machine_products: List[dict],  # List of {ean, position, ...}
    candidate_products: List[str],
    top_n_per_product: int = 3
) -> dict:
    """Get recommendations for all products in a machine."""
    predictor = SwapPredictor()
    all_recommendations = {}
    
    for slot in machine_products:
        product_ean = slot['ean']
        recommendations = get_recommendations_for_product(
            product_ean, 
            candidate_products, 
            top_n=top_n_per_product
        )
        all_recommendations[slot['position']] = recommendations
    
    return all_recommendations
```

## Dependencies

The `SwapPredictor` class requires:
- `models/` directory with trained models (in project root)
- `data/` directory with product data (in project root)

These are automatically resolved relative to the project root when using default initialization.

