# Location-Based Sales Forecasting

Predict sales for products at **new locations** using geo-weighted sales data from similar machine categories.

## Overview

The Location-Based Forecasting system predicts how well a product will sell at a hypothetical new location by:

1. **Finding similar machines** - Identifies existing machines in the same category (e.g., WORK, GYM, SCHOOLS)
2. **Retrieving historical sales** - Gets sales data for those machines (using FallbackAlgorithm)
3. **Calculating geographic similarity** - Compares the new location to each reference machine based on:
   - Direct distance
   - ICA supermarket proximity similarity
   - Company density similarity (weighted by employee counts)
4. **Weighting predictions** - Creates weighted average of sales, with higher weight for more similar locations
5. **Confidence scoring** - Provides confidence based on similarity, sample size, and consistency

---

## Key Features

- **Geographic similarity scoring** - Multi-factor similarity based on distance, ICA proximity, and company density
- **Category-specific** - Different configurations for WORK, GYM, SCHOOLS, etc.
- **Multiple weighting strategies** - Inverse distance, Gaussian, top-k, adaptive
- **Confidence scoring** - Quantifies prediction reliability
- **Standalone system** - Works independently (uses FallbackAlgorithm only for data retrieval)
- **Mock data included** - Ready to test with 37 ICA stores and 100 companies

---

## Installation

No additional dependencies beyond the existing project requirements.

```python
import sys
sys.path.append('/path/to/Rule-Based Models (RBM)')

from LocationBasedForecasting import predict_sales_at_location
```

---

## Quick Start

### Simple One-Liner Prediction

```python
from LocationBasedForecasting import predict_sales_at_location
import pandas as pd

# Load your data
sales_df = pd.read_parquet('Sales_2024.parquet')
machines_df = pd.read_parquet('app_machines.parquet')

# Predict sales at a new office location in Stockholm
prediction = predict_sales_at_location(
    product_name='Coca-Cola 33cl',
    new_latitude=59.3293,  # Stockholm coordinates
    new_longitude=18.0686,
    machine_category='WORK',  # Office environment
    sales_df=sales_df,
    machines_df=machines_df,
    time_period='2024-03'  # March 2024
)

print(f"Predicted monthly sales: {prediction['predicted_sales']:.0f}")
print(f"Confidence: {prediction['confidence']:.2%}")
print(f"Based on {len(prediction['reference_machines'])} similar machines")
```

### Using the Engine Directly

```python
from LocationBasedForecasting import LocationBasedForecaster

# Create forecaster
forecaster = LocationBasedForecaster(
    sales_df=sales_df,
    machines_df=machines_df,
    ica_stores_df=ica_stores,  # Optional: auto-loads if None
    companies_df=companies      # Optional: auto-loads if None
)

# Make predictions
prediction = forecaster.predict_sales(
    product_name='Coca-Cola 33cl',
    new_latitude=59.3293,
    new_longitude=18.0686,
    machine_category='WORK'
)
```

---

## How It Works

### Step-by-Step Process

```
1. Find Reference Machines
   ↓
   Filter machines_df for category='WORK'
   → Found 50 WORK machines

2. Get Historical Sales (using FallbackAlgorithm)
   ↓
   For each machine, retrieve Coca-Cola sales
   → Machine 123: 150 sales/month
   → Machine 456: 120 sales/month
   → Machine 789: 180 sales/month
   → ... (50 machines total)

3. Calculate Geographic Similarity
   ↓
   Compare new location to each reference machine:
   - Direct distance
   - ICA proximity similarity
   - Company density similarity
   →
   → Machine 123: similarity = 0.92 (very similar)
   → Machine 456: similarity = 0.65 (moderately similar)
   → Machine 789: similarity = 0.45 (less similar)

4. Weight Predictions
   ↓
   Weighted average = Σ(sales_i × similarity_i) / Σ(similarity_i)
   → Predicted sales: 142 units/month

5. Calculate Confidence
   ↓
   Based on: avg similarity, sample size, consistency
   → Confidence: 0.87 (high)
```

### Geographic Similarity Components

**Overall Similarity = 40% Distance + 30% ICA + 30% Company**

1. **Distance Similarity** (40%)
   - Exponential decay with distance
   - Closer locations = higher similarity
   - Formula: `exp(-distance / max_distance * 3.0)`

2. **ICA Proximity Similarity** (30%)
   - Compares ICA store proximity profiles
   - Similar access to ICA = more similar customer base
   - Based on k=3 nearest ICA stores

3. **Company Proximity Similarity** (30%)
   - Compares nearby company density (weighted by employees)
   - Similar business density = similar sales potential
   - Radius: 2km

---

## Available Weighting Strategies

### 1. Inverse Distance (Default)
Simple and intuitive - weight proportional to similarity.

```python
prediction = predict_sales_at_location(
    ...,
    weighting_strategy='inverse_distance'
)
```

### 2. Gaussian
Smoother falloff, reduces influence of distant machines.

```python
prediction = predict_sales_at_location(
    ...,
    weighting_strategy='gaussian',
    sigma=2.0  # Kernel width
)
```

### 3. Top-K
Uses only the k most similar machines.

```python
prediction = predict_sales_at_location(
    ...,
    weighting_strategy='top_k',
    k=5  # Use top 5 only
)
```

### 4. Adaptive
Automatically chooses strategy based on similarity distribution.

```python
prediction = predict_sales_at_location(
    ...,
    weighting_strategy='adaptive'
)
```

---

## Configuration

### Category-Specific Parameters

Different machine categories have different configurations:

```python
CATEGORY_CONFIGS = {
    'WORK': {
        'max_distance': 20.0,      # Offices are localized
        'min_machines': 5,
        'ica_weight': 0.2,         # Less ICA-dependent
        'company_weight': 0.5      # Heavily company-dependent
    },
    'GYM': {
        'max_distance': 30.0,
        'min_machines': 3,
        'ica_weight': 0.3,
        'company_weight': 0.3
    },
    'SCHOOLS': {
        'max_distance': 15.0,
        'min_machines': 4,
        'ica_weight': 0.4,
        'company_weight': 0.2
    }
}
```

### Customizing Weights

```python
from LocationBasedForecasting import get_geo_similarity_weights

# Get category-specific weights
weights = get_geo_similarity_weights('WORK')
# {'distance': 0.3, 'ica_profile': 0.2, 'company_profile': 0.5}
```

---

## Advanced Usage

### Compare Multiple Locations

```python
from LocationBasedForecasting import compare_locations

locations = [
    {'name': 'Stockholm Office', 'latitude': 59.3293, 'longitude': 18.0686},
    {'name': 'Göteborg Office', 'latitude': 57.7089, 'longitude': 11.9746},
    {'name': 'Malmö Office', 'latitude': 55.6050, 'longitude': 13.0038}
]

comparison = compare_locations(
    product_name='Coca-Cola 33cl',
    locations=locations,
    machine_category='WORK',
    sales_df=sales_df,
    machines_df=machines_df
)

print(comparison[['name', 'predicted_sales', 'confidence']])
```

### Find Best Location

```python
from LocationBasedForecasting import get_best_location

best = get_best_location(
    product_name='Coca-Cola 33cl',
    candidate_locations=locations,
    machine_category='WORK',
    sales_df=sales_df,
    machines_df=machines_df,
    min_confidence=0.6
)

print(f"Best location: {best['best_location']['name']}")
print(f"Expected sales: {best['prediction']['predicted_sales']:.0f}")
```

### Inspect Prediction Breakdown

```python
prediction = predict_sales_at_location(...)

for machine in prediction['breakdown']:
    print(f"Machine {machine['machine_id']}:")
    print(f"  Similarity: {machine['geo_similarity']:.2f}")
    print(f"  Sales: {machine['sales']:.0f}")
    print(f"  Weight: {machine['weight']:.2%}")
    print(f"  Contribution: {machine['contribution']:.0f}")
    print(f"  Distance: {machine['distance_km']:.1f} km")
```

---

## Data Requirements

### Required DataFrames

**sales_df** - Historical sales data
- Columns: `machine_id`, `product_name`, (optional: `quantity`, `local_timestamp`)

**machines_df** - Machine locations and metadata
- Columns: `machine_id`, `latitude`, `longitude`, `machine_eva_group`

**ica_stores_df** - ICA store locations (auto-loads if not provided)
- Columns: `latitude`, `longitude`

**companies_df** - Company locations (auto-loads if not provided)
- Columns: `latitude`, `longitude`, `employee_count`

### Mock Data

The system includes mock data for testing:
- **37 ICA stores** across Swedish cities
- **100 companies** with realistic employee distributions

```python
from LocationBasedForecasting import load_default_ica_stores, load_default_companies

ica_stores = load_default_ica_stores()
companies = load_default_companies()
```

---

## Integration with FallbackAlgorithm

Location-Based Forecasting uses FallbackAlgorithm for historical sales retrieval:

```python
from FallbackAlgorithm import FallbackEngine, PRODUCT_FIRST_STRATEGY
from LocationBasedForecasting import LocationBasedForecaster

# Create FallbackEngine
fallback_engine = FallbackEngine(sales_df, PRODUCT_FIRST_STRATEGY)

# Pass to forecaster for intelligent data retrieval
forecaster = LocationBasedForecaster(
    sales_df=sales_df,
    machines_df=machines_df,
    ica_stores_df=ica_stores,
    companies_df=companies,
    fallback_engine=fallback_engine  # Optional but recommended
)
```

**Role of FallbackAlgorithm:**
- Retrieves sales data for reference machines
- Handles sparse data through hierarchical fallback
- Provides fallback level information for confidence scoring

**LocationBasedForecaster's Role:**
- Geographic similarity calculations
- Weighting predictions by location similarity
- Confidence scoring based on geographic factors

---

## Result Structure

```python
{
    'predicted_sales': 142.5,           # Predicted average sales
    'confidence': 0.87,                 # Confidence score (0.0-1.0)
    'reference_machines': [123, 456],   # Machine IDs used
    'geo_similarities': [0.92, 0.65],   # Similarity to each machine
    'weights': [0.59, 0.41],            # Weight for each machine
    'breakdown': [                       # Detailed breakdown
        {
            'machine_id': 123,
            'geo_similarity': 0.92,
            'sales': 150.0,
            'weight': 0.59,
            'contribution': 88.5,
            'distance_km': 2.3,
            'sample_size': 166
        },
        ...
    ],
    'success': True,
    'message': 'Successfully predicted using 2 reference machines'
}
```

---

## Examples

See `example_location_prediction.ipynb` for comprehensive examples including:
1. Basic prediction at a new location
2. Comparing multiple candidate locations
3. Testing different weighting strategies
4. Visualizing geographic similarity
5. Analyzing prediction breakdown

---

## Architecture

```
LocationBasedForecasting/
├── location_forecaster.py       # Core prediction engine
├── prediction_pipeline.py       # High-level API
├── geo_utils.py                 # Distance calculations
├── geo_similarity.py            # Geographic similarity scoring
├── ica_proximity.py             # ICA proximity scoring
├── company_proximity.py         # Company proximity scoring
├── weighting_strategies.py      # Prediction weighting strategies
├── config.py                    # Configuration
├── __init__.py                  # Package exports
├── README.md                    # This file
└── data/
    ├── mock_ica_stores.parquet
    └── mock_companies.parquet
```