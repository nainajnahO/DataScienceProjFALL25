# Fallback Architecture for Rule-Based Models

A modular, hierarchical fallback system for handling data sparsity in sales analysis. The architecture progressively generalizes queries across machine and product dimensions until sufficient data is found.

## Overview

When analyzing sales data for a specific product in a specific machine, you may not have enough data for reliable analysis. This system automatically falls back to broader categories (e.g., similar machines, product subcategories) following a defined strategy.

### Key Features

- **Multiple fallback strategies** (Product-First, Machine-First, Location-Affinity)
- **Time-period fixed** - no temporal relaxation, returns NULL if insufficient data
- **Configurable thresholds** - customize minimum sample requirements per level
- **Confidence scoring** - quantifies reliability of fallback results
- **Extensible** - easily add custom strategies

## Architecture

```
FallbackAlgorithm/
├── config.py              # Configuration and thresholds
├── queries.py             # 12 low-level query functions (one per pair)
├── fallback_engine.py     # Mid-level orchestration engine
├── strategies.py          # High-level strategy definitions
└── __init__.py            # Package exports
```

### The Grid

All strategies traverse a 4×3 grid of (machine_level, product_level) pairs:

```
                 product_name    subcategory      category
              ┌──────────────┬──────────────┬──────────────┐
machine_id    │   Pair #1    │   Pair #2    │   Pair #3    │
              ├──────────────┼──────────────┼──────────────┤
sub_group     │   Pair #4    │   Pair #5    │   Pair #6    │
              ├──────────────┼──────────────┼──────────────┤
eva_group     │   Pair #7    │   Pair #8    │   Pair #9    │
              ├──────────────┼──────────────┼──────────────┤
ALL machines  │   Pair #10   │   Pair #11   │   Pair #12   │
              └──────────────┴──────────────┴──────────────┘
```

Different strategies traverse this grid in different orders.


## Quick Start

```python
import pandas as pd
from FallbackAlgorithm import FallbackEngine, PRODUCT_FIRST_STRATEGY

# Load your sales data
sales_data = pd.read_parquet('Sales_2024.parquet')

# Create fallback engine
engine = FallbackEngine(sales_data, PRODUCT_FIRST_STRATEGY)

# Execute fallback query (hierarchy values auto-fetched from DataFrame)
result = engine.execute_fallback(
    machine_id=477466190,
    product_name='Twix',
    time_period='2024-10'
)

# Check results
if result['success']:
    print(f"Found {result['sample_size']} records at level {result['level']}")
    print(f"Using: {result['pair_used']}")
    print(f"Confidence: {result['confidence']:.2%}")
    data = result['data']  # Use the filtered DataFrame
else:
    print("No data found (NULL)")
```

## How Auto-Fetch Works

**What you provide:**
- `machine_id` - The specific machine identifier
- `product_name` - The specific product name
- `time_period` - The time period for filtering (optional)

**What gets auto-fetched from the DataFrame:**
- `machine_sub_group` - Looked up from rows matching `machine_id`
- `machine_eva_group` - Looked up from rows matching `machine_id`
- `subcategory` - Looked up from rows matching `product_name`
- `category` - Looked up from rows matching `product_name`


## Available Strategies

### 1. Product-First Strategy
**Philosophy**: "Find where THIS product sells, anywhere"

Expands machine scope fully before broadening product definition.

```
Level 0: machine_id + product_name
Level 1: machine_sub_group + product_name
Level 2: machine_eva_group + product_name
Level 3: ALL machines + product_name
Level 4: machine_id + subcategory
...
```

**Best for**: Product performance analysis, assortment optimization

### 2. Machine-First Strategy
**Philosophy**: "Find what sells in THIS location, any product type"

Expands product scope fully before broadening machine scope.

```
Level 0: machine_id + product_name
Level 1: machine_id + subcategory
Level 2: machine_id + category
Level 3: machine_sub_group + product_name
Level 4: machine_sub_group + subcategory
...
```

**Best for**: Machine-specific restocking, location optimization

### 3. Location/Environment Affinity Strategy
**Philosophy**: "Environment drives behavior - prioritize similar environments"

Jumps to `machine_eva_group` early, assuming environment is highly predictive.

```
Level 0: machine_id + product_name
Level 1: machine_eva_group + product_name  (JUMP to environment!)
Level 2: machine_sub_group + product_name
Level 3: ALL machines + product_name
...
```

**Best for**: New machine placement, demographic-based recommendations

## Usage Examples

### Basic Usage

```python
from FallbackAlgorithm import FallbackEngine, PRODUCT_FIRST_STRATEGY

engine = FallbackEngine(sales_data, PRODUCT_FIRST_STRATEGY)
result = engine.execute_fallback(
    machine_id=477466190,
    product_name='Twix',
    time_period='2024-10'
)
```

### Quick Fallback (One-liner)

```python
from FallbackAlgorithm import quick_fallback, MACHINE_FIRST_STRATEGY

result = quick_fallback(
    df=sales_data,
    strategy=MACHINE_FIRST_STRATEGY,
    machine_id=477466190,
    product_name='Twix',
    time_period='2024-10'
)
```

### Compare Strategies

```python
from FallbackAlgorithm import (
    PRODUCT_FIRST_STRATEGY,
    MACHINE_FIRST_STRATEGY,
    LOCATION_AFFINITY_STRATEGY,
    FallbackEngine
)

strategies = {
    'Product-First': PRODUCT_FIRST_STRATEGY,
    'Machine-First': MACHINE_FIRST_STRATEGY,
    'Location-Affinity': LOCATION_AFFINITY_STRATEGY,
}

for name, strategy in strategies.items():
    engine = FallbackEngine(sales_data, strategy)
    result = engine.execute_fallback(**query_params)
    print(f"{name}: Level {result['level']}, {result['sample_size']} records")
```

### Custom Strategy

```python
from FallbackAlgorithm import register_custom_strategy, get_strategy

# Define custom strategy
my_strategy = [
    ('machine_id', 'product_name'),
    ('machine_eva_group', 'product_name'),
    ('ALL', 'category'),
]

# Register it
register_custom_strategy('my_strategy', my_strategy)

# Use it
strategy = get_strategy('my_strategy')
engine = FallbackEngine(sales_data, strategy)
```

### Visualize Strategy

```python
from FallbackAlgorithm import visualize_strategy

print(visualize_strategy('product_first'))
```

### Get Strategy Information

```python
engine = FallbackEngine(sales_data, PRODUCT_FIRST_STRATEGY)
info = engine.get_strategy_info()
print(info)
```

## Result Structure

The `execute_fallback()` method returns a dictionary with:

```python
{
    'data': pd.DataFrame,           # Filtered data (or None if failed)
    'level': int,                   # Fallback level used (0-11, or -1)
    'confidence': float,            # Confidence score (0.0-1.0)
    'pair_used': str,               # E.g., "machine_id + product_name"
    'machine_level': str,           # Machine aggregation level used
    'product_level': str,           # Product aggregation level used
    'sample_size': int,             # Number of records found
    'threshold': int,               # Minimum threshold required
    'success': bool,                # True if data found, False if NULL
    'message': str                  # Error/info message (if failed)
}
```

## Data Requirements

Your DataFrame must have these columns:
- `machine_id`
- `machine_sub_group`
- `machine_eva_group`
- `product_name`
- `subcategory`
- `category`

Optional columns:
- `time_period` - for temporal filtering (format: 'YYYY-MM' or custom)
- Any other columns for analysis

## Advanced Usage

### Override Minimum Samples

```python
# Use same threshold (10) for all levels
engine = FallbackEngine(
    sales_data,
    PRODUCT_FIRST_STRATEGY,
    min_samples=10,
    use_adaptive_thresholds=False
)
```