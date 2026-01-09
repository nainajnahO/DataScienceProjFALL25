"""
Fallback Strategies

High-level strategy definitions for different fallback approaches.

Each strategy defines the order in which fallback levels are tried,
represented as a list of (machine_level, product_level) tuples.

Strategies available:
- PRODUCT_FIRST_STRATEGY: Expand machine scope before product scope
- MACHINE_FIRST_STRATEGY: Expand product scope before machine scope
- LOCATION_AFFINITY_STRATEGY: Prioritize machine_eva_group early
"""

from typing import List, Tuple, Dict

# STRATEGY 1: PRODUCT-FIRST
# ----------------------------------------------------------------

PRODUCT_FIRST_STRATEGY: List[Tuple[str, str]] = [
    # Level 0: Start with most specific
    ('machine_key', 'product_name'),

    # Levels 1-3: Expand machine scope, keep product specific
    ('machine_sub_group', 'product_name'),
    ('machine_eva_group', 'product_name'),
    ('ALL', 'product_name'),

    # Levels 4-6: Broaden product to subcategory, restart machine expansion
    ('machine_key', 'subcategory'),
    ('machine_sub_group', 'subcategory'),
    ('machine_eva_group', 'subcategory'),
    ('ALL', 'subcategory'),

    # Levels 8-11: Broaden product to category, restart machine expansion
    ('machine_key', 'category'),
    ('machine_sub_group', 'category'),
    ('machine_eva_group', 'category'),
    ('ALL', 'category'),
]

# STRATEGY 2: MACHINE-FIRST
# ----------------------------------------------------------------
MACHINE_FIRST_STRATEGY: List[Tuple[str, str]] = [
    # Level 0: Start with most specific
    ('machine_key', 'product_name'),

    # Levels 1-2: Expand product scope, keep machine specific
    ('machine_key', 'subcategory'),
    ('machine_key', 'category'),

    # Levels 3-5: Expand to sub_group, restart product expansion
    ('machine_sub_group', 'product_name'),
    ('machine_sub_group', 'subcategory'),
    ('machine_sub_group', 'category'),

    # Levels 6-8: Expand to eva_group, restart product expansion
    ('machine_eva_group', 'product_name'),
    ('machine_eva_group', 'subcategory'),
    ('machine_eva_group', 'category'),

    # Levels 9-11: Expand to ALL machines, restart product expansion
    ('ALL', 'product_name'),
    ('ALL', 'subcategory'),
    ('ALL', 'category'),
]

# STRATEGY 3: LOCATION/ENVIRONMENT AFFINITY
# ----------------------------------------------------------------

LOCATION_AFFINITY_STRATEGY: List[Tuple[str, str]] = [
    # Level 0: Start with most specific
    ('machine_key', 'product_name'),

    # Level 1: JUMP to eva_group early (environment is predictive)
    ('machine_eva_group', 'product_name'),

    # Level 2: Fill in sub_group
    ('machine_sub_group', 'product_name'),

    # Level 3: Expand to all machines
    ('ALL', 'product_name'),

    # Levels 4-6: Broaden product to subcategory, prioritize environment
    ('machine_eva_group', 'subcategory'),
    ('machine_sub_group', 'subcategory'),
    ('ALL', 'subcategory'),

    # Levels 7-9: Broaden product to category, prioritize environment
    ('machine_eva_group', 'category'),
    ('machine_sub_group', 'category'),
    ('ALL', 'category'),

    # Levels 10-11: Fallback to machine_key with broader products (last resort)
    ('machine_key', 'subcategory'),
    ('machine_key', 'category'),
]

# STRATEGY REGISTRY
# ----------------------------------------------------------------
STRATEGY_REGISTRY: Dict[str, List[Tuple[str, str]]] = {
    'product_first': PRODUCT_FIRST_STRATEGY,
    'machine_first': MACHINE_FIRST_STRATEGY,
    'location_affinity': LOCATION_AFFINITY_STRATEGY,
}


def get_strategy(name: str) -> List[Tuple[str, str]]:
    """
    Get a strategy by name.

    Args:
        name: Strategy name ('product_first', 'machine_first', 'location_affinity')

    Returns:
        Strategy as list of (machine_level, product_level) tuples
    """
    if name not in STRATEGY_REGISTRY:
        available = ', '.join(STRATEGY_REGISTRY.keys())
        raise ValueError(
            f"Unknown strategy '{name}'. Available strategies: {available}"
        )

    return STRATEGY_REGISTRY[name]


def list_strategies() -> List[str]:
    """
    Get list of available strategy names.

    Returns:
        List of strategy names
    """
    return list(STRATEGY_REGISTRY.keys())


def register_custom_strategy(
        name: str,
        strategy: List[Tuple[str, str]],
        overwrite: bool = False
) -> None:
    """
    Register a custom fallback strategy.

    Args:
        name: Strategy name
        strategy: List of (machine_level, product_level) tuples
        overwrite: If True, allow overwriting existing strategies
    """
    if name in STRATEGY_REGISTRY and not overwrite:
        raise ValueError(
            f"Strategy '{name}' already exists. Use overwrite=True to replace it."
        )

    # Validate strategy format
    if not isinstance(strategy, list):
        raise TypeError("Strategy must be a list of tuples")

    for item in strategy:
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError(
                "Each strategy item must be a tuple of (machine_level, product_level)"
            )

    STRATEGY_REGISTRY[name] = strategy


# STRATEGY COMPARISON UTILITIES
# ----------------------------------------------------------------

def compare_strategies() -> Dict[str, Dict[str, any]]:
    """
    Compare characteristics of all registered strategies.

    Returns:
        Dictionary with strategy comparison data
    """
    comparison = {}

    for name, strategy in STRATEGY_REGISTRY.items():
        # Count unique machine and product levels
        machine_levels = set(m for m, p in strategy)
        product_levels = set(p for m, p in strategy)

        # Find which level product/category broadening occurs
        first_subcategory = next(
            (i for i, (m, p) in enumerate(strategy) if p == 'subcategory'),
            None
        )
        first_category = next(
            (i for i, (m, p) in enumerate(strategy) if p == 'category'),
            None
        )

        comparison[name] = {
            'num_levels': len(strategy),
            'unique_machine_levels': len(machine_levels),
            'unique_product_levels': len(product_levels),
            'machine_levels_used': sorted(machine_levels),
            'product_levels_used': sorted(product_levels),
            'first_subcategory_level': first_subcategory,
            'first_category_level': first_category,
            'strategy': strategy
        }

    return comparison


def visualize_strategy(strategy_name: str) -> str:
    """
    Create a text visualization of a strategy.

    Args:
        strategy_name: Name of the strategy to visualize

    Returns:
        Formatted string showing the strategy progression
    """
    strategy = get_strategy(strategy_name)

    output = [f"\n{strategy_name.upper().replace('_', ' ')} STRATEGY"]
    output.append("=" * 60)

    for level, (machine_level, product_level) in enumerate(strategy):
        output.append(f"Level {level:2d}: {machine_level:20s} + {product_level}")

    return "\n".join(output)


if __name__ == "__main__":
    # Print all strategies when run as a script
    print("Available Fallback Strategies")

    for strategy_name in list_strategies():
        print(visualize_strategy(strategy_name))
        print()
