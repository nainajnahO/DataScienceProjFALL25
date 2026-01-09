"""Autofill module for product suggestions based on predicted revenue.

This module provides functionality to suggest products for machines:
- Fill empty slots with high-revenue products
- Suggest swaps for full machines (replace low-revenue with high-revenue products)
"""

from .autofill import (
    run_autofill_workflow,
    run_swap_workflow,
    recalculate_ranking_with_scores,
)
from ._pipeline_helpers import (
    enrich_slots_with_eans,
    _calculate_revenue,
)

__all__ = [
    'run_autofill_workflow',
    'run_swap_workflow',
    'recalculate_ranking_with_scores',
    'enrich_slots_with_eans',
    '_calculate_revenue',
]
