"""
Query Functions

Low-level query functions for each (machine_level, product_level) pair.

Each function filters sales data based on a specific combination of machine
and product aggregation levels. These are the building blocks used by all
fallback strategies.
"""

import pandas as pd
from typing import Optional


# HELPER FUNCTION
# ----------------------------------------------------------------

def _filter_by_time_period(df: pd.DataFrame, time_period: Optional[str]) -> pd.DataFrame:
    """
    Filter DataFrame by time period using the local_timestamp column.

    Args:
        df: DataFrame to filter
        time_period: Time period value in 'YYYY-MM' format (e.g., '2024-03')

    Returns:
        Filtered DataFrame
    """

    # EDGE CASE
    if time_period is None:
        return df

    if 'local_timestamp' in df.columns:
        # Parse the time period (format: YYYY-MM)
        # PARSE THE TIME PERIOD
        try:
            year, month = time_period.split('-')
            start_date = f"{year}-{month}-01"

            # CALCULATE END DATE
            next_month = int(month) + 1
            next_year = int(year)
            if next_month > 12:
                next_month = 1
                next_year += 1
            end_date = f"{next_year}-{next_month:02d}-01"

            # FILTER DATAFRAME
            return df[
                (df['local_timestamp'] >= start_date) &
                (df['local_timestamp'] < end_date)
                ]
        except (ValueError, AttributeError):
            # TODO: IS THIS HOW WE WANT IT TO BEHAVE?
            return df

    # Fallback: check for 'time_period' column (for synthetic data)
    elif 'time_period' in df.columns:
        return df[df['time_period'] == time_period]

    else:
        # SHOULD NOT HAPPEN
        return df


# PAIR 1-3: machine_key level
# ----------------------------------------------------------------

def query_machine_key_product(
        df: pd.DataFrame,
        machine_key: str,
        product_name: str,
        time_period: Optional[str] = None,
        **kwargs
) -> pd.DataFrame:
    """
    Query: machine_key + product_name (Pair #1, Most Specific)

    Args:
        df: Sales DataFrame
        machine_key: Specific machine identifier
        product_name: Specific product name
        time_period: Optional time period filter (e.g., '2024-10' for October 2024)
        **kwargs: Additional filter parameters

    Returns:
        Filtered DataFrame
    """
    result = df[
        (df['machine_key'] == machine_key) &
        (df['product_name'] == product_name)
        ]

    return _filter_by_time_period(result, time_period)


def query_machine_key_subcategory(
        df: pd.DataFrame,
        machine_key: str,
        subcategory: str,
        time_period: Optional[str] = None,
        **kwargs
) -> pd.DataFrame:
    """
    Query: machine_key + subcategory (Pair #2)

    Args:
        df: Sales DataFrame
        machine_key: Specific machine identifier
        subcategory: Product subcategory (e.g., 'Choklad')
        time_period: Optional time period filter
        **kwargs: Additional filter parameters

    Returns:
        Filtered DataFrame
    """
    result = df[
        (df['machine_key'] == machine_key) &
        (df['subcategory'] == subcategory)
        ]

    return _filter_by_time_period(result, time_period)


def query_machine_key_category(
        df: pd.DataFrame,
        machine_key: str,
        category: str,
        time_period: Optional[str] = None,
        **kwargs
) -> pd.DataFrame:
    """
    Query: machine_key + category (Pair #3)

    Args:
        df: Sales DataFrame
        machine_key: Specific machine identifier
        category: Product category (e.g., 'Snacks', 'Dryck')
        time_period: Optional time period filter
        **kwargs: Additional filter parameters

    Returns:
        Filtered DataFrame
    """
    result = df[
        (df['machine_key'] == machine_key) &
        (df['category'] == category)
        ]

    return _filter_by_time_period(result, time_period)


# PAIR 4-6: machine_sub_group level
# ----------------------------------------------------------------

def query_subgroup_product(
        df: pd.DataFrame,
        machine_sub_group: str,
        product_name: str,
        time_period: Optional[str] = None,
        **kwargs
) -> pd.DataFrame:
    """
    Query: machine_sub_group + product_name (Pair #4)

    Args:
        df: Sales DataFrame
        machine_sub_group: Machine sub-group (e.g., 'School', 'Office')
        product_name: Specific product name
        time_period: Optional time period filter
        **kwargs: Additional filter parameters

    Returns:
        Filtered DataFrame
    """
    result = df[
        (df['machine_sub_group'] == machine_sub_group) &
        (df['product_name'] == product_name)
        ]

    return _filter_by_time_period(result, time_period)


def query_subgroup_subcategory(
        df: pd.DataFrame,
        machine_sub_group: str,
        subcategory: str,
        time_period: Optional[str] = None,
        **kwargs
) -> pd.DataFrame:
    """
    Query: machine_sub_group + subcategory (Pair #5)

    Args:
        df: Sales DataFrame
        machine_sub_group: Machine sub-group
        subcategory: Product subcategory
        time_period: Optional time period filter
        **kwargs: Additional filter parameters

    Returns:
        Filtered DataFrame
    """
    result = df[
        (df['machine_sub_group'] == machine_sub_group) &
        (df['subcategory'] == subcategory)
        ]

    return _filter_by_time_period(result, time_period)


def query_subgroup_category(
        df: pd.DataFrame,
        machine_sub_group: str,
        category: str,
        time_period: Optional[str] = None,
        **kwargs
) -> pd.DataFrame:
    """
    Query: machine_sub_group + category (Pair #6)

    Args:
        df: Sales DataFrame
        machine_sub_group: Machine sub-group
        category: Product category
        time_period: Optional time period filter
        **kwargs: Additional filter parameters

    Returns:
        Filtered DataFrame
    """
    result = df[
        (df['machine_sub_group'] == machine_sub_group) &
        (df['category'] == category)
        ]

    return _filter_by_time_period(result, time_period)


# PAIR 7-9: machine_eva_group level
# ----------------------------------------------------------------

def query_evagroup_product(
        df: pd.DataFrame,
        machine_eva_group: str,
        product_name: str,
        time_period: Optional[str] = None,
        **kwargs
) -> pd.DataFrame:
    """
    Query: machine_eva_group + product_name (Pair #7)

    Args:
        df: Sales DataFrame
        machine_eva_group: Machine EVA group (e.g., 'WORK', 'GYM', 'SCHOOLS UNIV')
        product_name: Specific product name
        time_period: Optional time period filter
        **kwargs: Additional filter parameters

    Returns:
        Filtered DataFrame
    """
    result = df[
        (df['machine_eva_group'] == machine_eva_group) &
        (df['product_name'] == product_name)
        ]

    return _filter_by_time_period(result, time_period)


def query_evagroup_subcategory(
        df: pd.DataFrame,
        machine_eva_group: str,
        subcategory: str,
        time_period: Optional[str] = None,
        **kwargs
) -> pd.DataFrame:
    """
    Query: machine_eva_group + subcategory (Pair #8)

    Args:
        df: Sales DataFrame
        machine_eva_group: Machine EVA group
        subcategory: Product subcategory
        time_period: Optional time period filter
        **kwargs: Additional filter parameters

    Returns:
        Filtered DataFrame
    """
    result = df[
        (df['machine_eva_group'] == machine_eva_group) &
        (df['subcategory'] == subcategory)
        ]

    return _filter_by_time_period(result, time_period)


def query_evagroup_category(
        df: pd.DataFrame,
        machine_eva_group: str,
        category: str,
        time_period: Optional[str] = None,
        **kwargs
) -> pd.DataFrame:
    """
    Query: machine_eva_group + category (Pair #9)

    Args:
        df: Sales DataFrame
        machine_eva_group: Machine EVA group
        category: Product category
        time_period: Optional time period filter
        **kwargs: Additional filter parameters

    Returns:
        Filtered DataFrame
    """
    result = df[
        (df['machine_eva_group'] == machine_eva_group) &
        (df['category'] == category)
        ]

    return _filter_by_time_period(result, time_period)


# PAIR 10-12: ALL machines level
# ----------------------------------------------------------------

def query_all_product(
        df: pd.DataFrame,
        product_name: str,
        time_period: Optional[str] = None,
        **kwargs
) -> pd.DataFrame:
    """
    Query: ALL machines + product_name (Pair #10)

    Args:
        df: Sales DataFrame
        product_name: Specific product name
        time_period: Optional time period filter
        **kwargs: Additional filter parameters

    Returns:
        Filtered DataFrame
    """
    result = df[df['product_name'] == product_name]

    return _filter_by_time_period(result, time_period)


def query_all_subcategory(
        df: pd.DataFrame,
        subcategory: str,
        time_period: Optional[str] = None,
        **kwargs
) -> pd.DataFrame:
    """
    Query: ALL machines + subcategory (Pair #11)

    Args:
        df: Sales DataFrame
        subcategory: Product subcategory
        time_period: Optional time period filter
        **kwargs: Additional filter parameters

    Returns:
        Filtered DataFrame
    """
    result = df[df['subcategory'] == subcategory]

    return _filter_by_time_period(result, time_period)


def query_all_category(
        df: pd.DataFrame,
        category: str,
        time_period: Optional[str] = None,
        **kwargs
) -> pd.DataFrame:
    """
    Query: ALL machines + category (Pair #12, Most General)

    Args:
        df: Sales DataFrame
        category: Product category
        time_period: Optional time period filter
        **kwargs: Additional filter parameters

    Returns:
        Filtered DataFrame
    """
    result = df[df['category'] == category]

    return _filter_by_time_period(result, time_period)


# QUERY FUNCTION REGISTRY
# ----------------------------------------------------------------


# MAP PAIR NUMBERS TO QUERY FUNCTIONS
QUERY_FUNCTIONS = {
    1: query_machine_key_product,
    2: query_machine_key_subcategory,
    3: query_machine_key_category,
    4: query_subgroup_product,
    5: query_subgroup_subcategory,
    6: query_subgroup_category,
    7: query_evagroup_product,
    8: query_evagroup_subcategory,
    9: query_evagroup_category,
    10: query_all_product,
    11: query_all_subcategory,
    12: query_all_category,
}
