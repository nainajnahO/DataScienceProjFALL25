"""
Fallback Engine

Mid-level orchestration logic for executing fallback strategies.

The FallbackEngine class handles the execution of fallback queries, validation
of results against minimum thresholds, and calculation of confidence scores.
"""

# IMPORTS
import pandas as pd
from typing import Dict, List, Tuple, Callable, Optional, Any, Union
import warnings

from .config import (
    MIN_SAMPLE_THRESHOLDS,
    DEFAULT_MIN_SAMPLES,
    CONFIDENCE_WEIGHTS
)
from .queries import QUERY_FUNCTIONS


# CREATE THE ENGINE CLASS
class FallbackEngine:
    """
    Executes fallback queries following a defined strategy.

    The engine tries each fallback level in order until it finds sufficient
    data or exhausts all options. Results include metadata about which level
    was used and the confidence score.
    """

    def __init__(
            self,
            df: pd.DataFrame,  # SALES DATA
            strategy: List[Tuple[str, str]],  # FALLBACK STRATEGY
            min_samples: Optional[int] = None,  # MINIMUM SAMPLE THRESHOLD (OVERRIDE CONFIG)
            use_adaptive_thresholds: bool = True,  # USE DIFFERENT THRESHOLDS PER LEVEL
            custom_thresholds: Optional[Union[int, Dict[str, Dict[str, int]]]] = None  # CUSTOM THRESHOLDS
    ):
        """
        INITIALIZE THE FALLBACK ENGINE

        Args:
            df: -
            strategy: List of (machine_level, product_level) tuples defining
                     fallback order
            min_samples: -
            use_adaptive_thresholds: -
            custom_thresholds: Custom thresholds - either:
                              - int: Single value applied to all levels
                              - dict: {machine_level: {product_level: threshold}}
                              Takes priority over config defaults when provided
        """
        self.df = df
        self.strategy = strategy
        self.min_samples = min_samples
        self.use_adaptive_thresholds = use_adaptive_thresholds
        self.custom_thresholds = custom_thresholds

        # INITIALIZE CACHES FOR HIERARCHY LOOKUPS
        self._machine_hierarchy_cache: Dict[str, Dict[str, str]] = {}
        self._product_hierarchy_cache: Dict[str, Dict[str, str]] = {}

        # VALIDATE THAT DATAFRAME HAS REQUIRED COLUMNS
        self._validate_dataframe()

    def _validate_dataframe(self) -> None:
        required_columns = {
            'machine_key', 'machine_sub_group', 'machine_eva_group',
            'product_name', 'subcategory', 'category'
        }

        missing = required_columns - set(self.df.columns)
        if missing:
            raise ValueError(f"DataFrame missing required columns: {missing}")

    def _get_threshold(self, machine_level: str, product_level: str) -> int:
        """
        Get the minimum sample threshold for a given level.

        Args:
            machine_level: Machine aggregation level
            product_level: Product aggregation level

        Returns:
            Minimum number of samples required
        """
        # CHECK CUSTOM THRESHOLDS FIRST
        if self.custom_thresholds is not None:
            if isinstance(self.custom_thresholds, int):
                return self.custom_thresholds
            try:
                return self.custom_thresholds[machine_level][product_level]
            except KeyError:
                pass  # Fall through to other options

        # CONFIG FILE VALUE
        if not self.use_adaptive_thresholds:
            return self.min_samples or DEFAULT_MIN_SAMPLES

        if self.min_samples is not None:
            return self.min_samples

        # USE ADAPTIVE THRESHOLDS FROM CONFIG FILE
        try:
            return MIN_SAMPLE_THRESHOLDS[machine_level][product_level]
        except KeyError:
            warnings.warn(
                f"No threshold defined for ({machine_level}, {product_level}). "
                f"Using default: {DEFAULT_MIN_SAMPLES}"
            )
            return DEFAULT_MIN_SAMPLES

    def _get_query_function(
            self,
            machine_level: str,
            product_level: str
    ) -> Callable:
        """
        Get the appropriate query function for a given level combination.

        Args:
            machine_level: Machine aggregation level
            product_level: Product aggregation level

        Returns:
            Query function

        """
        # MAPS LEVELS TO QUERY FUNCTIONS
        function_map = {
            ('machine_key', 'product_name'): QUERY_FUNCTIONS[1],
            ('machine_key', 'subcategory'): QUERY_FUNCTIONS[2],
            ('machine_key', 'category'): QUERY_FUNCTIONS[3],
            ('machine_sub_group', 'product_name'): QUERY_FUNCTIONS[4],
            ('machine_sub_group', 'subcategory'): QUERY_FUNCTIONS[5],
            ('machine_sub_group', 'category'): QUERY_FUNCTIONS[6],
            ('machine_eva_group', 'product_name'): QUERY_FUNCTIONS[7],
            ('machine_eva_group', 'subcategory'): QUERY_FUNCTIONS[8],
            ('machine_eva_group', 'category'): QUERY_FUNCTIONS[9],
            ('ALL', 'product_name'): QUERY_FUNCTIONS[10],
            ('ALL', 'subcategory'): QUERY_FUNCTIONS[11],
            ('ALL', 'category'): QUERY_FUNCTIONS[12],
        }

        key = (machine_level, product_level)

        # SHOULD NOT HAPPEN
        if key not in function_map:
            raise ValueError(f"No query function for combination: {key}")

        return function_map[key]

    def _lookup_hierarchy_values(
            self,
            machine_key: Optional[str] = None,
            product_name: Optional[str] = None
    ) -> Dict[str, Optional[str]]:
        """
        Automatically look up hierarchy values from the DataFrame.

        Args:
            machine_key: Specific machine identifier
            product_name: Specific product name

        Returns:
            Dictionary with all hierarchy values:
                - machine_sub_group
                - machine_eva_group
                - subcategory
                - category
        """
        hierarchy_values = {
            'machine_sub_group': None,
            'machine_eva_group': None,
            'subcategory': None,
            'category': None
        }

        # LOOKUP MACHINE HIERARCHICAL VALUES
        # Caching here is important, code execution time got reduced by several times!
        # The original times were not feasible... (+17 min)
        if machine_key is not None:
            if machine_key in self._machine_hierarchy_cache:
                # CACHE HIT
                cached = self._machine_hierarchy_cache[machine_key]
                hierarchy_values['machine_sub_group'] = cached['machine_sub_group']
                hierarchy_values['machine_eva_group'] = cached['machine_eva_group']
            else:
                # IF CACHE MISS, PERFORM THE SCAN AND UPDATE CACHE
                machine_rows = self.df[self.df['machine_key'].astype(str) == str(machine_key)]
                if machine_rows.empty:
                    raise ValueError(f"machine_key '{machine_key}' not found in DataFrame")

                # GET FIRST ROW
                machine_row = machine_rows.iloc[0]

                # STORE IN CACHE
                cache_entry = {
                    'machine_sub_group': machine_row['machine_sub_group'],
                    'machine_eva_group': machine_row['machine_eva_group']
                }
                self._machine_hierarchy_cache[machine_key] = cache_entry

                # UPDATE RESULT
                hierarchy_values['machine_sub_group'] = cache_entry['machine_sub_group']
                hierarchy_values['machine_eva_group'] = cache_entry['machine_eva_group']

        # LOOKUP PRODUCT HIERARCHICAL VALUES (SAME AS THE CODE ABOVE)
        if product_name is not None:
            if product_name in self._product_hierarchy_cache:
                # CACHE HIT
                cached = self._product_hierarchy_cache[product_name]
                hierarchy_values['subcategory'] = cached['subcategory']
                hierarchy_values['category'] = cached['category']
            else:
                # CACHE MISS
                product_rows = self.df[self.df['product_name'] == product_name]
                if product_rows.empty:
                    raise ValueError(f"product_name '{product_name}' not found in DataFrame")

                # GET FIRST ROW
                product_row = product_rows.iloc[0]

                # STORE IN CACHE
                cache_entry = {
                    'subcategory': product_row['subcategory'],
                    'category': product_row['category']
                }
                self._product_hierarchy_cache[product_name] = cache_entry

                # UPDATE RESULT
                hierarchy_values['subcategory'] = cache_entry['subcategory']
                hierarchy_values['category'] = cache_entry['category']

        return hierarchy_values

    def _calculate_confidence(
            self,
            result: pd.DataFrame,
            level: int,
            threshold: int
    ) -> float:
        """
        Calculate confidence score for a fallback result.

        Confidence is based on:
        - Fallback level (lower levels = higher confidence)
        - Sample size relative to threshold
        - Data variance (if applicable)

        Args:
            result: Query result DataFrame
            level: Fallback level (0-11)
            threshold: Minimum threshold that was used

        Returns:
            Confidence score between 0.0 and 1.0
        """
        # LOAD CONFIDENCE WEIGHT FROM CONFIG FILE
        base_confidence = CONFIDENCE_WEIGHTS.get(level, 0.5)

        # CHECK SAMPLE SIZE
        sample_size = len(result)
        if sample_size == 0:
            return 0.0

        # COMPARE IT TO THRESHOLD
        volume_ratio = min(sample_size / threshold, 2.0) / 2.0  # Cap at 2x threshold
        volume_score = volume_ratio

        # WEIGHT SCORES (70% level-based, 30% volume-based)
        confidence = 0.7 * base_confidence + 0.3 * volume_score

        # AS YOU CAN SEE, THESE VALUES OR DERIVATIONS ARE NOT SCIENTIFICALLY VALIDATED!
        # INSTEAD, THIS SERVES THE PURPOSE OF COMMUNICATING WITH THE USER HOW DEEP WE HAVE "FALLEN" IN THE ALGORITHM
        return min(confidence, 1.0)

    def execute_fallback(
            self,
            machine_key: Optional[str] = None,
            product_name: Optional[str] = None,
            time_period: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute the fallback strategy.

        Tries each level in the strategy until sufficient data is found.
        Hierarchy values (machine_sub_group, machine_eva_group, subcategory,
        category) are automatically looked up from the DataFrame.

        Args:
            machine_key: Specific machine identifier
            product_name: Specific product name
            time_period: Time period filter (e.g., '2024-10')

        Returns:
            Dictionary with:
                - data: Filtered DataFrame (or None if no data found)
                - level: Fallback level used (0-11, or -1 if failed)
                - confidence: Confidence score (0.0-1.0)
                - pair_used: Description of the pair
                - machine_level: Machine aggregation level used
                - product_level: Product aggregation level used
                - sample_size: Number of records found
                - threshold: Minimum threshold that was required
                - success: True if data found, False otherwise
                - message: Error message (if failed)

        """
        # LOOKUP HIERARCHICAL VALUES FROM DATAFRAME
        hierarchy_values = self._lookup_hierarchy_values(machine_key, product_name)

        # BUILD QUERY PARAMETERS WITH LOOKED-UP VALUES
        query_params = {
            'machine_key': machine_key,
            'machine_sub_group': hierarchy_values['machine_sub_group'],
            'machine_eva_group': hierarchy_values['machine_eva_group'],
            'product_name': product_name,
            'subcategory': hierarchy_values['subcategory'],
            'category': hierarchy_values['category'],
            'time_period': time_period
        }

        # MAIN LOOP (TRY EACH LEVEL IN STRATEGY UNTIL SUCCESSFUL)
        for level, (machine_level, product_level) in enumerate(self.strategy):

            # GET QUERY FUNCTION AND TRESHOLD
            query_func = self._get_query_function(machine_level, product_level)
            threshold = self._get_threshold(machine_level, product_level)

            # EXECUTE QUERY
            try:
                result = query_func(self.df, **query_params)
            except TypeError as e:
                # SHOULD NOT HAPPEN
                warnings.warn(
                    f"Level {level} ({machine_level}, {product_level}): "
                    f"Missing parameters - {e}"
                )
                continue

            # VALIDATE RESULT AGAINST TRESHOLD
            sample_size = len(result)

            if sample_size >= threshold:
                # SUCESS
                confidence = self._calculate_confidence(result, level, threshold)

                return {
                    'data': result,
                    'level': level,
                    'confidence': confidence,
                    'pair_used': f"{machine_level} + {product_level}",
                    'machine_level': machine_level,
                    'product_level': product_level,
                    'sample_size': sample_size,
                    'threshold': threshold,
                    'success': True
                }

        # NO FALLBACK LEVEL WAS SUCCESSFUL - RETURN NULL
        return {
            'data': None,
            'level': -1,
            'confidence': 0.0,
            'pair_used': 'NULL',
            'machine_level': None,
            'product_level': None,
            'sample_size': 0,
            'threshold': None,
            'success': False,
            'message': 'Insufficient data at all fallback levels'
        }

    def get_strategy_info(self) -> pd.DataFrame:

        # FUNCTION FOR FETCHING STRATEGY INFO

        strategy_info = []

        for level, (machine_level, product_level) in enumerate(self.strategy):
            threshold = self._get_threshold(machine_level, product_level)
            confidence_weight = CONFIDENCE_WEIGHTS.get(level, 0.5)

            strategy_info.append({
                'level': level,
                'machine_level': machine_level,
                'product_level': product_level,
                'pair': f"{machine_level} + {product_level}",
                'min_threshold': threshold,
                'confidence_weight': confidence_weight
            })

        return pd.DataFrame(strategy_info)


def quick_fallback(
        df: pd.DataFrame,
        strategy: List[Tuple[str, str]],
        machine_key: Optional[str] = None,
        product_name: Optional[str] = None,
        time_period: Optional[str] = None
) -> Dict[str, Any]:
    # THIS FUNCTION AUTOMATES THE CREATION OF THE FALLBACK INSTANCE AND TUNNELS RESULTS BACK.
    # OFCOURSE THE DOWNSIDE IS THAT NO INTERACTIONS ARE POSSIBLE WITH THE INSTANCE, IT IS ISOLATED AND DISCARDED
    engine = FallbackEngine(df, strategy)
    return engine.execute_fallback(
        machine_key=machine_key,
        product_name=product_name,
        time_period=time_period
    )
