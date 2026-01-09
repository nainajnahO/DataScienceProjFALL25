"""
Hierarchical Naive Baseline Forecaster
========================================

A sophisticated baseline that uses hierarchical fallback averaging for 4-week ahead predictions.

Prediction Strategy:
1. Primary: Last 4 non-zero sales for (machine_key, ean) within 10 weeks
2. Machine context: Same subcategory/category products in this machine
3. Product context: Same product in similar machines (subgroup → eva_group → all)
4. Fallback: Subcategory × machine_subgroup average across dataset

Horizon: Limited to 10 weeks before target for recency
"""

import pandas as pd
import numpy as np
from typing import List, Tuple
from pathlib import Path
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class HierarchicalNaiveForecaster:
    """
    Hierarchical fallback naive baseline for 4-week forecasting.
    
    Uses increasingly general context when specific history is unavailable.
    """
    
    def __init__(self, lookback_weeks: int = 10, min_samples: int = 4):
        """
        Initialize forecaster.
        
        Args:
            lookback_weeks: Maximum weeks to look back for history
            min_samples: Minimum samples needed for direct prediction
        """
        self.lookback_weeks = lookback_weeks
        self.min_samples = min_samples
        self.train_df = None
        self.aggregates = {}
        
    def fit(self, train_df: pd.DataFrame):
        """
        Fit the model by storing training data and pre-computing aggregates.
        
        Args:
            train_df: Training DataFrame with columns: week_start, machine_key, ean,
                      weekly_sales, subcategory, category, machine_sub_group, machine_eva_group
        """
        print(f"\n{'='*60}")
        print(f"Fitting Hierarchical Naive Baseline")
        print(f"{'='*60}")
        
        # Sort once for efficient lookups
        train_df = train_df.sort_values(['machine_key', 'ean', 'week_start']).copy()
        
        print("Pre-computing aggregates for fast prediction...")
        
        # Pre-compute: (machine, ean) → most recent average (fast lookup)
        print("  Creating (machine, ean) recent averages...")
        self.recent_averages = {}
        for (machine, ean), group in train_df.groupby(['machine_key', 'ean']):
            recent_sales = group[group['weekly_sales'] > 0]['weekly_sales'].tail(self.min_samples).values
            if len(recent_sales) >= self.min_samples:
                self.recent_averages[(machine, ean)] = recent_sales.mean()
        
        # Pre-compute machine-category averages
        print("  Creating machine-category averages...")
        self.machine_subcat_avg = {}
        self.machine_cat_avg = {}
        for machine in train_df['machine_key'].unique():
            machine_df = train_df[train_df['machine_key'] == machine]
            self.machine_subcat_avg[machine] = machine_df.groupby('subcategory')['weekly_sales'].mean().to_dict()
            self.machine_cat_avg[machine] = machine_df.groupby('category')['weekly_sales'].mean().to_dict()
        
        # Pre-compute product-machine group averages
        print("  Creating product-machine group averages...")
        self.product_subgroup_avg = {}
        self.product_eva_avg = {}
        for ean in train_df['ean'].unique():
            product_df = train_df[train_df['ean'] == ean]
            self.product_subgroup_avg[ean] = product_df.groupby('machine_sub_group')['weekly_sales'].mean().to_dict()
            self.product_eva_avg[ean] = product_df.groupby('machine_eva_group')['weekly_sales'].mean().to_dict()
        
        # Fallback: Subcategory × machine_subgroup average
        subcat_machine_agg = (
            train_df.groupby(['subcategory', 'machine_sub_group'])['weekly_sales']
            .mean()
            .to_dict()
        )
        self.aggregates['subcat_machine'] = subcat_machine_agg
        
        # Store overall average for ultimate fallback
        self.overall_avg = train_df['weekly_sales'].mean()
        
        print(f"✓ Computed aggregates:")
        print(f"    - Recent averages: {len(self.recent_averages):,} pairs")
        print(f"    - Machine-category: {len(self.machine_subcat_avg):,} machines")
        print(f"    - Product-machine: {len(self.product_subgroup_avg):,} products")
        print(f"    - Subcat-machine: {len(subcat_machine_agg)} combinations")
        print(f"✓ Model fit complete (memory efficient)\n")
        
        # Clear training data to save memory (only keep aggregates)
        self.train_df = None
        
        return self
    
    def predict(self, test_df: pd.DataFrame, horizons: List[int] = [1, 2, 3, 4]) -> pd.DataFrame:
        """
        Generate predictions using hierarchical fallback strategy.
        
        Args:
            test_df: Test DataFrame with same structure as training data
            horizons: List of weeks ahead to predict [1, 2, 3, 4]
            
        Returns:
            DataFrame with predictions for each horizon
        """
        print(f"\n{'='*60}")
        print(f"Generating Naive Baseline Predictions")
        print(f"{'='*60}")
        
        # Vectorized approach: group once and reuse
        results = []
        
        # Pre-filter test_df for efficiency
        test_df = test_df.sort_values(['machine_key', 'ean', 'week_start']).copy()
        
        # Fast vectorized prediction using pre-computed aggregates
        for horizon in horizons:
            print(f"\nProcessing horizon +{horizon}...")
            
            for idx, row in test_df.iterrows():
                # Use pre-computed aggregates for fast prediction
                prediction = self._predict_single_row_fast(row)
                
                results.append({
                    'index': idx,
                    'horizon': horizon,
                    'prediction': prediction,
                    'machine_key': row['machine_key'],
                    'ean': row['ean']
                })
            
            print(f"✓ Completed {len(test_df)} predictions for week +{horizon}")
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Pivot to get predictions per horizon
        predictions = results_df.pivot_table(
            index='index',
            columns='horizon',
            values='prediction'
        ).add_prefix('pred_week_')
        
        print(f"\n✓ Generated predictions for {len(predictions)} samples")
        return predictions
    
    def _predict_single_row(
        self, 
        row: pd.Series,
        history: pd.DataFrame,
        machine_key: str,
        ean: str
    ) -> float:
        """
        Hierarchical prediction for one (machine, ean, week).
        
        Fallback strategy:
        1. Direct history: Last 4 non-zero sales for (machine, ean)
        2. Machine context: Same subcategory/category in machine
        3. Product context: Same product in similar machines
        4. Fallback: Subcategory × machine_subgroup average
        """
        
        # Step 1: Try direct history
        direct_history = history[
            (history['machine_key'] == machine_key) &
            (history['ean'] == ean) &
            (history['weekly_sales'] > 0)
        ]['weekly_sales'].values
        
        if len(direct_history) >= self.min_samples:
            return direct_history[-self.min_samples:].mean()
        
        # Step 2: Try machine context (same subcategory/category in machine)
        machine_history = history[
            (history['machine_key'] == machine_key) &
            (history['weekly_sales'] > 0)
        ]
        
        # Try subcategory first
        subcat = row.get('subcategory')
        if subcat:
            subcat_sales = machine_history[
                machine_history['subcategory'] == subcat
            ]['weekly_sales'].values
            
            if len(subcat_sales) >= self.min_samples:
                return subcat_sales[-self.min_samples:].mean()
        
        # Try category
        category = row.get('category')
        if category:
            cat_sales = machine_history[
                machine_history['category'] == category
            ]['weekly_sales'].values
            
            if len(cat_sales) >= self.min_samples:
                return cat_sales[-self.min_samples:].mean()
        
        # Use all machine sales as fallback
        if len(machine_history) >= self.min_samples:
            return machine_history['weekly_sales'].tail(self.min_samples).mean()
        
        # Step 3: Try product context (same product in similar machines)
        product_history = history[
            (history['ean'] == ean) &
            (history['weekly_sales'] > 0)
        ]
        
        # Try same machine_sub_group
        machine_sub = row.get('machine_sub_group')
        if machine_sub and len(product_history) > 0:
            sub_group_sales = product_history[
                product_history['machine_sub_group'] == machine_sub
            ]['weekly_sales'].values
            
            if len(sub_group_sales) >= self.min_samples:
                return sub_group_sales[-self.min_samples:].mean()
        
        # Try same machine_eva_group
        machine_eva = row.get('machine_eva_group')
        if machine_eva and len(product_history) > 0:
            eva_group_sales = product_history[
                product_history['machine_eva_group'] == machine_eva
            ]['weekly_sales'].values
            
            if len(eva_group_sales) >= self.min_samples:
                return eva_group_sales[-self.min_samples:].mean()
        
        # Use all product sales as fallback
        if len(product_history) >= self.min_samples:
            return product_history['weekly_sales'].tail(self.min_samples).mean()
        
        # Step 4: Fallback to subcategory × machine_subgroup aggregate
        subcat = row.get('subcategory')
        machine_sub = row.get('machine_sub_group')
        
        if subcat and machine_sub:
            key = (subcat, machine_sub)
            if key in self.aggregates['subcat_machine']:
                return self.aggregates['subcat_machine'][key]
        
        # Ultimate fallback: overall dataset average (pre-computed)
        return self.overall_avg
    
    def _predict_single_row_fast(self, row: pd.Series) -> float:
        """
        Fast hierarchical prediction using pre-computed aggregates.
        
        Fallback strategy:
        1. Recent average: Pre-computed (machine, ean) average
        2. Machine context: Pre-computed machine-category averages
        3. Product context: Pre-computed product-machine group averages
        4. Fallback: Pre-computed subcategory × machine_subgroup average
        """
        machine_key = row['machine_key']
        ean = row['ean']
        
        # Step 1: Try pre-computed recent average
        if (machine_key, ean) in self.recent_averages:
            return self.recent_averages[(machine_key, ean)]
        
        # Step 2: Try machine subcategory average
        subcat = row.get('subcategory')
        if machine_key in self.machine_subcat_avg and subcat:
            subcat_avg = self.machine_subcat_avg[machine_key].get(subcat)
            if subcat_avg:
                return subcat_avg
        
        # Try machine category average
        category = row.get('category')
        if machine_key in self.machine_cat_avg and category:
            cat_avg = self.machine_cat_avg[machine_key].get(category)
            if cat_avg:
                return cat_avg
        
        # Step 3: Try product in similar machine groups
        machine_sub = row.get('machine_sub_group')
        if ean in self.product_subgroup_avg and machine_sub:
            subgroup_avg = self.product_subgroup_avg[ean].get(machine_sub)
            if subgroup_avg:
                return subgroup_avg
        
        machine_eva = row.get('machine_eva_group')
        if ean in self.product_eva_avg and machine_eva:
            eva_avg = self.product_eva_avg[ean].get(machine_eva)
            if eva_avg:
                return eva_avg
        
        # Step 4: Fallback to subcategory × machine_subgroup aggregate
        if subcat and machine_sub:
            key = (subcat, machine_sub)
            if key in self.aggregates['subcat_machine']:
                return self.aggregates['subcat_machine'][key]
        
        # Ultimate fallback: overall average
        return self.overall_avg
    
    def save(self, path: Path):
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump({
                'lookback_weeks': self.lookback_weeks,
                'min_samples': self.min_samples,
                'aggregates': self.aggregates,
                'recent_averages': self.recent_averages if hasattr(self, 'recent_averages') else {},
                'machine_subcat_avg': self.machine_subcat_avg if hasattr(self, 'machine_subcat_avg') else {},
                'machine_cat_avg': self.machine_cat_avg if hasattr(self, 'machine_cat_avg') else {},
                'product_subgroup_avg': self.product_subgroup_avg if hasattr(self, 'product_subgroup_avg') else {},
                'product_eva_avg': self.product_eva_avg if hasattr(self, 'product_eva_avg') else {},
                'overall_avg': self.overall_avg if hasattr(self, 'overall_avg') else 0
            }, f)
        
        print(f"✓ Model saved to {path}")
    
    @classmethod
    def load(cls, path: Path) -> 'HierarchicalNaiveForecaster':
        """Load model from disk."""
        path = Path(path)
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        forecaster = cls(
            lookback_weeks=data['lookback_weeks'],
            min_samples=data['min_samples']
        )
        forecaster.aggregates = data['aggregates']
        
        # Load pre-computed aggregates if they exist
        if 'recent_averages' in data:
            forecaster.recent_averages = data['recent_averages']
        if 'machine_subcat_avg' in data:
            forecaster.machine_subcat_avg = data['machine_subcat_avg']
        if 'machine_cat_avg' in data:
            forecaster.machine_cat_avg = data['machine_cat_avg']
        if 'product_subgroup_avg' in data:
            forecaster.product_subgroup_avg = data['product_subgroup_avg']
        if 'product_eva_avg' in data:
            forecaster.product_eva_avg = data['product_eva_avg']
        if 'overall_avg' in data:
            forecaster.overall_avg = data['overall_avg']
        
        print(f"✓ Model loaded from {path}")
        return forecaster


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def evaluate_naive_baseline(
    test_df: pd.DataFrame,
    predictions: pd.DataFrame,
    horizons: List[int] = [1, 2, 3, 4]
) -> pd.DataFrame:
    """
    Evaluate naive baseline predictions.
    
    Args:
        test_df: Test DataFrame with actual values
        predictions: Predictions DataFrame from forecaster
        horizons: List of weeks ahead
        
    Returns:
        Metrics DataFrame
    """
    metrics = []
    
    for horizon in horizons:
        pred_col = f'pred_week_{horizon}'
        
        if pred_col not in predictions.columns:
            continue
        
        # Get actual values (target shifted by horizon)
        actual_col = f'target_week_{horizon}'
        if actual_col not in test_df.columns:
            # Calculate actuals on the fly
            actuals = test_df.groupby(['machine_key', 'ean'])['weekly_sales'].shift(-horizon)
        else:
            actuals = test_df[actual_col]
        
        # Align indices
        valid_mask = actuals.notna() & predictions[pred_col].notna()
        y_true = actuals[valid_mask]
        y_pred = predictions[pred_col][valid_mask]
        
        if len(y_true) > 0:
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            
            metrics.append({
                'horizon': f'Week +{horizon}',
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'samples': len(y_true)
            })
    
    return pd.DataFrame(metrics)

