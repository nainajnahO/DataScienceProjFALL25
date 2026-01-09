"""
Predict Swap Outcomes
=====================

Use trained models to predict the outcome of a potential product swap.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path


class SwapPredictor:
    """Predict outcomes of product swaps."""
    
    def __init__(self, model_dir=None, data_dir=None):
        """
        Load trained models and build product lookup database.
        
        Args:
            model_dir: Directory containing trained models (default: ../models relative to this file)
            data_dir: Directory containing product data files (default: ../data relative to this file)
        """
        # Default to parent directory (project root) if not specified
        if model_dir is None:
            model_dir = Path(__file__).parent.parent / 'models'
        if data_dir is None:
            data_dir = Path(__file__).parent.parent / 'data'
        
        model_dir = Path(model_dir)
        data_dir = Path(data_dir)
        
        print("Loading models...")
        self.revenue_model = joblib.load(model_dir / 'revenue_model.pkl')
        # Try to load profit model (primary), fallback to revenue if not available
        profit_model_path = model_dir / 'profit_model.pkl'
        if profit_model_path.exists():
            self.profit_model = joblib.load(profit_model_path)
            print("  ✓ Profit model loaded (PRIMARY)")
        else:
            self.profit_model = None
            print("  ⚠ Profit model not found, using revenue model as proxy")
        self.success_model = joblib.load(model_dir / 'success_model.pkl')
        self.encoders = joblib.load(model_dir / 'encoders.pkl')
        self.feature_cols = joblib.load(model_dir / 'feature_columns.pkl')
        print("✓ Models loaded successfully")
        
        # Build product lookup database
        print("\nBuilding product lookup database...")
        self._build_product_lookup(data_dir)
        print("✓ Product lookup database ready")
    
    def _build_product_lookup(self, data_dir):
        """Build lookup dictionaries for product metadata and historical performance."""
        # 1. Load product aggregates for metadata
        product_agg_path = data_dir / 'aggregates' / 'product_profit_revenue.parquet'
        if product_agg_path.exists():
            print(f"  Loading product aggregates from {product_agg_path.name}...")
            product_agg = pd.read_parquet(product_agg_path)
            
            # Build EAN lookup for metadata
            self.product_metadata = {}
            for _, row in product_agg.iterrows():
                ean = str(row.get('ean', '')) if pd.notna(row.get('ean')) else None
                if ean:
                    self.product_metadata[ean] = {
                        'subcategory': row.get('subcategory'),
                        'category': row.get('category'),
                        'provider': row.get('provider'),
                        'avg_revenue': row.get('avg_revenue', 0),
                        'avg_profit': row.get('avg_profit', 0),
                        'avg_price': row.get('avg_price', 0)
                    }
            
            # Build subcategory lookup (for fallback)
            self.subcategory_metadata = {}
            subcat_agg = product_agg.groupby('subcategory').agg({
                'avg_revenue': 'mean',
                'avg_profit': 'mean',
                'avg_price': 'mean',
                'provider': lambda x: x.mode()[0] if len(x.mode()) > 0 else None
            }).to_dict('index')
            self.subcategory_metadata = subcat_agg
            
            # Build category lookup (for final fallback)
            self.category_metadata = {}
            if 'category' in product_agg.columns:
                cat_agg = product_agg.groupby('category').agg({
                    'avg_revenue': 'mean',
                    'avg_profit': 'mean',
                    'avg_price': 'mean'
                }).to_dict('index')
                self.category_metadata = cat_agg
            
            print(f"    Loaded metadata for {len(self.product_metadata):,} products")
        else:
            print(f"  Warning: {product_agg_path} not found, using fallback lookups only")
            self.product_metadata = {}
            self.subcategory_metadata = {}
            self.category_metadata = {}
        
        # 2. Load enriched swaps for historical performance
        swaps_path = data_dir / 'swaps' / 'product_swaps_enriched.parquet'
        if swaps_path.exists():
            print(f"  Loading historical performance from {swaps_path.name}...")
            swaps = pd.read_parquet(swaps_path)
            
            # Build historical performance lookup by product (EAN or product name)
            self.historical_performance = {}
            
            # Try to match by EAN first
            if 'ean_before' in swaps.columns or 'product_before' in swaps.columns:
                ean_col = 'ean_before' if 'ean_before' in swaps.columns else None
                product_col = 'product_before' if 'product_before' in swaps.columns else None
                
                for _, row in swaps.iterrows():
                    # Try to get EAN
                    ean = None
                    if ean_col and pd.notna(row.get(ean_col)):
                        ean = str(row[ean_col])
                    elif product_col and pd.notna(row.get(product_col)):
                        # Check if product_before is numeric (might be EAN)
                        try:
                            prod_str = str(row[product_col])
                            if prod_str.isdigit() and len(prod_str) >= 8:
                                ean = prod_str
                        except:
                            pass
                    
                    # Also try subcategory as identifier if EAN not available
                    identifier = ean if ean else None
                    if not identifier and 'subcategory_before' in swaps.columns and pd.notna(row.get('subcategory_before')):
                        identifier = f"SUBCAT_{row['subcategory_before']}"
                    
                    if identifier:
                        # Aggregate historical performance
                        if identifier not in self.historical_performance:
                            self.historical_performance[identifier] = {
                                'revenue_before_4w': [],
                                'profit_before_4w': [],
                                'sales_count_before_4w': [],
                                'days_observed_before_4w': []
                            }
                        
                        if pd.notna(row.get('revenue_before_4w')):
                            self.historical_performance[identifier]['revenue_before_4w'].append(row['revenue_before_4w'])
                        if pd.notna(row.get('profit_before_4w')):
                            self.historical_performance[identifier]['profit_before_4w'].append(row['profit_before_4w'])
                        if pd.notna(row.get('sales_count_before_4w')):
                            self.historical_performance[identifier]['sales_count_before_4w'].append(row['sales_count_before_4w'])
                        if pd.notna(row.get('days_observed_before_4w')):
                            self.historical_performance[identifier]['days_observed_before_4w'].append(row['days_observed_before_4w'])
            
            # Calculate averages for each product
            for ean, data in self.historical_performance.items():
                self.historical_performance[ean] = {
                    'revenue_before_4w': np.mean(data['revenue_before_4w']) if data['revenue_before_4w'] else 0,
                    'profit_before_4w': np.mean(data['profit_before_4w']) if data['profit_before_4w'] else 0,
                    'sales_count_before_4w': np.mean(data['sales_count_before_4w']) if data['sales_count_before_4w'] else 0,
                    'days_observed_before_4w': np.mean(data['days_observed_before_4w']) if data['days_observed_before_4w'] else 28
                }
            
            # Build subcategory-level historical performance (for fallback)
            self.subcategory_performance = {}
            if 'subcategory_before' in swaps.columns:
                subcat_perf = swaps.groupby('subcategory_before').agg({
                    'revenue_before_4w': 'mean',
                    'profit_before_4w': 'mean',
                    'sales_count_before_4w': 'mean',
                    'days_observed_before_4w': 'mean'
                }).to_dict('index')
                self.subcategory_performance = subcat_perf
            
            print(f"    Loaded historical performance for {len(self.historical_performance):,} products")
        else:
            print(f"  Warning: {swaps_path} not found, using category averages only")
            self.historical_performance = {}
            self.subcategory_performance = {}
    
    def _lookup_product(self, product_identifier):
        """
        Look up product information and historical performance.
        
        Args:
            product_identifier: EAN code (str), or dict with {ean, subcategory, category}
        
        Returns:
            dict with product info: {ean, subcategory, category, provider, 
                                    revenue_before_4w, profit_before_4w, 
                                    sales_count_before_4w, days_observed_before_4w}
        """
        # Handle different input formats
        if isinstance(product_identifier, dict):
            ean = product_identifier.get('ean')
            subcategory = product_identifier.get('subcategory')
            category = product_identifier.get('category')
        else:
            # Assume it's an EAN code (string)
            ean = str(product_identifier) if product_identifier else None
            subcategory = None
            category = None
        
        result = {
            'ean': ean,
            'subcategory': None,
            'category': None,
            'provider': None,
            'revenue_before_4w': 0,
            'profit_before_4w': 0,
            'sales_count_before_4w': 0,
            'days_observed_before_4w': 28
        }
        
        # Try to get metadata from EAN
        if ean and ean in self.product_metadata:
            meta = self.product_metadata[ean]
            result['subcategory'] = meta.get('subcategory') or subcategory
            result['category'] = meta.get('category') or category
            result['provider'] = meta.get('provider')
        else:
            # Use provided subcategory/category or try to look up
            result['subcategory'] = subcategory
            result['category'] = category
            
            # Try to get metadata from subcategory
            if subcategory and subcategory in self.subcategory_metadata:
                meta = self.subcategory_metadata[subcategory]
                result['provider'] = meta.get('provider')
        
        # Get historical performance - try EAN first, then subcategory
        if ean and ean in self.historical_performance:
            perf = self.historical_performance[ean]
            result['revenue_before_4w'] = perf.get('revenue_before_4w', 0)
            result['profit_before_4w'] = perf.get('profit_before_4w', 0)
            result['sales_count_before_4w'] = perf.get('sales_count_before_4w', 0)
            result['days_observed_before_4w'] = perf.get('days_observed_before_4w', 28)
        elif result['subcategory']:
            # Try subcategory identifier
            subcat_id = f"SUBCAT_{result['subcategory']}"
            if subcat_id in self.historical_performance:
                perf = self.historical_performance[subcat_id]
                result['revenue_before_4w'] = perf.get('revenue_before_4w', 0)
                result['profit_before_4w'] = perf.get('profit_before_4w', 0)
                result['sales_count_before_4w'] = perf.get('sales_count_before_4w', 0)
                result['days_observed_before_4w'] = perf.get('days_observed_before_4w', 28)
            elif result['subcategory'] in self.subcategory_performance:
                # Fallback to subcategory averages
                perf = self.subcategory_performance[result['subcategory']]
                result['revenue_before_4w'] = perf.get('revenue_before_4w', 0)
                result['profit_before_4w'] = perf.get('profit_before_4w', 0)
                result['sales_count_before_4w'] = perf.get('sales_count_before_4w', 0)
                result['days_observed_before_4w'] = perf.get('days_observed_before_4w', 28)
        elif result['subcategory'] and result['subcategory'] in self.subcategory_metadata:
            # Use subcategory metadata averages
            meta = self.subcategory_metadata[result['subcategory']]
            # Estimate from average revenue/profit (scale to 4 weeks)
            avg_revenue = meta.get('avg_revenue', 0)
            avg_profit = meta.get('avg_profit', 0)
            result['revenue_before_4w'] = avg_revenue * 20  # Rough estimate for 4 weeks
            result['profit_before_4w'] = avg_profit * 20
            result['sales_count_before_4w'] = 10  # Default estimate
            result['days_observed_before_4w'] = 28
        
        return result
    
    def _get_recommendation(self, profit_change, revenue_change, success_proba, confidence):
        """Generate recommendation based on profit (primary) and revenue (secondary)."""
        if profit_change > 0:
            if revenue_change > 0:
                return "✅ Recommended - Increases both profit and revenue"
            elif revenue_change >= -10:
                return "✅ Recommended - Increases profit (revenue slightly down)"
            else:
                return "⚠️ Marginal - Increases profit but revenue down significantly"
        elif profit_change >= -5 and revenue_change > 10:
            return "⚠️ Marginal - Profit flat but revenue up"
        else:
            return "❌ Not recommended - Decreases profit"
    
    def _get_product_id(self, ean, subcategory, category, ean_counts, min_ean_count, prefix='before'):
        """
        Get product identifier using EAN-first fallback hierarchy.
        
        Hierarchy: EAN → subcategory → category
        
        Args:
            ean: EAN code (str or None)
            subcategory: Subcategory name (str or None)
            category: Category name (str or None)
            ean_counts: Dict of EAN occurrence counts from training
            min_ean_count: Minimum occurrences needed to use EAN
            prefix: 'before' or 'after' for error messages
            
        Returns:
            Product identifier string (EAN_xxx, SUBCAT_xxx, or CAT_xxx)
        """
        # Try EAN first
        if ean is not None and pd.notna(ean):
            ean_str = str(ean)
            # Check if EAN has sufficient history in training data
            if ean_str in ean_counts and ean_counts[ean_str] >= min_ean_count:
                return f"EAN_{ean_str}"
        
        # Fallback to subcategory
        if subcategory is not None and pd.notna(subcategory):
            return f"SUBCAT_{subcategory}"
        
        # Final fallback to category
        if category is not None and pd.notna(category):
            return f"CAT_{category}"
        
        # Ultimate fallback
        return "Unknown"
    
    def prepare_swap_features(self, swap_data):
        """
        Prepare features for a potential swap.
        
        Args:
            swap_data: dict with swap information:
                - ean_before: str (optional) - EAN code of product being removed
                - ean_after: str (optional) - EAN code of product being added
                - subcategory_before: str - Subcategory of product being removed
                - subcategory_after: str - Subcategory of product being added
                - category_before: str (optional) - Category of product being removed
                - category_after: str (optional) - Category of product being added
                - provider_before: str - Provider/brand of product being removed
                - provider_after: str - Provider/brand of product being added
                - revenue_before_4w: float - Total revenue (SEK) from current product over 4 weeks
                - profit_before_4w: float - Total profit (SEK) from current product over 4 weeks
                - sales_count_before_4w: int - Number of sales transactions in 4-week period
                - days_observed_before_4w: int (default: 28) - Days observed
                
        Returns:
            DataFrame with features
        """
        # Fill defaults
        if 'days_observed_before_4w' not in swap_data:
            swap_data['days_observed_before_4w'] = 28
        
        # Get EAN counts and min count from encoders (if available)
        ean_before_counts = self.encoders.get('ean_before_counts', {})
        ean_after_counts = self.encoders.get('ean_after_counts', {})
        min_ean_count = self.encoders.get('min_ean_count', 3)
        
        # Calculate derived features
        data = {}
        
        # 1. Product similarity
        data['same_category'] = int(swap_data.get('subcategory_before') == swap_data.get('subcategory_after'))
        data['same_provider'] = int(swap_data.get('provider_before') == swap_data.get('provider_after'))
        
        # 2. Historical performance
        data['revenue_per_day_before'] = swap_data['revenue_before_4w'] / swap_data['days_observed_before_4w']
        data['profit_per_day_before'] = swap_data['profit_before_4w'] / swap_data['days_observed_before_4w']
        data['sales_per_day_before'] = swap_data['sales_count_before_4w'] / swap_data['days_observed_before_4w']
        
        # 3. Profit margin
        if swap_data['revenue_before_4w'] > 0:
            data['profit_margin_before'] = (swap_data['profit_before_4w'] / swap_data['revenue_before_4w']) * 100
        else:
            data['profit_margin_before'] = 0
        
        data['sales_count_before_4w'] = swap_data['sales_count_before_4w']
        
        # 4. Interaction features
        data['revenue_profit_interaction'] = data['revenue_per_day_before'] * data['profit_per_day_before']
        data['category_provider_interaction'] = data['same_category'] * data['same_provider']
        data['sales_revenue_interaction'] = data['sales_per_day_before'] * data['revenue_per_day_before']
        
        # 5. Optional temporal features (defaults if not provided)
        # Use provided values or defaults (avoid datetime operations for speed)
        data['swap_month'] = swap_data.get('swap_month', 6)  # Default to June
        data['swap_quarter'] = swap_data.get('swap_quarter', 2)  # Default to Q2
        data['is_summer'] = swap_data.get('is_summer', 1)  # Default to summer
        data['is_winter'] = swap_data.get('is_winter', 0)
        data['day_of_year'] = swap_data.get('day_of_year', 180)  # Default to mid-year
        
        # 6. Optional position features (defaults if not provided)
        data['is_top_half'] = swap_data.get('is_top_half', 0)
        data['position_normalized'] = swap_data.get('position_normalized', 0)
        
        # 7. Encode product identifiers using EAN-first fallback hierarchy
        product_id_before = self._get_product_id(
            swap_data.get('ean_before'),
            swap_data.get('subcategory_before'),
            swap_data.get('category_before'),
            ean_before_counts,
            min_ean_count,
            'before'
        )
        
        product_id_after = self._get_product_id(
            swap_data.get('ean_after'),
            swap_data.get('subcategory_after'),
            swap_data.get('category_after'),
            ean_after_counts,
            min_ean_count,
            'after'
        )
        
        # Encode product IDs
        try:
            data['product_before_encoded'] = self.encoders['product_before'].transform([product_id_before])[0]
        except (KeyError, ValueError):
            # If product encoder doesn't exist (backward compatibility) or product not seen, use -1
            data['product_before_encoded'] = -1
        
        try:
            data['product_after_encoded'] = self.encoders['product_after'].transform([product_id_after])[0]
        except (KeyError, ValueError):
            data['product_after_encoded'] = -1
        
        # Also encode subcategory and provider (for backward compatibility and additional features)
        try:
            data['subcategory_before_encoded'] = self.encoders['subcategory_before'].transform(
                [swap_data.get('subcategory_before', 'Unknown')]
            )[0]
        except (KeyError, ValueError):
            data['subcategory_before_encoded'] = -1
        
        try:
            data['subcategory_after_encoded'] = self.encoders['subcategory_after'].transform(
                [swap_data.get('subcategory_after', 'Unknown')]
            )[0]
        except (KeyError, ValueError):
            data['subcategory_after_encoded'] = -1
        
        try:
            data['provider_before_encoded'] = self.encoders['provider_before'].transform(
                [swap_data.get('provider_before', 'Unknown')]
            )[0]
        except (KeyError, ValueError):
            data['provider_before_encoded'] = -1
        
        try:
            data['provider_after_encoded'] = self.encoders['provider_after'].transform(
                [swap_data.get('provider_after', 'Unknown')]
            )[0]
        except (KeyError, ValueError):
            data['provider_after_encoded'] = -1
        
        # Create DataFrame with all features in correct order
        # Use dict comprehension with get() for missing features (faster than checking)
        feature_dict = {col: data.get(col, 0.0) for col in self.feature_cols}
        df = pd.DataFrame([feature_dict])
        
        return df
    
    def predict(self, product_before, product_after):
        """
        Predict swap outcome from just two product identifiers.
        
        Args:
            product_before: EAN code (str) or dict with {ean, subcategory, category}
                          - Product being removed/currently in position
            product_after: EAN code (str) or dict with {ean, subcategory, category}
                         - Product being swapped to/new product
        
        Returns:
            dict with predictions:
                - predicted_revenue_change: float (SEK)
                - success_probability: float (0-1)
                - success_prediction: bool
                - confidence: str (Low/Medium/High)
                - product_before_info: dict with looked up product info
                - product_after_info: dict with looked up product info
        """
        # Look up product information
        before_info = self._lookup_product(product_before)
        after_info = self._lookup_product(product_after)
        
        # Build swap_data dict for feature preparation
        swap_data = {
            'ean_before': before_info['ean'],
            'ean_after': after_info['ean'],
            'subcategory_before': before_info['subcategory'],
            'subcategory_after': after_info['subcategory'],
            'category_before': before_info['category'],
            'category_after': after_info['category'],
            'provider_before': before_info['provider'],
            'provider_after': after_info['provider'],
            'revenue_before_4w': before_info['revenue_before_4w'],
            'profit_before_4w': before_info['profit_before_4w'],
            'sales_count_before_4w': before_info['sales_count_before_4w'],
            'days_observed_before_4w': before_info['days_observed_before_4w']
        }
        
        # Prepare features
        X = self.prepare_swap_features(swap_data)
        
        # Predict profit change (PRIMARY METRIC)
        if self.profit_model is not None:
            profit_change = self.profit_model.predict(X)[0]
        else:
            # Fallback: estimate profit from revenue (assuming 30% margin)
            revenue_change = self.revenue_model.predict(X)[0]
            profit_change = revenue_change * 0.30
        
        # Predict revenue change (secondary metric)
        revenue_change = self.revenue_model.predict(X)[0]
        
        # Predict success probability (based on profit)
        success_proba = self.success_model.predict_proba(X)[0, 1]
        success_pred = success_proba > 0.5
        
        # Confidence level
        if success_proba > 0.7 or success_proba < 0.3:
            confidence = "High"
        elif success_proba > 0.6 or success_proba < 0.4:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        # Recommendation based on profit (primary) and revenue (secondary)
        recommendation = self._get_recommendation(profit_change, revenue_change, success_proba, confidence)
        
        return {
            'predicted_profit_change': round(profit_change, 2),  # PRIMARY
            'predicted_revenue_change': round(revenue_change, 2),  # Secondary
            'success_probability': round(success_proba, 3),
            'success_prediction': bool(success_pred),
            'confidence': confidence,
            'recommendation': recommendation,
            'product_before_info': before_info,
            'product_after_info': after_info
        }
    
    def predict_legacy(self, swap_data):
        """
        Legacy predict method that takes full swap_data dict.
        Use predict() for simplified interface.
        
        Args:
            swap_data: dict with swap information (see prepare_swap_features)
            
        Returns:
            dict with predictions
        """
        # Prepare features
        X = self.prepare_swap_features(swap_data)
        
        # Predict profit change (PRIMARY METRIC)
        if self.profit_model is not None:
            profit_change = self.profit_model.predict(X)[0]
        else:
            # Fallback: estimate profit from revenue (assuming 30% margin)
            revenue_change = self.revenue_model.predict(X)[0]
            profit_change = revenue_change * 0.30
        
        # Predict revenue change (secondary metric)
        revenue_change = self.revenue_model.predict(X)[0]
        
        # Predict success probability (based on profit)
        success_proba = self.success_model.predict_proba(X)[0, 1]
        success_pred = success_proba > 0.5
        
        # Confidence level
        if success_proba > 0.7 or success_proba < 0.3:
            confidence = "High"
        elif success_proba > 0.6 or success_proba < 0.4:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        # Recommendation
        recommendation = self._get_recommendation(profit_change, revenue_change, success_proba, confidence)
        
        return {
            'predicted_profit_change': round(profit_change, 2),  # PRIMARY
            'predicted_revenue_change': round(revenue_change, 2),  # Secondary
            'success_probability': round(success_proba, 3),
            'success_prediction': bool(success_pred),
            'confidence': confidence,
            'recommendation': recommendation
        }
    
    def predict_multiple(self, products_before, products_after):
        """
        Predict outcomes for multiple swaps.
        
        Args:
            products_before: List of product identifiers (EAN or dict) for products being removed
            products_after: List of product identifiers (EAN or dict) for products being swapped to
        
        Returns:
            DataFrame with predictions
        """
        if len(products_before) != len(products_after):
            raise ValueError("products_before and products_after must have the same length")
        
        predictions = []
        for prod_before, prod_after in zip(products_before, products_after):
            pred = self.predict(prod_before, prod_after)
            # Remove detailed info for cleaner output
            pred_clean = {
                'predicted_profit_change': pred['predicted_profit_change'],  # PRIMARY
                'predicted_revenue_change': pred['predicted_revenue_change'],  # Secondary
                'success_probability': pred['success_probability'],
                'success_prediction': pred['success_prediction'],
                'confidence': pred['confidence'],
                'recommendation': pred['recommendation']
            }
            predictions.append(pred_clean)
        
        return pd.DataFrame(predictions)


def example_usage():
    """Example: Predict swap outcome with simplified interface."""
    
    # Initialize predictor (uses default paths relative to this file)
    predictor = SwapPredictor()
    
    print("="*70)
    print("SWAP PREDICTION EXAMPLE")
    print("="*70)
    
    # Example 1: Simple EAN-based prediction
    print("\n--- Example 1: EAN-based prediction ---")
    product_before = "7310070961234"  # Just an EAN code
    product_after = "7310070961241"    # Just an EAN code
    
    print(f"Swapping from: {product_before}")
    print(f"Swapping to:   {product_after}")
    
    result = predictor.predict(product_before, product_after)
    
    print(f"\nProduct Before Info:")
    print(f"  Subcategory: {result['product_before_info'].get('subcategory', 'Unknown')}")
    print(f"  Provider:    {result['product_before_info'].get('provider', 'Unknown')}")
    print(f"  Historical Revenue (4w): {result['product_before_info']['revenue_before_4w']:.2f} SEK")
    
    print(f"\nProduct After Info:")
    print(f"  Subcategory: {result['product_after_info'].get('subcategory', 'Unknown')}")
    print(f"  Provider:    {result['product_after_info'].get('provider', 'Unknown')}")
    
    print(f"\n--- Predictions ---")
    print(f"Predicted Profit Change:  {result['predicted_profit_change']:+.2f} SEK (PRIMARY ⭐)")
    print(f"Predicted Revenue Change: {result['predicted_revenue_change']:+.2f} SEK (secondary)")
    print(f"Success Probability:      {result['success_probability']:.1%}")
    print(f"Success Prediction:       {'✓ Yes' if result['success_prediction'] else '✗ No'}")
    print(f"Confidence:               {result['confidence']}")
    print(f"Recommendation:           {result['recommendation']}")
    
    # Example 2: Fallback to subcategory if EAN not found
    print("\n" + "="*70)
    print("--- Example 2: Fallback to subcategory ---")
    product_before = {'subcategory': 'Läsk & Saft', 'category': 'Drycker'}
    product_after = {'subcategory': 'Läsk & Saft', 'category': 'Drycker'}
    
    print(f"Swapping from: {product_before}")
    print(f"Swapping to:   {product_after}")
    
    result2 = predictor.predict(product_before, product_after)
    
    print(f"\n--- Predictions ---")
    print(f"Predicted Profit Change:  {result2['predicted_profit_change']:+.2f} SEK (PRIMARY ⭐)")
    print(f"Predicted Revenue Change: {result2['predicted_revenue_change']:+.2f} SEK (secondary)")
    print(f"Success Probability:      {result2['success_probability']:.1%}")
    print(f"Success Prediction:       {'✓ Yes' if result2['success_prediction'] else '✗ No'}")
    print(f"Confidence:               {result2['confidence']}")
    print(f"Recommendation:           {result2['recommendation']}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    example_usage()




