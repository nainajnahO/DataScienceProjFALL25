import sys
import os
import logging
from pathlib import Path
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure functions root is in path to allow 'planogram' imports
# This file is in functions/planogram/new_location_engine/
# We need functions/ to be in sys.path
FILE_DIR = Path(__file__).resolve().parent
FUNCTIONS_DIR = FILE_DIR.parents[1]
if str(FUNCTIONS_DIR) not in sys.path:
    sys.path.append(str(FUNCTIONS_DIR))

try:
    from planogram import data_loader
    from planogram.new_location_engine.LocationBasedForecasting.geo_similarity import calculate_similarity_to_multiple
    from planogram.prediction_aggregator import aggregate_predictions_weighted
except ImportError as e:
    logger.error(f"Failed to import planogram modules: {e}")
    # Fallback/Error handling if imports fail
    raise

class LocationForecaster:
    def __init__(self):
        self.artifacts_dir = FUNCTIONS_DIR / 'data' / 'pipeline_artifacts'
        self.machines_df = None
        self.static_predictions = None
        self.ica_stores_df = None
        self.companies_df = None
        
        self.load_data()

    def load_data(self):
        """Load all necessary data artifacts."""
        logger.info("Loading artifacts for LocationForecaster...")
        try:
            self.machines_df = pd.read_parquet(self.artifacts_dir / 'machines.parquet')
            self.static_predictions = pd.read_parquet(self.artifacts_dir / 'static_predictions.parquet')
            
            # Load aux data
            self.ica_stores_df = data_loader.load_ica_stores()
            self.companies_df = data_loader.load_companies()
            
            logger.info("Artifacts loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading artifacts: {e}")
            raise

    def predict(self, latitude: float, longitude: float, products: list, machine_category: str = None) -> pd.DataFrame:
        """
        Generate sales predictions for a specific location.
        """
        logger.info(f"Running prediction for Lat: {latitude}, Lon: {longitude}")
        
    def _get_weighted_predictions(self, latitude: float, longitude: float):
        """
        Helper: Calculate weighted avg predictions for ALL products based on location similarity.
        Returns DataFrame with 'product_name' and 'pred_week_4'.
        """
        # 1. Similarity Calculation
        all_similarities = calculate_similarity_to_multiple(
            target_lat=latitude,
            target_lon=longitude,
            reference_locations_df=self.machines_df,
            ica_stores_df=self.ica_stores_df,
            companies_df=self.companies_df,
            distance_weight=0.1,
            ica_weight=0.45,
            company_weight=0.45
        )
        
        # 2. Filter
        filtered_sim = all_similarities[all_similarities['similarity'] > 0.65]
        
        # DEBUG: Print top similarities
        print("\n--- DEBUG: Similarity Calculation ---")
        if not filtered_sim.empty:
            print(f"Found {len(filtered_sim)} machines with similarity > 0.65")
            print("Top 5 matches:")
            print(filtered_sim[['machine_key', 'similarity']].head(5).to_string(index=False))
        else:
            print("No machines found with similarity > 0.65")
        
        if filtered_sim.empty:
            logger.warning("No machines found with similarity > 0.65.")
            return pd.DataFrame()

        # Take top 20
        top_20_sim = filtered_sim.head(20)
        top_20_keys = top_20_sim['machine_key']
        
        # Filter static predictions
        filtered_static_predictions = self.static_predictions[
            self.static_predictions['machine_key'].isin(top_20_keys)
        ]
        
        if filtered_static_predictions.empty:
             logger.warning("No static predictions found for the similar machines.")
             return pd.DataFrame()

        # 3. Aggregate
        aggregated_predictions = aggregate_predictions_weighted(
            predictions_df=filtered_static_predictions,
            similarity_df=top_20_sim,
            product_col='product_name',
            machine_key_col='machine_key',
            similarity_col='similarity'
        )
        return aggregated_predictions

    def get_all_predictions(self, latitude: float, longitude: float) -> pd.DataFrame:
        """
        Public method to get predictions for ALL products (for Autofill/Optimize).
        """
        agg_df = self._get_weighted_predictions(latitude, longitude)
        return agg_df

    def predict(self, latitude: float, longitude: float, products: list, machine_category: str = None) -> pd.DataFrame:
        """
        Generate sales predictions for a specific location and list of products.
        """
        logger.info(f"Running prediction for Lat: {latitude}, Lon: {longitude}")
        
        agg_df = self._get_weighted_predictions(latitude, longitude)
        
        if agg_df.empty:
             return pd.DataFrame(columns=['product', 'rule_based_prediction', 'explanation'])
        
        # Ensure 'product_name' is available
        if 'product_name' not in agg_df.columns and agg_df.index.name == 'product_name':
             agg_df = agg_df.reset_index()
             
        results = []
        for product_name in products:
            match = agg_df[agg_df['product_name'] == product_name]
            
            prediction_val = 0.0
            explanation = "No data"
            
            if not match.empty:
                if 'pred_week_4' in match.columns:
                    prediction_val = float(match['pred_week_4'].iloc[0])
                    explanation = "Weighted avg of similar machines"
                else:
                    explanation = "Missing week 4 prediction"
            
            results.append({
                'product': product_name,
                'rule_based_prediction': prediction_val,
                'explanation': explanation
            })
            
        return pd.DataFrame(results)
