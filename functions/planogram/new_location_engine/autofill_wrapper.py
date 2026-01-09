import sys
import logging
from pathlib import Path
import pandas as pd
import joblib

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure functions root is in path to allow 'planogram' imports
FILE_DIR = Path(__file__).resolve().parent
FUNCTIONS_DIR = FILE_DIR.parents[1]
if str(FUNCTIONS_DIR) not in sys.path:
    sys.path.append(str(FUNCTIONS_DIR))

try:
    from planogram.autofill import run_autofill_workflow, run_swap_workflow
except ImportError as e:
    logger.error(f"Failed to import autofill modules: {e}")
    raise

class AutofillService:
    def __init__(self):
        self.artifacts_dir = FUNCTIONS_DIR / 'data' / 'pipeline_artifacts'
        self.machines_df = None
        self.products_df = None
        self.predictions_df = None
        self.artifacts = None
        self.forecaster = None
        
        # Default Weights (from run_autofill_pipeline.ipynb)
        self.static_weights = {
            'healthiness': 0.6, 
            'location': 1.0,
            'confidence': 0.8     
        }
        self.dynamic_weights = {
            'uniqueness': 1.0, 
            'cousin': 0.0,      
            'inventory': 0.8   
        }
        
        self.load_data()

    def load_data(self):
        """Load all necessary data artifacts."""
        logger.info("Loading artifacts for AutofillService...")
        try:
            # Load DataFrames
            self.machines_df = pd.read_parquet(self.artifacts_dir / 'machines.parquet')
            self.products_df = pd.read_parquet(self.artifacts_dir / 'products.parquet')
            
            # Load Predictions (Static fallback)
            self.predictions_df = pd.read_parquet(self.artifacts_dir / 'static_predictions.parquet')

            # Load Artifacts (Mappings, Models) - Saved as pickle/joblib
            artifacts_path = self.artifacts_dir / 'artifacts.pkl'
            if artifacts_path.exists():
                self.artifacts = joblib.load(artifacts_path)
            else:
                logger.warning(f"artifacts.pkl not found at {artifacts_path}. Some features may fail.")
                self.artifacts = {}
                
            # Initialize LocationForecaster
            from planogram.new_location_engine.location_forecast import LocationForecaster
            self.forecaster = LocationForecaster()

            logger.info("Autofill artifacts loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading autofill artifacts: {e}")
            raise

    def get_machine_config(self, slots: list, machine_id: str) -> dict:
        """Reconstruct machine config dictionary from slots list."""
        return {
            'machine_key': machine_id,
            'slots': slots
        }
        
    def _get_predictions(self, lat: float, lon: float, machine_id: str):
        """Get location-specific predictions if lat/lon provided, else fallback."""
        if lat is not None and lon is not None:
            logger.info(f"Generating location-specific predictions for {lat}, {lon}")
            preds = self.forecaster.get_all_predictions(lat, lon)
            
            if preds.empty:
                logger.warning("Dynamic predictions empty.")
                return pd.DataFrame() # Will trigger fallback
            
            # Formatter for autofill compatibility
            # 1. Ensure EAN (Merge with products)
            if 'ean' not in preds.columns:
                logger.info(f"Merging EANs. Preds cols: {preds.columns}. Products cols: {self.products_df.columns}")
                # Assuming preds has 'product_name'
                # products_df has 'product_name' and 'ean'
                # We prioritize the products_df EANs
                product_lookup = self.products_df[['product_name', 'ean']].drop_duplicates('product_name')
                preds = preds.merge(product_lookup, on='product_name', how='left')
                logger.info(f"Merge complete. New cols: {preds.columns}")
            
            # 2. Set machine_key to match the request
            preds['machine_key'] = str(machine_id)
            
            # 3. Rename prediction column
            if 'pred_week_4' in preds.columns:
                preds['predicted_weekly_revenue'] = preds['pred_week_4']
            elif 'predicted_weekly_revenue' not in preds.columns:
                # Fallback if pred_week_4 missing?
                preds['predicted_weekly_revenue'] = 0.0
                
            return preds

        return self.predictions_df

    def autofill(self, slots: list, machine_id: str, lat: float = None, lon: float = None, static_weights: dict = None, dynamic_weights: dict = None) -> list:
        """Run the autofill workflow (Phase 1: Fill empty slots)."""
        logger.info(f"Running autofill for {machine_id}")
        
        predictions = self._get_predictions(lat, lon, machine_id)
        if predictions.empty:
            logger.warning("Predictions empty, using static fallback")
            predictions = self.predictions_df

        machine_config = self.get_machine_config(slots, machine_id)
        
        updated_config = run_autofill_workflow(
            machine_config=machine_config,
            predictions_df=predictions,
            products_df=self.products_df,
            static_weights=static_weights if static_weights else self.static_weights,
            dynamic_weights=dynamic_weights if dynamic_weights else self.dynamic_weights,
            artifacts=self.artifacts
        )
        
        return updated_config.get('slots', [])

    def optimize(self, slots: list, machine_id: str, lat: float = None, lon: float = None, static_weights: dict = None, dynamic_weights: dict = None) -> list:
        """Run the optimize workflow (Phase 2: Swap products)."""
        logger.info(f"Running optimize (swap) for {machine_id}")
        
        predictions = self._get_predictions(lat, lon, machine_id)
        if predictions.empty:
            logger.warning("Predictions empty, using static fallback")
            predictions = self.predictions_df

        machine_config = self.get_machine_config(slots, machine_id)
        
        updated_config = run_swap_workflow(
            machine_config=machine_config,
            predictions_df=predictions,
            products_df=self.products_df,
            static_weights=static_weights if static_weights else self.static_weights,
            dynamic_weights=dynamic_weights if dynamic_weights else self.dynamic_weights,
            artifacts=self.artifacts
        )
        
        return updated_config.get('slots', [])
