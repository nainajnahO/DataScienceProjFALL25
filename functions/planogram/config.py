# uno/functions/planogram/config.py
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Firebase collection names
FIREBASE_COLLECTIONS = {
    'MACHINES': 'app_machines',
    'PRODUCTS': 'products',
    'PURCHASE_PRICES': 'product_purchase_prices',
    'SALES_AWS': 'aws_sales',
    'CATEGORIES': 'subcategories',
    'PRODUCT_NAYAX_MAPPING': 'product_nayax_mapping',
}

# Path to local sales data (for now)
PROCESSED_SALES_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed')
SWEDISH_HOLIDAYS_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'SwedishHolidays(2017-26).csv')
PRODUCT_INFORMATION_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'products.json')

# Path to save/load trained artifacts
ARTIFACTS_DIR = os.path.join(BASE_DIR, 'planogram', 'artifacts')

# Target collection for the output
OUTPUT_COLLECTION_STATIC = 'app_machines_scores'
OUTPUT_COLLECTION_LETTER_GRADE_MAPPING = 'planogram_letter_grade_mapping'
# Snapshot Configuration
SNAPSHOT_N_DAYS = 30
SNAPSHOT_DATE_INTERVAL_DAYS = 7
SNAPSHOT_MIN_SALES = 10

# Training toggles for `train_all`. Set a model's flag to False to skip it.
TRAINING_TASKS = {
    'uniqueness_model': True,
    'location_mapping': True,
    'snapshot_model': False,
    'cousin_model': False,
    'healthiness_mapping': True,
    'moaaz_trend': False,
}

PREDICTION_STATIC_TASKS = {
    'location_mapping': True,
    'healthiness_mapping': True,
    'moaaz_trend': True,
}

PREDICTION_DYNAMIC_TASKS = {
    'uniqueness_model': True,
    'cousin_model': True,
    'inventory_score': True,
}

# New Location Engine Data Paths (new_location_engine directory)
ICA_STORES_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'scb_supermarkets.parquet')
COMPANIES_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'scb_companies.parquet')

