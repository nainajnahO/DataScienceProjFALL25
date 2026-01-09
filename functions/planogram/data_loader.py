import os
import pandas as pd
from datetime import datetime
from google.cloud.firestore import Client as FirestoreClient

from .config import FIREBASE_COLLECTIONS, PROCESSED_SALES_DATA_PATH, ICA_STORES_PATH, COMPANIES_PATH

# Direct import from sibling package (works when functions/ is in sys.path)
from sales_report.utils_firestore import firestore_to_df, get_products_df, get_machines_df

UNSPECIFIC_NAME = 'övrigt'
UNSPECIFIC_NAME_CF = UNSPECIFIC_NAME.casefold()


def _normalize_key(value):
    if isinstance(value, str):
        normalized = value.strip()
        if normalized:
            return normalized.casefold()
    return None


def _load_category_lookup(firestore_client: FirestoreClient | None) -> dict[str, str]:
    collection_name = FIREBASE_COLLECTIONS.get('CATEGORIES')
    if not firestore_client or not collection_name:
        return {}

    categories_df = firestore_to_df(collection_name, firestore_client, include_doc_id=False)
    if categories_df.empty or 'subcategory_name' not in categories_df.columns or 'category_name' not in categories_df.columns:
        return {}

    categories_df = categories_df[['subcategory_name', 'category_name']].dropna()
    if categories_df.empty:
        return {}

    categories_df['subcategory_key'] = categories_df['subcategory_name'].astype(str).str.strip().str.casefold()
    categories_df['category_value'] = categories_df['category_name'].astype(str).str.strip()

    return dict(zip(categories_df['subcategory_key'], categories_df['category_value']))


def _attach_category_column(df: pd.DataFrame, category_lookup: dict[str, str], subcategory_column: str = 'subcategory') -> pd.DataFrame:
    if df.empty or not category_lookup or subcategory_column not in df.columns:
        return df

    df = df.copy()

    def map_category(value):
        key = _normalize_key(value)
        return category_lookup.get(key) if key else None

    mapped = df[subcategory_column].apply(map_category)
    if 'category' in df.columns:
        df['category'] = mapped.combine_first(df['category'])
    else:
        df['category'] = mapped

    return df


def _remove_unspecific_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    mask = pd.Series(True, index=df.index)
    if 'too_unspecific' in df.columns:
        mask &= ~df['too_unspecific'].fillna(False)
    if 'product_name' in df.columns:
        mask &= df['product_name'].fillna('').astype(str).str.strip().str.casefold() != UNSPECIFIC_NAME_CF

    return df.loc[mask].copy()


def _clean_machine_slots(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or 'slots' not in df.columns:
        return df

    df = df.copy()

    def clean_slots(slots):
        if isinstance(slots, list):
            cleaned = []
            for slot in slots:
                if isinstance(slot, dict):
                    too_unspecific = slot.get('too_unspecific', False)
                    product_name = slot.get('product_name')
                    name_is_unspecific = False
                    if isinstance(product_name, str):
                        name_is_unspecific = product_name.strip().casefold() == UNSPECIFIC_NAME_CF
                    if too_unspecific or name_is_unspecific:
                        continue
                cleaned.append(slot)
            return cleaned
        return slots

    df['slots'] = df['slots'].apply(clean_slots)
    return df


def _count_slots(df: pd.DataFrame) -> int:
    if df.empty or 'slots' not in df.columns:
        return 0
    return df['slots'].apply(lambda s: len(s) if isinstance(s, list) else 0).sum()


def _log_cleaning_stats(label: str, before: int, after: int):
    removed = before - after
    pct = (removed / before * 100) if before else 0
    print(f"{label}: {before} -> {after} rows (removed {removed}, {pct:.2f}%)")

# get rid of to_unspecifik columns and product_name = övrigt
# make sure we add product category from the mapping in firestore
def load_firebase_collections(firestore_client: FirestoreClient = None) -> dict[str, pd.DataFrame]:
    """
    Loads specified Firebase collections (excluding sales) into a dictionary of DataFrames.
    """
    if firestore_client is None:
        raise Exception("Firestore client not provided in load_firebase_collections")

    category_lookup = _load_category_lookup(firestore_client)

    collections = {}
    purchase_prices_df = None
    for key, name in FIREBASE_COLLECTIONS.items():
        if key in ['CATEGORIES', 'SALES_AWS']:
            continue  # Categories are used internally for lookups and sales_aws needs special attention data is already loaded

        print(f"Loading collection: {name}")
        df = firestore_to_df(name, firestore_client, include_doc_id=True)
        raw_count = len(df)

        if key == 'PRODUCTS':
            df = _attach_category_column(df, category_lookup, subcategory_column='subcategory')
            df = _remove_unspecific_rows(df)
            _log_cleaning_stats(f"{name} (filtered unspecific rows)", raw_count, len(df))
        elif key == 'MACHINES':
            before_slots = _count_slots(df)
            df = _clean_machine_slots(df)
            after_slots = _count_slots(df)
            if before_slots or after_slots:
                removed_slots = before_slots - after_slots
                pct = (removed_slots / before_slots * 100) if before_slots else 0
                print(f"{name} slots cleaned: {before_slots} -> {after_slots} (removed {removed_slots}, {pct:.2f}%)")
            print(f"{name}: {raw_count} rows (machines themselves not filtered)")
            # Ensure we have a stable machine id column
            if 'id' not in df.columns and 'doc_id' in df.columns:
                df['id'] = df['doc_id']

        elif key == 'PURCHASE_PRICES':
            purchase_prices_df = df.copy()
            _log_cleaning_stats(name, raw_count, len(df))

        else:
            _log_cleaning_stats(name, raw_count, len(df))

        collections[key.lower()] = df

    # Enrich products with purchase prices
    if 'products' in collections:
        collections['products'] = enrich_with_purchase_prices(
            collections['products'],
            purchase_prices_df
        )
    
    return collections


def load_processed_sales() -> pd.DataFrame:
    """
    Loads and concatenates all processed sales data from Parquet files.
    """
    all_sales = []
    # Check if directory exists before trying to list its contents
    if not os.path.isdir(PROCESSED_SALES_DATA_PATH):
        print(f"Warning: Directory not found for processed sales data: {PROCESSED_SALES_DATA_PATH}")
        return pd.DataFrame()

    for filename in os.listdir(PROCESSED_SALES_DATA_PATH):
        if filename.endswith('.parquet') and filename not in {"scb_companies.parquet", "scb_supermarkets.parquet"}:
            file_path = os.path.join(PROCESSED_SALES_DATA_PATH, filename)
            all_sales.append(pd.read_parquet(file_path))
    
    if not all_sales:
        return pd.DataFrame()

    return pd.concat(all_sales, ignore_index=True)


def enrich_latest_sales_data(latest_sales_df: pd.DataFrame, firestore_client: FirestoreClient, firebase_data: dict = None) -> pd.DataFrame:
    """
    Enriches latest sales data with product and machine information to match historical columns.
    Uses firebase_data if provided to avoid re-fetching, otherwise fetches necessary collections.
    """
    if latest_sales_df.empty:
        return latest_sales_df

    print("Enriching latest sales data...")
    
    # 1. Get Products and Mapping
    products_df = None
    mapping_df = None
    
    if firebase_data:
        products_df = firebase_data.get('products')
        mapping_df = firebase_data.get('product_nayax_mapping')

    if products_df is None:
        print("Fetching products collection...")
        products_df = firestore_to_df(FIREBASE_COLLECTIONS['PRODUCTS'], firestore_client)

    if mapping_df is None:
        print("Fetching product_nayax_mapping collection...")
        mapping_df = firestore_to_df('product_nayax_mapping', firestore_client)

    # Prepare Product Lookup
    product_lookup = None
    if not products_df.empty:
        # Deduplicate products on product_name to ensure unique keys
        if 'product_name' in products_df.columns:
            if products_df.duplicated(subset=['product_name']).any():
                products_df = products_df.drop_duplicates(subset=['product_name'], keep='first')
        
        if not mapping_df.empty and 'nayax_name' in mapping_df.columns and 'product_name' in mapping_df.columns:
            # Clean mapping
            mapping_df = mapping_df[['nayax_name', 'product_name']].dropna()
            
            # Merge mapping with products to get nayax_name -> product details
            # This creates a dataframe keyed by nayax_name
            product_lookup = mapping_df.merge(
                products_df,
                on='product_name',
                how='left'
            )
        else:
            # Fallback: if no mapping, assume products_df might have nayax_name (rare)
            if 'nayax_name' in products_df.columns:
                product_lookup = products_df

    # 2. Get Machines
    machines_df = None
    if firebase_data:
        machines_df = firebase_data.get('machines')
    
    # Check if the cached machines_df has the necessary columns for enrichment (e.g., address)
    # If not (e.g., it's app_machines), we need to fetch the full machines dataset
    if machines_df is not None and 'address' not in machines_df.columns:
        print("Cached machines data missing 'address' column (likely app_machines). Fetching full machines data...")
        machines_df = None

    if machines_df is None:
        print("Fetching machines collection...")
        machines_df = get_machines_df(local_version=False, firestore_client=firestore_client)

    enriched_df = latest_sales_df.copy()

    # 3. Enrich with Product Info (merge on nayax_name)
    if product_lookup is not None and not product_lookup.empty and 'nayax_name' in enriched_df.columns:
        # Select relevant columns
        product_cols = [
            'nayax_name', 'product_name', 'provider', 'category', 'subcategory', 
            'moms', 'ean', 'purchase_price_kr', 'too_unspecific', 'pusher_friendly', 
            'width', 'spiral'
        ]
        # Only keep columns that exist
        product_cols = [c for c in product_cols if c in product_lookup.columns]
        
        enriched_df = enriched_df.merge(
            product_lookup[product_cols],
            on='nayax_name',
            how='left'
        )

    # 4. Enrich with Machine Info (merge on machine_key)
    if machines_df is not None and not machines_df.empty and 'machine_key' in enriched_df.columns:
        
        # Select relevant columns from machines
        machine_cols = [
            'machine_key', 'machine_name', 'machine_id', 
            'machine_eva_group', 'machine_sub_group', 'machine_group_tag', 
            'machine_model', 'sielaff_id', 'is_ICA_refiller', 
            'address', 'latitude', 'longitude'
        ]
        # Only keep columns that exist in machines_df
        machine_cols = [c for c in machine_cols if c in machines_df.columns]
        
        
        machines_merge = machines_df.copy()
        
        # Drop conflicting columns from enriched_df (except the join key)
        # This prevents duplication (e.g., machine_name_x, machine_name_y)
        cols_to_drop = [c for c in machine_cols if c in enriched_df.columns and c != 'machine_key']
        if cols_to_drop:
            enriched_df = enriched_df.drop(columns=cols_to_drop)

        enriched_df = enriched_df.merge(
            machines_merge[machine_cols],
            on='machine_key',
            how='left'
        )

    # 5. Ensure all historical columns exist and are ordered correctly
    target_columns = [
        'nayax_name', 'price', 'currency', 'position', 'machine_name', 'machine_id', 
        'refiller', 'customer_id', 'local_timestamp', 'card_type', 'card_brand', 
        'machine_key', 'product_name', 'provider', 'category', 'subcategory', 'moms', 
        'ean', 'purchase_price_kr', 'too_unspecific', 'pusher_friendly', 'width', 
        'spiral', 'machine_eva_group', 'machine_sub_group', 'machine_group_tag', 
        'machine_model', 'sielaff_id', 'is_ICA_refiller', 'address', 'latitude', 'longitude'
    ]

    # Add missing columns with None/NaN
    for col in target_columns:
        if col not in enriched_df.columns:
            enriched_df[col] = pd.NA

    # Fix Data Types to match Historical
    # EAN: Int64
    if 'ean' in enriched_df.columns:
        enriched_df['ean'] = pd.to_numeric(enriched_df['ean'], errors='coerce').astype('Int64')
    
    # Spiral: Int64
    if 'spiral' in enriched_df.columns:
        enriched_df['spiral'] = pd.to_numeric(enriched_df['spiral'], errors='coerce').astype('Int64')
    
    # Machine ID: string (standardized to str for now, historical seems to have some int64 but mixed often)
    # The error says historical is int64, latest is object. Let's try to align to numeric if possible, or str if historical is actually mixed.
    # Given historical is int64, we should convert to numeric.
    if 'machine_id' in enriched_df.columns:
         enriched_df['machine_id'] = pd.to_numeric(enriched_df['machine_id'], errors='coerce').astype('Int64') # Using Int64 for nullable integer

    # Latitude/Longitude: float64
    for col in ['latitude', 'longitude', 'moms']:
        if col in enriched_df.columns:
            enriched_df[col] = pd.to_numeric(enriched_df[col], errors='coerce').astype('float64')

    # Booleans: bool
    # We must treat is_ICA_refiller carefully because we just merged it from machines_df
    # In machines_df it might be boolean or object (from Firestore).
    # If it was missing in merge, it became NaN (float/object).
    # So we fillna(False) and convert to bool.
    for col in ['pusher_friendly', 'too_unspecific', 'is_ICA_refiller']:
        if col in enriched_df.columns:
            # Handle object/None types gracefully
            # Force conversion to boolean, handling various "truthy" values if necessary, but simple fillna(False) usually works for left joins
            enriched_df[col] = enriched_df[col].fillna(False).astype(bool)

    # Select only target columns in correct order
    return enriched_df[target_columns]


def load_all_sales_data(
    firestore_client: FirestoreClient = None,
    fetch_latest_sales: bool = False,
    firebase_data: dict = None
) -> pd.DataFrame:
    """
    Loads historical sales from Parquet files and extends it with newer sales data.
    
    Args:
        firestore_client: Firestore client to use for Firestore operations
        fetch_latest_sales: Whether to fetch and append latest sales data from Firestore
        firebase_data: Optional dictionary of pre-loaded Firebase collections (products, machines, etc.)
                      to avoid re-fetching during enrichment.
                      
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (historical_sales, latest_sales_df)
    """
    print("Loading historical sales from Parquet files...")
    historical_sales = load_processed_sales()
    historical_raw = len(historical_sales)

    if firestore_client is None:
        raise Exception("Firestore client not provided in load_all_sales_data")

    category_lookup = _load_category_lookup(firestore_client)
    historical_sales = _attach_category_column(historical_sales, category_lookup, subcategory_column='subcategory')
    historical_sales = _remove_unspecific_rows(historical_sales)
    _log_cleaning_stats("Historical sales (after filtering)", historical_raw, len(historical_sales))

    if fetch_latest_sales:
        print("Fetching latest sales data...")
        
        filters = None
        if not historical_sales.empty and 'local_timestamp' in historical_sales.columns:
            last_timestamp = historical_sales['local_timestamp'].max()
            if last_timestamp:
                print(f"Fetching sales after: {last_timestamp}")
                filters = [('local_timestamp', '>', last_timestamp)]

        latest_sales_df = firestore_to_df(
            FIREBASE_COLLECTIONS['SALES_AWS'], 
            firestore_client, 
            include_doc_id=True,
            order_by='local_timestamp', 
            descending=True,
            filters=filters
        )
        
        # Enrich latest sales with product and machine info
        if not latest_sales_df.empty:
            latest_sales_df = enrich_latest_sales_data(latest_sales_df, firestore_client, firebase_data=firebase_data)
            
            # Ensure local_timestamp is datetime64[ns, UTC]
            if 'local_timestamp' in latest_sales_df.columns:
                latest_sales_df['local_timestamp'] = pd.to_datetime(latest_sales_df['local_timestamp'], utc=True)
            
    else:
        print("Won't enrich with latest sales data! Set fetch_latest_sales to True to include latest sales.")
        latest_sales_df = pd.DataFrame()
        

    if historical_sales.empty and latest_sales_df.empty:
        print("Warning: No sales data found in Parquet files or latest input.")
        return pd.DataFrame(), pd.DataFrame()
    if historical_sales.empty:
        return latest_sales_df, pd.DataFrame()
    if latest_sales_df.empty:
        return historical_sales, pd.DataFrame()

    # Combine historical and latest sales
    combined_sales = pd.concat([historical_sales, latest_sales_df], ignore_index=True)

    return combined_sales

def enrich_with_purchase_prices(products_df: pd.DataFrame | None, purchase_prices_df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Simple EAN to purchase_price_kr mapping."""
    if products_df is None or products_df.empty:
        return products_df

    df = products_df.copy()

    # If no purchase prices, just ensure column exists
    if purchase_prices_df is None or purchase_prices_df.empty or 'ean' not in purchase_prices_df.columns:
        if 'purchase_price_kr' not in df.columns:
            df['purchase_price_kr'] = pd.NA
        return df

    prices = purchase_prices_df[['ean', 'purchase_price_kr']].copy()

    # Merge on EAN
    df = df.merge(
        prices[['ean', 'purchase_price_kr']],
        on='ean',
        how='left'
    )

    # Ensure purchase_price_kr column exists even if enrichment failed
    if 'purchase_price_kr' not in df.columns:
        df['purchase_price_kr'] = pd.NA

    return df


def _extract_lat_lon(df: pd.DataFrame) -> pd.DataFrame:
    """Extract latitude and longitude from Firestore GeoPoint 'location' column if needed."""
    if df.empty:
        return df
    
    df = df.copy()
    
    # If we already have lat/lon, ensure they are numeric
    if 'latitude' in df.columns and 'longitude' in df.columns:
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
        return df

    # Check for 'location' column (Firestore GeoPoint)
    if 'location' in df.columns:
        # Check if first non-null value has latitude/longitude attributes
        sample = df['location'].dropna().iloc[0] if not df['location'].dropna().empty else None
        if hasattr(sample, 'latitude') and hasattr(sample, 'longitude'):
            df['latitude'] = df['location'].apply(lambda x: x.latitude if x else None)
            df['longitude'] = df['location'].apply(lambda x: x.longitude if x else None)
        elif isinstance(sample, dict) and 'latitude' in sample and 'longitude' in sample:
             # Dict format (e.g. from JSON)
            df['latitude'] = df['location'].apply(lambda x: x.get('latitude') if x else None)
            df['longitude'] = df['location'].apply(lambda x: x.get('longitude') if x else None)
            
    return df

def load_ica_stores() -> pd.DataFrame:
    """
    Load supermarket locations from SCB data.
    """
    if os.path.exists(ICA_STORES_PATH):
        print(f"Loading ICA stores from {ICA_STORES_PATH}")
        return pd.read_parquet(ICA_STORES_PATH)
    else:
        print(f"Warning: ICA stores file not found at {ICA_STORES_PATH}")
        return pd.DataFrame(columns=['store_id', 'latitude', 'longitude'])


def load_companies() -> pd.DataFrame:
    """
    Load company locations from SCB data.
    """
    if os.path.exists(COMPANIES_PATH):
        print(f"Loading Companies from {COMPANIES_PATH}")
        return pd.read_parquet(COMPANIES_PATH)
    else:
        print(f"Warning: Companies file not found at {COMPANIES_PATH}")
        return pd.DataFrame(columns=['company_id', 'latitude', 'longitude', 'employee_count'])
