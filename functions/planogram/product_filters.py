# Helper utilities and filter definitions
import pandas as pd
import os

SUGAR_SAFE_PHRASES = {
    'utan tillsatt socker',
    'utan socker',
    'sockerfri',
    'sugar free',
    'no added sugar'
}
SUGAR_INDICATORS = {
    'socker',
    'sirap',
    'glukos',
    'fruktos',
    'sugar',
    'syrup',
    'honung',
    'dextros',
    'saccharose',
    'sucrose'
}
GLUTEN_SAFE_PHRASES = {
    'glutenfri',
    'gluten-free'
}
GLUTEN_INDICATORS = {
    'gluten',
    'vete',
    'råg',
    'korn',
    'dinkel',
    'spelt',
    'malt',
    'havre'
}
LACTOSE_SAFE_PHRASES = {
    'laktosfri',
    'lactose free',
    'lactose-free',
    'mjölkfri',
    'milk free',
    'vegansk',
    'vegan'
}
LACTOSE_INDICATORS = {
    'mjölk',
    'mjölkpulver',
    'mjölkprotein',
    'laktos',
    'lactose',
    'milk',
    'cream',
    'grädde',
    'smör',
    'ost',
    'cheese',
    'yoghurt',
    'yogurt',
    'butter',
    'whey',
    'vassle',
    'kasein'
}
MEAT_KEYWORDS = {
    'kött',
    'fläsk',
    'gris',
    'bacon',
    'skinka',
    'kyckling',
    'chicken',
    'nöt',
    'beef',
    'pork',
    'lamm',
    'lam',
    'lamml',
    'turkey',
    'kalkon',
    'salami',
    'korv',
    'ham',
    'prosciutto',
    'kebab'
    'pastrami'
}
FISH_KEYWORDS = {
    'fisk',
    'lax',
    'tonfisk',
    'torsk',
    'sill',
    'räk',
    'shrimp',
    'krab',
    'kräft',
    'scampi',
    'makrill',
    'sard',
    'hoki',
    'cod',
    'seafood'
}
EGG_KEYWORDS = {
    'ägg',
    'egg',
    'albumin'
}
HONEY_KEYWORDS = {
    'honung',
    'honey'
}
GELATIN_KEYWORDS = {
    'gelatin',
    'gelatine',
    'gelé'
}
NUT_KEYWORDS = {
    'nöt',
    'nötter',
    'jordnöt',
    'jordnötter',
    'peanut',
    'peanuts',
    'mandel',
    'almond',
    'cashew',
    'valnöt',
    'valnötter',
    'walnut',
    'walnuts',
    'pistage',
    'pistachio',
    'hassel',
    'hazelnut',
    'macadamia',
    'pecan'
}
ENERGY_KEYWORDS = {
    'energidryck',
    'energy drink',
    'nocco',
    'celsius',
    'red bull',
    'monster',
    'reign',
    'powerking',
    'clean drink',
    'battery',
    'hell energy'
}
MINIMUM_TRADE_ITEM_LIFESPAN_FROM_TIME_OF_PRODUCTION = 140

FRESH_SUBCATEGORIES = {
    'färdigmat',
    'macka',
    'wrap',
    'sallad',
    'yoghurt',
    'gröt',
    'kvarg',
    'keso',
    'mjölk',
    'pudding' #osäker
}

def _to_lower(text):
    return text.lower() if isinstance(text, str) else ''


def _combine_text_fields(row):
    parts = []
    for field in (
        'ingredientStatement',
        'allergenStatement',
        'tradeItemDescription',
        'descriptionShort',
        'product_name',
        'category',
        'subcategory',
        'provider'
    ):
        value = row.get(field)
        if isinstance(value, str):
            parts.append(value.lower())
    nayax_names = row.get('nayax_names')
    if isinstance(nayax_names, list):
        parts.extend(str(name).lower() for name in nayax_names if name is not None)
    return ' '.join(parts)

from pandas import isna
def _is_uncategorized(row):
    gtin = row.get('gtin')
    return isna(gtin)


def _contains_any(text, keywords):
    return any(keyword in text for keyword in keywords if keyword)


def _parse_float(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).replace(',', '.'))
    except (TypeError, ValueError):
        return None


def _get_nutrient_value(row, nutrient_code):
    headers = row.get('nutrientHeaders')
    if not isinstance(headers, list):
        return None
    nutrient_code = nutrient_code.upper()
    for header in headers:
        basis = header.get('nutrientBasisQuantity', {}) or {}
        basis_value = _parse_float(basis.get('value'))
        if basis_value not in (None, 100.0):
            continue
        for detail in header.get('nutrientDetails', []) or []:
            code = ((detail.get('nutrientTypeCode') or {}).get('code') or '').upper()
            if code != nutrient_code:
                continue
            for quantity in detail.get('quantityContained', []) or []:
                value = _parse_float(quantity.get('value'))
                if value is not None:
                    return value
    return None


def filter_gluten_free(row, default_counter=None):
    """Returns True if product IS gluten free (should be tagged)."""
    text = _combine_text_fields(row)
    if not text:
        if default_counter is not None:
            default_counter['gluten free'] = default_counter.get('gluten free', 0) + 1
        return False
    # If explicitly labeled gluten-free, tag it
    if _contains_any(text, GLUTEN_SAFE_PHRASES):
        return True
    # If contains gluten indicators, it's NOT gluten-free
    if _contains_any(text, GLUTEN_INDICATORS):
        return False
    # Default: if it had data, and made it here, it's gluten-free
    if default_counter is not None:
        default_counter['gluten free'] = default_counter.get('gluten free', 0) + 1
    if _is_uncategorized(row):
        return False
    return True


def filter_sugar_free(row, default_counter=None):
    """Returns True if product IS sugar free (should be tagged)."""
    sugar_value = _get_nutrient_value(row, 'SUGAR-')
    if sugar_value is not None:
        # If sugar <= 0.5g per 100g, it's sugar-free
        if sugar_value <= 0.5:
            return True
        # If sugar > 0.5g, it's NOT sugar-free
        return False
    text = _combine_text_fields(row)
    if not text:
        if default_counter is not None:
            default_counter['sugar free'] = default_counter.get('sugar free', 0) + 1
        return False
    # If explicitly labeled sugar-free, tag it
    if _contains_any(text, SUGAR_SAFE_PHRASES):
        return True
    # If contains sugar indicators, it's NOT sugar-free
    if _contains_any(text, SUGAR_INDICATORS):
        return False
    # Default: can't determine, don't tag
    if default_counter is not None:
        default_counter['sugar free'] = default_counter.get('sugar free', 0) + 1
    return False


def filter_lactose_free(row, default_counter=None):
    """Returns True if product IS lactose free (should be tagged)."""
    text = _combine_text_fields(row)
    if not text:
        if default_counter is not None:
            default_counter['lactose free'] = default_counter.get('lactose free', 0) + 1
        return False
    # If explicitly labeled lactose-free, tag it
    if _contains_any(text, LACTOSE_SAFE_PHRASES):
        return True
    # If contains lactose indicators, it's NOT lactose-free
    if _contains_any(text, LACTOSE_INDICATORS):
        return False
    # Default: can't determine, but tags anyways
    if default_counter is not None:
        default_counter['lactose free'] = default_counter.get('lactose free', 0) + 1
    if _is_uncategorized(row):
        return False
    return True


def filter_vegan(row, default_counter=None):
    """Returns True if product IS vegan (should be tagged)."""
    text = _combine_text_fields(row)
    if not text:
        if default_counter is not None:
            default_counter['vegan'] = default_counter.get('vegan', 0) + 1
        return False
    # If explicitly labeled vegan, tag it
    if 'vegansk' in text or 'vegan' in text:
        return True
    # If contains animal products, it's NOT vegan
    if _contains_any(text, MEAT_KEYWORDS | FISH_KEYWORDS):
        return False
    if _contains_any(text, EGG_KEYWORDS | HONEY_KEYWORDS | GELATIN_KEYWORDS):
        return False
    # Check if it contains lactose (dairy) - if it does, it's NOT vegan
    if _contains_any(text, LACTOSE_INDICATORS):
        return False
    # If it's lactose-free and has no animal products, might be vegan
    # But we can't be 100% sure without explicit label, but return True anyways
    if default_counter is not None:
        default_counter['vegan'] = default_counter.get('vegan', 0) + 1
    if _is_uncategorized(row):
        return False
    return True


def filter_vegetarian(row, default_counter=None):
    """Returns True if product IS vegetarian (should be tagged)."""
    text = _combine_text_fields(row)
    if not text:
        if default_counter is not None:
            default_counter['vegetarian'] = default_counter.get('vegetarian', 0) + 1
        return False
    # If explicitly labeled vegetarian, tag it
    if 'vegetarisk' in text or 'vegetarian' in text:
        return True
    # If contains meat or fish, it's NOT vegetarian
    if _contains_any(text, MEAT_KEYWORDS | FISH_KEYWORDS):
        return False
    # If contains gelatin (often from animals), it's NOT vegetarian
    if _contains_any(text, GELATIN_KEYWORDS):
        return False
    # Default: can't determine, but tags anyways
    if default_counter is not None:
        default_counter['vegetarian'] = default_counter.get('vegetarian', 0) + 1
    if _is_uncategorized(row):
        return False
    return True


def filter_no_nuts(row, default_counter=None):
    """Returns True if product IS nut-free (should be tagged)."""
    text = _combine_text_fields(row)
    if not text:
        if default_counter is not None:
            default_counter['no nuts'] = default_counter.get('no nuts', 0) + 1
        return False
    # If explicitly labeled nut-free, tag it
    if 'nötfri' in text or 'nut free' in text:
        return True
    # If contains nut keywords, it's NOT nut-free
    if _contains_any(text, NUT_KEYWORDS):
        return False
    # Default: can't determine, but tags anyways
    if default_counter is not None:
        default_counter['no nuts'] = default_counter.get('no nuts', 0) + 1
    if _is_uncategorized(row):
        return False
    return True


def filter_no_energy_drinks(row, default_counter=None):
    """Returns True if product IS NOT an energy drink (should be tagged)."""
    text = _combine_text_fields(row)
    if not text:
        if default_counter is not None:
            default_counter['no energy drinks'] = default_counter.get('no energy drinks', 0) + 1
        return False
    # If contains energy drink keywords, it IS an energy drink (don't tag)
    if _contains_any(text, ENERGY_KEYWORDS):
        return False
    subcategory = _to_lower(row.get('subcategory'))
    # If subcategory is "energi", it IS an energy drink (don't tag)
    if subcategory and 'energi' in subcategory:
        return False
    # Otherwise, it's NOT an energy drink (tag it)
    # Note: This is not a default case - we can determine it's not an energy drink
    if _is_uncategorized(row):
        return False
    return True


def filter_no_fresh_food(row, default_counter=None):
    """Returns True if product IS NOT fresh food (should be tagged)."""
    subcategory = _to_lower(row.get('subcategory'))
    
    # If it's uncategorized, we want to make other checks to see if it's fresh food
    if _is_uncategorized(row):
        # If subcategory matches fresh food categories, it IS fresh food (don't tag)
        if subcategory and any(key in subcategory for key in FRESH_SUBCATEGORIES):
            return False
    else:
        min_lifespan = _parse_float(row.get('minimumTradeItemLifespanFromTimeOfProduction'))
        if min_lifespan is not None and min_lifespan < MINIMUM_TRADE_ITEM_LIFESPAN_FROM_TIME_OF_PRODUCTION:
            return False
    # Otherwise, it's NOT fresh food (tag it)
    # Note: This is not a default case - we can determine it's not fresh food
    return True


FILTER_FUNCTIONS = {
    'gluten free': filter_gluten_free,
    'sugar free': filter_sugar_free,
    'lactose free': filter_lactose_free,
    'vegan': filter_vegan,
    'vegetarian': filter_vegetarian,
    'no energy drinks': filter_no_energy_drinks,
    'no fresh food': filter_no_fresh_food,
    'no nuts': filter_no_nuts,
}

def _evaluate_filters(row, filters=FILTER_FUNCTIONS, track_defaults=True, accumulated_defaults=None):
    """
    Evaluate filters for a product row.
    
    Args:
        row: Product row dictionary
        filters: Dictionary of filter name to filter function
        track_defaults: If True, track and return default counter statistics
        accumulated_defaults: Optional dict to accumulate default counts across multiple calls
    
    Returns:
        If track_defaults is True: (filter_results, default_counter_dict)
        If track_defaults is False: filter_results
    """
    existing = row.get('filters') # Use 'filters' for consistency
    if isinstance(existing, list):
        results = list(existing)
    else:
        results = []
    
    default_counter = {} if track_defaults else None
    
    for name, fn in filters.items():
        try:
            should_flag = bool(fn(row, default_counter=default_counter))
        except Exception as exc:
            should_flag = False
        if should_flag:
            results.append(name)
    if results:
        results = sorted(set(results))
    
    # Accumulate default counts if provided
    if accumulated_defaults is not None and default_counter is not None:
        for filter_name, count in default_counter.items():
            accumulated_defaults[filter_name] = accumulated_defaults.get(filter_name, 0) + count
    
    if track_defaults:
        return results, default_counter
    return results


def evaluate_filters_batch(rows, filters=FILTER_FUNCTIONS):
    """
    Evaluate filters for multiple product rows and return accumulated default counts.
    
    Args:
        rows: List of product row dictionaries or pandas DataFrame
        filters: Dictionary of filter name to filter function
    
    Returns:
        tuple: (list of filter results per row, accumulated_default_counter_dict)
    """
    accumulated_defaults = {}
    all_results = []
    
    # Handle both list and DataFrame
    if hasattr(rows, 'iterrows'):
        # It's a pandas DataFrame
        for idx, row in rows.iterrows():
            results, _ = _evaluate_filters(
                row.to_dict(), 
                filters=filters, 
                track_defaults=True,
                accumulated_defaults=accumulated_defaults
            )
            all_results.append(results)
    else:
        # It's a list of dicts
        for row in rows:
            results, _ = _evaluate_filters(
                row, 
                filters=filters, 
                track_defaults=True,
                accumulated_defaults=accumulated_defaults
            )
            all_results.append(results)
    
    return all_results, accumulated_defaults

def apply_all_filters(products_df: pd.DataFrame, product_information_df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies all defined filters to a products DataFrame after enriching it with data from products.json.
    """
    if products_df.empty:
        return products_df

    # Use a copy to avoid side effects on the original DataFrame
    products_df_copy = products_df.copy()

    # Define the path to products.json, relative to this script's location
    product_information_df = product_information_df.copy()

    if not product_information_df.empty:

        # Merge with products.json data, similar to the reference notebook
        if 'ean' in products_df_copy.columns and 'gtin' in product_information_df.columns:
            # Convert merge keys to string to handle potential type mismatches (e.g., float vs int)
            products_df_copy['ean_str'] = products_df_copy['ean'].astype(str)
            product_information_df['gtin_str'] = product_information_df['gtin'].astype(str)
            
            # Drop duplicates from product_information_df to avoid creating extra rows
            product_information_df.drop_duplicates(subset=['gtin_str'], inplace=True)

            merged_df = pd.merge(
                products_df_copy,
                product_information_df,
                left_on='ean_str',
                right_on='gtin_str',
                how='left',
                suffixes=('', '_from_json') # Avoid column name conflicts
            )
            merged_df.drop(columns=['ean_str', 'gtin_str'], inplace=True)
        else:
            print("Warning: Cannot merge with products.json. 'ean' or 'gtin' columns missing.")
            merged_df = products_df_copy
    else:
        print("Warning: product_information_df is empty. Skipping enrichment.")
        merged_df = products_df_copy

    print("Evaluating filters on enriched product data...")
    filter_results, _ = evaluate_filters_batch(merged_df)
    merged_df['filters'] = filter_results

    # Return only the original product columns plus the filters column
    base_columns = products_df.columns.tolist()
    return merged_df[base_columns + ['filters']]
