"""
Healthiness Scoring Module

This module calculates and assigns Nutri-Score letter grades (A-E) to products
based on their nutritional composition. The Nutri-Score is a nutritional rating
system that grades products from A (healthiest) to E (least healthy).

This module contains the core calculation functions. For train/predict functions,
see healthiness_model.py.
"""

import pandas as pd
import numpy as np
import re
import logging
from typing import Dict, List, Optional, Any, TypeAlias

logger = logging.getLogger(__name__)

# ============================================================================
# Nutri-Score Thresholds Configuration
# ============================================================================

CATEGORY_FATS_OILS_NUTS = "fats_oils_nuts"
CATEGORY_GENERAL_FOOD = "general_food"
CATEGORY_BEVERAGES = "beverages"

NUTRISCORE_THRESHOLDS = {
    CATEGORY_FATS_OILS_NUTS: {
        "components": {
            "energy_from_saturated_fatty_acids_kj_per_100g": {
                "pairs": [(120, 0), (240, 1), (360, 2), (480, 3), (600, 4), (720, 5), (840, 6), (960, 7), (1080, 8), (1200, 9)],
                "cap": 10,
            },
            "sugar_g_per_100g": {
                "pairs": [(3.4, 0), (6.8, 1), (10.0, 2), (14.0, 3), (17.0, 4), (20.0, 5), (24.0, 6), (27.0, 7), (31.0, 8), (34.0, 9), (37.0, 10), (41.0, 11), (44.0, 12), (48.0, 13), (51.0, 14)],
                "cap": 15,
            },
            "fat_g_per_100g": {
                "pairs": [(10, 0), (16, 1), (22, 2), (28, 3), (34, 4), (40, 5), (46, 6), (52, 7), (58, 8), (64, 9)],
                "cap": 10,
            },
            "salt_g_per_100g": {
                "pairs": [(0.20, 0), (0.40, 1), (0.60, 2), (0.80, 3), (1.00, 4), (1.20, 5), (1.40, 6), (1.60, 7), (1.80, 8), (2.00, 9), (2.20, 10), (2.40, 11), (2.60, 12), (2.80, 13), (3.00, 14), (3.20, 15), (3.40, 16), (3.60, 17), (3.80, 18), (4.00, 19)],
                "cap": 20,
            },
            "protein_g_per_100g": {
                "pairs": [(2.4, 0), (4.8, 1), (7.2, 2), (9.6, 3), (12.0, 4), (14.0, 5), (17.0, 6)],
                "cap": 7,
            },
            "dietary_fibre_g_per_100g": {
                "pairs": [(3.0, 0), (4.1, 1), (5.2, 2), (6.3, 3), (7.4, 4)],
                "cap": 5,
            },
            "fruit_vegetables_pulses_g_per_100g": {
                "pairs": [(40.0, 0), (60.0, 1), (80.0, 2)],
                "cap": 5,
            },
        },
        "letter_grade": {
            "A": {"min": -np.inf, "max": -6},
            "B": {"min": -5, "max": 2},
            "C": {"min": 3, "max": 10},
            "D": {"min": 11, "max": 18},
            "E": {"min": 19, "max": np.inf},
        },
    },
    CATEGORY_GENERAL_FOOD: {
        "components": {
            "energy_kj_per_100g": {
                "pairs": [(335, 0), (670, 1), (1005, 2), (1340, 3), (1675, 4), (2010, 5), (2345, 6), (2680, 7), (3015, 8), (3350, 9)],
                "cap": 10,
            },
            "saturated_fatty_acids_g_per_100g": {
                "pairs": [(1.0, 0), (2.0, 1), (3.0, 2), (4.0, 3), (5.0, 4), (6.0, 5), (7.0, 6), (8.0, 7), (9.0, 8), (10.0, 9)],
                "cap": 10,
            },
            "sugar_g_per_100g": {
                "pairs": [(3.4, 0), (6.8, 1), (10.0, 2), (14.0, 3), (17.0, 4), (20.0, 5), (24.0, 6), (27.0, 7), (31.0, 8), (34.0, 9), (37.0, 10), (41.0, 11), (44.0, 12), (48.0, 13), (51.0, 14)],
                "cap": 15,
            },
            "salt_g_per_100g": {
                "pairs": [(0.20, 0), (0.40, 1), (0.60, 2), (0.80, 3), (1.00, 4), (1.20, 5), (1.40, 6), (1.60, 7), (1.80, 8), (2.00, 9), (2.20, 10), (2.40, 11), (2.60, 12), (2.80, 13), (3.00, 14), (3.20, 15), (3.40, 16), (3.60, 17), (3.80, 18), (4.00, 19)],
                "cap": 20,
            },
            "protein_g_per_100g": {
                "pairs": [(2.4, 0), (4.8, 1), (7.2, 2), (9.6, 3), (12.0, 4), (14.0, 5), (17.0, 6)],
                "cap": 7,
            },
            "dietary_fibre_g_per_100g": {
                "pairs": [(3.0, 0), (4.1, 1), (5.2, 2), (6.3, 3), (7.4, 4)],
                "cap": 5,
            },
            "fruit_vegetables_pulses_g_per_100g": {
                "pairs": [(40.0, 0), (60.0, 1), (80.0, 2)],
                "cap": 5,
            },
        },
        "letter_grade": {
            "A": {"min": -np.inf, "max": 0},
            "B": {"min": 1, "max": 2},
            "C": {"min": 3, "max": 10},
            "D": {"min": 11, "max": 18},
            "E": {"min": 19, "max": np.inf},
        },
    },
    CATEGORY_BEVERAGES: {
        "components": {
            "energy_kj_per_100g": {
                "pairs": [(90, 0), (150, 1), (210, 2), (240, 3), (270, 4), (300, 5), (330, 6), (360, 7), (390, 8)],
                "cap": 10,
            },
            "saturated_fatty_acids_g_per_100g": {
                "pairs": [(1.0, 0), (2.0, 1), (3.0, 2), (4.0, 3), (5.0, 4), (6.0, 5), (7.0, 6), (8.0, 7), (9.0, 8), (10.0, 9)],
                "cap": 10,
            },
            "sugar_g_per_100g": {
                "pairs": [(0.5, 0), (2.0, 1), (3.5, 2), (5.0, 3), (6.0, 4), (7.0, 5), (8.0, 6), (9.0, 7), (10.0, 8), (11.0, 9)],
                "cap": 10,
            },
            "salt_g_per_100g": {
                "pairs": [(0.20, 0), (0.40, 1), (0.60, 2), (0.80, 3), (1.00, 4), (1.20, 5), (1.40, 6), (1.60, 7), (1.80, 8), (2.00, 9), (2.20, 10), (2.40, 11), (2.60, 12), (2.80, 13), (3.00, 14), (3.20, 15), (3.40, 16), (3.60, 17), (3.80, 18), (4.00, 19)],
                "cap": 20,
            },
            "has_sweeteners": {
                "pairs": [(False, 0), (True, 4)],
                "cap": 4,
            },
            "protein_g_per_100g": {
                "pairs": [(1.2, 0), (1.5, 1), (1.8, 2), (2.1, 3), (2.4, 4), (2.7, 5), (3.0, 6)],
                "cap": 7,
            },
            "dietary_fibre_g_per_100g": {
                "pairs": [(3.0, 0), (4.1, 1), (5.2, 2), (6.3, 3), (7.4, 4)],
                "cap": 5,
            },
            "fruit_vegetables_pulses_g_per_100g": {
                "pairs": [(40.0, 0), (60.0, 2), (80.0, 4)],
                "cap": 6,
            },
        },
        "letter_grade": {
            "B": {"min": -np.inf, "max": 2},
            "C": {"min": 3, "max": 6},
            "D": {"min": 7, "max": 9},
            "E": {"min": 10, "max": np.inf},
        },
    },
}

# Category mapping from product category names to threshold keys
CATEGORY_MAPPING = {
    "Fats, oils, nuts and seeds": CATEGORY_FATS_OILS_NUTS,
    "General food": CATEGORY_GENERAL_FOOD,
    "Beverages": CATEGORY_BEVERAGES,
}

# ============================================================================
# Helper Functions for Nutrition Extraction
# ============================================================================

def _normalize_identifier(value: Any) -> Optional[str]:
    """Normalize GTIN/EAN identifiers to comparable digit-only strings."""
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return None
        try:
            return format(float(value), '.0f')
        except (ValueError, TypeError):
            pass
    value_str = str(value).strip()
    if not value_str:
        return None
    digits_only = re.sub(r'\D', '', value_str)
    return digits_only or None


def _parse_float(value: Any) -> Optional[float]:
    """Helper function to parse float values from various types."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).replace(',', '.'))
    except (ValueError, TypeError):
        return None


def _get_nutrient_value(row: Dict[str, Any], nutrient_code: str, use_flexible_matching: bool = False) -> Optional[float]:
    """
    Extract nutrient value from a product row based on nutrient code.
    
    Args:
        row: Product row (dict)
        nutrient_code: Nutrient code to search for (e.g., 'SUGAR-', 'FAT-')
        use_flexible_matching: If True, also try prefix matching if exact match fails
    
    Returns:
        float or None: Nutrient value per 100g, or None if not found
    """
    headers = row.get('nutrientHeaders')
    if not isinstance(headers, list):
        return None
    
    nutrient_code = nutrient_code.upper().strip()
    
    # Try exact match first
    for header in headers:
        basis = header.get('nutrientBasisQuantity', {}) or {}
        basis_value = _parse_float(basis.get('value'))
        if basis_value is not None and abs(basis_value - 100.0) > 0.01:
            continue
        
        for detail in header.get('nutrientDetails', []) or []:
            code = ((detail.get('nutrientTypeCode') or {}).get('code') or '').upper().strip()
            if code == nutrient_code:
                for quantity in detail.get('quantityContained', []) or []:
                    value = _parse_float(quantity.get('value'))
                    if value is not None:
                        return value
    
    # If flexible matching is enabled and exact match failed, try prefix matching
    if use_flexible_matching:
        code_prefix = nutrient_code.rstrip('-')
        for header in headers:
            basis = header.get('nutrientBasisQuantity', {}) or {}
            basis_value = _parse_float(basis.get('value'))
            if basis_value is not None and abs(basis_value - 100.0) > 0.01:
                continue
            
            for detail in header.get('nutrientDetails', []) or []:
                code = ((detail.get('nutrientTypeCode') or {}).get('code') or '').upper().strip()
                if code.startswith(code_prefix):
                    for quantity in detail.get('quantityContained', []) or []:
                        value = _parse_float(quantity.get('value'))
                        if value is not None:
                            return value
    
    return None


def _get_first_available_value(row: Dict[str, Any], nutrient_codes: List[str], use_flexible_matching: bool = False) -> Optional[float]:
    """Return the first nutrient value found for any of the provided codes."""
    for code in nutrient_codes:
        value = _get_nutrient_value(row, code, use_flexible_matching=False)
        if value is not None:
            return value
    
    if use_flexible_matching:
        for code in nutrient_codes:
            value = _get_nutrient_value(row, code, use_flexible_matching=True)
            if value is not None:
                return value
    
    return None


def _get_energy_value_comprehensive(row: Dict[str, Any]) -> tuple[Optional[float], Optional[str]]:
    """
    Comprehensive energy extraction that searches through all nutrient codes
    to find energy values, not just predefined codes.
    """
    headers = row.get('nutrientHeaders')
    if not isinstance(headers, list):
        return None, None
    
    energy_keywords = ['ENER', 'ENERC', 'ENERGY', 'KCAL']
    energy_codes_kj = ['ENER-', 'ENER', 'ENERC', 'ENERC-', 'ENERC-KJ', 'ENERC-KJ-', 'ENERGY', 'ENERGY-', 'ENERGY-KJ', 'ENERGY-KJ-']
    
    value = _get_first_available_value(row, energy_codes_kj, use_flexible_matching=True)
    if value is not None:
        return value, 'kJ'
    
    energy_codes_kcal = ['ENERGY-KCAL', 'ENERGY-KCAL-', 'ENERC-KCAL', 'ENERC-KCAL-', 'KCAL', 'KCAL-']
    value = _get_first_available_value(row, energy_codes_kcal, use_flexible_matching=True)
    if value is not None:
        return value, 'kcal'
    
    # Search through all nutrient codes
    for header in headers:
        basis = header.get('nutrientBasisQuantity', {}) or {}
        basis_value = _parse_float(basis.get('value'))
        if basis_value is not None and abs(basis_value - 100.0) > 0.01:
            continue
        
        for detail in header.get('nutrientDetails', []) or []:
            code_info = detail.get('nutrientTypeCode', {}) or {}
            code = (code_info.get('code') or '').upper().strip()
            
            if not code:
                continue
            
            is_energy_code = False
            for keyword in energy_keywords:
                if keyword in code:
                    is_energy_code = True
                    break
            
            if is_energy_code:
                is_kcal = 'KCAL' in code
                for quantity in detail.get('quantityContained', []) or []:
                    value = _parse_float(quantity.get('value'))
                    if value is not None:
                        if is_kcal or (value > 0 and value < 1000):
                            return value, 'kcal'
                        else:
                            return value, 'kJ'
    
    return None, None


def _has_sweeteners(ingredient_statement: Optional[str]) -> bool:
    """Detect if a product contains sweeteners based on ingredient statement."""
    if not ingredient_statement or not isinstance(ingredient_statement, str):
        return False
    
    ingredients_lower = ingredient_statement.lower()
    sweetener_keywords = {
        'aspartam', 'aspartame', 'acesulfam', 'acesulfame', 'acesulfam k',
        'sucralos', 'sucralose', 'sacharin', 'saccharin', 'stevia', 'steviolglykosid',
        'steviol glycoside', 'rebaudiosid', 'xylitol', 'sorbitol', 'mannitol',
        'erythritol', 'maltitol', 'isomalt', 'lactitol', 'polyol', 'polyoler',
        'agave', 'agavsirap', 'agave syrup', 'honung', 'honey',
        'sockeralkohol', 'sugar alcohol', 'sötningsmedel', 'sotningsmedel',
        'sweetener', 'sweeteners', 'sötämne', 'sotamne', 'sweetening agent',
        'artificiell sötning', 'e950', 'e951', 'e952', 'e953', 'e954', 'e955',
        'e956', 'e957', 'e959', 'e960', 'e961', 'e962', 'e965', 'e966', 'e967', 'e968', 'e999',
    }
    
    for keyword in sweetener_keywords:
        if re.search(r'\b' + re.escape(keyword) + r'\b', ingredients_lower):
            return True
    
    return False


def _is_only_water(ingredient_statement: Optional[str]) -> bool:
    """Detect if an ingredient statement contains only water."""
    if not ingredient_statement or not isinstance(ingredient_statement, str):
        return False
    
    ingredients_lower = ingredient_statement.lower().strip()
    if not ingredients_lower:
        return False
    
    ingredients_lower = re.sub(r'^(ingredienser|ingredients|ingrédients)[:\s]*', '', ingredients_lower, flags=re.IGNORECASE)
    
    water_keywords = {
        'vatten', 'water', 'mineralvatten', 'mineral water', 'källvatten', 'spring water',
        'dricksvatten', 'drinking water', 'artesiskt vatten', 'artesian water',
        'destillerat vatten', 'distilled water', 'naturligt vatten', 'natural water',
        'renat vatten', 'purified water', 'osmotiskt vatten', 'osmotic water',
        'vann', 'mineralvann', 'kildevand', 'drikkevand', 'eau', 'eau minérale',
        'eau de source', 'aqua',
    }
    
    contains_water = False
    for keyword in water_keywords:
        if keyword in ingredients_lower:
            contains_water = True
            break
    
    if not contains_water:
        return False
    
    non_water_indicators = {
        'juice', 'juicekoncentrat', 'juice concentrate', 'saft', 'saftkoncentrat',
        'citron', 'lemon', 'lime', 'apelsin', 'orange', 'socker', 'sugar',
        'glukos', 'glucose', 'fruktos', 'fructose', 'sukker', 'sirup', 'syrup',
        'sirap', 'siroop', 'salt', 'kolsyra', 'carbon dioxide', 'co2', 'kullsyre',
        'syra', 'acid', 'surhetsreglerande', 'acid regulator', 'konserveringsmedel',
        'preservative', 'färgämne', 'color', 'färg', 'aroma', 'flavour', 'smakämne',
        'flavor', 'te', 'tea', 'kaffe', 'coffee', 'extract', 'extrakt', 'vitamin',
        'elektrolyt', 'electrolyte', 'protein', 'mjölk', 'milk', 'mjölkprotein',
        'milk protein', 'yoghurt', 'yogurt', 'kultur', 'culture',
    }
    
    descriptive_words = {
        'från', 'from', 'vid', 'at', 'på', 'on', 'i', 'in', 'av', 'of', 'till', 'to',
        'och', 'and', 'med', 'with', 'eller', 'or', 'naturligt', 'natural',
        'källa', 'source', 'spring', 'källan', 'the source', 'springen', 'sources',
        'brande', 'jylland', 'aquad\'or', 'aquador', 'aqua', 'mineral',
    }
    
    for indicator in non_water_indicators:
        if indicator in descriptive_words:
            continue
        pattern = r'\b' + re.escape(indicator) + r'\b'
        if re.search(pattern, ingredients_lower):
            if indicator == 'mineral' and ('mineralvatten' in ingredients_lower or 'mineral water' in ingredients_lower):
                continue
            return False
    
    return True


def _extract_fvp_from_ingredients(ingredient_statement: Optional[str]) -> Optional[float]:
    """
    Extract fruit, vegetables, and pulses (f, v, p) content from ingredient statement.
    Returns estimated f, v, p content (g/100g) or None if no f, v, p found.
    """
    if not ingredient_statement or not isinstance(ingredient_statement, str):
        return None
    
    # Basic FVP keywords (simplified version - full implementation would load from file)
    fvp_keywords = {
        'apple', 'pear', 'berry', 'grape', 'cherry', 'strawberry', 'lemon', 'orange',
        'banana', 'kiwi', 'pineapple', 'melon', 'fig', 'mango', 'tomato', 'cucumber',
        'pepper', 'onion', 'garlic', 'carrot', 'potato', 'spinach', 'lettuce', 'cabbage',
        'broccoli', 'cauliflower', 'celery', 'asparagus', 'pea', 'bean', 'lentil',
        'chickpea', 'avocado', 'olive', 'äpple', 'päron', 'apelsin', 'citron', 'banan',
        'jordgubbe', 'tomat', 'gurka', 'paprika', 'lök', 'vitlök', 'morot', 'potatis',
        'spenat', 'sallad', 'kål', 'broccoli', 'ärtor', 'bönor', 'linser', 'kikärtor',
    }
    
    ingredients_lower = ingredient_statement.lower()
    ingredients_lower = re.sub(r'^(ingredienser|ingredients|ingrédients)[:\s]*', '', ingredients_lower, flags=re.IGNORECASE)
    
    # Extract percentages if present
    percentage_pattern = r'(\d+(?:[.,]\d+)?)\s*%'
    percentages = {}
    for match in re.finditer(percentage_pattern, ingredients_lower):
        start = match.start()
        prev_comma = ingredients_lower.rfind(',', 0, start)
        prev_semicolon = ingredients_lower.rfind(';', 0, start)
        prev_colon = ingredients_lower.rfind(':', 0, start)
        prev_sep = max(prev_comma, prev_semicolon, prev_colon)
        
        if prev_sep > 0:
            ingredient_name = ingredients_lower[prev_sep + 1:start].strip()
            percentage = float(match.group(1).replace(',', '.'))
            percentages[ingredient_name] = percentage
    
    ingredients_clean = re.sub(percentage_pattern, '', ingredients_lower)
    
    # Split into ingredients
    ingredients_list = []
    current_ingredient = ""
    paren_depth = 0
    
    for char in ingredients_clean:
        if char == '(':
            paren_depth += 1
            current_ingredient += char
        elif char == ')':
            paren_depth -= 1
            current_ingredient += char
        elif char in ',;' and paren_depth == 0:
            if current_ingredient.strip():
                ingredients_list.append(current_ingredient.strip())
            current_ingredient = ""
        else:
            current_ingredient += char
    
    if current_ingredient.strip():
        ingredients_list.append(current_ingredient.strip())
    
    # Find f, v, p ingredients
    fvp_ingredients = []
    for idx, ingredient in enumerate(ingredients_list):
        ingredient_clean = re.sub(r'\([^)]*\)', '', ingredient).strip()
        ingredient_words = set(re.findall(r'\b\w+\b', ingredient_clean.lower()))
        ingredient_lower = ingredient_clean.lower()
        
        matches = False
        for keyword in fvp_keywords:
            if keyword in ingredient_words or keyword in ingredient_lower:
                matches = True
                break
        
        if matches:
            percentage = None
            for key, val in percentages.items():
                if any(word in key.lower() for word in ingredient_words):
                    percentage = val
                    break
            
            fvp_ingredients.append({
                'position': idx,
                'ingredient': ingredient_clean,
                'percentage': percentage,
            })
    
    if not fvp_ingredients:
        return None
    
    # Estimate f, v, p content
    explicit_percentages = [ing['percentage'] for ing in fvp_ingredients if ing['percentage'] is not None]
    if explicit_percentages:
        return sum(explicit_percentages)
    
    # Estimate based on position
    total_weight = 0
    estimated_percentage = 0
    
    for fvp_ing in fvp_ingredients:
        position = fvp_ing['position']
        if position == 0:
            weight = 35
        elif position == 1:
            weight = 15
        elif position == 2:
            weight = 10
        elif position < 5:
            weight = 5
        else:
            weight = 2
        
        total_weight += weight
        estimated_percentage += weight
    
    if len(fvp_ingredients) == 1 and fvp_ingredients[0]['position'] == 0:
        estimated_percentage = min(estimated_percentage, 80)
    elif len(fvp_ingredients) >= 2:
        estimated_percentage = min(estimated_percentage, 60)
    else:
        estimated_percentage = min(estimated_percentage, 30)
    
    return estimated_percentage


def _normalize_text_fields(row: Dict[str, Any], fields: List[str]) -> str:
    """Normalize text fields from a row into a single lowercase string."""
    parts = []
    for field in fields:
        value = row.get(field)
        if isinstance(value, str):
            parts.append(value.lower())
    return ' '.join(parts)


def _get_product_identifier(row_dict: Dict[str, Any]) -> Optional[str]:
    """Return the first normalized GTIN/EAN identifier found in the row."""
    for col in ('ean', 'EAN', 'gtin', 'GTIN'):
        if col in row_dict:
            candidate = _normalize_identifier(row_dict.get(col))
            if candidate:
                return candidate
    return None


def _text_contains_any(text: str, keywords: set) -> bool:
    """Check if text contains any of the keywords."""
    return any(keyword in text for keyword in keywords)


FATS_OILS_BASE_KEYWORDS = {
    'oil', 'olja', 'olive oil', 'rapeseed oil', 'rapsolja', 'sunflower oil', 'solrosolja',
    'cooking oil', 'smör', 'butter', 'ghee', 'margarine', 'spread', 'matfett',
    'kokosolja', 'coconut oil', 'vispgrädde', 'grädde', 'cream', 'whipped cream',
}
NUT_SEED_KEYWORDS = {
    'nut', 'nuts', 'nöt', 'nötter', 'jordnöt', 'jordnötter', 'peanut', 'peanuts',
    'peanut butter', 'jordnötssmör', 'mandel', 'almond', 'hazelnut', 'hasselnöt',
    'cashew', 'cashewnöt', 'pistachio', 'pistasch', 'valnöt', 'walnut', 'pecan',
    'macadamia', 'sesame', 'tahini', 'linfrö', 'linseed', 'chia', 'sunflower seed',
    'solrosfrö', 'pumpkin seed', 'fröblandning', 'seed mix', 'nut butter',
}
FATS_OILS_EXCLUDED_KEYWORDS = {'chestnut', 'kastanj', 'coconut', 'kokos'}
NON_FAT_PRODUCT_HINTS = {
    'bar', 'kaka', 'kex', 'cookie', 'godis', 'snack', 'choklad', 'chocolate',
    'dryck', 'drink', 'protein', 'mjölk', 'yoghurt', 'yogurt',
}


def _is_fats_oils_product_by_composition(fat_g_per_100g: Optional[float], text: str) -> bool:
    """Classify as fats/oils/nuts based primarily on nutritional composition."""
    if fat_g_per_100g is None or pd.isna(fat_g_per_100g):
        return False
    
    if fat_g_per_100g > 50:
        if _text_contains_any(text, FATS_OILS_BASE_KEYWORDS):
            if not _text_contains_any(text, NON_FAT_PRODUCT_HINTS):
                return True
        if _text_contains_any(text, NUT_SEED_KEYWORDS):
            if not _text_contains_any(text, FATS_OILS_EXCLUDED_KEYWORDS):
                if not _text_contains_any(text, NON_FAT_PRODUCT_HINTS):
                    return True
    
    if fat_g_per_100g > 30:
        if _text_contains_any(text, NUT_SEED_KEYWORDS):
            if not _text_contains_any(text, FATS_OILS_EXCLUDED_KEYWORDS):
                if not _text_contains_any(text, NON_FAT_PRODUCT_HINTS):
                    return True
    
    return False


DIRECT_BEVERAGE_CATEGORY_LABELS = {'dryck', 'beverage', 'beverages'}
BEVERAGE_SUBCATEGORY_LABELS = {
    'energidryck', 'energidrycker', 'läsk', 'lask', 'juice', 'juicer', 'vatten',
    'dricka', 'dryck', 'kaffe', 'iste', 'tea', 'te', 'proteinshake', 'proteinshakes',
    'shake', 'shakes', 'sport', 'sportdryck', 'vitamin', 'vitamindryck', 'tonic',
    'yoghurt', 'yogurt', 'mjölk', 'mjolk', 'chokladdryck', 'måltidsersättning', 'maltidsersattning',
}


def _label_is_beverage(label: Optional[str]) -> bool:
    """Return True if the provided label clearly represents a beverage."""
    if label is None:
        return False
    if not isinstance(label, str):
        if pd.isna(label):
            return False
        label = str(label)
    label = label.strip().lower()
    if not label:
        return False
    return (
        label in DIRECT_BEVERAGE_CATEGORY_LABELS
        or label in BEVERAGE_SUBCATEGORY_LABELS
        or 'dryck' in label
    )


def _classify_product_category(
    row: Dict[str, Any],
    fat_g_per_100g: Optional[float] = None,
    protein_g_per_100g: Optional[float] = None,
    carbs_g_per_100g: Optional[float] = None,
    fibre_g_per_100g: Optional[float] = None,
    product_database_df: Optional[pd.DataFrame] = None,
) -> str:
    """
    Classify product category using product database classification as the primary method
    for beverages. Only if a product is NOT classified as a beverage do we evaluate if it
    belongs to the fats/oils/nuts category; otherwise it defaults to general food.
    """
    CATEGORY_FATS_OILS = "Fats, oils, nuts and seeds"
    CATEGORY_BEVERAGES = "Beverages"
    CATEGORY_GENERAL_FOOD = "General food"
    
    text = _normalize_text_fields(
        row,
        ['descriptionShort', 'tradeItemDescription', 'product_name', 'brandName'],
    )
    
    # Check product database for beverage classification first
    if product_database_df is not None and not product_database_df.empty:
        identifier = _get_product_identifier(row)
        if identifier:
            # Check if identifier is in product database and classified as beverage
            identifier_cols = ['ean', 'EAN', 'gtin', 'GTIN']
            available_identifier_cols = [col for col in identifier_cols if col in product_database_df.columns]
            category_col = 'category' if 'category' in product_database_df.columns else None
            subcategory_col = 'subcategory' if 'subcategory' in product_database_df.columns else None
            
            for id_col in available_identifier_cols:
                matching_rows = product_database_df[product_database_df[id_col].apply(_normalize_identifier) == identifier]
                if not matching_rows.empty:
                    category_value = None
                    if category_col:
                        raw_category = matching_rows.iloc[0].get(category_col)
                        if isinstance(raw_category, str):
                            category_value = raw_category.strip().lower()
                    if not category_value and subcategory_col:
                        raw_subcategory = matching_rows.iloc[0].get(subcategory_col)
                        if isinstance(raw_subcategory, str):
                            category_value = raw_subcategory.strip().lower()
                        elif pd.notna(raw_subcategory):
                            category_value = str(raw_subcategory).strip().lower()
                    
                    if _label_is_beverage(category_value):
                        return CATEGORY_BEVERAGES
    
    # If not classified as beverage, check fats/oils/nuts classification
    if _is_fats_oils_product_by_composition(fat_g_per_100g, text):
        return CATEGORY_FATS_OILS
    
    # Default to general food for all other non-beverage products
    return CATEGORY_GENERAL_FOOD


def extract_nutritional_values(
    products_df: pd.DataFrame,
    product_database_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Extract nutritional values for each product in the dataframe.
    
    Args:
        products_df: DataFrame containing product information with 'nutrientHeaders' column
        product_database_df: Optional DataFrame with product database for beverage classification
    
    Returns:
        DataFrame: New dataframe containing extracted nutritional columns
    """
    products_df = products_df.copy()
    
    # Extract energy (kJ/100g)
    energy_results = products_df.apply(
        lambda row: _get_energy_value_comprehensive(row.to_dict() if isinstance(row, pd.Series) else row),
        axis=1,
    )
    
    energy_values = energy_results.apply(lambda x: x[0] if x[0] is not None else None)
    energy_units = energy_results.apply(lambda x: x[1] if x[1] is not None else None)
    energy_values = pd.to_numeric(energy_values, errors='coerce')
    
    # Convert kcal to kJ where needed (1 kcal = 4.184 kJ)
    energy = energy_values.copy()
    kcal_mask = energy_units == 'kcal'
    if kcal_mask.any():
        energy.loc[kcal_mask] = energy_values.loc[kcal_mask] * 4.184
    
    # Extract saturated fatty acids (g/100g)
    saturated_fat = pd.to_numeric(
        products_df.apply(
            lambda row: _get_first_available_value(
                row.to_dict() if isinstance(row, pd.Series) else row,
                ['FASAT', 'FASAT-', 'SATURATED', 'SATURATED-', 'SATFAT', 'SATFAT-'],
            ),
            axis=1,
        ),
        errors='coerce',
    )
    
    # Calculate energy from saturated fatty acids (kJ/100g) = saturated fat [g/100g] x 37 kJ/g
    energy_from_sat_fat = saturated_fat * 37.0
    
    # Extract sugar content (g/100g)
    sugar = pd.to_numeric(
        products_df.apply(
            lambda row: _get_first_available_value(
                row.to_dict() if isinstance(row, pd.Series) else row,
                ['SUGAR-', 'SUGAR', 'SUGARS-', 'SUGARS'],
            ),
            axis=1,
        ),
        errors='coerce',
    )
    
    # Extract fat content (g/100g)
    fat = pd.to_numeric(
        products_df.apply(
            lambda row: _get_first_available_value(
                row.to_dict() if isinstance(row, pd.Series) else row,
                ['FAT', 'FAT-', 'FATS', 'FATS-'],
            ),
            axis=1,
        ),
        errors='coerce',
    )
    
    # Extract carbohydrates content (g/100g)
    carbs = pd.to_numeric(
        products_df.apply(
            lambda row: _get_first_available_value(
                row.to_dict() if isinstance(row, pd.Series) else row,
                ['CHOAVL', 'CARB', 'CARB-', 'CARBOHYDRATE', 'CARBOHYDRATE-'],
            ),
            axis=1,
        ),
        errors='coerce',
    )
    
    # Extract salt content (g/100g)
    salt_direct = pd.to_numeric(
        products_df.apply(
            lambda row: _get_first_available_value(
                row.to_dict() if isinstance(row, pd.Series) else row,
                ['SALT', 'SALT-', 'SALTEQ'],
            ),
            axis=1,
        ),
        errors='coerce',
    )
    sodium = pd.to_numeric(
        products_df.apply(
            lambda row: _get_first_available_value(
                row.to_dict() if isinstance(row, pd.Series) else row,
                ['SODIUM', 'SODIUM-', 'NA', 'NA-'],
            ),
            axis=1,
        ),
        errors='coerce',
    )
    # Convert sodium to salt if salt not directly available (salt = sodium x 2.5)
    salt_from_sodium = sodium * 2.5
    salt = salt_direct.fillna(salt_from_sodium)
    
    # Extract protein content (g/100g)
    protein = pd.to_numeric(
        products_df.apply(
            lambda row: _get_first_available_value(
                row.to_dict() if isinstance(row, pd.Series) else row,
                ['PROTEIN', 'PROTEIN-', 'PROTEINS', 'PROTEINS-', 'PROT', 'PROT-', 'PRO', 'PRO-'],
            ),
            axis=1,
        ),
        errors='coerce',
    )
    
    # Extract dietary fibre content (g/100g)
    dietary_fibre = pd.to_numeric(
        products_df.apply(
            lambda row: _get_first_available_value(
                row.to_dict() if isinstance(row, pd.Series) else row,
                ['FIBTG', 'FIBRE', 'FIBRE-', 'FIBER', 'FIBER-', 'DIETARYFIBRE', 'DIETARYFIBRE-', 'DIETARYFIBER', 'DIETARYFIBER-'],
            ),
            axis=1,
        ),
        errors='coerce',
    ).fillna(0)
    
    # Extract fruit, vegetables and pulses (f, v, p) content (g/100g)
    fruit_veg_pulses_from_nutrients = pd.to_numeric(
        products_df.apply(
            lambda row: _get_first_available_value(
                row.to_dict() if isinstance(row, pd.Series) else row,
                ['FRUITVEGPULSES', 'FRUITVEGPULSES-', 'FVNP', 'FVNP-', 'FRUITVEG', 'FRUITVEG-', 'FVP', 'FVP-'],
            ),
            axis=1,
        ),
        errors='coerce',
    )
    
    # Extract from ingredients for products where nutrient codes don't provide values
    fruit_veg_pulses_from_ingredients = products_df.apply(
        lambda row: _extract_fvp_from_ingredients(row.get('ingredientStatement') if isinstance(row, pd.Series) else row.get('ingredientStatement')),
        axis=1,
    )
    fruit_veg_pulses_from_ingredients = pd.to_numeric(fruit_veg_pulses_from_ingredients, errors='coerce')
    
    # Use nutrient values if available, otherwise use ingredient-based estimates
    fruit_veg_pulses = (
        fruit_veg_pulses_from_nutrients.fillna(fruit_veg_pulses_from_ingredients).fillna(0)
    )
    
    # Get product names
    if 'descriptionShort' in products_df.columns:
        product_names = products_df['descriptionShort']
    elif 'tradeItemDescription' in products_df.columns:
        product_names = products_df['tradeItemDescription']
    elif 'product_name' in products_df.columns:
        product_names = products_df['product_name']
    else:
        product_names = pd.Series([None] * len(products_df), index=products_df.index)
    
    # Extract GTIN (check for gtin, GTIN, ean, EAN columns)
    gtin_series = None
    for col in ['gtin', 'GTIN', 'ean', 'EAN']:
        if col in products_df.columns:
            gtin_series = products_df[col]
            break
    
    if gtin_series is not None:
        gtin_series = pd.to_numeric(gtin_series, errors='coerce').astype('Int64')
    else:
        gtin_series = pd.Series([None] * len(products_df), dtype='Int64', index=products_df.index)
    
    # Classify products using nutritional composition, keywords, and ingredients
    product_categories = products_df.apply(
        lambda row: _classify_product_category(
            row.to_dict() if isinstance(row, pd.Series) else row,
            fat_g_per_100g=fat.get(row.name, None),
            protein_g_per_100g=protein.get(row.name, None),
            carbs_g_per_100g=carbs.get(row.name, None),
            fibre_g_per_100g=dietary_fibre.get(row.name, None),
            product_database_df=product_database_df,
        ),
        axis=1,
    )
    
    # Detect sweeteners in ingredient statements
    has_sweeteners = products_df.apply(
        lambda row: _has_sweeteners(row.get('ingredientStatement') if isinstance(row, pd.Series) else row.get('ingredientStatement')),
        axis=1,
    )
    
    # Detect beverages with only water (only true for beverages)
    is_only_water = products_df.apply(
        lambda row: (
            product_categories.get(row.name) == "Beverages"
            and _is_only_water(row.get('ingredientStatement') if isinstance(row, pd.Series) else row.get('ingredientStatement'))
        ),
        axis=1,
    )
    
    return pd.DataFrame(
        {
            'product_name': product_names,
            'gtin': gtin_series,
            'product_category': product_categories,
            'energy_kj_per_100g': energy,
            'saturated_fatty_acids_g_per_100g': saturated_fat,
            'energy_from_saturated_fatty_acids_kj_per_100g': energy_from_sat_fat,
            'sugar_g_per_100g': sugar,
            'fat_g_per_100g': fat,
            'salt_g_per_100g': salt,
            'protein_g_per_100g': protein,
            'dietary_fibre_g_per_100g': dietary_fibre,
            'fruit_vegetables_pulses_g_per_100g': fruit_veg_pulses,
            'has_sweeteners': has_sweeteners,
            'is_only_water': is_only_water,
        },
        index=products_df.index,
    )


# ============================================================================
# Nutri-Score Calculation Functions
# ============================================================================

def _score_from_upper_bounds(value: Optional[float], thresholds: List[tuple], cap_points: Optional[int] = None) -> int:
    """Assign points based on ordered (upper_bound, points) pairs."""
    if value is None or pd.isna(value):
        return 0
    for upper_bound, points in thresholds:
        if value <= upper_bound:
            return points
    return cap_points if cap_points is not None else 0


def _get_unfavourable_and_favourable_components(category_key: str) -> tuple[List[str], List[str]]:
    """Return lists of component names that are unfavourable and favourable for the given category."""
    if category_key == CATEGORY_FATS_OILS_NUTS:
        unfavourable = [
            "energy_from_saturated_fatty_acids_kj_per_100g",
            "sugar_g_per_100g",
            "fat_g_per_100g",
            "salt_g_per_100g",
        ]
        favourable = [
            "protein_g_per_100g",
            "dietary_fibre_g_per_100g",
            "fruit_vegetables_pulses_g_per_100g",
        ]
    elif category_key in (CATEGORY_GENERAL_FOOD, CATEGORY_BEVERAGES):
        unfavourable = [
            "energy_kj_per_100g",
            "saturated_fatty_acids_g_per_100g",
            "sugar_g_per_100g",
            "salt_g_per_100g",
        ]
        if category_key == CATEGORY_BEVERAGES:
            unfavourable.append("has_sweeteners")
        favourable = [
            "protein_g_per_100g",
            "dietary_fibre_g_per_100g",
            "fruit_vegetables_pulses_g_per_100g",
        ]
    else:
        cfg = NUTRISCORE_THRESHOLDS.get(category_key, {})
        all_components = list((cfg.get("components") or {}).keys())
        unfavourable = all_components
        favourable = []
    
    return unfavourable, favourable


def _calculate_nutri_score(
    unfavourable_points: int,
    favourable_points: int,
    fibre_points: int,
    fvp_points: int,
    category_key: str,
) -> int:
    """Calculate the nutri-score based on category-specific rules."""
    if category_key == CATEGORY_GENERAL_FOOD:
        if unfavourable_points > 11:
            return unfavourable_points - fibre_points - fvp_points
        else:
            return unfavourable_points - favourable_points
    elif category_key == CATEGORY_FATS_OILS_NUTS:
        if unfavourable_points > 7:
            return unfavourable_points - fibre_points - fvp_points
        else:
            return unfavourable_points - favourable_points
    elif category_key == CATEGORY_BEVERAGES:
        return unfavourable_points - favourable_points
    else:
        # Fallback: use general food rule
        if unfavourable_points > 11:
            return unfavourable_points - fibre_points - fvp_points
        else:
            return unfavourable_points - favourable_points


def _letter_grade_from_config(score: int, letter_cfg: Dict[str, Dict[str, float]]) -> Optional[str]:
    """Map a nutri-score to a letter grade using the category's letter grade configuration."""
    for grade, bounds in letter_cfg.items():
        lower = bounds.get("min", -np.inf)
        upper = bounds.get("max", np.inf)
        if lower <= score <= upper:
            return grade
    return None


def calculate_nutriscore(nutritional_values_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate component points and overall nutri-score for each product.
    
    Args:
        nutritional_values_df: DataFrame containing:
            - 'product_category': Product classification ("Fats, oils, nuts and seeds",
              "General food", or "Beverages")
            - Nutritional value columns (energy_kj_per_100g, sugar_g_per_100g, etc.)
    
    Returns:
        DataFrame: Input DataFrame with added columns:
            - 'nutrivalue_score': Numeric nutri-score
            - 'nutrivalue_letter_score': Letter grade (A-E, or B-E for beverages)
    """
    nutritional_values_df = nutritional_values_df.copy()
    
    if "product_category" not in nutritional_values_df.columns:
        raise ValueError("Input DataFrame must contain 'product_category' column")
    
    nutri_scores = []
    letter_grades = []
    
    # Process each row
    for idx, row in nutritional_values_df.iterrows():
        product_category = row.get("product_category")
        if pd.isna(product_category) or product_category not in CATEGORY_MAPPING:
            nutri_scores.append(None)
            letter_grades.append(None)
            continue
        
        category_key = CATEGORY_MAPPING[product_category]
        cfg = NUTRISCORE_THRESHOLDS.get(category_key, {})
        components_cfg = cfg.get("components", {})
        letter_cfg = cfg.get("letter_grade", {})
        
        # Calculate points for each component
        component_points = {}
        for component_name, component_cfg in components_cfg.items():
            value = row.get(component_name)
            pairs = component_cfg.get("pairs", [])
            cap = component_cfg.get("cap")
            points = _score_from_upper_bounds(value, pairs, cap_points=cap)
            component_points[component_name] = points
        
        # Get unfavourable and favourable component lists
        unfavourable_components, favourable_components = _get_unfavourable_and_favourable_components(
            category_key
        )
        
        # Sum unfavourable and favourable points
        unfavourable_points = sum(
            component_points.get(comp, 0) for comp in unfavourable_components
        )
        favourable_points = sum(
            component_points.get(comp, 0) for comp in favourable_components
        )
        
        # Get fibre and fvp points separately (needed for special calculation rules)
        fibre_points = component_points.get("dietary_fibre_g_per_100g", 0)
        fvp_points = component_points.get("fruit_vegetables_pulses_g_per_100g", 0)
        
        # Calculate nutri-score
        nutri_score = _calculate_nutri_score(
            unfavourable_points,
            favourable_points,
            fibre_points,
            fvp_points,
            category_key,
        )
        
        # Map to letter grade
        letter_grade = _letter_grade_from_config(nutri_score, letter_cfg)
        
        # Special rule: beverages that are only water should receive grade A
        if category_key == CATEGORY_BEVERAGES and bool(row.get("is_only_water")):
            letter_grade = "A"
        
        nutri_scores.append(nutri_score)
        letter_grades.append(letter_grade)
    
    # Add results to dataframe
    nutritional_values_df["nutrivalue_score"] = nutri_scores
    nutritional_values_df["nutrivalue_letter_score"] = letter_grades
    
    return nutritional_values_df
