"""
Healthiness Model - Train and Predict Functions

This module contains the train and predict functions for the healthiness scoring system.
It handles model training (creating EAN to letter grade mappings) and prediction
(assigning scores to machine slots and calculating overall machine grades).
"""

import pandas as pd
import logging
from typing import Dict, List, Optional, TypeAlias

from .healthiness_scoring import (
    extract_nutritional_values,
    calculate_nutriscore,
    _normalize_identifier,
)

logger = logging.getLogger(__name__)

# Type alias for the trained model artifact
# Contains mapping from normalized EAN to Nutri-Score letter grade
HealthinessModel: TypeAlias = Dict[str, Optional[str]]


def train_healthiness_model(
    product_information_df: pd.DataFrame,
    product_database_df: pd.DataFrame,
) -> HealthinessModel:
    """
    Trains the healthiness model by extracting nutritional values and calculating
    Nutri-Score grades for all products. Creates a mapping from EAN to letter grade.
    
    Args:
        product_information_df: DataFrame containing product information with 'nutrientHeaders' column
        product_database_df: DataFrame with product database containing EAN/GTIN column.
                            All products from this DataFrame will be included in the mapping,
                            with NaN if no letter grade could be calculated.
    
    Returns:
        Dictionary mapping normalized EAN to Nutri-Score letter grade (A-E) or None/NaN
    """
    product_information_df = product_information_df.copy()
    product_database_df = product_database_df.copy()
    
    logger.info(f"Starting healthiness model training with {len(product_information_df)} products")
    
    # Extract nutritional values
    nutritional_values_df = extract_nutritional_values(product_information_df, product_database_df)
    logger.info(f"Extracted nutritional values for {len(nutritional_values_df)} products")
    
    # Calculate Nutri-Scores
    nutriscore_df = calculate_nutriscore(nutritional_values_df)
    logger.info(f"Calculated Nutri-Scores for {len(nutriscore_df)} products")
    
    # Create mapping from EAN to letter score from calculated Nutri-Scores
    ean_to_letter_score = {}
    calculated_count = 0
    
    for _, row in nutriscore_df.iterrows():
        gtin = row.get('gtin')
        letter_score = row.get('nutrivalue_letter_score')
        
        if gtin is not None:
            normalized_ean = _normalize_identifier(gtin)
            if normalized_ean:
                ean_to_letter_score[normalized_ean] = letter_score
                if letter_score is not None:
                    calculated_count += 1
    
    logger.info(f"Created {calculated_count} EAN to letter score mappings from calculated Nutri-Scores")
    
    # Find EAN column in product_database_df
    ean_col = None
    for col in ['ean', 'EAN', 'gtin', 'GTIN']:
        if col in product_database_df.columns:
            ean_col = col
            break
    
    if ean_col is None:
        logger.warning("No EAN/GTIN column found in product_database_df. Returning only calculated mappings.")
        return ean_to_letter_score
    
    # Create complete mapping for all products in product_database_df
    complete_mapping = {}
    total_products = len(product_database_df)
    mapped_count = 0
    nan_count = 0
    
    for _, row in product_database_df.iterrows():
        ean_value = row.get(ean_col)
        if ean_value is not None:
            normalized_ean = _normalize_identifier(ean_value)
            if normalized_ean:
                # Use calculated score if available, otherwise None
                letter_score = ean_to_letter_score.get(normalized_ean)
                complete_mapping[normalized_ean] = letter_score
                
                if letter_score is not None:
                    mapped_count += 1
                else:
                    nan_count += 1
    
    logger.info(f"Created complete mapping for {len(complete_mapping)} EANs from product_database_df")
    logger.info(f"  - {mapped_count} EANs have letter scores")
    logger.info(f"  - {nan_count} EANs have NaN (no letter score available)")
    logger.info(f"  - {total_products - len(complete_mapping)} products skipped (no valid EAN)")
    
    return complete_mapping


def _calculate_machine_overall_letter_grade(letter_grades: List[Optional[str]]) -> Optional[str]:
    """
    Calculate overall letter grade for a machine based on its products' letter grades.
    
    Uses a weighted approach: A=5, B=4, C=3, D=2, E=1, None/NaN=0
    Returns the letter grade corresponding to the average score.
    
    Args:
        letter_grades: List of letter grades (A-E or None) for products in the machine
    
    Returns:
        Overall letter grade (A-E) or None if no valid grades
    """
    grade_to_score = {'A': 5, 'B': 4, 'C': 3, 'D': 2, 'E': 1}
    score_to_grade = {5: 'A', 4: 'B', 3: 'C', 2: 'D', 1: 'E'}
    
    scores = []
    for grade in letter_grades:
        if grade is not None and grade in grade_to_score:
            scores.append(grade_to_score[grade])
    
    if not scores:
        return None
    
    avg_score = sum(scores) / len(scores)
    # Round to nearest integer and map back to letter grade
    rounded_score = round(avg_score)
    # Clamp to valid range
    rounded_score = max(1, min(5, rounded_score))
    
    return score_to_grade.get(rounded_score)


def predict_healthiness_scores(
    machines_df: pd.DataFrame,
    product_df: pd.DataFrame,
    trained_model: HealthinessModel,
) -> pd.DataFrame:
    """
    Matches products in machine slots to their Nutri-Score letter grades and calculates
    overall machine healthiness scores.
    
    Args:
        machines_df: DataFrame containing machines with 'slots' column.
                     Each slot is a dict with 'product_name' field.
        trained_model: Trained model artifact from train_healthiness_model()
                      (mapping from normalized EAN to letter grade)
        product_df: DataFrame with 'ean' and 'product_name' columns for mapping
                    product_name to EAN.
    
    Returns:
        DataFrame: Input DataFrame with:
            - 'slots' column updated: Each slot dict now has 'nutrivalue_letter_score' field
            - 'machine_overall_letter_grade' column added: Overall letter grade for each machine
    """
    machines_df = machines_df.copy()
    
    if 'slots' not in machines_df.columns:
        logger.warning("No 'slots' column found in machines_df.")
        machines_df['machine_overall_letter_grade'] = None
        return machines_df
    
    # Create product_name to EAN mapping
    if 'ean' not in product_df.columns or 'product_name' not in product_df.columns:
        logger.error("product_df must have 'ean' and 'product_name' columns")
        machines_df['machine_overall_letter_grade'] = None
        return machines_df
    
    product_name_to_ean = {}
    product_df_unique = product_df.drop_duplicates(subset=['product_name'], keep='first')
    for _, row in product_df_unique.iterrows():
        product_name = row.get('product_name')
        ean = row.get('ean')
        if pd.notna(product_name) and pd.notna(ean):
            normalized_name = str(product_name).strip().lower()
            original_name = str(product_name).strip()
            normalized_ean = _normalize_identifier(ean)
            if normalized_ean:
                if normalized_name:
                    product_name_to_ean[normalized_name] = normalized_ean
                if original_name:
                    product_name_to_ean[original_name] = normalized_ean
    
    logger.info(f"Processing {len(machines_df)} machines for healthiness scoring")
    
    machine_overall_grades = []
    total_slots = 0
    slots_matched = 0
    slots_no_product_name = 0
    slots_no_ean_mapping = 0
    slots_no_letter_score = 0
    
    for idx, row in machines_df.iterrows():
        slots = row.get('slots', [])
        
        if not isinstance(slots, list) or len(slots) == 0:
            machine_overall_grades.append(None)
            continue
        
        slot_letter_grades = []
        
        # Process each slot
        for slot in slots:
            if not isinstance(slot, dict):
                slot_letter_grades.append(None)
                continue
            
            total_slots += 1
            
            # Get product_name and map to EAN
            product_name = slot.get('product_name')
            if product_name is None:
                slot_letter_grades.append(None)
                slot['nutrivalue_letter_score'] = None
                slots_no_product_name += 1
                continue
            
            # Map product_name to EAN
            normalized_name = str(product_name).strip().lower()
            original_name = str(product_name).strip()
            ean_value = product_name_to_ean.get(normalized_name) or product_name_to_ean.get(original_name)
            
            if not ean_value:
                slot_letter_grades.append(None)
                slot['nutrivalue_letter_score'] = None
                slots_no_ean_mapping += 1
                continue
            
            letter_score = trained_model.get(ean_value)
            slot_letter_grades.append(letter_score)
            slot['nutrivalue_letter_score'] = letter_score
            
            if letter_score is not None:
                slots_matched += 1
            else:
                slots_no_letter_score += 1
        
        # Calculate overall machine letter grade
        overall_grade = _calculate_machine_overall_letter_grade(slot_letter_grades)
        machine_overall_grades.append(overall_grade)
        
        # Update slots in the row
        machines_df.at[idx, 'slots'] = slots
    
    machines_df['machine_overall_letter_grade'] = machine_overall_grades
    
    logger.info(f"Processed {total_slots} slots")
    logger.info(f"  - {slots_matched} slots matched to letter grades")
    logger.info(f"  - {slots_no_product_name} slots without product_name")
    logger.info(f"  - {slots_no_ean_mapping} slots could not map product_name to EAN")
    logger.info(f"  - {slots_no_letter_score} slots have EAN but no letter score in model")
    
    return machines_df
