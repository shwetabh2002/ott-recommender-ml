"""
Baseline Content-Based Recommendation (Old: Type/Format Only)

This is the simple version that only uses type and format for comparison
"""

import pandas as pd
import logging
from collections import defaultdict


def compute_baseline_content_scores(user_item_matrix, user_id, content_metadata):
    """
    Baseline content-based scoring using ONLY type and format (old approach).
    
    This is for comparison with the enhanced 6D model.
    
    Args:
        user_item_matrix: DataFrame with userId as index, contentSlug as columns
        user_id: User to compute scores for
        content_metadata: Dict mapping contentSlug to {type, format}
    
    Returns:
        Dict mapping contentSlug to content-based score (0-1)
    """
    # Handle empty matrix or user not in matrix
    if user_item_matrix.empty or user_id not in user_item_matrix.index:
        logging.info(f"🔍 Baseline: User {user_id} not in matrix (cold start)")
        scores = {}
        if content_metadata:
            for slug in content_metadata.keys():
                scores[slug] = 0.5  # Neutral score
        return scores
    
    user_row = user_item_matrix.iloc[list(user_item_matrix.index).index(user_id)]
    user_interacted_items = user_row[user_row > 0]
    
    if len(user_interacted_items) == 0:
        logging.info(f"🔍 Baseline: User {user_id} has no interactions (cold start)")
        scores = {}
        if content_metadata:
            for slug in content_metadata.keys():
                scores[slug] = 0.5  # Neutral score
        return scores
    
    # Calculate user preferences based on interacted items (ONLY type and format)
    type_preferences = defaultdict(float)
    format_preferences = defaultdict(float)
    total_rating = 0.0
    
    for slug, rating in user_interacted_items.items():
        if slug in content_metadata:
            metadata = content_metadata[slug]
            content_type = metadata.get('type', 'unknown')
            content_format = metadata.get('format', 'standard')
            
            # Weight by rating
            type_preferences[content_type] += float(rating)
            format_preferences[content_format] += float(rating)
            total_rating += float(rating)
    
    # Normalize preferences
    if total_rating > 0:
        for key in type_preferences:
            type_preferences[key] /= total_rating
        for key in format_preferences:
            format_preferences[key] /= total_rating
    
    # Compute baseline content-based scores (ONLY type and format)
    content_scores = {}
    items_to_score = set(content_metadata.keys()) if content_metadata else set(user_item_matrix.columns)
    
    for slug in items_to_score:
        if slug not in content_metadata:
            content_scores[slug] = 0.5
            continue
        
        metadata = content_metadata[slug]
        content_type = metadata.get('type', 'unknown')
        content_format = metadata.get('format', 'standard')
        
        # Simple weighted combination (60% type, 40% format)
        type_score = type_preferences.get(content_type, 0.0)
        format_score = format_preferences.get(content_format, 0.0)
        combined_score = 0.6 * type_score + 0.4 * format_score
        
        content_scores[slug] = min(1.0, max(0.0, combined_score))
    
    return content_scores

