import pandas as pd
import numpy as np
import logging
from collections import defaultdict

def compute_content_based_scores(user_item_matrix, user_id, content_metadata, user_interactions_df=None):
    """
    Computes multi-dimensional content-based recommendation scores using:
    - Genres (most important)
    - Artists
    - Compliance/Censorship preferences
    - Premium preference
    - Type (show/movie)
    - Format (standard/microdrama)
    - Moods
    - Themes
    
    Args:
        user_item_matrix: DataFrame with userId as index, contentSlug as columns
        user_id: User to compute scores for
        content_metadata: Dict mapping contentSlug to enriched metadata
        user_interactions_df: Optional DataFrame with user interactions
    
    Returns:
        Dict mapping contentSlug to content-based score (0-1)
    """
    # Handle empty matrix or user not in matrix
    if user_item_matrix.empty or user_id not in user_item_matrix.index:
        # Cold start: no user data, return neutral scores for all content
        logging.info(f"🔍 Content-based: User {user_id} not in matrix (cold start)")
        scores = {}
        # Use content_metadata to get all available content
        if content_metadata:
            for slug in content_metadata.keys():
                scores[slug] = 0.5  # Neutral score
        else:
            # Fallback: if no metadata, return empty (shouldn't happen)
            logging.warning("⚠️ No content metadata available for content-based scoring")
        return scores
    
    user_row = user_item_matrix.iloc[list(user_item_matrix.index).index(user_id)]
    user_interacted_items = user_row[user_row > 0]
    
    if len(user_interacted_items) == 0:
        # Cold start: no interactions, return neutral scores
        logging.info(f"🔍 Content-based: User {user_id} has no interactions (cold start)")
        scores = {}
        # Use content_metadata if available, otherwise use matrix columns
        if content_metadata:
            for slug in content_metadata.keys():
                scores[slug] = 0.5  # Neutral score
        else:
            for slug in user_item_matrix.columns:
                scores[slug] = 0.5  # Neutral score
        return scores
    
    # Calculate user preferences based on interacted items (multi-dimensional)
    genre_preferences = defaultdict(float)
    artist_preferences = defaultdict(float)
    compliance_preferences = defaultdict(float)
    type_preferences = defaultdict(float)
    format_preferences = defaultdict(float)
    mood_preferences = defaultdict(float)
    theme_preferences = defaultdict(float)
    premium_preference = defaultdict(float)  # Track free vs premium
    total_rating = 0.0
    
    for slug, rating in user_interacted_items.items():
        if slug in content_metadata:
            metadata = content_metadata[slug]
            rating_val = float(rating)
            total_rating += rating_val
            
            # Genres (most important)
            for genre in metadata.get('genres', []):
                if genre:
                    genre_preferences[genre] += rating_val
            
            # Sub-genres (also important)
            for subgenre in metadata.get('subgenres', []):
                if subgenre:
                    genre_preferences[subgenre] += rating_val  # Merge with genres
            
            # Artists
            for artist in metadata.get('artists', []):
                if artist:
                    artist_preferences[artist] += rating_val
            
            # Compliance/Censorship
            for comp in metadata.get('compliance', []):
                if comp:
                    compliance_preferences[comp] += rating_val
            
            # Type and Format
            content_type = metadata.get('type', 'unknown')
            content_format = metadata.get('format', 'standard')
            type_preferences[content_type] += rating_val
            format_preferences[content_format] += rating_val
            
            # Moods
            for mood in metadata.get('moods', []):
                if mood:
                    mood_preferences[mood] += rating_val
            
            # Themes
            for theme in metadata.get('themes', []):
                if theme:
                    theme_preferences[theme] += rating_val
            
            # Premium preference
            is_premium = metadata.get('isPremium', False)
            premium_key = 'premium' if is_premium else 'free'
            premium_preference[premium_key] += rating_val
    
    # Normalize all preferences
    if total_rating > 0:
        for key in genre_preferences:
            genre_preferences[key] /= total_rating
        for key in artist_preferences:
            artist_preferences[key] /= total_rating
        for key in compliance_preferences:
            compliance_preferences[key] /= total_rating
        for key in type_preferences:
            type_preferences[key] /= total_rating
        for key in format_preferences:
            format_preferences[key] /= total_rating
        for key in mood_preferences:
            mood_preferences[key] /= total_rating
        for key in theme_preferences:
            theme_preferences[key] /= total_rating
        for key in premium_preference:
            premium_preference[key] /= total_rating
    
    logging.info(f"🔍 Content-based preferences for {user_id}:")
    logging.info(f"   Top Genres: {dict(list(sorted(genre_preferences.items(), key=lambda x: x[1], reverse=True))[:5])}")
    logging.info(f"   Top Artists: {dict(list(sorted(artist_preferences.items(), key=lambda x: x[1], reverse=True))[:3])}")
    logging.info(f"   Types: {dict(type_preferences)}")
    logging.info(f"   Formats: {dict(format_preferences)}")
    logging.info(f"   Premium: {dict(premium_preference)}")
    
    # Compute multi-dimensional content-based scores for all items
    content_scores = {}
    items_to_score = set(content_metadata.keys()) if content_metadata else set(user_item_matrix.columns)
    
    for slug in items_to_score:
        if slug not in content_metadata:
            content_scores[slug] = 0.5
            continue
        
        metadata = content_metadata[slug]
        
        # 1. Genre Score (40% weight) - Most important
        genre_score = 0.0
        genre_matches = 0
        for genre in metadata.get('genres', []):
            if genre and genre in genre_preferences:
                genre_score += genre_preferences[genre]
                genre_matches += 1
        for subgenre in metadata.get('subgenres', []):
            if subgenre and subgenre in genre_preferences:
                genre_score += genre_preferences[subgenre] * 0.8  # Slightly less weight
                genre_matches += 1
        if genre_matches > 0:
            genre_score = genre_score / max(genre_matches, 1)  # Average of matches
        else:
            genre_score = 0.0
        
        # 2. Artist Score (20% weight)
        artist_score = 0.0
        artist_matches = 0
        for artist in metadata.get('artists', []):
            if artist and artist in artist_preferences:
                artist_score += artist_preferences[artist]
                artist_matches += 1
        if artist_matches > 0:
            artist_score = artist_score / max(artist_matches, 1)
        else:
            artist_score = 0.0
        
        # 3. Type & Format Score (15% weight combined)
        content_type = metadata.get('type', 'unknown')
        content_format = metadata.get('format', 'standard')
        type_score = type_preferences.get(content_type, 0.0)
        format_score = format_preferences.get(content_format, 0.0)
        type_format_score = 0.6 * type_score + 0.4 * format_score
        
        # 4. Mood Score (10% weight)
        mood_score = 0.0
        mood_matches = 0
        for mood in metadata.get('moods', []):
            if mood and mood in mood_preferences:
                mood_score += mood_preferences[mood]
                mood_matches += 1
        if mood_matches > 0:
            mood_score = mood_score / max(mood_matches, 1)
        
        # 5. Theme Score (10% weight)
        theme_score = 0.0
        theme_matches = 0
        for theme in metadata.get('themes', []):
            if theme and theme in theme_preferences:
                theme_score += theme_preferences[theme]
                theme_matches += 1
        if theme_matches > 0:
            theme_score = theme_score / max(theme_matches, 1)
        
        # 6. Premium Preference (5% weight)
        is_premium = metadata.get('isPremium', False)
        premium_key = 'premium' if is_premium else 'free'
        premium_score = premium_preference.get(premium_key, 0.5)  # Default neutral if no preference
        
        # Weighted combination
        combined_score = (
            0.40 * genre_score +           # Genres most important
            0.20 * artist_score +          # Artists second
            0.15 * type_format_score +     # Type & Format
            0.10 * mood_score +            # Moods
            0.10 * theme_score +           # Themes
            0.05 * premium_score           # Premium preference
        )
        
        # Ensure score is in 0-1 range
        content_scores[slug] = min(1.0, max(0.0, combined_score))
    
    return content_scores

