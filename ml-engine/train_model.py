from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix
import pandas as pd
import numpy as np
import logging

def train_model(df_interactions: pd.DataFrame, df_shows: pd.DataFrame, df_episodes: pd.DataFrame):
    """
    Trains an ALS collaborative filtering model using contentSlug.
    Includes ALL content from shows/episodes tables (hybrid approach).
    Items without interactions get 0 rating.
    
    Returns model, user_item DataFrame, sparse_matrix, and content_metadata
    
    Note: implicit library requires float64 dtype for CSR matrices
    """
    # Step 1: Get all unique content slugs from shows and episodes
    all_content_slugs = set()
    content_metadata = {}  # Map slug to enriched metadata
    
    def extract_genres(genre_list):
        """Extract genre names from genreList array"""
        if genre_list is None:
            return []
        # Handle pandas Series or numpy arrays
        if isinstance(genre_list, (pd.Series, np.ndarray)):
            genre_list = genre_list.tolist()
        if not isinstance(genre_list, list):
            return []
        return [g.get('name', '').lower() if isinstance(g, dict) else str(g).lower() 
                for g in genre_list if g]
    
    def extract_subgenres(subgenre_list):
        """Extract sub-genre names from subGenreList array"""
        if subgenre_list is None:
            return []
        # Handle pandas Series or numpy arrays
        if isinstance(subgenre_list, (pd.Series, np.ndarray)):
            subgenre_list = subgenre_list.tolist()
        if not isinstance(subgenre_list, list):
            return []
        return [sg.get('name', '').lower() if isinstance(sg, dict) else str(sg).lower() 
                for sg in subgenre_list if sg]
    
    def extract_artists(artist_list):
        """Extract artist slugs from artistList array"""
        if artist_list is None:
            return []
        # Handle pandas Series or numpy arrays
        if isinstance(artist_list, (pd.Series, np.ndarray)):
            artist_list = artist_list.tolist()
        if not isinstance(artist_list, list):
            return []
        return [a.get('slug', '') if isinstance(a, dict) else str(a) 
                for a in artist_list if a and (isinstance(a, dict) and a.get('slug') or True)]
    
    def extract_compliance(compliance_list):
        """Extract compliance/censorship tags"""
        if compliance_list is None:
            return []
        # Handle pandas Series or numpy arrays
        if isinstance(compliance_list, (pd.Series, np.ndarray)):
            compliance_list = compliance_list.tolist()
        if not isinstance(compliance_list, list):
            return []
        return [c.get('name', '').lower() if isinstance(c, dict) else str(c).lower() 
                for c in compliance_list if c]
    
    def extract_moods(moods_list):
        """Extract mood names"""
        if moods_list is None:
            return []
        # Handle pandas Series or numpy arrays
        if isinstance(moods_list, (pd.Series, np.ndarray)):
            moods_list = moods_list.tolist()
        if not isinstance(moods_list, list):
            return []
        return [m.get('name', '').lower() if isinstance(m, dict) else str(m).lower() 
                for m in moods_list if m]
    
    def extract_themes(themes_list):
        """Extract theme names"""
        if themes_list is None:
            return []
        # Handle pandas Series or numpy arrays
        if isinstance(themes_list, (pd.Series, np.ndarray)):
            themes_list = themes_list.tolist()
        if not isinstance(themes_list, list):
            return []
        return [t.get('name', '').lower() if isinstance(t, dict) else str(t).lower() 
                for t in themes_list if t]
    
    # Add shows
    for idx, show in df_shows.iterrows():
        try:
            if 'slug' not in show or pd.isna(show.get('slug')):
                continue
            
            slug = str(show['slug'])
            all_content_slugs.add(slug)
            
            genres = extract_genres(show.get('genreList'))
            subgenres = extract_subgenres(show.get('subGenreList'))
            artists = extract_artists(show.get('artistList'))
            compliance = extract_compliance(show.get('complianceList'))
            moods = extract_moods(show.get('moods'))
            themes = extract_themes(show.get('themes'))
            
            # Handle scalar values safely
            format_val = show.get('format', 'standard')
            if pd.notna(format_val):
                format_val = str(format_val)
            else:
                format_val = 'standard'
            
            compliance_rating = show.get('complianceRating', '')
            if pd.notna(compliance_rating):
                compliance_rating = str(compliance_rating)
            else:
                compliance_rating = ''
            
            is_premium = show.get('isPremium', False)
            if pd.notna(is_premium):
                is_premium = bool(is_premium)
            else:
                is_premium = False
            
            duration = show.get('duration', 0)
            if pd.notna(duration):
                try:
                    duration = int(duration)
                except (ValueError, TypeError):
                    duration = 0
            else:
                duration = 0
            
            primary_dialect = show.get('primaryDialect', '')
            if pd.notna(primary_dialect):
                primary_dialect = str(primary_dialect)
            else:
                primary_dialect = ''
            
            tags = show.get('tags', '')
            if pd.notna(tags):
                tags = str(tags)
            else:
                tags = ''
            
            content_metadata[slug] = {
                'type': 'show',
                'format': format_val,
                'genres': genres,
                'subgenres': subgenres,
                'artists': artists,
                'compliance': compliance,
                'complianceRating': compliance_rating,
                'isPremium': is_premium,
                'duration': duration,
                'moods': moods,
                'themes': themes,
                'primaryDialect': primary_dialect,
                'tags': tags
            }
        except Exception as e:
            logging.warning(f"⚠️ Error processing show at index {idx}: {e}")
            continue
    
    # Add episodes (movies)
    for idx, episode in df_episodes.iterrows():
        try:
            if 'slug' not in episode or pd.isna(episode.get('slug')):
                continue
            
            slug = str(episode['slug'])
            all_content_slugs.add(slug)
            
            genres = extract_genres(episode.get('genreList'))
            subgenres = extract_subgenres(episode.get('subGenreList'))
            artists = extract_artists(episode.get('artistList'))
            compliance = extract_compliance(episode.get('complianceList'))
            moods = extract_moods(episode.get('moods'))
            themes = extract_themes(episode.get('themes'))
            
            # Handle scalar values safely
            format_val = episode.get('format', 'standard')
            if pd.notna(format_val):
                format_val = str(format_val)
            else:
                format_val = 'standard'
            
            compliance_rating = episode.get('complianceRating', '')
            if pd.notna(compliance_rating):
                compliance_rating = str(compliance_rating)
            else:
                compliance_rating = ''
            
            is_premium = episode.get('isPremium', False)
            if pd.notna(is_premium):
                is_premium = bool(is_premium)
            else:
                is_premium = False
            
            duration = episode.get('duration', 0)
            if pd.notna(duration):
                try:
                    duration = int(duration)
                except (ValueError, TypeError):
                    duration = 0
            else:
                duration = 0
            
            tags = episode.get('tags', '')
            if pd.notna(tags):
                tags = str(tags)
            else:
                tags = ''
            
            content_metadata[slug] = {
                'type': 'movie',
                'format': format_val,
                'genres': genres,
                'subgenres': subgenres,
                'artists': artists,
                'compliance': compliance,
                'complianceRating': compliance_rating,
                'isPremium': is_premium,
                'duration': duration,
                'moods': moods,
                'themes': themes,
                'tags': tags
            }
        except Exception as e:
            logging.warning(f"⚠️ Error processing episode at index {idx}: {e}")
            continue
    
    logging.info(f"📦 Total available content: {len(all_content_slugs)} items (from shows + episodes)")
    
    # Step 2: Create pivot table from interactions
    if not df_interactions.empty and len(df_interactions) > 0:
        user_item = df_interactions.pivot_table(
            index="userId", 
            columns="contentSlug", 
            values="rating"
        ).fillna(0)
    else:
        # If no interactions, create empty DataFrame with no users
        # Model will be trained on empty matrix (not ideal but won't crash)
        user_item = pd.DataFrame(columns=sorted(all_content_slugs))
        logging.warning("⚠️ No interactions found. Creating empty user-item matrix.")
    
    # Step 3: Add columns for all content (even without interactions)
    # This ensures ALL content from shows/episodes is in the recommendation pool
    if not user_item.empty:
        missing_content = all_content_slugs - set(user_item.columns)
        if missing_content:
            for slug in missing_content:
                user_item[slug] = 0  # Add column with 0 ratings
        
        # Sort columns for consistency
        user_item = user_item.reindex(columns=sorted(user_item.columns))
    else:
        # If user_item is empty, ensure it has all content columns
        user_item = pd.DataFrame(columns=sorted(all_content_slugs))
    
    logging.info(f"📊 User-item matrix shape: {user_item.shape}")
    if not user_item.empty:
        missing_count = len(all_content_slugs - set(user_item.columns)) if set(user_item.columns) != all_content_slugs else 0
        logging.info(f"   - Users: {user_item.shape[0]}")
        logging.info(f"   - Content items: {user_item.shape[1]} (includes {missing_count} items with no interactions)")
    else:
        logging.info(f"   - Users: 0 (empty matrix - no interactions)")
        logging.info(f"   - Content items: {len(all_content_slugs)}")
    
    # Convert to float64 - implicit library STRICTLY requires float64 dtype
    sparse_matrix = csr_matrix(user_item.values.astype('float64'))
    
    # Ensure CSR format (required by implicit library)
    if sparse_matrix.format != 'csr':
        sparse_matrix = sparse_matrix.tocsr()
    
    # Ensure dtype is float64 (implicit library is strict about this)
    if sparse_matrix.dtype != np.float64:
        sparse_matrix = sparse_matrix.astype(np.float64)

    # Only train model if we have users and items
    if sparse_matrix.shape[0] > 0 and sparse_matrix.shape[1] > 0:
        model = AlternatingLeastSquares(factors=20, regularization=0.1, iterations=20)
        model.fit(sparse_matrix)
        logging.info(f"✅ Model trained successfully. Content pool: {len(user_item.columns)} items")
    else:
        model = None
        logging.warning("⚠️ Cannot train model: empty matrix (no users or items)")

    return model, user_item, sparse_matrix, content_metadata


if __name__ == "__main__":
    import data_loader, preprocess
    df_interactions, df_shows, df_episodes = data_loader.load_data()
    df_interactions = preprocess.preprocess_interactions(df_interactions)
    model, matrix, sparse, metadata = train_model(df_interactions, df_shows, df_episodes)
    print("✅ Model trained on", matrix.shape, "matrix")
    print(f"   Content metadata: {len(metadata)} items")
