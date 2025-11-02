import numpy as np
import logging
import pandas as pd
from scipy.sparse import csr_matrix
from content_based import compute_content_based_scores

def recommend_for_user(model, user_item_matrix, sparse_matrix, user_id, content_metadata=None, top_n=5, collaborative_weight=0.7, content_weight=0.3):
    """
    Hybrid recommendation: Combines collaborative filtering and content-based filtering.
    Returns top N recommended contentSlugs for a user.
    
    Args:
        model: Trained ALS model
        user_item_matrix: DataFrame with userId as index, contentSlug as columns
        sparse_matrix: Sparse CSR matrix used for training
        user_id: User to recommend for
        content_metadata: Dict mapping contentSlug to {type, format}
        top_n: Number of recommendations to return
        collaborative_weight: Weight for collaborative filtering score (default 0.7)
        content_weight: Weight for content-based score (default 0.3)
    
    Returns:
        List of contentSlug strings (top N recommendations)
    """
    # Validate inputs
    if user_item_matrix.empty:
        # Matrix is empty, use content-based only
        logging.info(f"User-item matrix is empty. Using content-based recommendations only.")
        if not content_metadata:
            logging.warning("No content metadata available. Cannot generate recommendations.")
            return []
        
        # Pure content-based recommendations
        content_scores = compute_content_based_scores(
            pd.DataFrame(),  # Empty matrix
            user_id, 
            content_metadata
        )
        
        # Sort and return top N
        sorted_items = sorted(content_scores.items(), key=lambda x: x[1], reverse=True)
        top_recommendations = [slug for slug, score in sorted_items[:top_n]]
        
        logging.info(f"✅ Content-based recommendations for {user_id}:")
        for i, (slug, score) in enumerate(sorted_items[:top_n], 1):
            logging.info(f"   {i}. {slug} (Score: {score:.4f})")
        
        return top_recommendations
    
    if user_id not in user_item_matrix.index:
        logging.warning(f"User {user_id} not found in matrix.")
        # Still try content-based if metadata available
        if content_metadata:
            logging.info("Attempting content-based recommendations for unknown user.")
            content_scores = compute_content_based_scores(
                user_item_matrix,
                user_id, 
                content_metadata
            )
            sorted_items = sorted(content_scores.items(), key=lambda x: x[1], reverse=True)
            return [slug for slug, score in sorted_items[:top_n]]
        return []
    
    if model is None:
        logging.warning(f"Model is None. Using content-based recommendations only.")
        if not content_metadata:
            logging.warning("No content metadata available. Cannot generate recommendations.")
            return []
        
        # Pure content-based recommendations
        content_scores = compute_content_based_scores(
            user_item_matrix,
            user_id, 
            content_metadata
        )
        
        sorted_items = sorted(content_scores.items(), key=lambda x: x[1], reverse=True)
        top_recommendations = [slug for slug, score in sorted_items[:top_n]]
        
        logging.info(f"✅ Content-based recommendations for {user_id}:")
        for i, (slug, score) in enumerate(sorted_items[:top_n], 1):
            logging.info(f"   {i}. {slug} (Score: {score:.4f})")
        
        return top_recommendations

    user_idx = list(user_item_matrix.index).index(user_id)
    
    # Get user's interaction vector (row from the matrix)
    user_row = user_item_matrix.iloc[user_idx]
    
    # Log user's interaction matrix (non-zero ratings only)
    user_interactions = user_row[user_row > 0]
    if len(user_interactions) > 0:
        logging.info(f"📊 User {user_id} interaction matrix ({len(user_interactions)} interactions):")
        # Log top interactions (sorted by rating, descending)
        sorted_interactions = user_interactions.sort_values(ascending=False)
        for content_slug, rating in sorted_interactions.head(10).items():
            logging.info(f"   - {content_slug}: {rating:.2f}")
        if len(user_interactions) > 10:
            logging.info(f"   ... and {len(user_interactions) - 10} more interactions")
    else:
        logging.info(f"📊 User {user_id} has no interactions (all ratings are 0)")
    
    # Ensure user_idx is within valid range
    if user_idx >= sparse_matrix.shape[0]:
        logging.error(f"❌ User index {user_idx} is out of bounds. Matrix has {sparse_matrix.shape[0]} rows.")
        return []
    
    # Log detailed information about inputs
    logging.info(f"🔍 DEBUG: Starting recommendation computation")
    logging.info(f"🔍 DEBUG: user_idx={user_idx} (type: {type(user_idx)})")
    logging.info(f"🔍 DEBUG: sparse_matrix.shape={sparse_matrix.shape}, format={sparse_matrix.format}, dtype={sparse_matrix.dtype}")
    logging.info(f"🔍 DEBUG: sparse_matrix.nnz (non-zero elements)={sparse_matrix.nnz}")
    logging.info(f"🔍 DEBUG: user_item_matrix.shape={user_item_matrix.shape}")
    logging.info(f"🔍 DEBUG: top_n={top_n}")
    
    # Try model.recommend() methods first with detailed logging
    # Note: implicit library STRICTLY requires float64 dtype AND correct shape for CSR matrices
    # IMPORTANT: model.recommend() expects a matrix with 1 row per user in userids
    # So if requesting 1 user, we need to pass a (1, n_items) matrix, not the full (n_users, n_items) matrix
    recs = None
    method_used = None
    
    # Ensure sparse matrix is float64 and CSR (required by implicit library)
    if sparse_matrix.dtype != np.float64:
        logging.info(f"🔍 DEBUG: Converting sparse matrix from {sparse_matrix.dtype} to float64 (required by implicit)")
        sparse_matrix = sparse_matrix.astype(np.float64)
    if sparse_matrix.format != 'csr':
        logging.info(f"🔍 DEBUG: Converting sparse matrix to CSR format")
        sparse_matrix = sparse_matrix.tocsr()
    
    # CRITICAL FIX: Slice the matrix to get only this user's row
    # implicit expects: if userids=[0], then user_items must have shape (1, n_items)
    user_row_matrix = sparse_matrix[user_idx:user_idx+1]  # Get single row as (1, n_items) matrix
    logging.info(f"🔍 DEBUG: Full matrix shape: {sparse_matrix.shape}, Sliced user matrix shape: {user_row_matrix.shape}")
    logging.info(f"🔍 DEBUG: User matrix dtype: {user_row_matrix.dtype}, format: {user_row_matrix.format}")
    
    # Use native model.recommend() with sliced matrix (correct approach)
    # CRITICAL: implicit library requires:
    # 1. Matrix dtype = float64
    # 2. Matrix format = CSR
    # 3. Matrix shape = (1, n_items) when requesting 1 user (not full matrix)
    try:
        logging.info(f"🔍 DEBUG: Using native model.recommend() with sliced matrix")
        logging.info(f"🔍 DEBUG: user_row_matrix.shape: {user_row_matrix.shape}, dtype: {user_row_matrix.dtype}")
        result = model.recommend(user_idx, user_row_matrix, N=top_n, filter_already_liked_items=True)
        
        # Handle return format: model.recommend() with single integer returns tuple (item_indices, scores)
        if isinstance(result, tuple) and len(result) == 2:
            # Tuple format: (item_indices_array, scores_array) - convert to list of tuples
            item_indices, scores = result
            recs = [[(int(item_idx), float(score)) for item_idx, score in zip(item_indices, scores)]]
            logging.info(f"✅ Native model.recommend() succeeded! Generated {len(recs[0])} recommendations")
        elif isinstance(result, list):
            # List format - already correct format
            recs = result if isinstance(result[0], list) else [result]
            logging.info(f"✅ Native model.recommend() succeeded! Returned {len(recs[0]) if recs else 0} recommendations")
        else:
            logging.warning(f"⚠️ Unexpected return format from model.recommend(): {type(result)}")
            recs = None
        
        method_used = "Native model.recommend()"
    except Exception as e1:
        logging.warning(f"⚠️ Native model.recommend() failed: {type(e1).__name__}: {str(e1)}")
        logging.warning(f"⚠️ Error details: {repr(e1)}")
        
        # Fallback: Manual computation using model factors (always works)
        logging.info(f"🔍 DEBUG: Falling back to manual computation")
        try:
            user_factors = model.user_factors[user_idx]
            item_factors = model.item_factors
            
            scores = user_factors.dot(item_factors.T)
            
            user_row = sparse_matrix.getrow(user_idx)
            user_liked = set(user_row.indices)
            
            item_scores = [(i, float(score)) for i, score in enumerate(scores) if i not in user_liked]
            item_scores.sort(key=lambda x: x[1], reverse=True)
            
            recs = [[item_scores[i] for i in range(min(top_n, len(item_scores)))]]
            method_used = "Manual computation (fallback)"
            logging.info(f"✅ Manual computation succeeded! Generated {len(recs[0])} recommendations")
        except Exception as e2:
            logging.error(f"❌ Manual computation also failed: {type(e2).__name__}: {str(e2)}")
            logging.error(f"❌ Error details: {repr(e2)}")
            import traceback
            logging.error(f"❌ Full traceback:\n{traceback.format_exc()}")
            raise
    
    # recs is a list of lists (one per user), get first user's recommendations
    # Handle both None and empty cases safely
    if recs is None:
        logging.warning(f"⚠️ No recommendations returned for user {user_id} (recs is None)")
        return []
    
    if not isinstance(recs, (list, tuple)) or len(recs) == 0:
        logging.warning(f"⚠️ No recommendations returned for user {user_id}")
        logging.warning(f"⚠️ recs value: {recs}, type: {type(recs)}")
        return []
    
    # Extract user recommendations from recs (handle tuple return format)
    user_recs = []
    if isinstance(recs, tuple) and len(recs) == 2:
        # Tuple format: (item_indices_array, scores_array)
        item_indices, scores = recs
        user_recs = [(int(idx), float(score)) for idx, score in zip(item_indices, scores)]
        logging.info(f"🔍 DEBUG: Converted tuple format to list of tuples")
    elif isinstance(recs, list) and len(recs) > 0:
        # List format: should be list of lists or list of tuples
        user_recs = recs[0] if isinstance(recs[0], (list, tuple, np.ndarray)) else recs
        # Convert numpy arrays to list if needed
        if isinstance(user_recs, np.ndarray):
            # If it's a 2D array, convert to list of tuples
            if user_recs.ndim == 2 and user_recs.shape[1] >= 2:
                user_recs = [(int(row[0]), float(row[1])) for row in user_recs]
            else:
                user_recs = user_recs.tolist()
    else:
        logging.warning(f"⚠️ Unexpected recs format: {type(recs)}, value: {recs}")
        return []
    
    # Check if user_recs is empty (handle numpy arrays and lists properly)
    if isinstance(user_recs, np.ndarray):
        if user_recs.size == 0:
            logging.warning(f"⚠️ Empty recommendation array for user {user_id}")
            return []
    elif isinstance(user_recs, (list, tuple)):
        if len(user_recs) == 0:
            logging.warning(f"⚠️ Empty recommendation list for user {user_id}")
            return []
    else:
        logging.warning(f"⚠️ Unexpected user_recs type: {type(user_recs)}")
        return []
    
    logging.info(f"🔍 DEBUG: Method used: {method_used}")
    logging.info(f"🔍 DEBUG: user_recs count: {len(user_recs)}")
    logging.info(f"🔍 DEBUG: user_recs sample (first 3): {user_recs[:3] if len(user_recs) >= 3 else user_recs}")
    logging.info(f"🔍 DEBUG: user_recs types: {[type(x) for x in user_recs[:3]] if len(user_recs) >= 3 else []}")
    
    # Step 1: Get collaborative filtering scores for all items
    collaborative_scores = {}
    for item in user_recs:
        # Handle tuple format: (item_idx, score)
        if isinstance(item, (list, tuple)) and len(item) >= 1:
            item_idx = int(item[0])
            score = float(item[1]) if len(item) > 1 else 0.0
            if item_idx < len(user_item_matrix.columns):
                slug = user_item_matrix.columns[item_idx]
                collaborative_scores[slug] = score
                logging.debug(f"🔍 Collaborative: {slug} → {score:.4f}")
            else:
                logging.warning(f"⚠️ Item index {item_idx} out of range")
        elif isinstance(item, (int, np.integer)):
            item_idx = int(item)
            if item_idx < len(user_item_matrix.columns):
                slug = user_item_matrix.columns[item_idx]
                collaborative_scores[slug] = 0.0  # No score available
            else:
                logging.warning(f"⚠️ Item index {item_idx} out of range")
    
    # Step 2: If no collaborative scores, compute manually for all items
    if not collaborative_scores and model is not None:
        logging.info("🔍 Computing collaborative scores for all items (manual)")
        try:
            user_factors = model.user_factors[user_idx]
            item_factors = model.item_factors
            
            # Get user's liked items to filter
            user_row = sparse_matrix.getrow(user_idx)
            user_liked = set(user_row.indices)
            
            # Compute scores for all items
            all_scores = user_factors.dot(item_factors.T)
            for item_idx, score in enumerate(all_scores):
                if item_idx not in user_liked and item_idx < len(user_item_matrix.columns):
                    slug = user_item_matrix.columns[item_idx]
                    collaborative_scores[slug] = float(score)
        except Exception as e:
            logging.warning(f"⚠️ Failed to compute manual collaborative scores: {e}")
    
    # Step 3: Compute content-based scores
    content_scores = {}
    if content_metadata:
        logging.info("🔍 Computing content-based scores")
        content_scores = compute_content_based_scores(
            user_item_matrix, 
            user_id, 
            content_metadata
        )
        logging.info(f"✅ Content-based scores computed for {len(content_scores)} items")
    else:
        logging.warning("⚠️ No content_metadata provided. Skipping content-based scoring.")
        # Set neutral content scores for all items
        for slug in user_item_matrix.columns:
            content_scores[slug] = 0.5
    
    # Step 4: Normalize collaborative scores (0-1 range)
    if collaborative_scores:
        max_collab = max(collaborative_scores.values()) if collaborative_scores.values() else 1.0
        min_collab = min(collaborative_scores.values()) if collaborative_scores.values() else 0.0
        range_collab = max_collab - min_collab if max_collab != min_collab else 1.0
        
        for slug in collaborative_scores:
            if range_collab > 0:
                collaborative_scores[slug] = (collaborative_scores[slug] - min_collab) / range_collab
            else:
                collaborative_scores[slug] = 0.5
    
    # Step 5: Combine scores using hybrid formula
    hybrid_scores = {}
    all_items = set(list(collaborative_scores.keys()) + list(content_scores.keys()))
    
    for slug in all_items:
        collab_score = collaborative_scores.get(slug, 0.0)
        content_score = content_scores.get(slug, 0.5)
        
        # Hybrid score: weighted combination
        hybrid_score = (collaborative_weight * collab_score) + (content_weight * content_score)
        hybrid_scores[slug] = hybrid_score
        
        if slug in collaborative_scores:
            logging.debug(f"🔍 Hybrid: {slug} → CF={collab_score:.3f}, CB={content_score:.3f}, Final={hybrid_score:.3f}")
    
    # Step 6: Sort by hybrid score and return top N
    sorted_items = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
    top_recommendations = [slug for slug, score in sorted_items[:top_n]]
    
    logging.info(f"✅ Hybrid recommendations for {user_id}:")
    for i, (slug, score) in enumerate(sorted_items[:top_n], 1):
        collab = collaborative_scores.get(slug, 0.0)
        content = content_scores.get(slug, 0.5)
        logging.info(f"   {i}. {slug} (Hybrid: {score:.4f}, CF: {collab:.4f}, CB: {content:.4f})")
    
    return top_recommendations


if __name__ == "__main__":
    import data_loader, preprocess, train_model
    df_interactions, df_shows, df_episodes = data_loader.load_data()
    df_interactions = preprocess.preprocess_interactions(df_interactions)
    model, matrix, sparse, metadata = train_model.train_model(df_interactions, df_shows, df_episodes)
    if not matrix.empty and len(matrix.index) > 0:
        print("Top recommendations:", recommend_for_user(model, matrix, sparse, matrix.index[0], content_metadata=metadata))
    else:
        print("⚠️ No users in matrix to test recommendations")
