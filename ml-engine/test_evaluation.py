"""
Evaluation Script for Recommendation System

Compares:
1. Baseline model (type/format only)
2. Enhanced model (6D multi-dimensional)

Measures: Precision@K, Recall@K, NDCG@K, Coverage
"""

import logging
import pandas as pd
from data_loader import load_data
from preprocess import preprocess_interactions
from train_model import train_model
from recommend import recommend_for_user
from evaluation import (
    split_interactions,
    prepare_ground_truth,
    compare_models,
    print_evaluation_report,
    print_comparison_report
)
from content_based import compute_content_based_scores
from baseline_content_based import compute_baseline_content_scores

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def generate_baseline_recommendations(
    model, user_item_matrix, sparse_matrix, user_ids, content_metadata, top_n=10
):
    """
    Generate recommendations using baseline (old) approach
    """
    recommendations = {}
    
    for user_id in user_ids:
        try:
            if user_id not in user_item_matrix.index:
                recommendations[user_id] = []
                continue
            
            # Get collaborative filtering scores
            user_idx = list(user_item_matrix.index).index(user_id)
            
            # Compute collaborative scores (same for both models)
            try:
                user_row_matrix = sparse_matrix[user_idx:user_idx+1]
                result = model.recommend(user_idx, user_row_matrix, N=top_n*2, filter_already_liked_items=True)
                
                if isinstance(result, tuple) and len(result) == 2:
                    item_indices, scores = result
                    collaborative_scores = {}
                    for item_idx, score in zip(item_indices, scores):
                        if item_idx < len(user_item_matrix.columns):
                            slug = user_item_matrix.columns[item_idx]
                            collaborative_scores[slug] = float(score)
                else:
                    collaborative_scores = {}
            except:
                # Fallback to manual computation
                user_factors = model.user_factors[user_idx]
                item_factors = model.item_factors
                scores = user_factors.dot(item_factors.T)
                user_row = sparse_matrix.getrow(user_idx)
                user_liked = set(user_row.indices)
                
                collaborative_scores = {}
                for item_idx, score in enumerate(scores):
                    if item_idx not in user_liked and item_idx < len(user_item_matrix.columns):
                        slug = user_item_matrix.columns[item_idx]
                        collaborative_scores[slug] = float(score)
            
            # Normalize collaborative scores
            if collaborative_scores:
                max_collab = max(collaborative_scores.values())
                min_collab = min(collaborative_scores.values())
                range_collab = max_collab - min_collab if max_collab != min_collab else 1.0
                for slug in collaborative_scores:
                    if range_collab > 0:
                        collaborative_scores[slug] = (collaborative_scores[slug] - min_collab) / range_collab
            
            # Get baseline content scores (ONLY type/format)
            content_scores = compute_baseline_content_scores(
                user_item_matrix, user_id, content_metadata
            )
            
            # Combine scores (70% collaborative, 30% content)
            hybrid_scores = {}
            all_items = set(list(collaborative_scores.keys()) + list(content_scores.keys()))
            
            for slug in all_items:
                collab_score = collaborative_scores.get(slug, 0.0)
                content_score = content_scores.get(slug, 0.5)
                hybrid_score = 0.7 * collab_score + 0.3 * content_score
                hybrid_scores[slug] = hybrid_score
            
            # Sort and get top N
            sorted_items = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
            recommendations[user_id] = [slug for slug, score in sorted_items[:top_n]]
            
        except Exception as e:
            logging.warning(f"⚠️ Error generating baseline recommendations for {user_id}: {e}")
            recommendations[user_id] = []
    
    return recommendations


def generate_enhanced_recommendations(
    model, user_item_matrix, sparse_matrix, user_ids, content_metadata, top_n=10
):
    """
    Generate recommendations using enhanced (6D multi-dimensional) approach
    """
    recommendations = {}
    
    for user_id in user_ids:
        try:
            recs = recommend_for_user(
                model, user_item_matrix, sparse_matrix, user_id,
                content_metadata=content_metadata, top_n=top_n
            )
            recommendations[user_id] = recs
        except Exception as e:
            logging.warning(f"⚠️ Error generating enhanced recommendations for {user_id}: {e}")
            recommendations[user_id] = []
    
    return recommendations


def main():
    """
    Main evaluation function
    """
    print("\n" + "="*80)
    print("🔬 Recommendation System Evaluation")
    print("="*80)
    
    # Step 1: Load data
    print("\n📊 Step 1: Loading data...")
    df_interactions, df_shows, df_episodes = load_data()
    
    if df_interactions.empty:
        print("❌ No interactions found. Cannot evaluate.")
        return
    
    # Step 2: Preprocess
    print("\n📊 Step 2: Preprocessing interactions...")
    df_interactions = preprocess_interactions(df_interactions)
    
    # Step 3: Split into train/test
    print("\n📊 Step 3: Splitting into train/test sets...")
    train_df, test_df = split_interactions(df_interactions, test_ratio=0.2, min_interactions=2)
    
    if len(test_df) == 0:
        print("❌ No test data available. Need users with at least 2 interactions.")
        return
    
    # Step 4: Prepare ground truth (exclude items from training set)
    print("\n📊 Step 4: Preparing ground truth...")
    ground_truth = prepare_ground_truth(test_df, train_df=train_df, min_rating=0.5)
    
    if len(ground_truth) == 0:
        print("❌ No ground truth available. Need interactions with rating >= 0.5.")
        return
    
    # Step 5: Train model on training set
    print("\n📊 Step 5: Training model on training set...")
    model, user_item_matrix, sparse_matrix, content_metadata = train_model(
        train_df, df_shows, df_episodes
    )
    
    if model is None:
        print("❌ Model training failed.")
        return
    
    # Step 6: Get user IDs to evaluate
    user_ids_to_evaluate = list(ground_truth.keys())
    user_ids_to_evaluate = [uid for uid in user_ids_to_evaluate if uid in user_item_matrix.index]
    
    if len(user_ids_to_evaluate) == 0:
        print("❌ No users to evaluate.")
        return
    
    print(f"\n✅ Evaluating {len(user_ids_to_evaluate)} users...")
    
    # Step 7: Generate baseline recommendations
    print("\n📊 Step 7: Generating baseline (type/format only) recommendations...")
    baseline_recs = generate_baseline_recommendations(
        model, user_item_matrix, sparse_matrix, user_ids_to_evaluate,
        content_metadata, top_n=20
    )
    
    # Step 8: Generate enhanced recommendations
    print("\n📊 Step 8: Generating enhanced (6D multi-dimensional) recommendations...")
    enhanced_recs = generate_enhanced_recommendations(
        model, user_item_matrix, sparse_matrix, user_ids_to_evaluate,
        content_metadata, top_n=20
    )
    
    # Step 9: Diagnostic - Check recommendation quality
    print("\n📊 Step 9: Diagnostic analysis...")
    print(f"   Ground truth items per user:")
    for user_id, items in list(ground_truth.items())[:5]:
        print(f"     {user_id}: {len(items)} items - {items[:3]}")
    
    print(f"\n   Sample recommendations:")
    for user_id in list(user_ids_to_evaluate)[:3]:
        if user_id in enhanced_recs:
            print(f"     {user_id}: {enhanced_recs[user_id][:5]}")
    
    # Check overlap
    print(f"\n   Checking overlaps...")
    total_overlaps = 0
    for user_id in user_ids_to_evaluate:
        if user_id in enhanced_recs and user_id in ground_truth:
            rec_set = set(enhanced_recs[user_id])
            gt_set = set(ground_truth[user_id])
            overlap = rec_set & gt_set
            if len(overlap) > 0:
                total_overlaps += 1
                print(f"     {user_id}: {len(overlap)} matches - {list(overlap)[:3]}")
    
    print(f"   Users with matches: {total_overlaps}/{len(user_ids_to_evaluate)}")
    
    # Step 10: Compare models
    print("\n📊 Step 10: Comparing models...")
    all_items = list(user_item_matrix.columns)
    comparison_df = compare_models(
        baseline_recs, enhanced_recs, ground_truth, all_items, k_values=[5, 10, 20]
    )
    
    # Step 11: Print results
    print("\n" + "="*80)
    print("📈 EVALUATION RESULTS")
    print("="*80)
    print_comparison_report(comparison_df)
    
    # Step 12: Save results
    output_file = "evaluation_results.csv"
    comparison_df.to_csv(output_file, index=False)
    print(f"\n💾 Results saved to: {output_file}")
    
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()

