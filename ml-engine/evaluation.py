"""
Evaluation Framework for Recommendation System

Measures:
- Precision@K: Fraction of recommended items that are relevant
- Recall@K: Fraction of relevant items that are recommended
- NDCG@K: Quality of ranking considering position (Normalized Discounted Cumulative Gain)
- Coverage: Fraction of items that can be recommended

Compares:
- Baseline (old: type/format only) vs Enhanced (new: 6D multi-dimensional model)
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


def precision_at_k(recommended: List[str], relevant: List[str], k: int) -> float:
    """
    Precision@K: Fraction of top-K recommendations that are relevant
    
    Args:
        recommended: List of recommended item slugs
        relevant: List of relevant item slugs (ground truth)
        k: Number of top recommendations to consider
    
    Returns:
        Precision@K score (0-1)
    """
    if k == 0 or len(recommended) == 0:
        return 0.0
    
    top_k = recommended[:k]
    relevant_set = set(relevant)
    
    # Count how many recommended items are relevant
    relevant_recommended = sum(1 for item in top_k if item in relevant_set)
    
    return relevant_recommended / min(k, len(recommended))


def recall_at_k(recommended: List[str], relevant: List[str], k: int) -> float:
    """
    Recall@K: Fraction of relevant items that are in top-K recommendations
    
    Args:
        recommended: List of recommended item slugs
        relevant: List of relevant item slugs (ground truth)
        k: Number of top recommendations to consider
    
    Returns:
        Recall@K score (0-1)
    """
    if len(relevant) == 0:
        return 0.0
    
    top_k = recommended[:k]
    relevant_set = set(relevant)
    top_k_set = set(top_k)
    
    # Count how many relevant items are in recommendations
    relevant_found = len(relevant_set & top_k_set)
    
    return relevant_found / len(relevant) if len(relevant) > 0 else 0.0


def dcg_at_k(recommended: List[str], relevant: List[str], k: int) -> float:
    """
    Discounted Cumulative Gain at K
    Relevance score decreases logarithmically with position
    
    Args:
        recommended: List of recommended item slugs
        relevant: List of relevant item slugs (ground truth)
        k: Number of top recommendations to consider
    
    Returns:
        DCG@K score
    """
    if k == 0 or len(recommended) == 0:
        return 0.0
    
    relevant_set = set(relevant)
    dcg = 0.0
    
    for i, item in enumerate(recommended[:k], start=1):
        if item in relevant_set:
            # Relevance = 1 if item is relevant, 0 otherwise
            # Position discount = log2(i+1)
            relevance = 1.0
            dcg += relevance / np.log2(i + 1)
    
    return dcg


def ndcg_at_k(recommended: List[str], relevant: List[str], k: int) -> float:
    """
    Normalized Discounted Cumulative Gain at K
    Normalized version of DCG (divides by ideal DCG)
    
    Args:
        recommended: List of recommended item slugs
        relevant: List of relevant item slugs (ground truth)
        k: Number of top recommendations to consider
    
    Returns:
        NDCG@K score (0-1)
    """
    dcg = dcg_at_k(recommended, relevant, k)
    
    if len(relevant) == 0:
        return 0.0
    
    # Ideal DCG: all relevant items at the top, sorted by relevance
    ideal_order = relevant[:k]
    ideal_dcg = dcg_at_k(ideal_order, relevant, k)
    
    if ideal_dcg == 0:
        return 0.0
    
    return dcg / ideal_dcg


def coverage(recommended_lists: List[List[str]], all_items: List[str]) -> float:
    """
    Coverage: Fraction of all items that appear in at least one recommendation list
    
    Args:
        recommended_lists: List of recommendation lists (one per user)
        all_items: List of all available items
    
    Returns:
        Coverage score (0-1)
    """
    if len(all_items) == 0:
        return 0.0
    
    # Find all items that were recommended at least once
    recommended_items = set()
    for rec_list in recommended_lists:
        recommended_items.update(rec_list)
    
    coverage_score = len(recommended_items) / len(all_items)
    return coverage_score


def evaluate_recommendations(
    recommendations: Dict[str, List[str]],  # user_id -> list of recommended slugs
    ground_truth: Dict[str, List[str]],      # user_id -> list of relevant slugs
    all_items: List[str],                    # All available items
    k_values: List[int] = [5, 10, 20]
) -> Dict[str, Dict[str, float]]:
    """
    Comprehensive evaluation of recommendations
    
    Args:
        recommendations: Dict mapping user_id to recommended item slugs
        ground_truth: Dict mapping user_id to relevant item slugs (test set)
        all_items: List of all available items for coverage calculation
        k_values: List of K values to evaluate (e.g., [5, 10, 20])
    
    Returns:
        Dictionary with metrics for each K value
    """
    results = {}
    
    # Evaluate for each K
    for k in k_values:
        metrics = {
            'precision': [],
            'recall': [],
            'ndcg': []
        }
        
        # Evaluate each user
        users_with_matches = 0
        for user_id in recommendations.keys():
            if user_id not in ground_truth:
                continue
            
            rec = recommendations[user_id]
            rel = ground_truth[user_id]
            
            if len(rec) == 0:
                logging.debug(f"⚠️ No recommendations for user {user_id}")
                continue
            
            if len(rel) == 0:
                logging.debug(f"⚠️ No ground truth for user {user_id}")
                continue
            
            # Check for matches
            rec_set = set(rec)
            rel_set = set(rel)
            matches = rec_set & rel_set
            
            if len(matches) > 0:
                users_with_matches += 1
                logging.debug(f"✅ User {user_id}: {len(matches)} matches out of {len(rel)} relevant items")
                logging.debug(f"   Recommended: {rec[:min(5, len(rec))]}")
                logging.debug(f"   Relevant: {rel[:min(5, len(rel))]}")
                logging.debug(f"   Matches: {list(matches)[:5]}")
            
            metrics['precision'].append(precision_at_k(rec, rel, k))
            metrics['recall'].append(recall_at_k(rec, rel, k))
            metrics['ndcg'].append(ndcg_at_k(rec, rel, k))
        
        logging.info(f"📊 K={k}: {users_with_matches}/{len(metrics['precision'])} users had matching recommendations")
        
        # Average across all users
        results[f'k={k}'] = {
            'precision': np.mean(metrics['precision']) if metrics['precision'] else 0.0,
            'recall': np.mean(metrics['recall']) if metrics['recall'] else 0.0,
            'ndcg': np.mean(metrics['ndcg']) if metrics['ndcg'] else 0.0,
            'users_evaluated': len(metrics['precision'])
        }
    
    # Calculate coverage
    all_rec_lists = list(recommendations.values())
    coverage_score = coverage(all_rec_lists, all_items)
    results['coverage'] = coverage_score
    
    return results


def split_interactions(
    df_interactions: pd.DataFrame,
    test_ratio: float = 0.2,
    min_interactions: int = 2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split interactions into train and test sets
    
    Args:
        df_interactions: DataFrame with userId, contentSlug, rating
        test_ratio: Fraction of interactions to use for testing
        min_interactions: Minimum interactions per user to include in test set
    
    Returns:
        (train_df, test_df) tuple
    """
    # Only include users with enough interactions
    user_counts = df_interactions['userId'].value_counts()
    valid_users = user_counts[user_counts >= min_interactions].index
    
    df_filtered = df_interactions[df_interactions['userId'].isin(valid_users)].copy()
    
    # Group by user and split
    train_list = []
    test_list = []
    
    for user_id, user_df in df_filtered.groupby('userId'):
        user_df = user_df.sample(frac=1, random_state=42).reset_index(drop=True)
        n_test = max(1, int(len(user_df) * test_ratio))
        
        test_df = user_df.head(n_test)
        train_df = user_df.tail(len(user_df) - n_test)
        
        train_list.append(train_df)
        test_list.append(test_df)
    
    train_df = pd.concat(train_list, ignore_index=True)
    test_df = pd.concat(test_list, ignore_index=True)
    
    logging.info(f"📊 Split interactions: Train={len(train_df)}, Test={len(test_df)}")
    logging.info(f"   Users in train: {train_df['userId'].nunique()}")
    logging.info(f"   Users in test: {test_df['userId'].nunique()}")
    
    return train_df, test_df


def prepare_ground_truth(
    test_df: pd.DataFrame, 
    train_df: pd.DataFrame = None,
    min_rating: float = 0.5
) -> Dict[str, List[str]]:
    """
    Prepare ground truth from test interactions
    Only include items with rating >= min_rating as relevant
    Exclude items that user already interacted with in training set
    
    Args:
        test_df: Test interactions DataFrame
        train_df: Training interactions DataFrame (to filter out already seen items)
        min_rating: Minimum rating to consider item relevant
    
    Returns:
        Dict mapping user_id to list of relevant content slugs
    """
    ground_truth = defaultdict(list)
    
    # Get training interactions per user to exclude them from ground truth
    train_interactions = defaultdict(set)
    if train_df is not None and not train_df.empty:
        for _, row in train_df.iterrows():
            user_id = row['userId']
            content_slug = row['contentSlug']
            train_interactions[user_id].add(content_slug)
    
    # Build ground truth from test set
    for _, row in test_df.iterrows():
        user_id = row['userId']
        content_slug = row['contentSlug']
        rating = row['rating']
        
        # Only include if:
        # 1. Rating is high enough
        # 2. User didn't interact with this item in training set (cold-start scenario)
        if rating >= min_rating:
            if user_id not in train_interactions or content_slug not in train_interactions[user_id]:
                ground_truth[user_id].append(content_slug)
            else:
                logging.debug(f"Excluding {content_slug} from ground truth for {user_id} (already in train)")
    
    logging.info(f"📊 Ground truth: {len(ground_truth)} users with relevant items")
    
    # Log some stats
    total_items = sum(len(items) for items in ground_truth.values())
    logging.info(f"   Total relevant items: {total_items}")
    
    return dict(ground_truth)


def compare_models(
    baseline_recommendations: Dict[str, List[str]],
    enhanced_recommendations: Dict[str, List[str]],
    ground_truth: Dict[str, List[str]],
    all_items: List[str],
    k_values: List[int] = [5, 10, 20]
) -> pd.DataFrame:
    """
    Compare baseline vs enhanced model performance
    
    Args:
        baseline_recommendations: Recommendations from old model (type/format only)
        enhanced_recommendations: Recommendations from new model (6D multi-dimensional)
        ground_truth: Test set ground truth
        all_items: All available items
        k_values: K values to evaluate
    
    Returns:
        DataFrame with comparison results
    """
    # Evaluate baseline
    baseline_results = evaluate_recommendations(
        baseline_recommendations, ground_truth, all_items, k_values
    )
    
    # Evaluate enhanced
    enhanced_results = evaluate_recommendations(
        enhanced_recommendations, ground_truth, all_items, k_values
    )
    
    # Create comparison DataFrame
    comparison_data = []
    
    for k in k_values:
        k_key = f'k={k}'
        comparison_data.append({
            'Metric': 'Precision@K',
            'K': k,
            'Baseline': baseline_results[k_key]['precision'],
            'Enhanced': enhanced_results[k_key]['precision'],
            'Improvement': enhanced_results[k_key]['precision'] - baseline_results[k_key]['precision'],
            'Improvement %': ((enhanced_results[k_key]['precision'] - baseline_results[k_key]['precision']) 
                            / baseline_results[k_key]['precision'] * 100) if baseline_results[k_key]['precision'] > 0 else 0
        })
        
        comparison_data.append({
            'Metric': 'Recall@K',
            'K': k,
            'Baseline': baseline_results[k_key]['recall'],
            'Enhanced': enhanced_results[k_key]['recall'],
            'Improvement': enhanced_results[k_key]['recall'] - baseline_results[k_key]['recall'],
            'Improvement %': ((enhanced_results[k_key]['recall'] - baseline_results[k_key]['recall']) 
                            / baseline_results[k_key]['recall'] * 100) if baseline_results[k_key]['recall'] > 0 else 0
        })
        
        comparison_data.append({
            'Metric': 'NDCG@K',
            'K': k,
            'Baseline': baseline_results[k_key]['ndcg'],
            'Enhanced': enhanced_results[k_key]['ndcg'],
            'Improvement': enhanced_results[k_key]['ndcg'] - baseline_results[k_key]['ndcg'],
            'Improvement %': ((enhanced_results[k_key]['ndcg'] - baseline_results[k_key]['ndcg']) 
                            / baseline_results[k_key]['ndcg'] * 100) if baseline_results[k_key]['ndcg'] > 0 else 0
        })
    
    comparison_data.append({
        'Metric': 'Coverage',
        'K': None,
        'Baseline': baseline_results['coverage'],
        'Enhanced': enhanced_results['coverage'],
        'Improvement': enhanced_results['coverage'] - baseline_results['coverage'],
        'Improvement %': ((enhanced_results['coverage'] - baseline_results['coverage']) 
                        / baseline_results['coverage'] * 100) if baseline_results['coverage'] > 0 else 0
    })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    return comparison_df


def print_evaluation_report(results: Dict[str, Dict[str, float]], model_name: str = "Model"):
    """
    Print formatted evaluation report
    
    Args:
        results: Results dictionary from evaluate_recommendations
        model_name: Name of the model being evaluated
    """
    print(f"\n{'='*60}")
    print(f"📊 Evaluation Report: {model_name}")
    print(f"{'='*60}")
    
    for key, metrics in results.items():
        if key == 'coverage':
            print(f"\n📈 Coverage: {metrics:.4f}")
        else:
            print(f"\n{key}:")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  NDCG:      {metrics['ndcg']:.4f}")
            print(f"  Users:     {metrics['users_evaluated']}")


def print_comparison_report(comparison_df: pd.DataFrame):
    """
    Print formatted comparison report
    
    Args:
        comparison_df: Comparison DataFrame from compare_models
    """
    print(f"\n{'='*80}")
    print(f"📊 Model Comparison: Baseline vs Enhanced")
    print(f"{'='*80}")
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    
    print("\n" + comparison_df.to_string(index=False))
    
    print(f"\n{'='*80}")
    print("✅ Summary:")
    
    for metric in ['Precision@K', 'Recall@K', 'NDCG@K']:
        metric_df = comparison_df[comparison_df['Metric'] == metric]
        avg_improvement = metric_df['Improvement %'].mean()
        print(f"  {metric}: Average improvement: {avg_improvement:.2f}%")
    
    coverage_row = comparison_df[comparison_df['Metric'] == 'Coverage'].iloc[0]
    print(f"  Coverage: Improvement: {coverage_row['Improvement %']:.2f}%")
    
    print(f"{'='*80}\n")

