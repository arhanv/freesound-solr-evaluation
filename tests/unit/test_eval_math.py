import pytest
from eval.evaluate_similarity_search import calculate_ndcg_weighted

def test_calculate_ndcg_weighted_perfect_match():
    """Test NDCG when the retrieved items are in perfect order matching ground truth."""
    retrieved = [10, 20, 30, 40, 50]
    ground_truth = [10, 20, 30, 40, 50]
    k = 5
    
    score = calculate_ndcg_weighted(retrieved, ground_truth, k)
    # A perfect alignment should yield an NDCG of 1.0
    assert score == pytest.approx(1.0)

def test_calculate_ndcg_weighted_complete_miss():
    """Test NDCG when none of the retrieved items are in the ground truth."""
    retrieved = [99, 98, 97, 96, 95]
    ground_truth = [10, 20, 30, 40, 50]
    k = 5
    
    score = calculate_ndcg_weighted(retrieved, ground_truth, k)
    assert score == 0.0

def test_calculate_ndcg_weighted_partial_match_out_of_order():
    """Test NDCG penalizes correct results that appear lower in the ranking."""
    ground_truth = [10, 20, 30, 40, 50]
    
    # 1. We retrieve the #1 result (ID 10) correctly at the top
    retrieved_good = [10, 99, 98, 97, 96]
    score_good = calculate_ndcg_weighted(retrieved_good, ground_truth, k=5)
    
    # 2. We retrieve the #1 result, but it's at the bottom
    retrieved_poor = [99, 98, 97, 96, 10]
    score_poor = calculate_ndcg_weighted(retrieved_poor, ground_truth, k=5)
    
    # Both sets contain one correct ID, but the one ranked higher should score better
    assert score_good > score_poor
    assert score_good > 0.0
    
def test_calculate_ndcg_handles_empty():
    """Test NDCG gracefully handles empty lists."""
    score1 = calculate_ndcg_weighted([], [10, 20], k=2)
    assert score1 == 0.0
    
    score2 = calculate_ndcg_weighted([10, 20], [], k=2)
    assert score2 == 0.0
