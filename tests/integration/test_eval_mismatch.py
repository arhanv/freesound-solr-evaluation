import pytest
import numpy as np

from eval.evaluate_similarity_search import evaluate_similarity_search, load_target_sounds

def test_evaluation_space_size_mismatch(solr_test_collection, solr_test_client):
    """
    Test scenario where the source space has more vectors than the PCA target space.
    If the IDs are not correctly aligned between ground_truth and target, 
    the evaluation math will silently fail by comparing wrong vectors.
    """
    # 1. Setup Data
    # 3 Real sounds: 10, 20, 30
    # Space "large_source": has all 3 vectors
    # Space "small_target": only has vectors for 10 and 30 (20 is missing)
    docs = [
        {
            "id": "10", "content_type": "s",
            "_childDocuments_": [
                {"id": "10_large", "content_type": "v", "similarity_space": "large_source", "sim_vector128_l2": [0.1]*128},
                {"id": "10_small", "content_type": "v", "similarity_space": "small_target", "sim_vector128_l2": [0.1]*128}
            ]
        },
        {
            "id": "20", "content_type": "s",
            "_childDocuments_": [
                {"id": "20_large", "content_type": "v", "similarity_space": "large_source", "sim_vector128_l2": [0.2]*128}
            ]
        },
        {
            "id": "30", "content_type": "s",
            "_childDocuments_": [
                {"id": "30_large", "content_type": "v", "similarity_space": "large_source", "sim_vector128_l2": [0.3]*128},
                {"id": "30_small", "content_type": "v", "similarity_space": "small_target", "sim_vector128_l2": [0.3]*128}
            ]
        }
    ]
    solr_test_client.add(docs, commitWithin=1000)
    solr_test_client.commit()
    
    # 2. Extract Ground Truth
    gt_vecs, gt_target_ids = load_target_sounds("large_source", num_sounds=3, solr_url=solr_test_collection)
    
    # Run eval to get actual ground truth nearest neighbors (using self-space just to get a dummy list)
    gt_result = evaluate_similarity_search(
        gt_vecs,
        gt_target_ids,
        similarity_space="large_source",
        retrieve_n=2,
        metric_k=2,
        solr_url=solr_test_collection,
        dashboard=False
    )
    ground_truth = gt_result.retrieved_neighbors
    # For ID 10 -> [10, ...], ID 20 -> [20, ...], ID 30 -> [30, ...]
    
    # 3. Load Target Space (with the mismatch)
    # The script passes gt_target_ids = [10, 20, 30] to load_target_sounds
    target_vecs, target_ids = load_target_sounds(
        "small_target",
        num_sounds=3,
        specific_sound_ids=gt_target_ids,
        solr_url=solr_test_collection
    )
    
    # Notice: target_ids will be [10, 30] because 20 is missing in Solr.
    # Notice: ground_truth is still length 3.
    
    # 4. Run Evaluation
    # If there's an alignment bug in evaluate_similarity_search, query i=1 (which is ID 30 in target_ids)
    # will be compared against ground_truth[1] (which is for ID 20).
    result = evaluate_similarity_search(
        target_vecs,
        target_ids,
        ground_truth_results=ground_truth,
        similarity_space="small_target",
        retrieve_n=2,
        metric_k=2,
        solr_url=solr_test_collection,
        dashboard=False
    )
    
    # For an alignment bug, we'd have to inspect the internal metric calculations, 
    # but initially, let's just assert the function doesn't crash 
    # and maybe expect it to be broken (so we can fix it later).
    assert result is not None
    assert len(result.query_times_ms) == 2
    print(result.metrics)
