import pytest
from search.generate_and_index_synthetics import cleanup_synthetic
from search.pca import delete_pca_vectors
from unittest.mock import patch

def test_synthetic_cleanup_safety(solr_test_collection, solr_test_client):
    """
    Test that cleanup_synthetic only deletes documents with is_synthetic:true
    or similarity_space containing 'synthetic', and leaves real data untouched.
    """
    docs = [
        # Real Parent & Vector
        {
            "id": "real_1", "content_type": "s", "name": "Real Sound",
            "_childDocuments_": [
                {"id": "real_1_laion_clap", "content_type": "v", "similarity_space": "laion_clap", "sim_vector128_l2": [0.1]*128}
            ]
        },
        # Synthetic Parent & Vector
        {
            "id": "1000000000", "content_type": "s", "is_synthetic": True, "name": "Synth Parent",
            "_childDocuments_": [
                {"id": "1000000000_laion_clap_synthetic", "content_type": "v", "similarity_space": "laion_clap_synthetic", "sim_vector128_l2": [0.2]*128}
            ]
        }
    ]
    solr_test_client.add(docs, commitWithin=1000)
    solr_test_client.commit()
    
    # Verify setup
    assert solr_test_client.search("*:*").hits == 4

    # 2. Run Cleanup
    cleanup_synthetic(solr_url=solr_test_collection)
    
    # 3. Assert Real Data Survives
    real_parent = solr_test_client.search("id:real_1").docs
    assert len(real_parent) == 1
    
    real_vector = solr_test_client.search("id:real_1_laion_clap").docs
    assert len(real_vector) == 1
    
    # 4. Assert Synthetic Data is Gone
    assert solr_test_client.search("id:1000000000").hits == 0
    assert solr_test_client.search("id:1000000000_laion_clap_synthetic").hits == 0

def test_synthetic_cleanup_query_string_safety():
    """
    Specifically test that the cleanup query string doesn't contain dangerous 
    number ranges that might catch IDs starting with 2 or 9.
    (Preserved from original tests/test_cleanup_safety.py)
    """
    from search.generate_and_index_synthetics import SYNTHETIC_CLEANUP_QUERY
    assert '[' not in SYNTHETIC_CLEANUP_QUERY, "The cleanup query contains a dangerous range '[...]'"
    assert 'TO' not in SYNTHETIC_CLEANUP_QUERY, "The cleanup query contains a dangerous 'TO' range"

def test_pca_cleanup_safety(solr_test_collection, solr_test_client):
    """
    Test that delete_pca_vectors only deletes vectors from the specific
    target space, leaving the parent document and other vector spaces untouched.
    """
    # 1. Setup Data
    docs = [
        {
            "id": "real_2", "content_type": "s", "name": "Another Real Sound",
            "_childDocuments_": [
                {"id": "real_2_laion_clap", "content_type": "v", "similarity_space": "laion_clap", "sim_vector128_l2": [0.1]*128},
                {"id": "real_2_target_pca", "content_type": "v", "similarity_space": "target_pca", "sim_vector128_l2": [0.2]*128}
            ]
        }
    ]
    solr_test_client.add(docs, commitWithin=1000)
    solr_test_client.commit()
    
    # 2. Run Cleanup (Patch input to auto-confirm 'y')
    with patch('builtins.input', return_value='y'):
        delete_pca_vectors("target_pca", solr_url=solr_test_collection)
        
    # 3. Assert Parent Survives
    assert solr_test_client.search("id:real_2").hits == 1
    
    # 4. Assert Sister Vector Space Survives
    assert solr_test_client.search("id:real_2_laion_clap").hits == 1
    
    # 5. Assert Target PCA Space is Gone
    assert solr_test_client.search("id:real_2_target_pca").hits == 0
