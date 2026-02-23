import pytest
import numpy as np

from search.generate_and_index_synthetics import (
    generate_and_index, cleanup_synthetic, delete_synthetic_space
)
from search.index_to_solr import SolrIndexer

@pytest.fixture
def mock_gm_model():
    """Creates a simple mock GMModel for testing."""
    class MockModel:
        def sample(self, n):
            return np.random.rand(n, 128).astype(np.float32)
    return MockModel()

def test_granular_cleanup(solr_test_collection, solr_test_client, mock_gm_model):
    """Verifies that deleting one synthetic space doesn't affect another."""
    
    # 1. Generate data for two different spaces
    # Note: generate_and_index appends '_synthetic' to the source_space name it receives.
    generate_and_index("space_a", 10, mock_gm_model, solr_url=solr_test_collection, batch_size=5)
    generate_and_index("space_b", 10, mock_gm_model, solr_url=solr_test_collection, batch_size=5)
    
    solr_test_client.commit()

    # Verify both exist
    hits_a = solr_test_client.search("similarity_space:space_a_synthetic").hits
    hits_b = solr_test_client.search("similarity_space:space_b_synthetic").hits
    assert hits_a == 10
    assert hits_b == 10
    
    # Verify parent documents exist
    parents_a_before = solr_test_client.search("{!parent which='content_type:s'}similarity_space:space_a_synthetic").hits
    parents_b_before = solr_test_client.search("{!parent which='content_type:s'}similarity_space:space_b_synthetic").hits
    assert parents_a_before > 0
    assert parents_b_before > 0

    # 2. Delete space_a_synthetic
    delete_synthetic_space("space_a_synthetic", solr_url=solr_test_collection)
    
    solr_test_client.commit()
    
    # 3. Verify results: A vectors should be gone, B vectors should remain
    hits_a_after = solr_test_client.search("similarity_space:space_a_synthetic").hits
    hits_b_after = solr_test_client.search("similarity_space:space_b_synthetic").hits
    parents_a_after = solr_test_client.search("{!parent which='content_type:s'}similarity_space:space_a_synthetic").hits
    parents_b_after = solr_test_client.search("{!parent which='content_type:s'}similarity_space:space_b_synthetic").hits
    
    assert hits_a_after == 0
    assert hits_b_after == 10
    
    # Parents should still exist even for A, because we only deleted vectors
    total_parents_after = solr_test_client.search("is_synthetic:true").hits
    assert total_parents_after == 20
    
    # 4. Full cleanup
    cleanup_synthetic(solr_url=solr_test_collection)
    solr_test_client.commit()
    assert solr_test_client.search("is_synthetic:true").hits == 0
