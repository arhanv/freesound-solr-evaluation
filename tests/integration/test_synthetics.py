import pytest
import numpy as np
import pysolr
from unittest.mock import patch

from search.generate_and_index_synthetics import generate_and_index, get_synthetic_count
from search.gm_model import GMModel
from search.index_to_solr import SolrIndexer
from search.configs import SOLR_URL

# Path to mock instead of the real SOLR_URL
@pytest.fixture
def mock_solr_url(solr_test_collection):
    """Overrides the global SOLR_URL in generation script for integration tests."""
    with patch('search.generate_and_index_synthetics.SOLR_URL', solr_test_collection):
        yield solr_test_collection

def test_generate_and_index_synthetics(mock_solr_url, solr_test_client):
    """
    Integration test:
    Verifies that calling generate_and_index actually creates documents 
    in the Solr index with the expected parent/child structure and tags.
    """
    # 1. Setup mock GMModel
    class MockGMModel:
        def sample(self, n):
            # Return n dummy 10-dimensional vectors
            return np.random.rand(n, 10).astype(np.float32)
            
    mock_gmm = MockGMModel()
    num_to_generate = 5
    source_space = "test_space"
    
    # 2. Execute
    generate_and_index(source_space, num_to_generate, mock_gmm, solr_url=mock_solr_url, batch_size=2)
    
    # 3. Assert Count using the existing utility
    indexer = SolrIndexer(mock_solr_url)
    synthetic_count = get_synthetic_count(indexer)
    assert synthetic_count == num_to_generate, f"Expected {num_to_generate} parents, found {synthetic_count}"
    
    def get_field(doc, field):
        val = doc.get(field)
        return val[0] if isinstance(val, list) else val

    # 4. Assert Structure (Parents)
    parents = solr_test_client.search('is_synthetic:true', rows=10).docs
    assert len(parents) == num_to_generate
    for d in parents:
        assert get_field(d, 'content_type') == 's', f"Expected 's', got {d.get('content_type')} in doc {d}"
    
    # 5. Assert Structure (Children)
    # The target space name is source_space + "_synthetic"
    expected_target_space = f"{source_space}_synthetic"
    children = solr_test_client.search(f'similarity_space:{expected_target_space}', rows=10).docs
    assert len(children) == num_to_generate
    for d in children:
        assert get_field(d, 'content_type') == 'v', f"Expected 'v', got {d.get('content_type')} in doc {d}"
    
    # Verify the vector field exists in the child docs
    # Our mock generated 10-d vectors
    vector_field = "sim_vector10_l2" 
    assert all(vector_field in d for d in children), f"Expected field {vector_field} in all children"
