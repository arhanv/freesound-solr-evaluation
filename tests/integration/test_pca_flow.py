import pytest
import numpy as np

from search.pca import load_vectors_from_solr, add_pca_child_documents
from search.generate_and_index_synthetics import generate_and_index
from search.index_to_solr import SolrIndexer

def test_pca_read_write_flow(solr_test_collection, solr_test_client):
    """
    Integration test:
    1. Generates 5 dummy documents in the test collection.
    2. Reads them back using `load_vectors_from_solr`.
    3. Simulates a PCA dimension reduction.
    4. Writes them back using `add_pca_child_documents`.
    5. Verifies the new specific child documents exist.
    """
    # 1. Setup initial data
    class MockGMModel:
        def sample(self, n):
            return np.random.rand(n, 10).astype(np.float32)
            
    source_space = "pca_src"
    generate_and_index(source_space, 5, MockGMModel(), solr_url=solr_test_collection, batch_size=5)
    
    # 2. Test Reading Vectors
    source_target = f"{source_space}_synthetic"
    vectors, sound_ids, child_docs = load_vectors_from_solr(source_target, solr_url=solr_test_collection)
    
    assert len(vectors) == 5
    assert len(sound_ids) == 5
    assert len(child_docs) == 5
    assert vectors.shape == (5, 10)
    
    # 3. Test Writing "Reduced" Vectors
    target_space = "pca_test_2d"
    reduced_vectors = np.random.rand(5, 2).astype(np.float32)
    
    add_pca_child_documents(sound_ids, reduced_vectors, child_docs, target_space, solr_url=solr_test_collection, batch_size=2)
    
    # SolrIndexer commitments are used internally, but `add_pca_child_documents` uses pysolr directly and relies on commitWithin=10000.
    # Force hard commit for testing:
    solr_test_client.commit()
    
    # 4. Assert new documents
    new_children = solr_test_client.search(f'similarity_space:{target_space}', rows=10).docs
    assert len(new_children) == 5
    
    def get_field(doc, field):
        val = doc.get(field)
        return val[0] if isinstance(val, list) else val
        
    for doc in new_children:
        assert get_field(doc, 'content_type') == 'v'
        assert "sim_vector2_l2" in doc
        # Verify the parent ID link works 
        root_val = get_field(doc, '_root_')
        assert root_val is not None
        assert int(root_val) in sound_ids
