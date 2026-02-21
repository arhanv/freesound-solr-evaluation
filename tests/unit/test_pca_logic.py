import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from search.pca import load_vectors_from_solr, add_pca_child_documents

class MockSolrResults:
    def __init__(self, docs, nextCursorMark=None, hits=None):
        self.docs = docs
        self.nextCursorMark = nextCursorMark or "*"
        self.hits = hits if hits is not None else 100

def test_load_vectors_pagination_logic(mocker):
    """
    Tests that load_vectors_from_solr correctly loops through cursorMarks
    and concatenates the pages into a single dataset.
    """
    # Create 3 pages of dummy responses
    page1 = [
        {"id": "1_v", "similarity_space": "test", "sim_vector8_l2": [0.1]*8, "_root_": "1"},
        {"id": "2_v", "similarity_space": "test", "sim_vector8_l2": [0.2]*8, "_root_": "2"}
    ]
    page2 = [
        {"id": "3_v", "similarity_space": "test", "sim_vector8_l2": [0.3]*8, "_root_": "3"},
        {"id": "4_v", "similarity_space": "test", "sim_vector8_l2": [0.4]*8, "_root_": "4"}
    ]
    # Page 3 is empty, indicating the end
    page3 = []

    m_solr = MagicMock()
    # The first call gets total hits. The second gets sample field. Then the loop starts.
    m_solr.search.side_effect = [
        MockSolrResults([], hits=4),  # Total hits query
        MockSolrResults(page1),       # Sample query
        MockSolrResults(page1, nextCursorMark="page2"),
        MockSolrResults(page2, nextCursorMark="page3"),
        MockSolrResults(page3, nextCursorMark="page3"), # Stop condition
    ]
    
    m_pysolr = mocker.patch("search.pca.pysolr.Solr", return_value=m_solr)
    
    vectors, sound_ids, child_docs = load_vectors_from_solr("test")
    
    assert len(vectors) == 4
    assert sound_ids == [1, 2, 3, 4]
    assert len(child_docs) == 4

def test_add_pca_surgical_replacement(mocker):
    """
    Tests that add_pca_child_documents correctly fetches a parent and its children,
    removes the OLD PCA space of the same name, and injects the new one,
    without destroying other children.
    """
    m_solr = MagicMock()
    m_pysolr = mocker.patch("search.pca.pysolr.Solr", return_value=m_solr)
    
    # 1. Define the input data
    sound_ids = [100]
    reduced_vectors = np.array([[0.5, 0.5]], dtype=np.float32)
    original_child_docs = [{"timestamp_start": 0, "timestamp_end": -1}]
    target_space = "pca_test"
    
    # 2. Mock Solr's responses when the script asks for parents and children
    # It fetches the parent document
    mock_parent = {"id": "100", "content_type": "s", "name": "Fake Sound"}
    
    # It fetches existing children
    # We pretend this parent already has a 'pca_test' child (which should be overwritten)
    # and a 'laion_clap' child (which should be kept!)
    mock_children = [
        {"id": "100_laion_clap", "similarity_space": "laion_clap", "_root_": "100"},
        {"id": "100_pca_test_old", "similarity_space": "pca_test", "_root_": "100"}
    ]
    
    m_solr.search.side_effect = [
        [mock_parent],   # Parent search
        mock_children    # Children search
    ]
    
    # 3. Call the function
    add_pca_child_documents(sound_ids, reduced_vectors, original_child_docs, target_space, batch_size=1)
    
    # 4. Assert how it tried to update Solr
    m_solr.add.assert_called_once()
    
    # Extract the payload that was sent to Solr.add()
    docs_to_index, = m_solr.add.call_args[0]
    payload = docs_to_index[0]
    
    assert payload["id"] == "100"
    
    # Verify the surgical replacement logic
    final_children = payload["_childDocuments_"]
    
    # We should have exactly 2 children: the untouched 'laion_clap' and the newly injected 'pca_test'
    assert len(final_children) == 2
    spaces_in_payload = [c["similarity_space"] for c in final_children]
    
    assert "laion_clap" in spaces_in_payload
    assert "pca_test" in spaces_in_payload
    
    # Verify the old pca_test document was NOT kept (we overwrote it with the new one we just generated)
    new_pca_child = next(c for c in final_children if c["similarity_space"] == "pca_test")
    assert new_pca_child["id"] == "100_pca_test"
    assert "sim_vector2_l2" in new_pca_child
    assert new_pca_child["sim_vector2_l2"] == [0.5, 0.5]
