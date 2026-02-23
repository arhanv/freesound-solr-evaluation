import pytest
import pysolr
import os
import sys

# Ensure the search module is discoverable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search.configs import SOLR_BASE_URL
from search.solrapi import SolrManagementAPI

TEST_COLLECTION = "freesound_test"
TEST_SOLR_URL = f"{SOLR_BASE_URL}/solr/{TEST_COLLECTION}"

@pytest.fixture(scope="session")
def solr_test_collection():
    """
    Session-scoped fixture to create a temporary Solr collection for testing,
    and destroy it after all tests complete.
    """
    api = SolrManagementAPI(SOLR_BASE_URL, TEST_COLLECTION)
    
    import json
    from search import configs
    
    # Setup: Create collection with proper schema
    try:
        if not api.collection_exists():
            print(f"\nSetting up test collection & schema: {TEST_COLLECTION}")
            
            # Load schema definitions
            schema = json.load(open(configs.SEARCH_SCHEMA_PATH))
            delete_defaults = json.load(open(configs.DELETE_DEFAULT_FIELDS_SCHEMA_PATH))
            
            # Apply schema
            api.create_collection_and_schema(delete_defaults, schema, test_field_name='id')
            # Wait briefly for collection to be fully ready
            import time
            time.sleep(2)
    except Exception as e:
        pytest.fail(f"Failed to setup test collection: {e}")

    yield TEST_SOLR_URL

    # Teardown: Delete collection
    try:
        if os.environ.get("KEEP_TEST_CORE") != "1":
            print(f"\nTearing down test collection: {TEST_COLLECTION}")
            api.delete_collection()
        else:
            print(f"\nKEEP_TEST_CORE=1, skipping teardown of {TEST_COLLECTION}")
    except Exception as e:
        print(f"Warning: Failed to delete test collection: {e}")

@pytest.fixture
def solr_test_client(solr_test_collection):
    """
    Function-scoped fixture returning a connected Solr client to the test collection,
    ensuring the collection is empty before each test.
    """
    client = pysolr.Solr(solr_test_collection, always_commit=True)
    
    # Ensure empty index before test
    client.delete(q="*:*")
    
    return client
