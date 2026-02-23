import json
import os
import sys

# Add project root to sys.path so 'search' package is found when running script directly
if __name__ == "__main__" or __name__.startswith("search"):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search.solrapi import SolrManagementAPI
import search.configs as configs

schema = json.load(open(configs.SEARCH_SCHEMA_PATH))
delete_defaults = json.load(open(configs.DELETE_DEFAULT_FIELDS_SCHEMA_PATH))

api = SolrManagementAPI(configs.SOLR_BASE_URL, configs.COLLECTION_NAME, logging_verbose=1)
api.create_collection_and_schema(delete_defaults, schema, test_field_name='id')