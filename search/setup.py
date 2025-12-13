import json
from solrapi import SolrManagementAPI
import configs

schema = json.load(open(configs.SEARCH_SCHEMA_PATH))
delete_defaults = json.load(open(configs.DELETE_DEFAULT_FIELDS_SCHEMA_PATH))

api = SolrManagementAPI(configs.SOLR_BASE_URL, configs.COLLECTION_NAME, logging_verbose=1)
api.create_collection_and_schema(delete_defaults, schema, test_field_name='id')