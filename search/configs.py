import os

# Base directory (freesound-solr-evaluation/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SEARCH_DOCUMENTS_DIR = "/Volumes/T7 Shield/Datasets/search_documents_export_100k"
SOLR_BASE_URL = "http://localhost:8983"
COLLECTION_NAME = "freesound"

# Absolute paths
SEARCH_SCHEMA_PATH = os.path.join(BASE_DIR, "schema", "freesound.json")
DELETE_DEFAULT_FIELDS_SCHEMA_PATH = os.path.join(BASE_DIR, "schema", "delete_default_fields.json")

SOLR_URL = SOLR_BASE_URL + "/solr/" + COLLECTION_NAME
BATCH_SIZE = 1000