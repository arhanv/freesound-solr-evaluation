SEARCH_DOCUMENTS_DIR = "/Volumes/T7 Shield/Datasets/search_documents_export_100k"
SOLR_BASE_URL = "http://localhost:8983"
COLLECTION_NAME = "freesound"
SEARCH_SCHEMA_PATH = "../schema/freesound.json"
DELETE_DEFAULT_FIELDS_SCHEMA_PATH = "../schema/delete_default_fields.json"
SOLR_URL = SOLR_BASE_URL + "/solr/" + COLLECTION_NAME
BATCH_SIZE = 1000