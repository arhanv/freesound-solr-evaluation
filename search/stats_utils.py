import requests
from pysolr import Solr
from datetime import datetime
from search.configs import SOLR_URL, SOLR_BASE_URL, COLLECTION_NAME

def get_solr_health():
    """Returns basic health metrics for the Solr core/collection."""
    try:
        # 1. Ping test
        ping_url = f"{SOLR_URL}/admin/ping"
        resp = requests.get(ping_url, params={'wt': 'json'}, timeout=2)
        resp.raise_for_status()
        ping_data = resp.json()
        ping_status = str(ping_data.get('status', 'error')).upper()
        
        # Normalize status (case-insensitive check for Solr "OK")
        status = "ONLINE" if ping_status == "OK" else "UNREACHABLE"
        
        # 2. Core status (Size, Version)
        admin_url = f"{SOLR_BASE_URL}/solr/admin/cores"
        resp = requests.get(admin_url, params={'action': 'STATUS', 'wt': 'json'}, timeout=5)
        resp.raise_for_status()
        cores = resp.json().get('status', {})
        
        # Robust lookup: Find core that exactly matches OR starts with the collection name
        # In cloud mode, collection 'freesound' might be core 'freesound_shard1_replica_n1'
        core_data = cores.get(COLLECTION_NAME)
        if not core_data:
            for name, data in cores.items():
                if name.startswith(COLLECTION_NAME):
                    core_data = data
                    break
        
        if not core_data:
            # Fallback: take the first available core if only one exists
            if len(cores) == 1:
                core_data = list(cores.values())[0]

        index_size_bytes = 0
        num_docs = 0
        last_modified = "Unknown"
        current_core = COLLECTION_NAME

        if core_data:
            index_size_bytes = core_data.get('index', {}).get('sizeInBytes', 0)
            num_docs = core_data.get('index', {}).get('numDocs', 0)
            last_modified = core_data.get('index', {}).get('lastModified', 'Unknown')
            current_core = core_data.get('name', COLLECTION_NAME)
        
        result = {
            'status': status,
            'collection': current_core,
            'num_docs': num_docs,
            'size_mb': round(index_size_bytes / (1024 * 1024), 2),
            'last_update': last_modified,
            'refresh_time': datetime.now().strftime("%H:%M:%S")
        }
        if status == "UNREACHABLE":
            result['error'] = f"Solr responded with status: {ping_status}"
            
        return result
    except requests.exceptions.ConnectionError:
        err_msg = 'Connection refused (Docker down?)'
        return {
            'status': 'DOWN', 
            'error': err_msg,
            'refresh_time': datetime.now().strftime("%H:%M:%S")
        }
    except requests.exceptions.Timeout:
        err_msg = 'Request timed out (Solr slow or hanging?)'
        return {
            'status': 'UNREACHABLE', 
            'error': err_msg,
            'refresh_time': datetime.now().strftime("%H:%M:%S")
        }
    except Exception as e:
        err_msg = str(e)
        return {
            'status': 'UNREACHABLE', 
            'error': err_msg,
            'refresh_time': datetime.now().strftime("%H:%M:%S")
        }

def get_content_distribution():
    """Returns counts for Real vs Synthetic parent sounds."""
    solr = Solr(SOLR_URL)
    try:
        # Search for parent documents by content_type:s
        total_parents = solr.search('content_type:s', rows=0).hits
        
        real_count = 0
        synthetic_count = 0
        
        try:
            # Real sounds: parent docs where is_synthetic is NOT true
            real_count = solr.search('content_type:s AND -is_synthetic:true', rows=0).hits
            # Synthetic sounds: parent docs marked is_synthetic:true
            synthetic_count = solr.search('content_type:s AND is_synthetic:true', rows=0).hits
        except Exception as e:
            # Catch "undefined field is_synthetic" 400 errors
            if "undefined field" in str(e).lower():
                real_count = total_parents
                synthetic_count = 0
            else:
                raise e
            
        return {
            'real': real_count,
            'synthetic': synthetic_count,
            'total_parents': total_parents,
            'descriptions': {
                'optimize': "Merges Lucene segments on disk to reclaim space and improve search speed. Crucial after large deletions.",
                'refresh': "Fetches the latest statistics and health status from the Solr cluster.",
                'purge': "Permanently deletes all synthetic data documents and triggers an index optimization."
            }
        }
    except Exception as e:
        print(f"Error in distribution check: {e}")
        return {'real': 0, 'synthetic': 0, 'total_parents': 0}

def get_similarity_spaces():
    """Groups documents by similarity space and returns counts/dimensions."""
    solr = Solr(SOLR_URL)
    try:
        params = {'facet': 'on', 'facet.field': 'similarity_space', 'rows': 0}
        resp = solr.search('content_type:v', **params)
        facets = resp.facets.get('facet_fields', {}).get('similarity_space', [])
        
        spaces = []
        for i in range(0, len(facets), 2):
            name = facets[i]
            count = facets[i+1]
            
            dim = "Unknown"
            est_size_mb = 0.0
            sample = solr.search(f'similarity_space:{name}', rows=1).docs
            if sample:
                doc = sample[0]
                vec_keys = [k for k in doc.keys() if k.startswith('sim_vector')]
                if vec_keys:
                    dim_str = vec_keys[0].replace('sim_vector', '').split('_')[0]
                    dim = int(dim_str)
                    est_size_mb = calculate_space_size_mb(count, dim)
                    
            spaces.append({
                'name': name,
                'count': count,
                'dimension': dim,
                'size_mb': est_size_mb,
                'avg_size_kb': round((est_size_mb * 1024) / count, 2) if count > 0 else 0,
                'type': 'Synthetic' if 'synthetic' in name else 'Real'
            })
        return spaces
    except Exception as e:
        print(f"Error fetching spaces: {e}")
        return []

def calculate_space_size_mb(count, dim):
    """Calculates estimated size of a vector space in MB."""
    if not isinstance(dim, int) or count <= 0:
        return 0.0
    # Lucene HNSW overhead is approx 2x raw vector size (4 bytes/float)
    bytes_per_doc = dim * 4 * 2.0 
    return round((count * bytes_per_doc) / (1024 * 1024), 2)
