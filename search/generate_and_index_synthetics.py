import argparse
import os
import sys
import numpy as np
from tqdm import tqdm
import requests
import json

from pca import load_vectors_from_solr
from index_to_solr import SolrIndexer
from configs import SOLR_URL
from gm_model import GMModel

# Starting ID for synthetic data to avoid collision with real Freesound IDs
SYNTHETIC_ID_START = 1_000_000_000 

def generate_and_index(source_space, n_synthetic, gm_model, solr_url=SOLR_URL, batch_size=2000):
    indexer = SolrIndexer(solr_url)
    print(f"Generating {n_synthetic} synthetic embeddings...")
    target_space = f"{source_space}_synthetic"
    # Generate synthetic data in chunks
    current_id = SYNTHETIC_ID_START
    total_batches = (n_synthetic // batch_size) + 1
    
    for _ in tqdm(range(total_batches), desc="Indexing synthetic batches"):
        
        # Sample a batch of vectors from the GMM
        # Determine actual size for last batch
        current_batch_size = min(batch_size, n_synthetic - (current_id - SYNTHETIC_ID_START))
        if current_batch_size <= 0:
            break
            
        vectors = gm_model.sample(current_batch_size)
        
        # Construct Solr documents for indexing (parent-child structure)
        docs_to_index = []
        
        for i, vec in enumerate(vectors):
            parent_id = current_id + i
            
            child_doc = {
                'id': f"{parent_id}_{target_space}",
                'content_type': 'v',
                'similarity_space': target_space,
                f"sim_vector{vec.shape[0]}_l2": vec.tolist()
            }
            
            parent_doc = {
                'id': parent_id,
                'name': f"Synthetic Sound {parent_id}",
                'content_type': 's',
                'is_synthetic': True,  # easy to delete later
                '_childDocuments_': [child_doc]
            }
            
            docs_to_index.append(parent_doc)
            
        # Index to Solr
        indexer.index_documents(docs_to_index, batch_size=batch_size)
        
        current_id += current_batch_size
    
    indexer.commit()
    print(f"Successfully indexed {n_synthetic} synthetic documents.")

def get_synthetic_count(indexer):
    """Counts the number of synthetic documents in an initialized SolrIndexer instance."""
    return indexer.solr.search('is_synthetic:true', rows=0).hits

def cleanup_synthetic(solr_url=SOLR_URL):
    "Deletes synthetic data from Solr."
    indexer = SolrIndexer(solr_url)
    print(f"Current count before cleanup: {get_synthetic_count(indexer)}")
    
    # 1. Delete by the flag (Parents)
    # 2. Delete by the ID range we set for synthetics (1,000,000,000+)
    # 3. Delete by the specific child ID pattern
    delete_query = f'is_synthetic:true OR id:[{int(SYNTHETIC_ID_START)} TO *]'
    
    print(f"Executing deep cleanup with query: {delete_query}")
    indexer.solr.delete(q=delete_query)
    indexer.commit()
    
    # Force Solr to clear the deleted documents from disk immediately
    requests.get(f"{solr_url}/update?optimize=true")
    
    print(f"Count after cleanup: {get_synthetic_count(indexer)}")

def get_index_size_mb(solr_url):
    """
    Returns the total size of all Solr cores in MB.
    """
    try:
        # Extract base URL
        base_url = solr_url.split('/solr')[0] + "/solr"
        admin_url = f"{base_url}/admin/cores"
        
        resp = requests.get(admin_url, params={'action': 'STATUS', 'wt': 'json'}, timeout=5)
        resp.raise_for_status()
        status = resp.json().get('status', {})

        # Sum the size of all cores to handle sharded collections
        total_bytes = sum(
            core.get('index', {}).get('sizeInBytes', 0) 
            for core in status.values()
        )
        
        return float(total_bytes) / (1024 * 1024)
    except Exception as e:
        print(f"Warning: Could not fetch index size ({e})")
        return 0.0

def main():
    parser = argparse.ArgumentParser(description='Generate and Index Synthetic Embeddings via GMM')
    parser.add_argument('--fit', action='store_true', help='Fit a new GMM model')
    parser.add_argument('--find-k', action='store_true', help='Run BIC search to find best K')
    parser.add_argument('--generate', type=int, default=0, help='Number of synthetic vectors to generate')
    parser.add_argument('--cleanup', action='store_true', help='Remove previously generated synthetic data')
    parser.add_argument('--subsample', type=int, default=None, 
                        help='Number of random samples to use for fitting (defaults to all)')
    
    parser.add_argument('--source-space', default='laion_clap', help='Source similarity space')
    parser.add_argument('--model-path', default='models/gmm_laion_clap.pkl', help='Path to save/load GMM')
    parser.add_argument('--k', type=int, default=64, help='Number of components (if not running find-k)')

    # These two flags allow checking current counts for analytics/debugging
    parser.add_argument('--check-count', action='store_true', help='Print current number of synthetic docs')
    parser.add_argument('--check-size', action='store_true', help='Print current index size (MB)')
    
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)

    if args.check_count or args.check_size:
        indexer = SolrIndexer(SOLR_URL)
        if args.check_count:
            print(f"Number of synthetic docs: {get_synthetic_count(indexer)}")
        if args.check_size:
            print(f"Index size: {get_index_size_mb(SOLR_URL):.2f} MB")
        if not (args.fit or args.find_k or args.generate > 0 or args.cleanup):
            return

    if args.cleanup:
        cleanup_synthetic()
        if args.generate == 0 and not args.fit and not args.find_k:
            return

    manager = None

    # Case 1: Training a new model
    if args.fit or args.find_k:
        # initialize new GMM helper
        manager = GMModel(n_components=args.k)
        
        vectors, _, _ = load_vectors_from_solr(args.source_space)

        num_available = len(vectors)

        if args.subsample is not None and args.subsample < num_available:
            print(f"Subsampling {args.subsample} vectors from available {num_available} for fitting...")
            indices = np.random.choice(num_available, size=args.subsample, replace=False)
            vectors = vectors[indices]

        if args.find_k:
            best_k = manager.find_optimal_k(
                vectors, 
                plot_path=args.model_path.replace('.pkl', '_bic_plot.png')
            )
            manager.n_components = best_k
            
        manager.fit(vectors)
        manager.save(args.model_path)
    
    # Case 2: Using an existing model
    elif args.generate > 0:
        if not os.path.exists(args.model_path):
            print(f"Error: Model not found at {args.model_path}. Run with --fit first.")
            return

        manager = GMModel.load(args.model_path)

    # Generate data
    if args.generate > 0 and manager is not None:
        mb_before = get_index_size_mb(SOLR_URL)
        print(f"Index size (before adding new synthetics): {mb_before:.2f} MB")
        generate_and_index(args.source_space, args.generate, manager)
        mb_after = get_index_size_mb(SOLR_URL)
        print(f"Index size (after adding new synthetics): {mb_after:.2f} MB")
        print(f"Approximate size added by synthetics: {mb_after - mb_before:.2f} MB")


if __name__ == "__main__":
    main()