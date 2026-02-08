"""
Generate and index synthetic sound embeddings for Solr scalability testing.

This script manages the creation of fake sound entries with realistic
embedding vectors generated from a Gaussian Mixture Model (GMM). These synthetic documents
populate the Solr index to test performance (indexing speed, search latency, memory usage)
at scales larger than the actual Freesound database.

Key Features:
- Fits a GMM to existing real embedding vectors using GMModel.
- Generates synthetic vectors distributed similarly to the real data (or a sampled subset).
- Indexes synthetic data as Solr documents to a new similarity space and is_synthetic tag.
- Provides utilities to check index size and cleanup synthetic data.

Examples:
    # 1. Fit a GMM model on existing data (finding best K automatically)
    python search/generate_and_index_synthetics.py --fit --find-k --source-space laion_clap

    # 2. Generate and index 1 million synthetic vectors using the saved model
    python search/generate_and_index_synthetics.py --generate 1000000

    # 3. Clean up (delete) all synthetic data from Solr
    python search/generate_and_index_synthetics.py --cleanup
"""

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
from datetime import datetime

# Starting ID for synthetic data to avoid collision with real Freesound IDs
SYNTHETIC_ID_START = 1_000_000_000 

def generate_and_index(source_space, n_synthetic, gm_model, solr_url=SOLR_URL, batch_size=2000):
    """Generates synthetic vectors and indexes them into Solr in batches.

    Creates both parent documents (representing the fake sound) and child documents
    (containing the vector).

    Args:
        source_space (str): The name of the original similarity space (used for naming).
        n_synthetic (int): Total number of synthetic documents to generate.
        gm_model (GMModel): The fitted Gaussian Mixture Model to sample from.
        solr_url (str): Solr endpoint URL.
        batch_size (int): Number of documents per indexing batch.
    """
    indexer = SolrIndexer(solr_url)
    print(f"Generating {n_synthetic} synthetic embeddings...")
    target_space = f"{source_space}_synthetic"
    
    current_id = SYNTHETIC_ID_START
    total_batches = (n_synthetic // batch_size) + 1
    
    for _ in tqdm(range(total_batches), desc="Indexing synthetic batches"):
        
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
            
        indexer.index_documents(docs_to_index, batch_size=batch_size)
        indexer.commit()  # Flush logs and reclaim space every batch!
        
        current_id += current_batch_size
    
    print("Finalizing index (optimizing)...")
    requests.get(f"{solr_url}/update?optimize=true")
    print(f"Successfully indexed {n_synthetic} synthetic documents.")

def get_synthetic_count(indexer):
    """Returns the current number of documents flagged as 'is_synthetic:true'."""
    return indexer.solr.search('is_synthetic:true', rows=0).hits

def cleanup_synthetic(solr_url=SOLR_URL):
    """Removes all synthetic data from the Solr index.
    
    Deletes documents based on the `is_synthetic` flag and the reserved ID range.
    Triggers a hard commit and optimization to reclaim disk space.
    """
    indexer = SolrIndexer(solr_url)
    print(f"Current count before cleanup: {get_synthetic_count(indexer)}")
    
    # 1. Delete by the flag (Parents)
    # 2. Delete by the ID range reserved for synthetics
    delete_query = f'is_synthetic:true OR id:[{int(SYNTHETIC_ID_START)} TO *]'
    
    print(f"Executing deep cleanup with query: {delete_query}")
    indexer.solr.delete(q=delete_query)
    indexer.commit()
    
    # Request Solr optimization to merge segments and purge deleted docs from disk
    requests.get(f"{solr_url}/update?optimize=true")
    
    print(f"Count after cleanup: {get_synthetic_count(indexer)}")

def get_index_size_mb(solr_url):
    """Fetches the total on-disk size of the Solr index (across all cores) in MB.
    
    Args:
        solr_url (str): The Solr base URL.
        
    Returns:
        float: Size in Megabytes (MB).
    """
    try:
        # Construct Admin API URL
        base_url = solr_url.split('/solr')[0] + "/solr"
        admin_url = f"{base_url}/admin/cores"
        
        resp = requests.get(admin_url, params={'action': 'STATUS', 'wt': 'json'}, timeout=5)
        resp.raise_for_status()
        status = resp.json().get('status', {})

        # Sum the size of all cores (handling potential sharding/replicas)
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
    parser.add_argument('--min-k', type=int, default=100, help='Min K for BIC search')
    parser.add_argument('--max-k', type=int, default=1000, help='Max K for BIC search')
    parser.add_argument('--step', type=int, default=100, help='Step size for BIC search')
    parser.add_argument('--max-iter', type=int, default=100, help='Max iterations for GMM fitting')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    parser.add_argument('--check-count', action='store_true', help='Print current number of synthetic docs')
    parser.add_argument('--check-size', action='store_true', help='Print current index size (MB)')
    
    args = parser.parse_args()
    
    np.random.seed(args.seed)

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
        manager = GMModel(n_components=args.k, seed=args.seed, max_iter=args.max_iter)
        
        vectors, _, _ = load_vectors_from_solr(args.source_space)
        num_available = len(vectors)

        if args.subsample is not None and args.subsample < num_available:
            print(f"Subsampling {args.subsample} vectors from available {num_available} for fitting...")
            indices = np.random.choice(num_available, size=args.subsample, replace=False)
            vectors = vectors[indices]

        if args.find_k:
            best_k = manager.find_optimal_k(
                vectors, 
                min_k=args.min_k,
                max_k=args.max_k,
                step=args.step,
                plot_path=args.model_path.replace('.pkl', '_bic_plot.png')
            )
            manager.n_components = best_k
            
        if not manager.model:
            manager.fit(vectors)
        
        manager.save(args.model_path)
        
        # Save metadata
        meta_path = args.model_path.replace('.pkl', '.json')
        with open(meta_path, 'w') as f:
            json.dump({
                'k': manager.n_components,
                'seed': args.seed,
                'n_vectors': len(vectors),
                'timestamp': datetime.now().isoformat(),
                'source_space': args.source_space
            }, f, indent=2)
        print(f"Model metadata saved to {meta_path}")
    
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