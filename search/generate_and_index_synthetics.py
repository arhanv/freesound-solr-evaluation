import argparse
import os
import sys
import numpy as np
from tqdm import tqdm

from pca import load_vectors_from_solr
from index_to_solr import SolrIndexer
from configs import SOLR_URL
from gm_model import GMModel

# Starting ID for synthetic data to avoid collision with real Freesound IDs
SYNTHETIC_ID_START = 1_000_000_000 

def generate_and_index(source_space, n_synthetic, gm_model, solr_url=SOLR_URL, batch_size=5000):
    indexer = SolrIndexer(solr_url)
    print(f"Generating {n_synthetic} synthetic embeddings...")
    
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
                'id': f"{parent_id}_{source_space}",
                'content_type': 'v',
                'similarity_space': source_space,
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
    """Deletes only synthetic data from Solr."""
    indexer = SolrIndexer(solr_url)
    print(f"Number of synthetic documents in index: {get_synthetic_count(indexer)}")
    print("Deleting synthetic documents...")
    indexer.solr.delete(q='is_synthetic:true')
    indexer.commit()
    print(f"Number of synthetic documents in index: {get_synthetic_count(indexer)}")

def main():
    parser = argparse.ArgumentParser(description='Generate and Index Synthetic Embeddings via GMM')
    parser.add_argument('--fit', action='store_true', help='Fit a new GMM model')
    parser.add_argument('--find-k', action='store_true', help='Run BIC search to find best K')
    parser.add_argument('--generate', type=int, default=0, help='Number of synthetic vectors to generate')
    parser.add_argument('--cleanup', action='store_true', help='Remove previously generated synthetic data')
    
    parser.add_argument('--source-space', default='laion_clap', help='Source similarity space')
    parser.add_argument('--model-path', default='models/gmm_laion_clap.pkl', help='Path to save/load GMM')
    parser.add_argument('--k', type=int, default=64, help='Number of components (if not running find-k)')
    
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)

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
        
        if args.find_k:
            best_k = manager.find_optimal_k(
                vectors, 
                plot_path=args.model_path.replace('.pkl', '_bic_plot.png')
            )
            manager.n_components = best_k
            
        manager.fit(vectors)
        manager.save_model(args.model_path)
    
    # Case 2: Using an existing model
    elif args.generate > 0:
        if not os.path.exists(args.model_path):
            print(f"Error: Model not found at {args.model_path}. Run with --fit first.")
            return

        manager = GMModel.load_model(args.model_path)

    # Generate data
    if args.generate > 0 and manager is not None:
        generate_and_index(args.source_space, args.generate, manager)

if __name__ == "__main__":
    main()