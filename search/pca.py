"""
Reduce dimensionality of sound similarity vectors using PCA and re-index Solr with new similarity spaces.

This script performs two main functions:
1. Training (fitting) a PCA model on existing high-dimensional vectors from Solr.
2. Transforming existing vectors and re-indexing them as new child documents in Solr.

Key Features:
- Implements a robust "Read-Modify-Write" cycle to ensure atomic updates to Solr blocks.
- Uses append-only text checkpointing to resume long-running re-indexing jobs.
- Batches updates (capping at 1000) to avoid Solr Boolean query limits.
- Optimized I/O by limiting field lists and reusing connections.
- Standalone cleanup utility to reset similarity spaces.
"""

import argparse
import json
import os
import pickle
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import pysolr
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from tqdm import tqdm

# Add project root to sys.path so 'search' package is found when running script directly
if __name__ == "__main__" or __name__.startswith("search"):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search.configs import SOLR_URL, SOLR_BASE_URL, COLLECTION_NAME
from search.solrapi import SolrManagementAPI


def ensure_solr_field_exists(dimensions, solr_base_url=SOLR_BASE_URL, collection_name=COLLECTION_NAME):
    """Ensures Solr has the required field type and field for the given dimensions (Memoized)."""
    if not hasattr(ensure_solr_field_exists, "_checked"):
        ensure_solr_field_exists._checked = set()
    
    field_name = f"sim_vector{dimensions}_l2"
    if field_name in ensure_solr_field_exists._checked:
        return

    api = SolrManagementAPI(solr_base_url, collection_name)
    field_type_name = f"knn_vector{dimensions}_l2"

    if api.check_collection_has_field(field_name):
        ensure_solr_field_exists._checked.add(field_name)
        return

    print(f"Adding {dimensions}-d vector field to Solr...")
    field_type = {
        "name": field_type_name,
        "class": "solr.DenseVectorField",
        "vectorDimension": str(dimensions),
        "similarityFunction": "dot_product",
        "knnAlgorithm": "hnsw",
        "hnswMaxConnections": "10",
        "hnswBeamWidth": "40"
    }
    field = {
        "name": field_name,
        "type": field_type_name,
        "indexed": True,
        "stored": True,
        "required": False
    }

    try:
        api.add_field_types([field_type], bulk=True)
    except Exception: pass
    try:
        api.add_fields([field], bulk=True)
        print(f"Added field '{field_name}'")
    except Exception as e:
        raise Exception(f"Failed to add field: {e}")
    
    ensure_solr_field_exists._checked.add(field_name)


import requests

def load_vectors_from_solr(similarity_space, solr_url=SOLR_URL, max_vectors=None):
    """Loads similarity vectors from Solr by querying child documents using cursor pagination."""
    session = requests.Session()
    solr = pysolr.Solr(solr_url, always_commit=False, session=session)
    query = f'content_type:v AND similarity_space:{similarity_space}'

    try:
        total_hits = solr.search(query, rows=0).hits
    except Exception: total_hits = 0

    limit = max_vectors if max_vectors else total_hits
    print(f"Loading {limit} vectors (space={similarity_space})...")

    # --- Step 1: Find the exact vector field name to avoid wildcard slowness ---
    vector_field = None
    try:
        # Fetch just one doc to identify field
        sample = solr.search(query, rows=1, fl='sim_vector*')
        if sample.docs:
            vector_field = next((k for k in sample.docs[0].keys() if k.startswith('sim_vector')), None)
    except Exception: pass

    if not vector_field:
        print(f"Warning: Could not identify vector field for {similarity_space}. Falling back to wildcard.")
        vector_field_fl = 'sim_vector*'
    else:
        vector_field_fl = vector_field

    # --- Step 2: Download batches ---
    vectors = []
    sound_ids = []
    child_docs = []
    cursor_mark = '*'
    page_size = 10000 
    
    # Metadata fields to preserve
    metadata_fl = 'id,similarity_space,timestamp_start,timestamp_end,pack_grouping_child'
    full_fl = f"{metadata_fl},{vector_field_fl}"

    pbar = tqdm(total=limit, desc="Downloading vectors")

    while len(vectors) < limit:
        rows_to_fetch = min(page_size, limit - len(vectors))
        try:
            results = solr.search(query, rows=rows_to_fetch, fl=full_fl, 
                                  cursorMark=cursor_mark, sort='id asc')
        except Exception as e:
            print(f"Error fetching page: {e}"); break

        if not results.docs: break

        for doc in results.docs:
            # If we used wildcard, find it now (first time only)
            if vector_field is None:
                vector_field = next((k for k in doc.keys() if k.startswith('sim_vector')), None)

            if vector_field and vector_field in doc:
                vectors.append(doc[vector_field])
                child_id = doc['id']
                try:
                    sound_ids.append(int(child_id.split('_')[0]))
                    child_docs.append(doc)
                except Exception: continue

        pbar.update(len(results.docs))
        if cursor_mark == results.nextCursorMark: break
        cursor_mark = results.nextCursorMark

    pbar.close()
    session.close() # Clean up session
    if not vectors: raise Exception(f"No vectors found for similarity_space={similarity_space}")
    return np.array(vectors, dtype=np.float32), sound_ids, child_docs


def fit_pca(vectors, n_components):
    """Fits a PCA model on the provided vectors."""
    print(f"Fitting PCA to {n_components} components...")
    pca = PCA(n_components=n_components)
    pca.fit(vectors)
    print(f"Explained variance: {pca.explained_variance_ratio_.sum():.4f}")
    return pca


def transform_vectors(pca, vectors, normalize_output=True):
    """Transforms vectors using a fitted PCA model."""
    transformed = pca.transform(vectors).astype(np.float32)
    if normalize_output:
        transformed = normalize(transformed, norm='l2')
    return transformed


def save_pca_model(pca, filepath):
    """Saves a fitted PCA model to disk."""
    with open(filepath, 'wb') as f:
        pickle.dump(pca, f)
    print(f"Saved PCA model to {filepath}")


def load_pca_model(filepath):
    """Loads a PCA model from disk."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def archive_existing_model(filepath):
    """Renames an existing model file and its sidecar to include a timestamp."""
    if not filepath or not os.path.exists(filepath):
        return
    
    timestamp = datetime.fromtimestamp(os.path.getmtime(filepath)).strftime("%Y%m%d-%H%M%S")
    archive_path = f"{filepath}.{timestamp}.archive"
    os.rename(filepath, archive_path)
    print(f"Archived existing model to {archive_path}")

    # Check for metadata sidecar
    meta_path = filepath.replace('.pkl', '.json')
    if os.path.exists(meta_path):
        meta_archive_path = f"{meta_path}.{timestamp}.archive"
        os.rename(meta_path, meta_archive_path)


def fit_and_save_pca(source_space, n_components, output_model_path, max_vectors=None):
    """Orchestrates loading data, fitting PCA, and saving the model."""
    vectors, _, _ = load_vectors_from_solr(source_space, max_vectors=max_vectors)
    pca = fit_pca(vectors, n_components)
    
    os.makedirs(os.path.dirname(os.path.abspath(output_model_path)), exist_ok=True)
    archive_existing_model(output_model_path)
    save_pca_model(pca, output_model_path)
    
    # Save sidecar metadata
    meta_path = output_model_path.replace('.pkl', '.json')
    with open(meta_path, 'w') as f:
        json.dump({
            'n_components': n_components,
            'n_training_samples': len(vectors),
            'source_space': source_space,
            'explained_variance': float(pca.explained_variance_ratio_.sum()),
            'timestamp': datetime.now().isoformat(),
            'model_filename': os.path.basename(output_model_path)
        }, f, indent=4)
    print(f"Model metadata saved to {meta_path}")
    
    return pca


def load_checkpoint(filepath):
    """Loads processed parent IDs from a text-based checkpoint file."""
    if filepath and os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                return set(line.strip() for line in f if line.strip())
        except Exception as e:
            print(f"Warning: Failed to load checkpoint {filepath}: {e}")
    return set()


def save_checkpoint(filepath, new_ids):
    """Appends new processed IDs to the checkpoint file."""
    if not filepath: return
    try:
        with open(filepath, 'a') as f:
            for nid in new_ids:
                f.write(f"{nid}\n")
    except Exception as e:
        print(f"Warning: Failed to save checkpoint: {e}")


def delete_pca_vectors(target_space, solr_url=SOLR_URL):
    """Safely deletes vectors from a specific PCA space."""
    solr = pysolr.Solr(solr_url, always_commit=True)
    query = f'content_type:v AND similarity_space:{target_space}'
    
    print(f"Searching for vectors in '{target_space}' to delete...")
    try:
        hits = solr.search(query, rows=0).hits
        if hits == 0:
            print(f"No vectors found in space '{target_space}'. Nothing to clean.")
            return

        confirm = input(f"DANGER: This will delete {hits} child documents in '{target_space}'. Proceed? (y/n): ")
        if confirm.lower() != 'y':
            print("Cleanup aborted.")
            return

        print(f"Deleting {hits} vectors...")
        solr.delete(q=query)
        print("Cleanup complete. Index optimized (hard commit).")
    except Exception as e:
        print(f"Cleanup failed: {e}")


def _process_batch(chunk_ids, parent_data_map, target_similarity_space, vector_field,
                   solr_url, checkpoint_path, checkpoint_lock, pbar):
    """Processes a single batch of parent IDs: fetch, build, and index."""
    FIELDS_TO_EXCLUDE = ['_version_', 'type_facet', 'username_facet', 'tagfacet', 'created_range',
                         'timestamp_start', 'timestamp_end']
    # Each thread gets its own pysolr connection (not thread-safe to share one)
    solr = pysolr.Solr(solr_url, always_commit=False, timeout=300)
    id_query = " OR ".join(chunk_ids)

    # --- 1. Fetch Parents ---
    try:
        results = solr.search(f"content_type:s AND id:({id_query})", rows=len(chunk_ids), fl='*')
    except Exception as e:
        print(f"\nError fetching parents for chunk starting at {chunk_ids[0]}: {e}")
        pbar.update(len(chunk_ids))
        return

    parent_map = {str(doc['id']): doc for doc in results}
    for doc in parent_map.values():
        for field in FIELDS_TO_EXCLUDE:
            if field in doc:
                del doc[field]

    # --- 2. Fetch Existing Children ---
    try:
        child_results = solr.search(
            f"content_type:v AND _root_:({' OR '.join(chunk_ids)})",
            rows=len(chunk_ids) * 20, fl='*')
    except Exception as e:
        print(f"\nError fetching children for chunk starting at {chunk_ids[0]}: {e}")
        pbar.update(len(chunk_ids))
        return

    parent_children_map = {}
    for child in child_results:
        root_id = str(child.get('_root_'))
        if root_id not in parent_children_map:
            parent_children_map[root_id] = []
        if '_version_' in child:
            del child['_version_']
        parent_children_map[root_id].append(child)

    # --- 3. Build & Send Updates ---
    docs_to_index = []
    for pid in chunk_ids:
        if pid not in parent_map:
            continue
        parent_doc = parent_map[pid]
        new_data_list = parent_data_map[pid]
        final_children = [c for c in parent_children_map.get(pid, [])
                          if c.get('similarity_space') != target_similarity_space]
        for vec, orig_doc in new_data_list:
            final_children.append({
                'id': f"{pid}_{target_similarity_space}",
                'content_type': 'v',
                'similarity_space': target_similarity_space,
                'timestamp_start': orig_doc.get('timestamp_start', 0),
                'timestamp_end': orig_doc.get('timestamp_end', -1),
                vector_field: vec.tolist(),
            })
        parent_doc['_childDocuments_'] = final_children
        docs_to_index.append(parent_doc)

    if docs_to_index:
        try:
            solr.add(docs_to_index, commitWithin=10000)
            with checkpoint_lock:
                save_checkpoint(checkpoint_path, chunk_ids)
        except Exception as e:
            print(f"\nError indexing batch starting at {chunk_ids[0]}: {e}")

    pbar.update(len(chunk_ids))


def add_pca_child_documents(sound_ids, reduced_vectors, original_child_docs, target_similarity_space,
                        solr_url=SOLR_URL, batch_size=1000, checkpoint_path=None, max_workers=1):
    """Adds PCA-reduced child documents to Solr using an optimized Read-Modify-Write strategy.
    
    Args:
        max_workers: Number of parallel threads for batch processing. Default is 1 (serial).
                     Values >1 parallelise independent Solr fetch+index cycles; useful when
                     Solr round-trip latency dominates. Each thread uses its own connection.
    """
    print("Grouping data by parent ID...")
    parent_data_map = {}
    for sid, vec, orig_doc in tqdm(zip(sound_ids, reduced_vectors, original_child_docs),
                                   total=len(sound_ids), desc="Grouping by parent"):
        sid_str = str(sid)
        if sid_str not in parent_data_map:
            parent_data_map[sid_str] = []
        parent_data_map[sid_str].append((vec, orig_doc))

    processed_parents = load_checkpoint(checkpoint_path)
    unique_parent_ids = sorted(list(parent_data_map.keys()))
    parents_to_process = [pid for pid in unique_parent_ids if pid not in processed_parents]

    print(f"Total parents: {len(unique_parent_ids)}. Parents to process: {len(parents_to_process)}")

    if not parents_to_process:
        print("All parents processed!")
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                os.remove(checkpoint_path)
            except Exception:
                pass
        return

    if batch_size > 1000:
        batch_size = 1000

    n_dims = reduced_vectors.shape[1]
    vector_field = f"sim_vector{n_dims}_l2"
    checkpoint_lock = threading.Lock()

    batches = [parents_to_process[i:i + batch_size]
               for i in range(0, len(parents_to_process), batch_size)]

    effective_workers = min(max_workers, len(batches))
    if effective_workers > 1:
        print(f"Using {effective_workers} parallel workers for {len(batches)} batches.")

    with tqdm(total=len(parents_to_process), desc="Processing & Indexing") as pbar:
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            futures = [
                executor.submit(
                    _process_batch,
                    chunk_ids, parent_data_map, target_similarity_space, vector_field,
                    solr_url, checkpoint_path, checkpoint_lock, pbar
                )
                for chunk_ids in batches
            ]
            for future in as_completed(futures):
                exc = future.exception()
                if exc:
                    print(f"\nBatch raised an exception: {exc}")

    print("Processing complete.")
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            os.remove(checkpoint_path)
        except Exception:
            pass


def reduce_and_reindex(pca_model_path, source_similarity_space, target_similarity_space, normalize_output=True,
                       batch_size=1000, checkpoint_path=None, cache_path=None, max_workers=1):
    """Executes the full pipeline with optional vector caching."""
    pca = load_pca_model(pca_model_path)
    ensure_solr_field_exists(pca.n_components_)

    loaded_from_cache = False
    if cache_path and os.path.exists(cache_path):
        print(f"Loading transformed vectors from cache: {cache_path}...")
        try:
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
                sound_ids = cache_data['sound_ids']
                reduced = cache_data['reduced_vectors']
                child_docs = cache_data['child_docs']
                print(f"Cache loaded: {len(sound_ids)} vectors.")
                loaded_from_cache = True
        except Exception as e:
            print(f"Warning: Failed to load cache: {e}. Falling back to Solr download.")

    if not loaded_from_cache:
        vectors, sound_ids, child_docs = load_vectors_from_solr(source_similarity_space)
        print("Transforming vectors...")
        reduced = transform_vectors(pca, vectors, normalize_output)

        if cache_path:
            print(f"Saving transformed vectors to cache: {cache_path}...")
            try:
                os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
                with open(cache_path, 'wb') as f:
                    pickle.dump({'sound_ids': sound_ids, 'reduced_vectors': reduced, 'child_docs': child_docs}, f)
            except Exception as e: print(f"Warning: Failed to save cache: {e}")

    add_pca_child_documents(sound_ids, reduced, child_docs, target_similarity_space, batch_size=batch_size,
                            checkpoint_path=checkpoint_path, max_workers=max_workers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PCA dimensionality reduction for similarity vectors')
    parser.add_argument('--fit', action='store_true', help='Fit PCA model')
    parser.add_argument('--reindex', action='store_true', help='Transform and add child documents to Solr')
    parser.add_argument('--cleanup', action='store_true', help='Delete vectors from the target PCA space')
    parser.add_argument('--dims', type=int, nargs='+', default=[128], help='Target dimensions (can provide multiple)')
    parser.add_argument('--source-space', default='laion_clap', help='Source similarity space')
    parser.add_argument('--target-space', default=None, help='Target similarity space name (if single dimension)')
    parser.add_argument('--model-path', default=None, help='Path to PCA model file (if single dimension)')
    parser.add_argument('--checkpoint', default=None, help='Path to checkpoint file (if single dimension)')
    parser.add_argument('--batch-size', type=int, default=1000, help='Batch size for indexing')
    parser.add_argument('--workers', type=int, default=1, help='Number of parallel worker threads for batch indexing (default: 1)')
    parser.add_argument('--cache', action='store_true', help='Cache transformed vectors to disk')
    parser.add_argument('--cache-path', default=None, help='Path to vector cache file (if single dimension)')

    args = parser.parse_args()

    pbar = tqdm(args.dims)
    for dim in pbar:
        # 1. Determine Target Space Name
        if len(args.dims) == 1 and args.target_space:
            target = args.target_space
        else:
            target = f"{args.source_space}_pca{dim}"
            if args.target_space and len(args.dims) > 1:
                print(f"Warning: Multiple dimensions provided. Ignoring target-space '{args.target_space}' for dim {dim} and using '{target}' instead.")

        # 2. Determine Model Path
        if len(args.dims) == 1 and args.model_path:
            model_path = args.model_path
        else:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, "models", "pca", f"{target}.pkl")

        pbar.set_description(f"Dim: {dim} ({target})")

        if args.cleanup:
            # Note: delete_pca_vectors has a confirmation prompt
            delete_pca_vectors(target)

        if args.fit:
            fit_and_save_pca(args.source_space, dim, model_path)

        if args.reindex:
            # Determine Cache Path
            if len(args.dims) == 1 and args.cache_path:
                cache_p = args.cache_path
            elif args.cache:
                cache_p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", f"cache_{target}.pkl")
            else:
                cache_p = None

            # Determine Checkpoint Path
            if len(args.dims) == 1 and args.checkpoint:
                ckpt = args.checkpoint
            else:
                ckpt = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", f"pca_checkpoint_{target}.txt")

            reduce_and_reindex(model_path, args.source_space, target,
                               batch_size=args.batch_size, checkpoint_path=ckpt,
                               cache_path=cache_p, max_workers=args.workers)
        
    print("\nAll dimensions processed.")
