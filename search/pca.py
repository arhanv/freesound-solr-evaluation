"""
Reduce dimensionality of sound similarity vectors using PCA and re-index Solr with new similarity spaces.

This script performs two main functions:
1. Training (fitting) a PCA model on existing high-dimensional vectors from Solr.
2. Transforming existing vectors and re-indexing them as new child documents in Solr.

Key Features:
- Implements a robust "Read-Modify-Write" cycle to ensure atomic updates to Solr blocks.
- Uses checkpointing to resume long-running re-indexing jobs.
- Batches updates to avoid overwhelming Solr.
"""

import argparse
import json
import os
import pickle

import numpy as np
import pysolr
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from tqdm import tqdm

from configs import SOLR_URL, SOLR_BASE_URL, COLLECTION_NAME
from solrapi import SolrManagementAPI


def ensure_solr_field_exists(dimensions, solr_base_url=SOLR_BASE_URL, collection_name=COLLECTION_NAME):
    """Ensures Solr has the required field type and field for the given dimensions.

    Dynamically adds a DenseVectorField type and the corresponding field definition
    to the Solr schema if they do not already exist.

    Args:
        dimensions (int): Vector dimensions (e.g., 128).
        solr_base_url (str): Solr base URL.
        collection_name (str): Solr collection name.
    """
    api = SolrManagementAPI(solr_base_url, collection_name)
    field_name = f"sim_vector{dimensions}_l2"
    field_type_name = f"knn_vector{dimensions}_l2"

    if api.check_collection_has_field(field_name):
        print(f"Field '{field_name}' already exists")
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
        api.add_field_types([field_type], bulk=False)
        print(f"Added field type '{field_type_name}'")
    except Exception as e:
        print(f"Field type may already exist: {e}")

    try:
        api.add_fields([field], bulk=False)
        print(f"Added field '{field_name}'")
    except Exception as e:
        raise Exception(f"Failed to add field: {e}")


def load_vectors_from_solr(similarity_space, solr_url=SOLR_URL, max_vectors=None):
    """Loads similarity vectors from Solr by querying child documents.

    Uses cursor pagination to efficiently retrieve large sets of vectors.

    Args:
        similarity_space (str): Similarity space name (e.g., 'laion_clap').
        solr_url (str): Solr collection URL.
        max_vectors (int, optional): Max vectors to load (None = all).

    Returns:
        tuple: A tuple containing:
            - vectors (np.ndarray): Array of vectors.
            - sound_ids (list): List of parent sound IDs.
            - child_docs (list): List of original child document dictionaries.
    """
    solr = pysolr.Solr(solr_url, always_commit=False)

    # Query for child documents with this similarity space
    query = f'content_type:v AND similarity_space:{similarity_space}'

    print(f"Counting vectors for {similarity_space}...")
    try:
        total_hits = solr.search(query, rows=0).hits
    except Exception as e:
        print(f"Error checking count: {e}")
        total_hits = 0

    limit = max_vectors if max_vectors else total_hits
    print(f"Loading {limit} vectors from Solr (similarity_space={similarity_space})...")

    vectors = []
    sound_ids = []
    child_docs = []

    cursor_mark = '*'
    page_size = 5000
    vector_field = None

    pbar = tqdm(total=limit, desc="Downloading vectors")

    while True:
        if len(vectors) >= limit:
            break

        rows_to_fetch = page_size
        remaining = limit - len(vectors)
        if remaining < page_size:
            rows_to_fetch = remaining

        try:
            # Cursor pagination requires sort by unique key
            results = solr.search(query, rows=rows_to_fetch, fl='*,[child]', cursorMark=cursor_mark, sort='id asc')
        except Exception as e:
            print(f"Error fetching page: {e}")
            break

        if not results.docs:
            break

        batch_count = 0
        for doc in results.docs:
            # Identify the vector field dynamically (e.g., sim_vector512_l2)
            if vector_field is None:
                for key in doc.keys():
                    if key.startswith('sim_vector') and key.endswith('_l2'):
                        vector_field = key
                        break

            if vector_field and vector_field in doc:
                vectors.append(doc[vector_field])
                
                # Extract parent ID from the composite child ID ({parent_id}_{space})
                child_id = doc['id']
                try:
                    sound_id = int(child_id.split('_')[0])
                except Exception:
                    continue  # Skip malformed IDs

                sound_ids.append(sound_id)
                child_docs.append(doc)
                batch_count += 1

        pbar.update(batch_count)

        next_cursor = results.nextCursorMark
        if cursor_mark == next_cursor:
            break
        cursor_mark = next_cursor

    pbar.close()

    if not vectors:
        raise Exception(f"No vectors found for similarity_space={similarity_space}")

    print(f"Loaded {len(vectors)} vectors (dimensionality: {len(vectors[0])})")

    return np.array(vectors, dtype=np.float32), sound_ids, child_docs


def fit_pca(vectors, n_components):
    """Fits a PCA model on the provided vectors.

    Args:
        vectors (np.ndarray): Input vectors (N, D).
        n_components (int): Number of target dimensions.

    Returns:
        sklearn.decomposition.PCA: The fitted PCA model.
    """
    print(f"Fitting PCA to {n_components} components...")
    pca = PCA(n_components=n_components)
    pca.fit(vectors)

    explained_var = pca.explained_variance_ratio_.sum()
    print(f"Explained variance: {explained_var:.4f}")

    return pca


def transform_vectors(pca, vectors, normalize_output=True):
    """Transforms vectors using a fitted PCA model.

    Args:
        pca (sklearn.decomposition.PCA): Fitted PCA model.
        vectors (np.ndarray): Input vectors to transform.
        normalize_output (bool): Whether to L2-normalize the output vectors.

    Returns:
        np.ndarray: Transformed (and optionally normalized) vectors.
    """
    transformed = pca.transform(vectors).astype(np.float32)

    if normalize_output:
        transformed = normalize(transformed, norm='l2')

    return transformed


def save_pca_model(pca, filepath):
    """Saves a fitted PCA model to disk using pickle."""
    with open(filepath, 'wb') as f:
        pickle.dump(pca, f)
    print(f"Saved PCA model to {filepath}")


def load_pca_model(filepath):
    """Loads a PCA model from disk."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def fit_and_save_pca(source_space, n_components, output_model_path, max_vectors=None):
    """Orchestrates loading data, fitting PCA, and saving the model.

    Args:
        source_space (str): Source similarity space to train on.
        n_components (int): Target dimensions.
        output_model_path (str): File path to save the model.
        max_vectors (int, optional): Limit on number of training vectors.
    
    Returns:
        sklearn.decomposition.PCA: The fitted model.
    """
    vectors, _, _ = load_vectors_from_solr(source_space, max_vectors=max_vectors)
    pca = fit_pca(vectors, n_components)
    save_pca_model(pca, output_model_path)
    return pca


def load_checkpoint(filepath):
    """Loads the set of processed parent IDs from a checkpoint file."""
    if filepath and os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                return set(data.get('processed_ids', []))
        except Exception as e:
            print(f"Warning: Failed to load checkpoint {filepath}: {e}")
    return set()


def save_checkpoint(filepath, new_ids):
    """Appends new processed IDs to the checkpoint file."""
    if not filepath:
        return

    processed_ids = set()
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                processed_ids = set(data.get('processed_ids', []))
        except Exception:
            pass

    processed_ids.update(new_ids)

    with open(filepath, 'w') as f:
        json.dump({'processed_ids': list(processed_ids)}, f)


def add_pca_child_documents(sound_ids, reduced_vectors, original_child_docs, target_similarity_space,
                            solr_url=SOLR_URL, batch_size=100, checkpoint_path=None):
    """Adds PCA-reduced child documents to Solr using a Read-Modify-Write strategy.

    To safely add nested documents without overwriting existing ones (a limitation 
    of standard Solr indexing), this function:
    1. Fetches the full parent document and all its existing children.
    2. Appends the new PCA-reduced vector as a new child.
    3. Re-indexes the entire block (Parent + Old Children + New Child).

    Args:
        sound_ids (list): List of parent sound IDs corresponding to the vectors.
        reduced_vectors (np.ndarray): Array of reduced vectors.
        original_child_docs (list): Original child documents (used for metadata inheritance).
        target_similarity_space (str): Name of the new similarity space (e.g., 'laion_clap_pca128').
        solr_url (str): Solr collection URL.
        batch_size (int): Number of PARENTS to process per batch.
        checkpoint_path (str): Path to the checkpoint file.
    """
    solr = pysolr.Solr(solr_url, always_commit=False, timeout=300)

    # Group data by parent ID first to handle batching by parent
    print("Grouping data by parent ID...")
    parent_data_map = {}  # parent_id -> list of (vector, original_child_doc)

    for sid, vec, orig_doc in zip(sound_ids, reduced_vectors, original_child_docs):
        sid_str = str(sid)
        if sid_str not in parent_data_map:
            parent_data_map[sid_str] = []
        parent_data_map[sid_str].append((vec, orig_doc))

    # Load checkpoint to skip already processed parents
    processed_parents = load_checkpoint(checkpoint_path)
    if processed_parents:
        print(f"Loaded checkpoint: {len(processed_parents)} parents already processed.")

    unique_parent_ids = sorted(list(parent_data_map.keys()))
    parents_to_process = [pid for pid in unique_parent_ids if pid not in processed_parents]

    print(f"Total parents: {len(unique_parent_ids)}. Parents to process: {len(parents_to_process)}")

    if not parents_to_process:
        print("All parents processed!")
        # Cleanup checkpoint if it exists
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                os.remove(checkpoint_path)
                print(f"Checkpoint file {checkpoint_path} deleted.")
            except Exception as e:
                print(f"Warning: Could not delete checkpoint file: {e}")
        return

    # Determine vector field name based on dimensionality
    n_dims = reduced_vectors.shape[1]
    vector_field = f"sim_vector{n_dims}_l2"

    # Fields to exclude when fetching parent docs to avoid schema conflicts or unnecessary data
    FIELDS_TO_EXCLUDE = ['_version_', 'type_facet', 'username_facet', 'tagfacet', 'created_range', 'timestamp_start',
                         'timestamp_end']

    print(f"Processing in chunks of {batch_size} parents...")

    for i in tqdm(range(0, len(parents_to_process), batch_size), desc="Processing & Indexing"):
        chunk_ids = parents_to_process[i:i + batch_size]

        # --- 1. Fetch Parents ---
        # We need the parent documents to re-index the block correctly
        id_query = " OR ".join(chunk_ids)
        q = f"content_type:s AND id:({id_query})"

        try:
            results = solr.search(q, rows=len(chunk_ids), fl='*')
        except Exception as e:
            print(f"\nError fetching parents for chunk {i}: {e}")
            continue

        parent_map = {}
        for doc in results:
            for field in FIELDS_TO_EXCLUDE:
                if field in doc:
                    del doc[field]
            parent_map[str(doc['id'])] = doc

        # --- 2. Fetch Existing Children ---
        # Fetch all children associated with these parents to preserve them
        child_query = f"content_type:v AND _root_:({' OR '.join(chunk_ids)})"
        try:
            # Request extra rows to ensure we get all children
            child_results = solr.search(child_query, rows=len(chunk_ids) * 20, fl='*')
        except Exception as e:
            print(f"\nError fetching children for chunk {i}: {e}")
            continue

        parent_children_map = {}
        for child in child_results:
            root_id = str(child.get('_root_'))
            if root_id not in parent_children_map:
                parent_children_map[root_id] = []

            if '_version_' in child:
                del child['_version_']
            parent_children_map[root_id].append(child)

        # --- 3. Build Updates ---
        docs_to_index = []

        for pid in chunk_ids:
            if pid not in parent_map:
                print(f"Warning: Parent {pid} not found in Solr. Skipping.")
                continue

            parent_doc = parent_map[pid]
            new_data_list = parent_data_map[pid]  # list of (vec, orig_doc) for this parent

            # Get existing children
            existing_children = parent_children_map.get(pid, [])

            # Keep existing children EXCEPT those from the target space (replacing old versions)
            final_children = [
                c for c in existing_children
                if c.get('similarity_space') != target_similarity_space
            ]

            # Append NEW children (the PCA vectors)
            for vec, orig_doc in new_data_list:
                child_doc = {
                    'id': f"{pid}_{target_similarity_space}",
                    'content_type': 'v',
                    'similarity_space': target_similarity_space,
                    'timestamp_start': orig_doc.get('timestamp_start', 0),
                    'timestamp_end': orig_doc.get('timestamp_end', -1),
                    vector_field: vec.tolist(),
                }
                # Preserve grouping info if present
                if 'pack_grouping_child' in orig_doc:
                    child_doc['pack_grouping_child'] = orig_doc['pack_grouping_child']

                final_children.append(child_doc)

            # Assign the full list of children to the parent document
            parent_doc['_childDocuments_'] = final_children
            docs_to_index.append(parent_doc)

        # --- 4. Index & Commit ---
        if docs_to_index:
            try:
                solr.add(docs_to_index)
                solr.commit()

                # --- 5. Save Checkpoint ---
                save_checkpoint(checkpoint_path, chunk_ids)

            except Exception as e:
                print(f"\nError indexing/committing batch: {e}")

    print("Processing complete.")
    
    # Cleanup checkpoint on success
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            os.remove(checkpoint_path)
            print(f"Checkpoint file {checkpoint_path} deleted.")
        except Exception as e:
            print(f"Warning: Could not delete checkpoint file: {e}")


def reduce_and_reindex(pca_model_path, source_similarity_space, target_similarity_space, normalize_output=True,
                       batch_size=100, checkpoint_path=None):
    """Executes the full pipeline: load model, transform vectors, and re-index.

    Args:
        pca_model_path (str): Path to the pickle file containing the PCA model.
        source_similarity_space (str): Source space to read vectors from.
        target_similarity_space (str): Target space name for the new vectors.
        normalize_output (bool): Whether to normalize vectors after transformation.
        batch_size (int): Batch size for Solr updates.
        checkpoint_path (str): Path to checkpoint file.
    """
    pca = load_pca_model(pca_model_path)

    # Ensure Solr has the required field for the reduced dimensions
    ensure_solr_field_exists(pca.n_components_)

    # Load ALL original vectors into memory to perform PCA transform.
    vectors, sound_ids, child_docs = load_vectors_from_solr(source_similarity_space)

    print("Transforming vectors...")
    reduced = transform_vectors(pca, vectors, normalize_output)

    add_pca_child_documents(sound_ids, reduced, child_docs, target_similarity_space, batch_size=batch_size,
                            checkpoint_path=checkpoint_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PCA dimensionality reduction for similarity vectors')
    parser.add_argument('--fit', action='store_true', help='Fit PCA model')
    parser.add_argument('--reindex', action='store_true', help='Transform and add child documents to Solr')
    parser.add_argument('--dims', type=int, default=128, help='Target dimensions')
    parser.add_argument('--source-space', default='laion_clap', help='Source similarity space')
    parser.add_argument('--target-space', default=None, help='Target similarity space name')
    parser.add_argument('--model-path', default=None, help='Path to PCA model file')
    parser.add_argument('--checkpoint', default=None,
                        help='Path to checkpoint file (default: pca_checkpoint.json if reindexing)')
    parser.add_argument('--batch-size', type=int, default=100, help='Batch size for indexing')

    args = parser.parse_args()

    # Set default model path if not provided
    if args.model_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        args.model_path = os.path.join(base_dir, "models", f"pca_{args.dims}.pkl")

    if args.fit:
        fit_and_save_pca(args.source_space, args.dims, args.model_path)

    if args.reindex:
        target = args.target_space or f"{args.source_space}_pca{args.dims}"
        # Use a unique checkpoint file per target space in the models directory
        if args.checkpoint:
            ckpt = args.checkpoint
        else:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ckpt = os.path.join(base_dir, "models", f"pca_checkpoint_{target}.json")
        
        reduce_and_reindex(args.model_path, args.source_space, target, batch_size=args.batch_size, checkpoint_path=ckpt)
