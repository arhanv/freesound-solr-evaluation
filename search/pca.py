"""
Reduce dimensionality of sound similarity vectors using PCA and re-index Solr with new similarity spaces.
"""
import argparse
import os
import numpy as np
import pickle
import pysolr
import requests
import json
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from tqdm import tqdm
from configs import SOLR_URL, SOLR_BASE_URL, COLLECTION_NAME
from solrapi import SolrManagementAPI

def ensure_solr_field_exists(dimensions, solr_base_url=SOLR_BASE_URL, collection_name=COLLECTION_NAME):
	"""
	Ensure Solr has the required field type and field for the given dimensions.

	Args:
		dimensions (int): Vector dimensions (e.g., 128)
		solr_base_url (str): Solr base URL
		collection_name (str): Solr collection name
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
	"""
	Load similarity vectors from Solr by querying child documents.

	Args:
		similarity_space (str): Similarity space name (e.g., 'laion_clap')
		solr_url (str): Solr collection URL
		max_vectors (int, optional): Max vectors to load (None = all)

	Returns:
		tuple: (vectors array, sound_ids list, child_docs list)
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
		# Check limit
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
			# Find the vector field (sim_vector512_l2, sim_vector100_l2, etc.)
			if vector_field is None:
				for key in doc.keys():
					if key.startswith('sim_vector') and key.endswith('_l2'):
						vector_field = key
						break

			if vector_field and vector_field in doc:
				vectors.append(doc[vector_field])
				# Extract parent sound ID from child doc ID (format: {sound_id}_{similarity_space})
				child_id = doc['id']
				# Robust ID parsing
				try:
					sound_id = int(child_id.split('_')[0])
				except:
					continue # Skip malformed ID
					
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
	"""
	Fit PCA model on vectors.

	Args:
		vectors (np.ndarray): Input vectors (N, D)
		n_components (int): Target dimensions

	Returns:
		PCA: Fitted PCA model
	"""
	print(f"Fitting PCA to {n_components} components...")
	pca = PCA(n_components=n_components)
	pca.fit(vectors)

	explained_var = pca.explained_variance_ratio_.sum()
	print(f"Explained variance: {explained_var:.4f}")

	return pca

def transform_vectors(pca, vectors, normalize_output=True):
	"""Transform vectors using fitted PCA.

	Args:
		pca (PCA): Fitted PCA model
		vectors (np.ndarray): Input vectors
		normalize_output (bool): L2-normalize after transformation

	Returns:
		np.ndarray: Transformed vectors
	"""
	transformed = pca.transform(vectors).astype(np.float32)

	if normalize_output:
		transformed = normalize(transformed, norm='l2')

	return transformed

def save_pca_model(pca, filepath):
	"""Save PCA model to disk."""
	with open(filepath, 'wb') as f:
		pickle.dump(pca, f)
	print(f"Saved PCA model to {filepath}")

def load_pca_model(filepath):
	"""Load PCA model from disk."""
	with open(filepath, 'rb') as f:
		return pickle.load(f)

def fit_and_save_pca(source_space, n_components, output_model_path, max_vectors=None):
	"""
	Fit PCA on vectors from source_space and save model.

	Args:
		source_space (str): Source similarity space
		n_components (int): Number of target dimensions for PCA model
		output_model_path (str): Path to save PCA model
		max_vectors (int, optional): Maximum number of vectors to include

	Returns:
		sklearn.decomposition.PCA: Fitted PCA model
	"""
	vectors, _, _ = load_vectors_from_solr(source_space, max_vectors=max_vectors)
	pca = fit_pca(vectors, n_components)
	save_pca_model(pca, output_model_path)
	return pca

def load_checkpoint(filepath):
	"""Load set of processed parent IDs from checkpoint file."""
	if filepath and os.path.exists(filepath):
		try:
			with open(filepath, 'r') as f:
				data = json.load(f)
				return set(data.get('processed_ids', []))
		except Exception as e:
			print(f"Warning: Failed to load checkpoint {filepath}: {e}")
	return set()

def save_checkpoint(filepath, new_ids):
	"""Append new processed IDs to checkpoint file."""
	if not filepath:
		return
	
	processed_ids = set()
	if os.path.exists(filepath):
		try:
			with open(filepath, 'r') as f:
				data = json.load(f)
				processed_ids = set(data.get('processed_ids', []))
		except:
			pass
	
	processed_ids.update(new_ids)
	
	with open(filepath, 'w') as f:
		json.dump({'processed_ids': list(processed_ids)}, f)

def add_pca_child_documents(sound_ids, reduced_vectors, original_child_docs, target_similarity_space, solr_url=SOLR_URL, batch_size=100, checkpoint_path=None):
	"""
	Add PCA-reduced child documents to Solr with batching and checkpointing.

	Args:
		sound_ids (list): List of parent sound IDs
		reduced_vectors (np.ndarray): Array of reduced vectors
		original_child_docs (list): Original child documents
		target_similarity_space (str): Name of the new similarity space
		solr_url (str): Solr collection URL
		batch_size (int): Number of PARENTS to process per batch (chunk size)
		checkpoint_path (str): Path to checkpoint file
	"""
	solr = pysolr.Solr(solr_url, always_commit=False, timeout=300)
	
	# Group data by parent ID first
	print("Grouping data by parent ID...")
	parent_data_map = {} # parent_id -> list of (vector, original_child_doc)
	
	for sid, vec, orig_doc in zip(sound_ids, reduced_vectors, original_child_docs):
		sid_str = str(sid)
		if sid_str not in parent_data_map:
			parent_data_map[sid_str] = []
		parent_data_map[sid_str].append((vec, orig_doc))

	# Load checkpoint
	processed_parents = load_checkpoint(checkpoint_path)
	if processed_parents:
		print(f"Loaded checkpoint: {len(processed_parents)} parents already processed.")

	unique_parent_ids = sorted(list(parent_data_map.keys()))
	parents_to_process = [pid for pid in unique_parent_ids if pid not in processed_parents]
	
	print(f"Total parents: {len(unique_parent_ids)}. Parents to process: {len(parents_to_process)}")
	
	if not parents_to_process:
		print("All parents processed!")
		return

	# Determine vector field name based on dimensionality
	n_dims = reduced_vectors.shape[1]
	vector_field = f"sim_vector{n_dims}_l2"
	
	# FIELDS_TO_EXCLUDE from fetched parent docs
	FIELDS_TO_EXCLUDE = ['_version_', 'type_facet', 'username_facet', 'tagfacet', 'created_range', 'timestamp_start', 'timestamp_end'] 

	# Process in chunks (batches)
	# batch_size here matches the 'chunk_size' logic we had before
	chunk_size = batch_size # Renaming for clarity
	
	print(f"Processing in chunks of {chunk_size} parents...")
	
	for i in tqdm(range(0, len(parents_to_process), chunk_size), desc="Processing & Indexing"):
		chunk_ids = parents_to_process[i:i+chunk_size]
		
		# --- 1. Fetch Parents ---
		id_query = " OR ".join(chunk_ids)
		q = f"content_type:s AND id:({id_query})"
		
		try:
			results = solr.search(q, rows=len(chunk_ids), fl='*')
		except Exception as e:
			print(f"\nError fetching parents for chunk {i}: {e}")
			continue

		parent_map = {}
		for doc in results:
			# Clean up valid parent doc
			for field in FIELDS_TO_EXCLUDE:
				if field in doc:
					del doc[field]
			parent_map[str(doc['id'])] = doc
			
		# --- 2. Fetch Existing Children ---
		child_query = f"content_type:v AND _root_:({' OR '.join(chunk_ids)})"
		try:
			child_results = solr.search(child_query, rows=len(chunk_ids) * 20, fl='*')
		except Exception as e:
			print(f"\nError fetching children for chunk {i}: {e}")
			# We proceed? No, risky. 
			# In a robust script we might skip this chunk, but keeping it simple.
			continue

		parent_children_map = {}
		for child in child_results:
			root_id = str(child.get('_root_'))
			if root_id not in parent_children_map:
				parent_children_map[root_id] = []
			
			if '_version_' in child: del child['_version_']
			parent_children_map[root_id].append(child)

		# --- 3. Build Updates ---
		docs_to_index = []
		
		for pid in chunk_ids:
			if pid not in parent_map:
				print(f"Warning: Parent {pid} not found in Solr. Skipping.")
				continue
				
			parent_doc = parent_map[pid]
			new_data_list = parent_data_map[pid] # list of (vec, orig_doc)
			
			# Get existing children
			existing_children = parent_children_map.get(pid, [])
			
			# Filter out OLD documents of the target space (replacing them)
			final_children = [
				c for c in existing_children 
				if c.get('similarity_space') != target_similarity_space
			]
			
			# Create NEW children
			for vec, orig_doc in new_data_list:
				child_doc = {
					'id': f"{pid}_{target_similarity_space}", # Note: assuming 1 vec per space per parent usually? 
					# If multiple vectors per space (unlikely for PCA), ID conflict? 
					# Typically 1 vector per space. 
					'content_type': 'v',
					'similarity_space': target_similarity_space,
					'timestamp_start': orig_doc.get('timestamp_start', 0),
					'timestamp_end': orig_doc.get('timestamp_end', -1),
					vector_field: vec.tolist(),
				}
				if 'pack_grouping_child' in orig_doc:
					child_doc['pack_grouping_child'] = orig_doc['pack_grouping_child']
				
				final_children.append(child_doc)
			
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

def reduce_and_reindex(pca_model_path, source_similarity_space, target_similarity_space, normalize_output=True, batch_size=100, checkpoint_path=None):
	"""
	Reduce dimensionality and add to Solr.
	"""
	pca = load_pca_model(pca_model_path)

	# Ensure Solr has the required field for the reduced dimensions
	ensure_solr_field_exists(pca.n_components_)

	# Note: We still load ALL original vectors into memory to perform PCA transform.
	# If this is too large for RAM (Memory Error), we would need to batch load_vectors too.
	# Assuming 50k - 500k vectors fits in RAM (500k * 512 floats * 4 bytes ~= 1GB). Should be fine.
	vectors, sound_ids, child_docs = load_vectors_from_solr(source_similarity_space)
	
	print("Transforming vectors...")
	reduced = transform_vectors(pca, vectors, normalize_output)
	
	add_pca_child_documents(sound_ids, reduced, child_docs, target_similarity_space, batch_size=batch_size, checkpoint_path=checkpoint_path)

# CLI interface
if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='PCA dimensionality reduction for similarity vectors')
	parser.add_argument('--fit', action='store_true', help='Fit PCA model')
	parser.add_argument('--reindex', action='store_true', help='Transform and add child documents to Solr')
	parser.add_argument('--dims', type=int, default=128, help='Target dimensions')
	parser.add_argument('--source-space', default='laion_clap', help='Source similarity space')
	parser.add_argument('--target-space', default=None, help='Target similarity space name')
	parser.add_argument('--model-path', default=None, help='Path to PCA model file')
	parser.add_argument('--checkpoint', default=None, help='Path to checkpoint file (default: pca_checkpoint.json if reindexing)')
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
		ckpt = args.checkpoint if args.checkpoint else "pca_checkpoint.json"
		reduce_and_reindex(args.model_path, args.source_space, target, batch_size=args.batch_size, checkpoint_path=ckpt)
