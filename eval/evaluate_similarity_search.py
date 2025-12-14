"""
Evaluate similarity search performance for different similarity spaces.
"""
import sys
import os
import numpy as np
import pandas as pd
import time
import pickle
import pysolr
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'search'))
from configs import SOLR_URL

NUM_SOUNDS_FOR_EVAL = 2000
N_NEIGHBORS = 50

class SearchEvaluationResult:
	def __init__(self, configs, target_sound_ids, query_times_ms, retrieved_neighbors, metrics):
		self.configs = configs
		self.target_sound_ids = target_sound_ids
		self.timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
		self.query_times_ms = query_times_ms
		self.retrieved_neighbors = retrieved_neighbors
		self.metrics = metrics

	def save(self, filepath):
		with open(filepath, 'wb') as f:
			pickle.dump(self, f)

	@staticmethod
	def load(filepath):
		with open(filepath, 'rb') as f:
			return pickle.load(f)

	def to_csv(self, filepath="eval_results.csv", append=True):
		"""Save configs + metrics as a single row to CSV for comparison."""
		row = {
			'timestamp': self.timestamp,
			**self.configs,
			**self.metrics
		}

		df = pd.DataFrame([row])

		if append and os.path.exists(filepath):
			df.to_csv(filepath, mode='a', header=False, index=False)
		else:
			df.to_csv(filepath, index=False)

def similarity_search_solr(query_vector, similarity_space, k, solr_url=SOLR_URL):
	"""
	Perform KNN vector search in Solr.

	Args:
		query_vector (list): Query vector
		similarity_space (str): Similarity space name
		k (int): Number of neighbors to retrieve

	Returns:
		tuple: (results object with .docs and .q_time, query_time_ms)
	"""
	solr = pysolr.Solr(solr_url, always_commit=False)

	# Find the vector field by sampling a document from this similarity space
	sample_query = f'content_type:v AND similarity_space:{similarity_space}'
	sample_results = solr.search(sample_query, rows=1, fl='*')

	if len(sample_results.docs) == 0:
		raise Exception(f"No vectors found for similarity_space={similarity_space}")

	vector_field = None
	for key in sample_results.docs[0].keys():
		if key.startswith('sim_vector') and key.endswith('_l2'):
			vector_field = key
			break

	if not vector_field:
		raise Exception(f"Could not find vector field for similarity_space={similarity_space}")

	# Build KNN query: {!knn f=FIELD topK=K}[vector]
	fq = f'content_type:v AND similarity_space:{similarity_space}'
	vector_str = ','.join([str(v) for v in query_vector])
	knn_query = f'{{!knn f={vector_field} topK={k+1}}}[{vector_str}]'

	start = time.time()
	results = solr.search(knn_query, fq=fq, rows=k+1, fl='id,score,_root_')
	query_time = (time.time() - start) * 1000

	if hasattr(results, 'qtime'):
		query_time = results.qtime

	return results, query_time

def evaluate_similarity_search(target_sound_vecs, target_sound_ids, ground_truth_results=None, similarity_space='laion_clap', k=N_NEIGHBORS, results_csv="eval/results/eval_results.csv", save_to_csv=True, warmup=0, solr_url=SOLR_URL, seed=42):
	"""
	Evaluate similarity search latency and recall.

	Args:
		target_sound_vecs (np.ndarray): Array of query vectors (N, D)
		target_sound_ids (list): Sound IDs corresponding to target_sound_vecs
		ground_truth_results (list, optional): Ground truth nearest neighbors for recall calculation
		similarity_space (str): Similarity space to evaluate
		k (int): Number of nearest neighbors to retrieve
		results_csv (str): CSV file to save results
		save_to_csv (bool): Whether to save results to CSV
		warmup (int): Number of warmup queries before measurement
		solr_url (str): Solr collection URL

	Returns:
		SearchEvaluationResult: Object containing configs, retrieved neighbors, and metrics
	"""
	# Fetch index stats
	try:
		stat_solr = pysolr.Solr(solr_url)
		index_size = stat_solr.search('*:*', rows=0).hits
		index_num_sounds = stat_solr.search('content_type:s', rows=0).hits
	except Exception as e:
		print(f"Warning: Failed to fetch index stats: {e}")
		index_size = -1
		index_num_sounds = -1

	query_times = []
	retrieved_neighbors = []
	recalls = []

	if warmup > 0:
		print(f"Running {warmup} warmup queries...")
		for _ in range(warmup):
			similarity_search_solr(target_sound_vecs[0].tolist(), similarity_space, k, solr_url)

	print(f"Querying Solr with {len(target_sound_vecs)} target sounds...")

	empty_result_count = 0
	for i, query_vec in enumerate(tqdm(target_sound_vecs, desc="Running Solr queries")):
		results, query_time = similarity_search_solr(query_vec.tolist(), similarity_space, k, solr_url)

		# Extract parent sound IDs from child documents (_root_ field points to parent)
		query_sound_id = target_sound_ids[i]
		neighbors = []
		for doc in results.docs:
			if '_root_' in doc:
				parent_id = int(doc['_root_'])
				if parent_id != query_sound_id:
					neighbors.append(parent_id)
				if len(neighbors) >= k:
					break

		if len(neighbors) == 0:
			empty_result_count += 1
			if empty_result_count <= 3:
				print(f"\nWarning: Query {i} (sound_id={query_sound_id}) returned 0 results")

		query_times.append(query_time)
		retrieved_neighbors.append(neighbors)

		if ground_truth_results and i < len(ground_truth_results):
			gt_neighbors = [sid for sid in ground_truth_results[i] if sid != query_sound_id][:k]
			gt_set = set(gt_neighbors)
			retrieved_set = set(neighbors[:k])
			recall = len(gt_set & retrieved_set) / k if k > 0 else 0.0
			recalls.append(recall)

	if empty_result_count > 0:
		print(f"\n\nWarning: {empty_result_count}/{len(target_sound_vecs)} queries returned 0 results")
		print(f"This will negatively impact recall metrics.")

	performance_metrics = {
		'latency_p50': np.percentile(query_times, 50),
		'latency_p95': np.percentile(query_times, 95),
		'latency_p99': np.percentile(query_times, 99),
		'mean_latency': np.mean(query_times),
		'qps': 1000 / np.mean(query_times),
		'recall_mean': np.mean(recalls) if recalls else 0.0,
		'recall_std': np.std(recalls) if recalls else 0.0,
		'empty_results': empty_result_count
	}

	search_evaluation_result = SearchEvaluationResult(
		configs={
			'similarity_space': similarity_space,
			'n_neighbors': k,
			'query_size': len(target_sound_ids),
			'index_num_docs': index_size,
			'index_num_sounds': index_num_sounds,
			'random_seed': seed,
			'warmup': warmup,
		},
		target_sound_ids=target_sound_ids,
		query_times_ms=query_times,
		retrieved_neighbors=retrieved_neighbors,
		metrics=performance_metrics
	)

	if save_to_csv:
		search_evaluation_result.to_csv(filepath=results_csv, append=True)
		print(f"Saved results to {results_csv}.")

	print("Performance metrics:")
	df = pd.DataFrame([performance_metrics])
	print(df.to_string(index=False))

	return search_evaluation_result

def load_target_sounds(similarity_space, num_sounds=NUM_SOUNDS_FOR_EVAL, specific_sound_ids=None, solr_url=SOLR_URL):
	"""
	Load sound vectors from Solr for evaluation.

	Args:
		similarity_space (str): Similarity space name
		num_sounds (int): Number of sounds to load
		specific_sound_ids (list, optional): Specific sound IDs to load
		solr_url (str): Solr collection URL

	Returns:
		tuple: (vectors array, sound_ids list)
	"""
	solr = pysolr.Solr(solr_url, always_commit=False)

	print(f"Loading vectors from '{similarity_space}'...")
	query = f'content_type:v AND similarity_space:{similarity_space}'

	if specific_sound_ids is not None:
		# Load all vectors and filter to specific IDs (avoids complex query)
		results = solr.search(query, rows=100000, fl='*')
	else:
		results = solr.search(query, rows=num_sounds, fl='*')

	vectors = []
	sound_ids = []

	if len(results.docs) == 0:
		raise Exception(f"No documents found for similarity_space={similarity_space}. Query returned 0 results.")

	vector_field = None
	for key in results.docs[0].keys():
		if key.startswith('sim_vector') and key.endswith('_l2'):
			vector_field = key
			break

	if not vector_field:
		raise Exception(f"Could not find vector field for similarity_space={similarity_space}. Available fields: {list(results.docs[0].keys())}")

	# Build lookup if filtering to specific IDs
	specific_ids_set = set(specific_sound_ids) if specific_sound_ids else None

	for doc in tqdm(results.docs, desc=f"Loading {similarity_space} vectors"):
		if vector_field in doc and '_root_' in doc:
			sound_id = int(doc['_root_'])
			# Filter to specific IDs if requested
			if specific_ids_set is None or sound_id in specific_ids_set:
				vectors.append(doc[vector_field])
				sound_ids.append(sound_id)

	if len(vectors) == 0:
		raise Exception(f"No vectors found for similarity_space={similarity_space}")

	# If specific IDs were requested, ensure we maintain the same order
	if specific_sound_ids is not None:
		# Create a dict for fast lookup
		vec_dict = {sid: vec for sid, vec in zip(sound_ids, vectors)}
		# Reorder to match specific_sound_ids
		ordered_vectors = []
		ordered_ids = []
		for sid in specific_sound_ids:
			if sid in vec_dict:
				ordered_vectors.append(vec_dict[sid])
				ordered_ids.append(sid)
		vectors = ordered_vectors
		sound_ids = ordered_ids

	print(f"Loaded {len(vectors)} vectors")

	return np.array(vectors, dtype=np.float32), sound_ids

if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(description='Evaluate similarity search performance')
	parser.add_argument('--space', required=True, help='Similarity space to evaluate')
	parser.add_argument('--ground-truth-space', default="laion_clap", help='Similarity space for ground truth')
	parser.add_argument('--ground-truth-file', default=None, help='Load ground truth from pickle file')
	parser.add_argument('--save-ground-truth', default=None, help='Save ground truth to pickle file')
	parser.add_argument('--num-sounds', type=int, default=NUM_SOUNDS_FOR_EVAL, help=f'Number of query sounds (default: {NUM_SOUNDS_FOR_EVAL})')
	parser.add_argument('--k', type=int, default=N_NEIGHBORS, help=f'Number of neighbors (default: {N_NEIGHBORS})')
	parser.add_argument('--results-csv', default='eval/results/eval_results.csv', help='CSV file to save results')
	parser.add_argument('--warmup', type=int, default=0, help='Number of warmup queries (default: 0)')
	parser.add_argument('--clear-cache', action='store_true', help='Reload collection to clear cache before running')
	parser.add_argument('--seed', type=int, default=42, help='Random seed for query selection')

	args = parser.parse_args()
	
	# Set random seed
	np.random.seed(args.seed)

	if args.clear_cache:
		print("Reloading collection to clear cache...")
		from configs import SOLR_BASE_URL, COLLECTION_NAME
		from solrapi import SolrManagementAPI
		api = SolrManagementAPI(SOLR_BASE_URL, COLLECTION_NAME)
		api.reload_collection()
		# Add a small delay to ensure reload propagates?
		time.sleep(2)

	if args.ground_truth_file and not args.ground_truth_file.startswith('/'):
		if not args.ground_truth_file.startswith('eval/'):
			args.ground_truth_file = f"eval/results/{args.ground_truth_file}"

	if args.save_ground_truth and not args.save_ground_truth.startswith('/'):
		if not args.save_ground_truth.startswith('eval/'):
			args.save_ground_truth = f"eval/results/{args.save_ground_truth}"

	ground_truth = None
	target_ids = None

	if args.ground_truth_file:
		print(f"Loading ground truth from {args.ground_truth_file}...")
		gt_result = SearchEvaluationResult.load(args.ground_truth_file)
		ground_truth = gt_result.retrieved_neighbors
		target_ids = gt_result.target_sound_ids
	elif args.ground_truth_space:
		print(f"Computing ground truth from '{args.ground_truth_space}'...")
		gt_vecs, target_ids = load_target_sounds(args.ground_truth_space, args.num_sounds)
		gt_result = evaluate_similarity_search(
			gt_vecs,
			target_ids,
			ground_truth_results=None,
			similarity_space=args.ground_truth_space,
			k=args.k,
			results_csv=args.results_csv,
			save_to_csv=False,
			warmup=args.warmup
		)
		ground_truth = gt_result.retrieved_neighbors
		
		# If we are evaluating the ground truth space itself (baseline),
		# use these results directly instead of running again (which would hit cache).
		if args.space == args.ground_truth_space:
			print(f"Target space '{args.space}' matches ground truth space.")
			print("Using ground truth generation run as evaluation result (Recall=1.0).")
			
			gt_result.metrics['recall_mean'] = 1.0
			gt_result.metrics['recall_std'] = 0.0
			
			# Print updated metrics
			print("Performance metrics:")
			df = pd.DataFrame([gt_result.metrics])
			print(df.to_string(index=False))
			
			gt_result.to_csv(filepath=args.results_csv, append=True)
			print(f"Saved results to {args.results_csv}.")
			
			if args.save_ground_truth:
				gt_result.save(args.save_ground_truth)
				print(f"Saved ground truth to {args.save_ground_truth}")
				
			sys.exit(0)

		if args.save_ground_truth:
			gt_result.save(args.save_ground_truth)
			print(f"Saved ground truth to {args.save_ground_truth}")

	target_vecs, target_ids = load_target_sounds(
		args.space,
		args.num_sounds,
		specific_sound_ids=target_ids
	)

	evaluate_similarity_search(
		target_vecs,
		target_ids,
		ground_truth_results=ground_truth,
		similarity_space=args.space,
		k=args.k,
		results_csv=args.results_csv,
		save_to_csv=True,
		warmup=args.warmup,
		seed=args.seed
	)

