"""
Evaluate similarity search performance for different similarity spaces.
"""
import argparse
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import pysolr
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'search'))
from configs import SOLR_URL, SOLR_BASE_URL, COLLECTION_NAME
from stats_utils import calculate_space_size_mb

NUM_SOUNDS_FOR_EVAL = 2000
DEFAULT_RETRIEVE_N = 50
DEFAULT_METRIC_K = 50

class SearchEvaluationResult:
    """Stores and serializes evaluation results."""

    def __init__(self, configs, target_sound_ids, query_times_ms, retrieved_neighbors, metrics):
        """Initializes the SearchEvaluationResult object.

        Args:
            configs (dict): Configuration parameters for the evaluation run.
            target_sound_ids (list): List of sound IDs used as queries.
            query_times_ms (list): List of query latencies in milliseconds.
            retrieved_neighbors (list): List of lists, where each inner list contains
                                        the IDs of retrieved neighbors for a query.
            metrics (dict): Dictionary of calculated performance metrics.
        """
        self.configs = configs
        self.target_sound_ids = target_sound_ids
        self.timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        self.query_times_ms = query_times_ms
        self.retrieved_neighbors = retrieved_neighbors
        self.metrics = metrics

    def save(self, filepath):
        """Save results object to a pickle file.

        Args:
            filepath (str): Path to the file where the object will be saved.
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath):
        """Load results object from a pickle file.

        Args:
            filepath (str): Path to the pickle file to load.

        Returns:
            SearchEvaluationResult: The loaded SearchEvaluationResult object.
        """
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    def to_csv(self, filepath="eval_results.csv", append=True):
        """Save configs + metrics as a single row to CSV for comparison.

        Args:
            filepath (str): Path to the CSV file.
            append (bool): If True, append to the file; otherwise, overwrite.
        """
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
    """Perform KNN vector search in Solr.

    Args:
        query_vector (list): Query vector.
        similarity_space (str): Similarity space name.
        k (int): Number of neighbors to retrieve (retrieve_n).
        solr_url (str): Solr collection URL.

    Returns:
        tuple: (results object, query_time_ms)
            results object: The Solr search results object with .docs and .q_time.
            query_time_ms (float): The time taken for the Solr query in milliseconds.
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
    # Request k+1 to account for the query sound itself usually being returned
    knn_query = f'{{!knn f={vector_field} topK={k+1}}}[{vector_str}]'

    start = time.time()
    results = solr.search(knn_query, fq=fq, rows=k+1, fl='id,score,_root_')
    query_time = (time.time() - start) * 1000

    if hasattr(results, 'qtime'):
        query_time = results.qtime

    return results, query_time

def calculate_ndcg_weighted(retrieved, ground_truth, k=50):
    """
    Calculates NDCG@K using the "true" rank as the relevance score.
    Items ranked higher by the original embeddings will have higher relevance scores.
    Essentially, the top item in ground_truth has relevance k, the second k-1, ..., the k-th item has relevance 1.

    Args:
        retrieved (list): List of retrieved item IDs.
        ground_truth (list): List of ground truth item IDs.
        k (int): Number of top items to consider for NDCG calculation.
    Returns:
        float: Weighted NDCG@K score.
    """
    # Map ground truth items to their relevance scores (k, k-1, k-2... 1)
    relevance_map = {item: (k - idx) for idx, item in enumerate(ground_truth[:k])}

    # Calculate Actual DCG
    dcg = 0.0
    for i, item in enumerate(retrieved[:k]):
        rel = relevance_map.get(item, 0)
        if rel > 0:
            # Formula: rel_i / log2(rank + 1)
            dcg += rel / np.log2(i + 2)
    
    # Calculate Ideal DCG (if the items were returned in perfect 1, 2, 3... order)
    idcg = sum((k - i) / np.log2(i + 2) for i in range(k))
    
    return dcg / idcg if idcg > 0 else 0.0

def evaluate_similarity_search(target_sound_vecs, target_sound_ids, ground_truth_results=None,
                               similarity_space='laion_clap', retrieve_n=DEFAULT_RETRIEVE_N, metric_k=DEFAULT_METRIC_K, output_dir=None,
                               warmup=0, solr_url=SOLR_URL, seed=42, save_details=False):
    """Evaluate similarity search latency and recall.

    Args:
        target_sound_vecs (np.ndarray): Array of query vectors (N, D).
        target_sound_ids (list): Sound IDs corresponding to target_sound_vecs.
        ground_truth_results (list, optional): Ground truth nearest neighbors for recall calculation.
                                                Each element is a list of ground truth neighbor IDs.
        similarity_space (str): Similarity space to evaluate.
        retrieve_n (int): Number of nearest neighbors to retrieve from Solr.
        metric_k (int): Number of neighbors to consider for recall/nDCG metrics.
        output_dir (str, optional): Directory to save results (results.csv, detailed metrics).
                                    If None, results are not saved to disk.
        warmup (int): Number of warmup queries before measurement.
        solr_url (str): Solr collection URL.
        seed (int): Random seed for reproducibility.
        save_details (bool): Whether to save detailed per-query metrics to a pickled dataframe.

    Returns:
        SearchEvaluationResult: Object containing configs, retrieved neighbors, and metrics.
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
    ndcgs = []
    warmup_times = []

    if warmup > 0:
        print(f"Running {warmup} warmup queries (sampled from target set)...")
        # Sample random indices for warmup to avoid caching effects of a single vector
        warmup_indices = np.random.choice(len(target_sound_vecs), size=warmup, replace=True)
        for idx in tqdm(warmup_indices, desc="Warmup"):
             _, w_time = similarity_search_solr(target_sound_vecs[idx].tolist(), similarity_space, retrieve_n, solr_url)
             warmup_times.append(w_time)

    print(f"Querying Solr with {len(target_sound_vecs)} target sounds...")

    empty_result_count = 0
    for i, query_vec in enumerate(tqdm(target_sound_vecs, desc="Running Solr queries")):
        results, query_time = similarity_search_solr(query_vec.tolist(), similarity_space, retrieve_n, solr_url)

        # Extract parent sound IDs from child documents (_root_ field points to parent)
        query_sound_id = target_sound_ids[i]
        neighbors = []
        for doc in results.docs:
            if '_root_' in doc:
                parent_id = int(doc['_root_'])
                if parent_id != query_sound_id:
                    neighbors.append(parent_id)
                if len(neighbors) >= max(retrieve_n, metric_k):
                    break

        if not neighbors:
            empty_result_count += 1
            if empty_result_count <= 3:
                print(f"\nWarning: Query {i} (sound_id={query_sound_id}) returned 0 results")

        query_times.append(query_time)
        retrieved_neighbors.append(neighbors)

        if ground_truth_results and i < len(ground_truth_results):
            gt_neighbors = [sid for sid in ground_truth_results[i] if sid != query_sound_id][:metric_k]
            gt_set = set(gt_neighbors)
            retrieved_set = set(neighbors[:metric_k])
            # Recall Calculation (Set-based)
            recall = len(gt_set & retrieved_set) / metric_k if metric_k > 0 else 0.0
            recalls.append(recall)
            # NDCG Calculation (Order-sensitive)
            ndcg_val = calculate_ndcg_weighted(neighbors, gt_neighbors, metric_k)
            ndcgs.append(ndcg_val)

    if empty_result_count > 0:
        print(f"\n\nWarning: {empty_result_count}/{len(target_sound_vecs)} queries returned 0 results")

    mean_latency = np.mean(query_times)
    warmup_mean = np.mean(warmup_times) if warmup_times else 0.0
    warmup_penalty_pct = ((warmup_mean - mean_latency) / mean_latency * 100) if mean_latency > 0 and warmup_times else 0.0

    # Calculate Space Size
    try:
        count_query = f'content_type:v AND similarity_space:{similarity_space}'
        total_vectors = pysolr.Solr(solr_url).search(count_query, rows=0).hits
        space_size_mb = calculate_space_size_mb(total_vectors, target_sound_vecs.shape[1])
    except Exception as e:
        print(f"Warning: Failed to calculate space size: {e}")
        space_size_mb = 0.0

    performance_metrics = {
        'latency_p50': np.percentile(query_times, 50),
        'latency_p95': np.percentile(query_times, 95),
        'latency_p99': np.percentile(query_times, 99),
        'mean_latency': mean_latency,
        'qps': 1000.0 / mean_latency if mean_latency > 0 else 0.0,
        'recall_mean': np.mean(recalls) if recalls else 0.0,
        'recall_std': np.std(recalls) if recalls else 0.0,
        'ndcg_mean': np.mean(ndcgs) if ndcgs else 0.0,
        'ndcg_std': np.std(ndcgs) if ndcgs else 0.0,
        'empty_results': empty_result_count,
        'warmup_mean': warmup_mean,
        'warmup_penalty_pct': warmup_penalty_pct,
        'space_size_mb': space_size_mb
    }

    result = SearchEvaluationResult(
        configs={
            'similarity_space': similarity_space,
            'retrieve_n': retrieve_n,
            'metric_k': metric_k,
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

    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save metrics to CSV
        result.to_csv(os.path.join(output_dir, "results.csv"))
        print(f"Saved summary results to {os.path.join(output_dir, 'results.csv')}.")
        
        # Save details if requested
        if save_details:
            details_data = []
            for i in range(len(target_sound_ids)):
                details_data.append({
                    'query_index': i,
                    'sound_id': target_sound_ids[i],
                    'latency_ms': query_times[i],
                    'recall': recalls[i] if i < len(recalls) else None,
                    'similarity_space': similarity_space,
                    'seed': seed
                })
            details_df = pd.DataFrame(details_data)
            details_path = os.path.join(output_dir, f"per_query_details_{similarity_space}.pkl")
            details_df.to_pickle(details_path)
            print(f"Saved detailed query results to {details_path}.")

    print("Performance metrics:")
    print(pd.DataFrame([performance_metrics]).to_string(index=False))

    return result

def load_target_sounds(similarity_space, num_sounds=NUM_SOUNDS_FOR_EVAL, specific_sound_ids=None, solr_url=SOLR_URL):
    """Load sound vectors from Solr for evaluation.

    Args:
        similarity_space (str): Similarity space name.
        num_sounds (int): Number of sounds to load.
        specific_sound_ids (list, optional): Specific sound IDs to load. If provided,
                                              only vectors for these IDs will be loaded
                                              and returned in the specified order.
        solr_url (str): Solr collection URL.

    Returns:
        tuple: (vectors array, sound_ids list)
            vectors (np.ndarray): Array of loaded vectors (N, D).
            sound_ids (list): List of sound IDs corresponding to the vectors.
    """
    solr = pysolr.Solr(solr_url, always_commit=False)

    print(f"Loading vectors from '{similarity_space}'...")
    query = f'content_type:v AND similarity_space:{similarity_space}'

    results_docs = []
    if specific_sound_ids is not None:
        # Load specific IDs using batching to avoid URL length limits
        # and to ensure we find exactly the requested IDs
        batch_size = 500
        all_ids = list(set(specific_sound_ids)) # Unique IDs
        
        for i in range(0, len(all_ids), batch_size):
            batch = all_ids[i:i + batch_size]
            # specific_sound_ids are ints, _root_ is string in Solr
            # Construct boolean OR query for _root_
            id_query = " OR ".join([str(sid) for sid in batch])
            fq = f"_root_:({id_query})"
            
            try:
                # We expect at most 'batch_size' results per batch
                res = solr.search(query, fq=fq, rows=batch_size, fl='*,_root_')
                results_docs.extend(res.docs)
            except Exception as e:
                print(f"Error loading batch {i}-{i+batch_size}: {e}")
                
        # Mock a results object with combined docs
        class MockResults:
            def __init__(self, docs):
                self.docs = docs
        results = MockResults(results_docs)
            
    else:
        results = solr.search(query, rows=num_sounds, fl='*,_root_')

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

    specific_ids_set = set(specific_sound_ids) if specific_sound_ids else None

    for doc in tqdm(results.docs, desc=f"Loading {similarity_space} vectors"):
        if vector_field in doc and '_root_' in doc:
            sound_id = int(doc['_root_'])
            # Filter to specific IDs if requested
            if specific_ids_set is None or sound_id in specific_ids_set:
                vectors.append(doc[vector_field])
                sound_ids.append(sound_id)

    if len(vectors) == 0:
        # Debugging aid
        sample_keys = list(results.docs[0].keys()) if results.docs else "None"
        raise Exception(f"No vectors loaded for {similarity_space}. \n"
                        f"Found {len(results.docs)} docs, but filtering retained 0. \n"
                        f"Vector field: {vector_field}. \n"
                        f"Sample keys: {sample_keys}. \n"
                        f"Specific IDs provided: {len(specific_ids_set) if specific_ids_set else 'None'}")

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
    parser = argparse.ArgumentParser(description='Evaluate similarity search performance')
    parser.add_argument('--space', required=True, help='Similarity space to evaluate')
    parser.add_argument('--ground-truth-space', default="laion_clap", help='Similarity space for ground truth')
    parser.add_argument('--ground-truth-file', default=None, help='Load ground truth from pickle file')
    parser.add_argument('--save-ground-truth', default=None, help='Save ground truth to pickle file')
    parser.add_argument('--num-sounds', type=int, default=NUM_SOUNDS_FOR_EVAL, help=f'Number of query sounds (default: {NUM_SOUNDS_FOR_EVAL})')
    parser.add_argument('--retrieve-n', type=int, default=DEFAULT_RETRIEVE_N, help=f'Number of neighbors to retrieve from Solr (default: {DEFAULT_RETRIEVE_N})')
    parser.add_argument('--metric-k', type=int, default=DEFAULT_METRIC_K, help=f'Top K elements to consider for metrics (default: {DEFAULT_METRIC_K})')
    parser.add_argument('--k', type=int, default=None, help='Legacy: overrides both --retrieve-n and --metric-k if provided.')
    parser.add_argument('--output-dir', default=None, help='Directory to save results (results.csv, details, etc.)')
    parser.add_argument('--results-csv', default='eval/results/eval_results.csv', help='Legacy: CSV file to save results if output-dir is not set')
    parser.add_argument('--warmup', type=int, default=500, help='Number of warmup queries (default: 500)')
    parser.add_argument('--clear-cache', action='store_true', help='Reload collection to clear cache before running')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for query selection')
    parser.add_argument('--save-details', action='store_true', help='Save detailed per-query metrics')

    args = parser.parse_args()

    # Handle legacy --k flag
    if args.k is not None:
        args.retrieve_n = args.k
        args.metric_k = args.k

    # Set random seed
    np.random.seed(args.seed)

    if args.clear_cache:
        print("Reloading collection to clear cache...")
        from solrapi import SolrManagementAPI
        api = SolrManagementAPI(SOLR_BASE_URL, COLLECTION_NAME)
        api.reload_collection()
        time.sleep(2)

    # Ensure output directory exists if specified
    if args.output_dir and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

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
        
        # If calculating ground truth, we can also save it if output_dir is provided
        gt_save_path = args.save_ground_truth
        if args.output_dir and not gt_save_path:
             gt_save_path = os.path.join(args.output_dir, "ground_truth.pkl")

        gt_result = evaluate_similarity_search(
            gt_vecs,
            target_ids,
            ground_truth_results=None,
            similarity_space=args.ground_truth_space,
            retrieve_n=args.retrieve_n,
            metric_k=args.metric_k,
            output_dir=None, # Don't save main metrics here yet
            warmup=args.warmup,
            seed=args.seed,
            save_details=False
        )
        ground_truth = gt_result.retrieved_neighbors
        
        if gt_save_path:
            gt_result.save(gt_save_path)
            print(f"Saved ground truth to {gt_save_path}")

    target_vecs, target_ids = load_target_sounds(
        args.space,
        args.num_sounds,
        specific_sound_ids=target_ids
    )

    # Determine the results CSV path for legacy saving if output_dir is not used
    results_csv_path = args.results_csv
    if args.output_dir:
        # If output_dir is set, evaluate_similarity_search will handle saving to output_dir/results.csv
        # So, we don't need to pass results_csv directly to it for saving.
        pass
    else:
        # If no output_dir, use the legacy results_csv path for saving
        # Ensure the directory for results_csv exists
        results_csv_dir = os.path.dirname(results_csv_path)
        if results_csv_dir and not os.path.exists(results_csv_dir):
            os.makedirs(results_csv_dir)

    evaluate_similarity_search(
        target_vecs,
        target_ids,
        ground_truth_results=ground_truth,
        similarity_space=args.space,
        retrieve_n=args.retrieve_n,
        metric_k=args.metric_k,
        output_dir=args.output_dir,
        warmup=args.warmup,
        seed=args.seed,
        save_details=args.save_details
    )
