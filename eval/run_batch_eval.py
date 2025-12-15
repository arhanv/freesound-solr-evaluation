"""
Script to batch run similarity search evaluations.
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime

import pysolr

# Add search directory to path to import configs
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'search'))
from configs import SOLR_URL


def run_command(cmd):
    """Run a shell command and print it.

    Args:
        cmd (str): The command to execute.
    """
    print(f"Running: {cmd}")
    subprocess.check_call(cmd, shell=True)


def fetch_index_stats(solr_url=SOLR_URL):
    """Fetch index stats (number of parent sounds) from Solr.

    Args:
        solr_url (str): Solr collection URL.

    Returns:
        int: Number of parent sound documents in the index, or 0 if fetch fails.
    """
    try:
        solr = pysolr.Solr(solr_url)
        # Search for content_type:s to count parent documents only
        results = solr.search('content_type:s', rows=0)
        return results.hits
    except Exception as e:
        print(f"Warning: Could not fetch index stats: {e}")
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Batch run evaluations')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num-sounds', type=int, default=2000, help='Number of sounds to evaluate')
    parser.add_argument('--source-space', default='laion_clap', help='Source space (probably laion_clap)')
    parser.add_argument('--dims', type=int, nargs='+', default=[128, 256, 384],
                        help='List of PCA dimensions to evaluate')
    parser.add_argument('--warmup', type=int, default=500, help='Number of warmup queries')
    parser.add_argument('--save-details', action='store_true', help='Save per-query metrics')
    args = parser.parse_args()

    # 1. Fetch Index Stats for Naming
    index_size = fetch_index_stats()
    index_size_k = int(index_size / 1000)

    # 2. Generate Run ID and Directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    run_id = f"seed{args.seed}_queries{args.num_sounds}_index{index_size_k}k_{timestamp}"

    # Base output directory
    output_dir = os.path.join(os.path.dirname(__file__), 'results', run_id)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Starting batch evaluation. Results will be saved to: {output_dir}")

    # 3. Save Configuration
    config = {
        'seed': args.seed,
        'num_sounds': args.num_sounds,
        'source_space': args.source_space,
        'dims': args.dims,
        'warmup': args.warmup,
        'timestamp': timestamp,
        'index_size': index_size,
        'run_id': run_id
    }

    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    # 4. Run baseline
    # We pass the output_dir. evaluate_similarity_search.py will handle saving csv and pickling GT.
    gt_filename = "ground_truth.pkl"

    print(f"\n--Generating queries and ground truth file for {args.source_space}--")
    
    cmd_gt = (
        f"python eval/evaluate_similarity_search.py "
        f"--space {args.source_space} "
        f"--ground-truth-space {args.source_space} "
        f"--output-dir {output_dir} "
        f"--seed {args.seed} "
        f"--clear-cache "
        f"--num-sounds {args.num_sounds} "
        f"--warmup {args.warmup}"
    )
    if args.save_details:
        cmd_gt += " --save-details"

    run_command(cmd_gt)

    gt_file_path = os.path.join(output_dir, gt_filename)

    # 5. Run PCA Evaluations
    for dim in args.dims:
        target_space = f"{args.source_space}_pca{dim}"
        print(f"\n--Evaluating {target_space}--")

        cmd_eval = (
            f"python eval/evaluate_similarity_search.py "
            f"--space {target_space} "
            f"--ground-truth-file {gt_file_path} "
            f"--output-dir {output_dir} "
            f"--seed {args.seed} "
            f"--clear-cache "
            f"--num-sounds {args.num_sounds} "
            f"--warmup {args.warmup}"
        )
        if args.save_details:
            cmd_eval += " --save-details"

        run_command(cmd_eval)

    print(f"\n\nAll evaluations completed. Results saved in {output_dir}")
