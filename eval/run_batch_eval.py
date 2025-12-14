"""
Script to batch run similarity search evaluations.
"""
import argparse
import subprocess
import sys
import os

def run_command(cmd):
	print(f"Running: {cmd}")
	subprocess.check_call(cmd, shell=True)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Batch run evaluations')
	parser.add_argument('--seed', type=int, default=42, help='Random seed')
	parser.add_argument('--num-sounds', type=int, default=2000, help='Number of sounds to evaluate')
	parser.add_argument('--source-space', default='laion_clap', help='Source space (probably laion_clap)')
	parser.add_argument('--dims', type=int, nargs='+', default=[128, 256, 384], help='List of PCA dimensions to evaluate')
	parser.add_argument('--results-csv', default='eval_results.csv', help='Results file')
	parser.add_argument('--save-details', action='store_true', help='Save per-query metrics')
	args = parser.parse_args()
	
	gt_file = f"gt_{args.source_space}_{args.seed}.pkl"
	
	# Baseline/LAION-CLAP
	print(f"\n--Generating queries and ground truth file for {args.source_space}--")
	cmd_gt = f"python eval/evaluate_similarity_search.py --space {args.source_space} --save-ground-truth {gt_file} --seed {args.seed} --clear-cache --results-csv eval/results/{args.results_csv} --num-sounds {args.num_sounds}"
	if args.save_details:
		cmd_gt += " --save-details"
	run_command(cmd_gt)
	
	# PCA Spaces
	for dim in args.dims:
		target_space = f"{args.source_space}_pca{dim}"
		print(f"\n--Evaluating {target_space}--")
		cmd_eval = f"python eval/evaluate_similarity_search.py --space {target_space} --ground-truth-file {gt_file} --seed {args.seed} --clear-cache --results-csv eval/results/{args.results_csv} --num-sounds {args.num_sounds}"
		if args.save_details:
			cmd_eval += " --save-details"
		run_command(cmd_eval)
		
	print("\n\nAll evaluations completed.")
