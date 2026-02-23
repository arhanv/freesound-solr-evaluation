import os
import json
import sys
import pickle
import pandas as pd
from eval.run_batch_eval import run_command
from eval.evaluate_similarity_search import SearchEvaluationResult

# Hack to correctly unpickle a class that the subprocess pickled as '__main__.SearchEvaluationResult'
sys.modules['__main__'].SearchEvaluationResult = SearchEvaluationResult

def test_run_batch_eval_pipeline(solr_test_collection, solr_test_client, tmp_path):
    """
    Test the full batch evaluation orchestrator using a small dummy dataset.
    This guarantees it runs, writes the correct files, handles the config,
    and produces correctly formatted pickled details and CSVs.
    """
    # 1. Setup Dummy Data in Test Core
    docs = [
        {
            "id": "1", "content_type": "s",
            "_childDocuments_": [
                {"id": "1_laion_clap", "content_type": "v", "similarity_space": "laion_clap", "sim_vector128_l2": [0.1]*128},
                {"id": "1_laion_clap_pca4", "content_type": "v", "similarity_space": "laion_clap_pca4", "sim_vector128_l2": [0.1]*128},
                {"id": "1_laion_clap_pca2", "content_type": "v", "similarity_space": "laion_clap_pca2", "sim_vector128_l2": [0.1]*128}
            ]
        },
        {
            "id": "2", "content_type": "s",
            "_childDocuments_": [
                {"id": "2_laion_clap", "content_type": "v", "similarity_space": "laion_clap", "sim_vector128_l2": [0.2]*128},
                {"id": "2_laion_clap_pca4", "content_type": "v", "similarity_space": "laion_clap_pca4", "sim_vector128_l2": [0.2]*128},
                {"id": "2_laion_clap_pca2", "content_type": "v", "similarity_space": "laion_clap_pca2", "sim_vector128_l2": [0.2]*128}
            ]
        },
        {
            "id": "3", "content_type": "s",
            "_childDocuments_": [
                {"id": "3_laion_clap", "content_type": "v", "similarity_space": "laion_clap", "sim_vector128_l2": [0.3]*128},
                {"id": "3_laion_clap_pca4", "content_type": "v", "similarity_space": "laion_clap_pca4", "sim_vector128_l2": [0.3]*128},
                {"id": "3_laion_clap_pca2", "content_type": "v", "similarity_space": "laion_clap_pca2", "sim_vector128_l2": [0.3]*128}
            ]
        }
    ]
    solr_test_client.add(docs, commitWithin=1000)
    solr_test_client.commit()

    # We need to temporarily patch the output_dir generation in run_batch_eval 
    # to point to tmp_path so we don't pollute the real results folder
    from unittest.mock import patch
    
    with patch('eval.run_batch_eval.os.path.join') as mock_join:
        # Mocking join is tricky because it's used heavily. 
        # A simpler way is to construct the command line execution manually in the test,
        # perfectly mirroring what the user would type, but mocking the script's internal save location.
        pass

    # Actually, we can just run the script via subprocess and pass a custom output dir via a temporary patch to the script itself,
    # OR we can just test the *evaluation script loops* mechanically.
    
    # Since run_batch_eval explicitly hardcodes its output_dir, let's run it, find the newest folder,
    # assert the files, and then clean up the newest folder.
    
    # Run the batch eval through standard subprocess, but forcefully override the URL and DIMS
    cmd = (
        f"python -u eval/run_batch_eval.py "
        f"--source-space laion_clap "
        f"--dims 4 2 "
        f"--num-sounds 3 "
        f"--warmup 0 "
        f"--retrieve-n 2 "
        f"--metric-k 2 "
        f"--save-details "
        f"--solr-url {solr_test_collection} "  # CRITICAL: Point to test collection
    )
    
    # Record folders before
    results_base_dir = "eval/results"
    if not os.path.exists(results_base_dir):
        os.makedirs(results_base_dir)
        
    folders_before = set(os.listdir(results_base_dir))
    
    # Execute
    import subprocess
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Subprocess failed with error: {result.stderr}")
        assert False, f"Batch eval failed: {result.stderr}"
    
    # Record folders after to find the new run
    folders_after = set(os.listdir(results_base_dir))
    new_folders = folders_after - folders_before
    
    assert len(new_folders) == 1, "Expected exactly one new results folder to be created"
    
    run_dir_name = list(new_folders)[0]
    run_dir = os.path.join(results_base_dir, run_dir_name)
    
    try:
        # --- VERIFICATIONS ---
        
        # 1. Check Config JSON
        config_path = os.path.join(run_dir, "config.json")
        assert os.path.exists(config_path)
        with open(config_path, 'r') as f:
            config = json.load(f)
            assert config['source_space'] == 'laion_clap'
            assert config['dims'] == [4, 2]
            assert config['num_sounds'] == 3
            assert config['index_size'] == 3  # Based on our dummy data!
            
        # 2. Check Ground Truth Pickle
        gt_path = os.path.join(run_dir, "ground_truth.pkl")
        assert os.path.exists(gt_path)
        with open(gt_path, 'rb') as f:
            gt_result = pickle.load(f)
            # Should contain 3 target IDs and 3 neighbor arrays
            assert len(gt_result.target_sound_ids) == 3
            assert len(gt_result.retrieved_neighbors) == 3
            
        # 3. Check CSV Summary
        csv_path = os.path.join(run_dir, "results.csv")
        assert os.path.exists(csv_path)
        df = pd.read_csv(csv_path)
        # Should contain 3 rows (ground_truth, pca4, pca2)
        assert len(df) == 3 
        assert set(df['similarity_space'].tolist()) == {'laion_clap', 'laion_clap_pca4', 'laion_clap_pca2'}
        assert 'recall_mean' in df.columns
        assert 'ndcg_mean' in df.columns
        
        # 4. Check Detailed Pickles
        details_pca4_path = os.path.join(run_dir, "per_query_details_laion_clap_pca4.pkl")
        assert os.path.exists(details_pca4_path)
        df_pca4 = pd.read_pickle(details_pca4_path)
        # Exactly 3 queries were run
        assert len(df_pca4) == 3
        assert 'sound_id' in df_pca4.columns
        assert 'recall' in df_pca4.columns
        assert df_pca4.iloc[0]['similarity_space'] == 'laion_clap_pca4'
        
    finally:
        # Post-test Cleanup: Remove the generated folder so we don't pollute the dev space
        import shutil
        shutil.rmtree(run_dir)
