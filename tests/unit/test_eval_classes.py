import os
import pickle
import pandas as pd
from eval.evaluate_similarity_search import SearchEvaluationResult

def test_search_evaluation_result_serialization(tmp_path):
    """
    Test that the SearchEvaluationResult class can correctly save to pickle,
    load from pickle, and export to CSV without data loss.
    """
    configs = {
        'similarity_space': 'test_space',
        'retrieve_n': 10,
        'metric_k': 5,
        'query_size': 2,
        'index_num_docs': 100,
        'index_num_sounds': 50,
        'random_seed': 42,
        'warmup': 0
    }
    
    target_ids = [100, 200]
    query_times = [10.5, 12.1]
    retrieved_neighbors = [[101, 102], [201, 202]]
    
    metrics = {
        'latency_p50': 11.0,
        'mean_latency': 11.3,
        'recall_mean': 0.8,
        'ndcg_mean': 0.75,
        'empty_results': 0,
        'space_size_mb': 5.5
    }

    # 1. Create Result Object
    result = SearchEvaluationResult(
        configs=configs,
        target_sound_ids=target_ids,
        query_times_ms=query_times,
        retrieved_neighbors=retrieved_neighbors,
        metrics=metrics
    )
    
    # Verify timestamp was auto-generated
    assert hasattr(result, 'timestamp')
    assert isinstance(result.timestamp, str)

    # 2. Test Pickle Save/Load
    pkl_path = tmp_path / "test_result.pkl"
    result.save(str(pkl_path))
    
    assert os.path.exists(pkl_path)
    
    loaded_result = SearchEvaluationResult.load(str(pkl_path))
    assert loaded_result.configs == configs
    assert loaded_result.target_sound_ids == target_ids
    assert loaded_result.metrics == metrics
    assert loaded_result.timestamp == result.timestamp

    # 3. Test CSV Export (Creation)
    csv_path = tmp_path / "results.csv"
    result.to_csv(str(csv_path), append=False)
    
    assert os.path.exists(csv_path)
    df = pd.read_csv(csv_path)
    
    # Assert correct number of rows and some specific columns
    assert len(df) == 1
    assert df.iloc[0]['similarity_space'] == 'test_space'
    assert df.iloc[0]['recall_mean'] == 0.8
    assert df.iloc[0]['index_num_docs'] == 100

    # 4. Test CSV Export (Append)
    # Give it a slightly different metric to prove the append worked
    result.metrics['recall_mean'] = 0.9
    result.to_csv(str(csv_path), append=True)
    
    df_appended = pd.read_csv(csv_path)
    assert len(df_appended) == 2
    assert df_appended.iloc[0]['recall_mean'] == 0.8
    assert df_appended.iloc[1]['recall_mean'] == 0.9
