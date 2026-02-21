# Future Testing Ideas (TO ADD)

This file serves as a backlog for integration, system, and end-to-end tests that we want to implement in the future, once the foundational unit and local integration tests are stable.

## Integration Tests
- **Streamlit UI Integrations:** Use Streamlit's built-in testing wrapper (`streamlit.testing.v1`) to verify that the dashboard pages render correctly when fed specific simulated Solr data.
    - Test that `1_System_Monitor.py` displays the correct status colors when Docker is mocked to be down.
    - Test that `3_Analysis.py` correctly plots metrics from a simulated evaluation run.
- **Complex Synthetics Pipeline:** An integration test that runs `generate_and_index_synthetics.py`, checks the generated distribution against `stats_utils.py`, and then runs `evaluate_similarity_search.py` against those synthetics.

## System / End-to-End Tests
- **Golden Subset Evaluation:** Create a static, small dataset (e.g., 1000 items). Run the full `run_batch_eval.py` pipeline against a staging Solr instance (or Docker container) and assert that the final recalled metrics (NDCG/Recall) fall within a statistically acceptable bound to detect breaking algorithmic changes.
- **Parallel Processing Safety:** If/when PCA indexing is parallelized, an integration test that bombards a test collection from multiple threads and asserts that the final document count exactly matches expectations, proving no race conditions exist.

## Refactoring Debt
- **Dependency Injection Migration:** Currently, scripts create `pysolr.Solr` instances internally. A future goal is to refactor all major functions to accept a `solr_client` argument to make dependency injection cleaner, reducing reliance on `unittest.mock.patch` in our unit tests.
