# freesound-solr-evaluation
toolbox for evaluating freesound's vector similarity and search performance

## Structure
- `search/`: scripts for initiating/managing Solr, indexing data, and training pca models
- `eval/`: scripts for evaluating search performance. also contains evaluation results.
- `models/`: saved PCA models
- `schema/`: schema definitions for Solr


## Basic setup
1. Create a Python virtual environment and install dependencies
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
2. Configure Environment (Optional)
Set the `SEARCH_DOCUMENTS_DIR` environment variable if your data is not in the default location.
```bash
export SEARCH_DOCUMENTS_DIR="[wherever the JSONs are stored]"
```

3. Start Docker engine and Solr
```
docker-compose up -d
```
3. Initialize Solr with Freesound schema
```
python search/setup.py
```

## Loading data into solr
The `index_to_solr.py` script indexes sound metadata and vectors from the configured `SEARCH_DOCUMENTS_DIR`.

**Common Operations**

| Goal | Command |
| :--- | :--- |
| **Check Progress** | `python search/index_to_solr.py --status` |
| **Index All** | `python search/index_to_solr.py --index-all` |
| **Index First N Files** | `python search/index_to_solr.py --index 10` |
| **Index Next N Files** | `python search/index_to_solr.py --index-new 50` |

You can also add `--clear` to wipe the collection before indexing (e.g. `python search/index_to_solr.py --clear --index-all`).

## Fitting compact similarity vectors (PCA)
To reduce the dimensionality of vectors via PCA, we use `search/pca.py`, which fits and saves a PCA matrix to `models/`.

`pca.py` includes a few arguments for process control (you can see the rest using 'python search/pca.py --help'):
- `--fit`: fit the PCA model
- `--reindex`: re-index the vectors with the PCA model
- `--dims`: the number of dimensions to reduce to
- `--checkpoint`: if the process is interrupted, you can use the checkpoint saved to `models/` to resume

e.g. `python search/pca.py --fit --reindex --dims 128` will fit a PCA model with 128 dimensions and index the reindex Solr to include these new similarity vectors as child documents of the parent sounds.

## Evaluating Search Performance
The `eval/` directory contains scripts for evaluating search performance. These can be run after fitting and indexing the compact vectors into Solr.

### Optional: Individual similarity space evaluation
`evaluate_similarity_search.py` can be used to evaluate search performance for a specific similarity space. It constructs a set of test queries from the database using `--seed` and `--num-sounds` (or via a `--ground-truth-file` which was generated from a previous run, to reuse the same test queries across different configs). The test queries are then evaluated for recall and latency in similarity search using the vector space specified in `--space`. 

If we run this line: `python eval/evaluate_similarity_search.py --space laion_clap --num-sounds 2000 --seed 100 --save-ground-truth`, it will execute the eval loop on the basic CLAP similarity vectors and save the results to `eval/results/{run_name}/ground_truth.pkl` so we can reuse these as the baseline for the compressed vectors. However, you can just use the batch eval script instead, which does all of this automatically.

### Recommended: Batch evaluation
`run_batch_eval.py` can be used to evaluate search performance at once for multiple similarity spaces, with warmup and cache-clearing to ensure fair comparison. You can run it like this:

`python eval/run_batch_eval.py --seed 100 --num-sounds 4000 --source-space laion_clap --dims 64 128 256 384 --warmup 500 --save-details`
1. Creates a set of test queries with `--seed` and `--num-sounds`.
2. Runs the eval loop for the `laion_clap` space, and saves the baseline results to `ground_truth`.
3. For each of the similarity spaces in PCA-{64, 128, 256, 384}, it will evaluate the search performance (using the same query set) against the baseline and save the results to `results.csv`.

**Results:**

Output is saved to `eval/results/` in run-specific directories (e.g., `seed42_queries2000_.../`).
- `results.csv`: Summary metrics (recall, latency, qps, configs).
- `ground_truth.pkl`: The similarity results returned by original similarity vectors (512-dim).
- `per_query_results.pkl`: Dataframes containing the query-by-query results. Useful for checking latency performance variations and impact of heaping/caching processes from Solr.