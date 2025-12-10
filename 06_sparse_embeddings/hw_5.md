# Homework №5: Sparse models

## Deadline: 19.12 23:59

## 0. Rules

### How to submit?

- Create a new branch, e.g. hw-5
- Create a pull request to main/master branch
- Request a review from both of us (this is counted as a date of submission)
- Merge the PR once it is approved by at least one of the reviewers

> If the task was sent 5 days before the first deadline, you can receive comments to address in order to improve the final mark

### Task requirements

- Have a clean repository (no data in the repo, no .idea and other garbage, etc.)
- Write clean well-structured code (pre-commit hooks, ruff, type hints, etc.)
- Python version: 3.10+
- Package manager: uv
- Create a new dependency group, e.g. hw5, common dependencies can be put without a group.
- Readme with a description of how to install dependencies, download the data and run the code. **Hint**: try installing it in a fresh environment.
- **HW should be uploaded in .py files**

## 1. Preparing the data

Download [SciFact](https://huggingface.co/datasets/mteb/scifact) dataset.

The dataset consists of 3 parts: `default`, `corpus` and `queries`.

`corpus` records consist of `_id`, `title`, and `text`, while `queries` consist just of `_id` and `text`.
`default` contains search results with `query-id`, `corpus-id` and the corresponding binary `score`.
In this particular dataset, `score` contains only 1.


## 2. Working with sparse vectors

### 2.1 Uploading the data

- Create a collection called `hw5`, configure 2 sparse vectors for it - `bm25` and `splade`. Don't forget to provide `Modifier.IDF` for `bm25`.  
- Upload points to the collection. Attach original data (`_id`, `title`, `text`) to the points as payload. You can compute embeddings manually with `fastembed` (splade can be computed with `sentence-transformers` or other libraries which support it as well) or you can use its integration in `qdrant-client` and compute vectors via `models.Document`.
- Compare the time used for computing embeddings and uploading data with `Bm25` and `prithivida/Splade_PP_en_v1`.

If you don't have enough resources on your machine to compute embeddings with Splade, you can use Colab, but provide the code for inference in a separate file like `splade_inference.py`.

The models are available in `fastembed` by the following names: `Qdrant/Bm25` and `prithivida/Splade_PP_en_v1`
If you're using GPU, install `fastembed-gpu` instead of `fastembed`.

### 2.2 Search everything

Find 10 nearest neighbours for each vector from `queries`.

Search results should be written into `bm25_search_results.jsonl` and `splade_search_results.jsonl`.

Each record in a file should contain:
- `MRR@10`,
- `QPS` (query per second),
- `total_search_time` (in seconds),

Example of a record (hnsw):
```json
  {
    "MRR@10": 0.25,
    "total_search_time": 2.9306865171529353,
    "QPS": 2121.69790754239
  }
```

### 2.3 Search pre-computed

Most of the time consumed by Splade was spent on inference.
Let's see how much time it takes to just search splade embeddings.

Take a sample of 100 queries and precompute Splade embeddings for them.
Find 10 nearest neighbours for each of these queries.

Save results into `splade_precomputed_search_results.jsonl`

### 2.4. Conclusion

Did Splade perform better? Why?