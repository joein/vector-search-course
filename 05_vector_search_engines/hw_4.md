# Homework №4: Vector search engines. Part 2.

## Deadline: 19.12 23:59

## 0. Rules

### How to submit?

- Create a new branch, e.g. hw-4
- Create a pull request to main/master branch
- Request a review from both of us (this is counted as a date of submission)
- Merge the PR once it is approved by at least one of the reviewers

> If the task was sent 5 days before the first deadline, you can receive comments to address in order to improve the final mark

### Task requirements

- Have a clean repository (no data in the repo, no .idea and other garbage, etc.)
- Write clean well-structured code (pre-commit hooks, ruff, type hints, etc.)
- Python version: 3.10+
- Package manager: uv
- Create a new dependency group, e.g. hw4, common dependencies can be put without a group.
- Readme with a description of how to install dependencies, download the data and run the code. **Hint**: try installing it in a fresh environment.
- **HW should be uploaded in .py files**

## 1. Preparing the data

In this hw you'll try working with payload filters in Qdrant.

Write a small `.sh` script to download and unpack the dataset (you can take the one you wrote in hw2, hw3): [laion-small-clip.tgz](https://storage.googleapis.com/ann-filtered-benchmark/datasets/laion-small-clip.tgz)
(Do not download again if the dataset has already been loaded)

The dataset contains 3 files, `payloads.jsonl`, `vectors.npy` and `tests.jsonl`.
We finally need all 3 files

You're already familiar with `vectors.npy` and `payloads.jsonl`.

`tests.jsonl` contains queries with filtering conditions and expected results. 
Record's structure is:
query - vector to be used for similarity search
conditions - filtering conditions with range filters
closest_ids - IDs of records, expected to be found with given query
closest_scores - similarity scores of associated IDs

Put the data under `hw4/data`, don't forget to add `hw4/data` to `.gitignore`
We don't need to compute ground-truth anymore, since it is provided with the dataset


## 2. Payload filters

### 2.1 Preparing the collection

- Create a collection called `hw4` or reuse the collection from `hw3` called `single_unnamed`. 
This collection should contain unnamed vectors of size 512 with a cosine similarity as a distance metric, leave it with the default params.
- Upload the data with any methods you've already learnt.

### 2.2 Filtrable HNSW

Find 10 nearest neighbours for each vector in every collection searching with filters from `tests.jsonl`.
Don't be afraid if you find fewer than 10 results, the number of results has just to be the same as in `tests.jsonl`.
In `conditions` you might find empty filters, or filters which contain `and` or `or` clauses while having only 1 condition - that's fine, just build filters as they are provided.
An example of building a filter:

```python
from qdrant_client import models

query_filter = models.Filter(must=[models.FieldCondition(key='field_a', match=models.MatchValue(value=30))])
# avoid calling a variable with filters just `filter`, because it's a builtin python name
```

In order to do filtered search you'd need to build payload indexes for the fields you are going to filter on.
It's better to create payload indexes first, and then upload the data, because filtrable hnsw affects how the graph is built. 

**Make sure that indexes are built (status is green and number of indexed payload points is as expected) before searching.**

If the number of indexed points in a payload index is less than anticipated - try using method `count` with a filter checking existence of the payload field (also set `exact=True` to do brute-force search).

Search results should be written into `filtered_search_results.jsonl`

Each record in a file should contain:
- `Precision@1`,
- `Precision@3`,
- `Precision@5`,
- `Precision@10`,
- `QPS` (query per second),
- `total_search_time` (in seconds),

Example of a record (hnsw):
```json
  {
    "Precision@1": 0.03165,
    "Precision@3": 0.5858033333327176,
    "Precision@5": 0.6851040000009077,
    "Precision@10": 0.7347350000002316,
    "total_search_time": 2.9306865171529353,
    "QPS": 2121.69790754239
  }
```

### 2.3 ACORN

Run the same experiment with ACORN and save results to `filtered_search_acorn_results.jsonl`
ACORN is configured per-query.

It's not the best practice, but for the sake of simplicity, upload `*.jsonl` files with results to `hw4/results` directory.  

### 2.4 Other filter types

The dataset we've been working with contains only one filter type - `range`, however, Qdrant supports much more types.
Let's try them out.

Create a random dataset with 200,000 points, you can choose vectors dimension on your own - something like 384, 512 or 768.
Generate payload via functions from `payload_generator.py` (available in the repository).
Feel free to add new fields to the generator if you'd like.

Try building filters for each of the datatypes (not for all at once), with `must`, `must_not`, `should` clauses.
Some of the fields might have the same type, build at least 10 different filters.

You can find all the available filters and conditions in [Qdrant's documentation](https://qdrant.tech/documentation/concepts/filtering/?q=filter)
