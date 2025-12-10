# Homework №3: Vector search engines. Part 1.

## Deadline: 19.12 23:59

## 0. Rules

### How to submit?

- Create a new branch, e.g. hw-3
- Create a pull request to main/master branch
- Request a review from both of us (this is counted as a date of submission)
- Merge the PR once it is approved by at least one of the reviewers

> If the task was sent 5 days before the first deadline, you can receive comments to address in order to improve the final mark

### Task requirements

- Have a clean repository (no data in the repo, no .idea and other garbage, etc.)
- Write clean well-structured code (pre-commit hooks, ruff, type hints, etc.)
- Python version: 3.10+
- Package manager: uv
- Create a new dependency group, e.g. hw3, common dependencies can be put without a group.
- Readme with a description of how to install dependencies, download the data and run the code. **Hint**: try installing it in a fresh environment.
- **HW should be uploaded in .py files**

## 1. Preparing the data

In this hw you'll try building ANN search pipelines with Qdrant, as well as check other capabilities vector search engines have.

Overall, you'll need to run the same experiments as in [hw2](https://github.com/joein/vector-search-course/blob/main/03_ann/hw_2.md), use CRUD operations, modify your collection and restore it from a snapshot.

### 1.1 Download the data 

Write a small `.sh` script to download and unpack the dataset (you can take the one you wrote in hw2): [laion-small-clip.tgz](https://storage.googleapis.com/ann-filtered-benchmark/datasets/laion-small-clip.tgz)
(Do not download again if the dataset has already been loaded)

The dataset contains 3 files, `payloads.jsonl`, `vectors.npy` and `tests.jsonl`.
In this hw we need only `payloads.jsonl` and `vectors.npy`.

`vectors.npy` has a shape of (100_000, 512) - it is a subset of CLIP embeddings computed on laion dataset.

Put it under `hw3/data`, don't forget to add `hw3/data` to `.gitignore`

### 1.2 Generate ground-truth 

Consider each vector in the dataset as a potential query. 

Write a script which will find 10 true nearest neighbours for each of the vectors in the dataset via brute-force.

Use Qdrant for this task. 
Create collection, upload data, search. (Hint: you'll need to set `exact=True` in `models.SearchParams` to enable brute-force search)

Save results into `ground_truth.jsonl`.

> _Hint: `jsonl` is a file format, each line of which is a correct `json`, while the whole file is not. One of its advantages is that it can be populated and read iteratively._

Example:
```json lines
{"0": [55183, 84834, 89211, 61363, 53912, 68685, 25178, 58597, 70092, 24357]}
{"1": [68058, 16046, 52287, 67035, 30169, 34689, 55349, 5335, 16625, 69318]}
```

_Do not include the vector itself into its nearest neighbours._


## 2. Qdrant 101

### 2.1 Collection configuration

- Create a collection called `single_unnamed`. This collection will contain unnamed vectors of size 512 with a cosine similarity as a distance metric, leave it with the default params.
- Create a collection called `multiple_named`. This collection will contain 2 named vectors, one should be called `clip_default`, another one is `clip_tuned`. 
Configure the default HNSW parameters on the collection level to be `m=32`, `ef_construct=256`.
Configure `clip_tuned` to have HNSW parameters `m=36`, `ef_construct=300` (you can set other values as well, except the default `m=16` and `ef_construct=100`, but don't set too large to avoid spending to much time on building the index)

### 2.2 Uploading the data

- Upload `vectors.npy` and `payloads.jsonl` using `upsert` to `single_unnamed`.
- Do the same for `multiple_named` collection, use the same data for both `clip_default` and `clip_tuned`.

You can try uploading it in a single call, but you'll be timed out. Write a function to split the data into batches (it should be a generator function to work with sequences which are too large to load into RAM). 
Find the best batch size for your needs.

- Recreate the collection to drop the data.
- Repeat the upload, use `upload_points` for the first collection, and `upload_collection` for the second one. Try a couple of different values for `parallel` to parallelize the workload across several processes. (don't forget to wrap the code into `if __name__ == '__main__'`)

### 2.3 Search

Find 10 nearest neighbours for each vector in every collection.

**Make sure that index is built (status is green) before searching.**

Each task results should be written into `{vector_name}_results.jsonl`

Each record in a file should contain:

- `Precision@1`,
- `Precision@3`,
- `Precision@5`,
- `Precision@10`,
- `QPS` (query per second),
- `total_search_time` (in seconds),
- `<hyperparameters>` (default values are m: 16, ef_construct: 100)

Example of a record (hnsw):
```json
  {
    "m": 8,
    "ef_construct": 40,
    "Precision@1": 0.03165,
    "Precision@3": 0.5858033333327176,
    "Precision@5": 0.6851040000009077,
    "Precision@10": 0.7347350000002316,
    "total_search_time": 2.9306865171529353,
    "QPS": 2121.69790754239
  }
```

Choose the fastest experiment, if you are satisfied with `precision` across the results, try out `ef=50` (search-time parameter configured in `query_points`, otherwise `ef=200`).
Put the results under `{vector_name}_ef_search_{value}_results.jsonl`, where `value` is the value of `ef` you used.

It's not the best practice, but for the sake of simplicity, upload obtained `*.jsonl` values to `hw3/*.jsonl` 

### 2.4 Other CRUD operations

We've already covered part of `CRUD` operations, `C` (creation) via `upsert`, `upload_collection`, `upload_points` and `R` (read) partially via `query_points`.
Let's learn how to use the remaining operations.

There are 2 more ways to read data in Qdrant: `scroll` and `retrieve`.
 

`scroll` - is just an iterator over a collection. It can be used when you want to read data, and you don't have `ids` of the points, nor you have vectors to search with.

`scroll` has a set of useful parameters, which can help you control the amount of data you read. Some of them are: 

- `limit` - number of points to return
- `scroll_filter` - skip points not meeting filtering condition,
- `with_payload` - whether to include payload in the response 
- `with_vectors` - whether to include vectors into the response 
- `offset` - for pagination.


`retrieve` - is a method to read a set of points from the collection by their ids, the data read can be only modified by `with_vectors` and `with_payload` parameters.


#### 2.4.1 `Read` tasks :

- Read 10 points from `single_unnamed` collection, setting `with_payload=False`, and `with_vectors=True`, save `ids` of the points
- Retrieve payloads for these points via `retrieve`
- Read 100 points from `multiple_collection` with `limit=10` (use `offset` for pagination), select only `clip_default` vectors, since we don't need to read the vectors for the points twice - they are the same and we want to reduce the network overhead.

#### 2.4.2 Spoiling the collection

You need to have a day-off tomorrow, but your manager has declined your request.
Let's find a way to look busy while doing nothing.
Modify some points in your collection (don't spoil prod collection) to return some unexplainable weird results, but leave yourself a quick way to fix it.

- Create a snapshot of your collection (`create_snapshot` method)
- Spoil collection as much as you want, but after calling any method to modify the state of vectors, payload or whole points - retrieve the affected points to check if they have actually been changed, add `assert` commands.
Methods to try: `upsert`, `update_vectors`, `set_payload`, `overwrite_payload`, `delete_payload`, `clear_payload`, `delete`, `delete_vectors`.

With that, the collection is in just the right condition for you to sigh deeply and inform everyone that further investigation is required.
After coming back restore the collection from a snapshot, check that everything has been fixed and that was a minor issue with some experiments modifying the dev collection.





