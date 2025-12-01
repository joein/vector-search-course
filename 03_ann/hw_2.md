# Homework №2: ANN exploration

## Deadline: 11.12 23:59

## 0. Rules

### How to submit?

- Create a new branch, e.g. hw-2
- Create a pull request to main/master branch
- Request a review from both of us (this is counted as a date of submission)
- Merge the PR once it is approved by at least one of the reviewers

> If the task was sent 5 days before the first deadline, you can receive comments to address in order to improve the final mark

### Task requirements

- Have a clean repository (no data in the repo, no .idea and other garbage, etc.)
- Write clean well-structured code (pre-commit hooks, ruff, type hints, etc.)
- Python version: 3.10+
- Package manager: uv
- Create a new dependency group, e.g. hw2, common dependencies can be put without a group.
- Readme with a description of how to install dependencies, download the data and run the code. **Hint**: try installing it in a fresh environment.
- **HW should be uploaded in .py files**

## 1. Preparing the data

In this hw you'll compare the performance of different ANN algorithms, such as ANNOY, IVFPQ and HNSW.

Besides time spent on index building and querying, it's also important to check the retrieval quality.

When there are no labels available, ANN algorithms can also be tested against labels obtained via brute-force search.

### 1.1 Download the data 

Write a small `.sh` script to download and unpack the dataset: [laion-small-clip.tgz](https://storage.googleapis.com/ann-filtered-benchmark/datasets/laion-small-clip.tgz)
(Do not download again if the dataset has already been loaded)

The dataset contains 3 files, `payloads.jsonl`, `vectors.npy` and `tests.jsonl`.
We don't need `payloads.jsonl` and `tests.jsonl` at the moment.

`vectors.npy` has a shape of (100_000, 512) - it is a subset of CLIP embeddings computed on laion dataset.

Put it under `hw2/data`, don't forget to add `hw2/data` to `.gitignore`

### 1.2 Generate ground-truth 

Consider each vector in the dataset as a potential query. 

Write a script which will find 10 true nearest neighbours for each of the vectors in the dataset via brute-force.

The whole distance matrix probably won't fit your RAM, you can implement brute-force manually (e.g. via numpy) or use some third-party tool for this (no restrictions, but make sure that a chosen tool is using brute-force search, not ANN).

Save results into `ground_truth.jsonl`.

> _Hint: `jsonl` is a file format, each line of which is a correct `json`, while the whole file is not. One of its advantages is that it can be populated and read iteratively._

Example:
```json lines
{"0": [55183, 84834, 89211, 61363, 53912, 68685, 25178, 58597, 70092, 24357]}
{"1": [68058, 16046, 52287, 67035, 30169, 34689, 55349, 5335, 16625, 69318]}
```

_Do not include the vector itself into its nearest neighbours._


## 2. ANN

Each task results should be written into `{algorithm_name}_results.jsonl`

Each record in a file should contain:
- `indexing_time`,
- `Precision@1`,
- `Precision@3`,
- `Precision@5`,
- `Precision@10`,
- `QPS` (query per second),
- `total_search_time` (in seconds),
- `<hyperparameters>`

Example of a record (hnsw):
```json
  {
    "M": 8,
    "efConstruction": 40,
    "efSearch": 10,
    "Precision@1": 0.03165,
    "Precision@3": 0.5858033333327176,
    "Precision@5": 0.6851040000009077,
    "Precision@10": 0.7347350000002316,
    "indexing_time": 1.412082000169903,
    "total_search_time": 2.9306865171529353,
    "QPS": 34121.69790754239
  }
```

### 2.1 ANNOY

Find 10 nearest neighbours for each of the vectors in the dataset using ANNOY.

Try out at least these parameters (all their combinations, like [(10, 100), (10, 500), (10, 1000), etc.]):

```python
n_trees = [10, 25, 50, 100, 200]  
search_k = [100, 500, 1000, 5000]
```

_Feel free to experiment and include other parameter values as well, though, it's not required._

Write an explanation for each of the hyperparameters and observed results.

### 2.2 IVFPQ

Find 10 nearest neighbours for each of the vectors in the dataset using IVFPQ (e.g. via faiss).

Try out at least these parameters:

```python
nlist = [64, 128, 256, 512, 1024]
m = [16, 32]
nbits = [8]
nprobe = [1, 2, 4, 8, 16, 32, 64, 128]
```

_Feel free to experiment and include other parameter values as well, though, it's not required._

Write an explanation for each of the hyperparameters and observed results.

### 2.3 HNSW

Find 10 nearest neighbours for each of the vectors in the dataset using HNSW (e.g. via faiss, hnswlib or nmslib).

Try out at least these parameters:
```python
m = [8, 16, 32, 64]             
ef_construction = [32, 64, 100, 128, 256]
ef_search = [32, 64, 100, 128, 256]
```

_Feel free to experiment and include other parameter values as well, though, it's not required._

Write an explanation for each of the hyperparameters and observed results.

It's not the best practice, but for the sake of simplicity, upload `ground_truth.jsonl`, and all 3 `{algorithm_name}_results.jsonl` to `hw2/results` directory. 