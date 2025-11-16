# Homework №1: How to search?

## Deadline: 27.11 23:59

## 0. Rules

### How to submit?

- Create a private repository at GitHub.com
- Invite [joein](https://github.com/joein) and [senyafeelsgood](https://github.com/senyafeelsgood) to become a contributor
- Each hw should be done in its own branch, e.g. hw-1
- Create a pull request to main/master branch
- Request a review from both of us (this is counted as a date of submission)
- Merge the PR once it is approved by at least one of the reviewers

> If the task was sent 5 days before the first deadline, you can receive comments to address in order to improve the final mark

### Task requirements

- Have a clean repository (no data in the repo, no .idea and other garbage, etc.)
- Write clean well-structured code (pre-commit hooks, ruff, type hints, etc.)
- Python version: 3.10+
- Package manager: uv
- Each hw has to have a dependency group, common dependencies can be put without a group (scikit-learn, matplotlib and similar libraries required for the first hw should be in a group, while numpy can be in the root.)
- Readme with installation instructions. Hint: try installing it in a fresh environment.

## 1. Distances


### 1.1 Let's measure it

Implement dot product, cosine similarity, euclidean and manhattan distances for vector comparison. 

These functions should accept query (or multiple queries) and matrix to compare to as input.

Example:

```python
import numpy as np


def cosine_similarity(queries: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return np.ndarray(...)

query = np.random.rand(3)
queries = np.random.rand(10, 3)
data = np.random.rand(100, 3)
cosine_similarity(query, data)
cosine_similarity(queries, data)
```

Make 2 versions of the functions:
1. Pure numpy
2. Pure python (no numpy, just for loops)

Make sure to test your results:

- Compare numpy functions to some ground truth implementation such as the one in scikit-learn.
- Compare numpy functions to pure python functions.

There is no need to write full-fledged tests, `assert np.allclose(..., ...)` is enough.

Hint: 
While doing asserts, make sure not to change the input data. E.g., do not normalize in-place.


### 1.2 How fast are we?

Benchmark pure python distance computation vs numpy

Make several experiments with different number of vectors and dimensionality.


## 2. Using vectors

### 2.1 Let's search

1. Choose your favourite text dataset in english and download it. Do not use large datasets, small ones (~50k records) or subsets of a bigger datasets are enough (let's not spend too much time on downloading the data).
2. Take any pre-trained english word2vec / fasttext / glove model or model2vec, if the former are slow to download.
3. Embed the sentences in your dataset. Use `memmap` to save RAM if required.
4. Try finding the nearest neighbours for the sentences from the dataset (10 samples is enough) with the functions you implemented previously.


### 2.2 Let's visualize 

1. Take samples of different sizes from your dataset (e.g. 500, 5_000, 25_000 records)
2. Visualize it using PCA, t-SNE and UMAP. Try using different sample sizes, which algorithm is the fastest? Which one do you like better?
3. Make conclusion out of the seen results (don't worry if your data and vectors do not visualize well, it can also happen)
