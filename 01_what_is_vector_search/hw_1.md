# Homework №1: How to Search?

**Deadline: December 3, 23:59**

## 0. Rules

### How to submit?

- Create a private repository on [GitHub](https://github.com).
- Invite [joein](https://github.com/joein) and all course assistants as collaborators.
- Complete each homework assignment in its own branch (e.g., `hw-1`).
- Create a pull request to the `main` or `master` branch.
- Request a review from all reviewers. **The date you request a review counts as your submission date.**
- Merge the PR once it has been approved by at least one reviewer.

### Task requirements

- Keep your repository clean (no datasets, `.idea` directories, generated files, or other unnecessary files).
- Write clean, well-structured code (pre-commit hooks, Ruff, type hints, etc.).
- Python version: **3.13+**
- Package manager: **uv**
- Ensure that dependencies can be installed separately for each homework assignment (e.g., using separate `pyproject.toml` files or uv workspaces).
- Include a README with installation and execution instructions. Hint: try following your own instructions in a fresh environment.

## 1. Distances

### 1.1 Let's measure it

Implement the following functions for vector comparison:

- Dot product
- Cosine similarity
- Euclidean distance
- Manhattan distance

Each function should accept either a single query vector or multiple query vectors, along with a matrix of vectors to compare against.

Example:

```python
import numpy as np


def cosine_similarity(
    queries: np.ndarray,
    matrix: np.ndarray,
) -> np.ndarray:
    ...


query = np.random.rand(3)
queries = np.random.rand(10, 3)
data = np.random.rand(100, 3)

cosine_similarity(query, data)
cosine_similarity(queries, data)
```

Implement **two versions** of each function:

1. Pure NumPy
2. Pure Python (no NumPy, only loops)

Validate your implementations:

- Compare your NumPy implementations against a reference implementation, such as scikit-learn.
- Compare your NumPy implementations against your pure Python implementations.

There is no need to write full-fledged tests. Using `assert np.allclose(...)` is sufficient.

**Hint:** Make sure your functions do not modify the input data. For example, avoid in-place normalization.

### 1.2 How fast are we?

Benchmark your pure Python implementations against their NumPy counterparts.

- Run several experiments with different numbers of vectors and vector dimensionalities.
- Compare the execution times.
- Summarize your observations.

## 2. Using vectors

### 2.1 Let's search

1. Choose an English text dataset and load it using the Hugging Face [`datasets`](https://huggingface.co/docs/datasets/) library. Use a relatively small dataset (~10k–50k records) or a subset of a larger one to avoid unnecessary download and processing time.
2. Choose any pre-trained English Word2Vec, FastText, or GloVe model.
3. Embed the sentences in your dataset. Explain how you obtain a single embedding for each sentence from the model's output. Use `memmap` to save RAM if needed.
4. Find the nearest neighbors for 10 sentences from your dataset using the functions you implemented in Section 1.

### 2.2 Let's visualize

1. Take samples of different sizes from your dataset (e.g., 500, 5,000, and 25,000 records).
2. Visualize the embeddings using PCA, t-SNE, and UMAP. Experiment with different sample sizes. Which algorithm is the fastest? Which visualization do you prefer?
3. If t-SNE takes too long, you may use smaller samples. Explain your choice and why you reduced the sample size.
4. Draw conclusions from your results. Don't worry if your embeddings do not produce clearly separated clusters — that's also a valid observation.