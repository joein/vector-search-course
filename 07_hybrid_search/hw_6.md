# Homework №6: Hybrid search

## Deadline: 25.12 23:59

## 0. Rules

### How to submit?

- Create a new branch, e.g. hw-6
- Create a pull request to main/master branch
- Request a review from both of us (this is counted as a date of submission)
- Merge the PR once it is approved by at least one of the reviewers

### Task requirements

- Have a clean repository (no data in the repo, no .idea and other garbage, etc.)
- Write clean well-structured code (pre-commit hooks, ruff, type hints, etc.)
- Python version: 3.10+
- Package manager: uv
- Create a new dependency group, e.g. hw6, common dependencies can be put without a group.
- Readme with a description of how to install dependencies, download the data and run the code. **Hint**: try installing it in a fresh environment.
- **HW should be uploaded in .py files**

## 1. Preparing the data

Download [SciFact](https://huggingface.co/datasets/mteb/scifact) dataset.

The dataset consists of 3 parts: `default`, `corpus` and `queries`.

`corpus` records consist of `_id`, `title`, and `text`, while `queries` consist just of `_id` and `text`.
`default` contains search results with `query-id`, `corpus-id` and the corresponding binary `score`.
In this particular dataset, `score` contains only 1.


## 2. Building hybrid search

### 2.1 Uploading the data

- Create a collection called `hw6`, configure dense and sparse vector for it. Let's use `"sentence-transformers/all-minilm-l6-v2"` and `"Qdrant/bm25"` for now. Don't forget to add `Modifier.IDF` to the config.  
- Upload points to the collection. Attach original data (`_id`, `title`, `text`) to the points as payload.

### 2.2 Hybrid search

Let's try implementing different strategies of building hybrid search.

#### 2.2.1 Fusion
For each record from `queries`:
Search for 10 nearest neighbours to the current query with dense and sparse embeddings simultaneously, fuse them with `RRF` and `DBSF`. (Use `RrfQuery` and try out several `k` values).

Use [MRR](https://en.wikipedia.org/wiki/Mean_reciprocal_rank) for evaluation.
No need to save the results, but they should be around `0.5-0.65` 

Just write the MRR you've computed as a comment.


#### 2.2.2 Late interaction

Create another collection (don't delete the first one, we'll reuse it later), configure `ColBERT` vectors.
We'll only do rescoring with ColBERT, in order not to build the HNSW graph and spend time and resources, set `m=0` for ColBERT (and only for it) to disable its graph construction.
Though, since we probably won't have enough points to pass optimizer's threshold, the HNSW graph won't be build anyway.

Upload the data the same way, but this time with ColBERT.

Do the same thing as in 2.2.1, rescore with ColBERT instead of fusion.

#### 2.2.3 Cross-encoders

We've tried rescoring with the embedding-based models, let's try using a dedicated reranking model now. 

Qdrant's `query_points` do not provide an interface to do reranking with cross-encoders.
So, you'd need to run your query first, and then manually apply cross-encoders to rerank the results.
`query_points` returns a **sorted** list of **scored** points. 

When dense and sparse models are used together, their scores come from different distributions. 
As a result, the absolute values are not directly comparable: it is unclear how important a given score is for each model, which makes it unclear how to properly sort the points.

For this reason, when at least one prefetch is provided, `query_points` requires you to explicitly define how the results from different models should be fused.

There is, however, a practical workaround: you can use `SampleQuery` to retrieve points in a _random_ order. 
The sampled candidates can then be reranked using a cross-encoder, which produces a single, comparable relevance score across all candidates.

Search like you did before, but rerank with a cross-encoder of your choice.

## 3. Competition

Since we’re running short on time before the course ends, let’s add an option to earn extra points toward your final grade.

Your task is to build a search pipeline and push accuracy as high as you can - minimum score to beat is `0.69`.
Train and test parts of the dataset are publicly available, thus, let's just combine them and evaluate the whole thing.

Constraints:

Use at most 5 models total (of any type: dense, sparse, rerankers, etc.). It can also be one-stage pipeline if you'd like.
P.S. don't use a model specifically fine-tuned for `SciFact` like (https://huggingface.co/Y-Research-Group/CSR-NV_Embed_v2-Retrieval-SciFACT)

After embeddings are precomputed, search time on CPU must be ≤ 1.5s end-to-end. (Real systems are often far stricter—think tens of milliseconds—but 1.5s is our limit.)

Don’t use huge “monster” models (e.g., ~8B parameters).

Do not delete experiments that did not improve the score, leave them in the code.
Save the final results in `final_hybrid_results.jsonl`, each record should contain `MRR`, `QPS`, `total_search_time`.

Leaderboard
There won’t be a public online leaderboard, but we can post the current best results in the course channel whenever someone asks.

1st place gets 2 points to the final grade.
2nd place gets 1.5 points to the final grade.
3rd place gets 1 point.
4th and 5th - 0.75
6-10 - 0.5
10-15 - 0.25
16-... - sorry...