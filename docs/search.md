# Search

`search` finds the chunks that best match a question:

```python
hits = rag.search("which filter fits the old pump?", limit=5)
```

It returns up to `limit` hits, 10 by default, with the best hit first.

## Search modes

| Mode | What it does | Good for |
| --- | --- | --- |
| `"hybrid"`, the default | Runs a dense search and a BM25 search, and merges the two lists | Most questions |
| `"dense"` | Dense vectors only: finds text with the same meaning, even when it shares no words with the question | Questions worded differently from your documents |
| `"sparse"` | BM25 only: finds exact words | Names, rare words and codes like `XJ-9042` |

```python
rag.search("XJ-9042", mode="sparse")
rag.search("a famous mosque in Pakistan", mode="dense")
```

Hybrid search merges the two lists with reciprocal rank fusion (RRF). Each list gives a chunk `1 / (position + 2)` points, with positions counted from 0. Chunks with the most points come first. A chunk near the top of both lists ranks above a chunk found by only one search. Qdrant merges the lists itself. On Postgres, BestRAG runs both searches in SQL and merges them with the same formula.

Search also handles these cases:

- A question with no letters or digits has no words for BM25. In hybrid mode, BestRAG then runs the dense search alone. In sparse mode, it returns an empty list.
- `limit=0` returns an empty list.
- Any other `mode` raises `ValueError`.

## The results

Each hit is a dict with five keys:

| Key | Type | What it holds |
| --- | --- | --- |
| `text` | `str` | The text of the chunk |
| `score` | `float` | Higher is better. The scale depends on the mode and on the reranker. |
| `source` | `str` | The name the document was indexed under |
| `page` | `int` or `None` | The PDF page, starting at 1, or `None` for plain text |
| `metadata` | `dict` or `None` | The metadata given when the document was indexed, or `None` |

The score means something different in each mode:

- In `"hybrid"` mode, it is the RRF score.
- In `"dense"` mode, it is the cosine similarity between the question and the chunk.
- In `"sparse"` mode, it is the BM25 score.
- With a reranker, it is the reranker's score.

Compare scores only within one search. BestRAG ranks results, but it does not decide whether the documents answer the question. To use a score cutoff for that, tune it for each language and model.

Each hit is a chunk, so a long document can appear more than once. To list each document once, keep its first hit:

```python
seen = set()
for hit in rag.search("annual leave", limit=10):
    if hit["source"] not in seen:
        seen.add(hit["source"])
        print(hit["source"], hit["page"])
```

## The reranker

A reranker is a cross-encoder. It reads the question together with each candidate chunk and scores how well the chunk answers it. This is slower than a vector search, but it often puts the right chunk higher.

The reranker is off by default. To turn it on, name a reranker when you create `BestRAG`, or pass your own function:

```python
# Free and local: a small reranker on your machine, downloaded once (about 150 MB).
rag = BestRAG("http://localhost:6334", "my-docs", reranker="jinaai/jina-reranker-v1-turbo-en")
# To use a provider such as Cohere Rerank, comment out the line above and uncomment this (see Embedding providers):
# rag = BestRAG("http://localhost:6334", "my-docs", reranker=rerank)
hits = rag.search("which filter fits the old pump?", limit=5)
```

With a reranker, `search` first collects `max(3 * limit, 30)` candidate chunks using your chosen mode. The reranker scores every candidate, then BestRAG returns the best `limit` results. A function `rerank(query, texts)` receives all candidates in one call. It returns one score per text, with higher scores ranked first. [Embedding providers](embedding-providers.md#reranker-recipes) has recipes for Cohere and Voyage.

!!! note "Local reranker speed and memory"
    With the local reranker, each search takes about 0.8 seconds instead of milliseconds. BestRAG also uses more memory because it loads a second model. The first use downloads that model, about 150 MB for `jina-reranker-v1-turbo-en`. [Benchmarks](benchmarks.md) shows how much it helps.

Open the same collection with and without a reranker to choose which to use for each search:

```python
reranker = "jinaai/jina-reranker-v1-turbo-en"
fast = BestRAG("http://localhost:6334", "my-docs")
careful = BestRAG("http://localhost:6334", "my-docs", reranker=reranker)
```

## Choosing models

### The embedding model

`dense_model` sets the model that makes the dense vectors. The default is `BAAI/bge-small-en-v1.5`, a small English model of about 130 MB. Choose any text embedding model that [fastembed](https://github.com/Anush008/fastembed-rs) supports. Use its Hugging Face name or fastembed model code. An unknown name raises `BestRAGError`.

`dense_model` can also be your own function, `embed(texts, query)`, for example one that calls OpenAI, Gemini or Cohere. Then nothing is downloaded. [Embedding providers](embedding-providers.md) has a recipe for each provider.

Use the same model for the life of a collection. When you open it, BestRAG checks that the model's vector size matches the stored vectors. A different size raises `BestRAGError`. BestRAG cannot tell apart two models with the same vector size. To change models, index your documents into a new collection.

### Other languages

BestRAG keeps text in any script, but its default models are English models. For other languages, choose multilingual ones:

```python
rag = BestRAG(
    "http://localhost:6334",
    "hr-docs",
    # "BAAI/bge-m3" is stronger, but about 2 GB.
    dense_model="intfloat/multilingual-e5-small",
    reranker="jinaai/jina-reranker-v2-base-multilingual",
)
```

### The reranker model

`reranker` takes any reranker that fastembed supports. `"jinaai/jina-reranker-v1-turbo-en"` is small and English, and `"jinaai/jina-reranker-v2-base-multilingual"` covers other languages. It can also be your own function, `rerank(query, texts)`.

### Where models are stored

Local models download on first use and are reused after that. BestRAG stores them in `cache_dir` if you pass it. Otherwise, it uses `$BESTRAG_CACHE_DIR` if set. Otherwise, it uses a `bestrag` folder in your user cache folder. If `$HF_HOME` is set, fastembed uses it instead. BestRAG checks the folder first and downloads only missing models. You can put models into a Docker image at build time. See [No model downloads on your servers](embedding-providers.md#no-model-downloads-on-your-servers).
