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

Hybrid search merges the two lists with reciprocal rank fusion (RRF). Each list gives a chunk `1 / (position + 2)` points, with positions counted from 0, and the chunks with the most points come first. So a chunk near the top of both lists beats a chunk that only one search found. On Qdrant, Qdrant merges the lists itself. On Postgres, BestRAG runs both searches in SQL and merges them with the same formula.

A few edge cases:

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

So compare scores only within one search. BestRAG ranks results, but it does not decide whether a question has an answer at all. A score cutoff for that has to be tuned for each language and model.

Hits are chunks, not whole documents, so one long document can appear more than once. To list each document once, keep its first hit:

```python
seen = set()
for hit in rag.search("annual leave", limit=10):
    if hit["source"] not in seen:
        seen.add(hit["source"])
        print(hit["source"], hit["page"])
```

## The reranker

A reranker is a cross-encoder. It reads the question together with each candidate chunk and scores how well the chunk answers it. This is slower than a vector search, but it often puts the right chunk higher.

The reranker is off by default. To turn it on, name a reranker when you create `BestRAG`:

```python
rag = BestRAG(
    "http://localhost:6334",
    "my-docs",
    reranker="jinaai/jina-reranker-v1-turbo-en",
)
hits = rag.search("which filter fits the old pump?", limit=5)
```

With a reranker, `search` first collects a larger pool of candidates, `max(3 * limit, 30)` chunks, with the mode you chose. The reranker scores every candidate, and BestRAG returns the best `limit`.

!!! note "What it costs"
    With the reranker, each search takes about 0.8 seconds instead of milliseconds, and BestRAG uses more memory, because it loads a second model. The first time, it also downloads that model, about 150 MB for `jina-reranker-v1-turbo-en`. [Benchmarks](benchmarks.md) shows how much it helps.

You can open the same collection twice, once with a reranker and once without, and choose for each search:

```python
reranker = "jinaai/jina-reranker-v1-turbo-en"
fast = BestRAG("http://localhost:6334", "my-docs")
careful = BestRAG("http://localhost:6334", "my-docs", reranker=reranker)
```

## Choosing models

### The embedding model

`dense_model` sets the model that makes the dense vectors. The default is `BAAI/bge-small-en-v1.5`, a small English model of about 130 MB. You can name any text embedding model that [fastembed](https://github.com/Anush008/fastembed-rs) supports, by its Hugging Face name or by fastembed's model code. An unknown name raises `BestRAGError`.

A collection keeps the model it was created with. When you open a collection, BestRAG checks that the model makes vectors of the same size as the stored ones, and raises `BestRAGError` if it does not. It cannot tell apart two models with the same vector size, so always open a collection with the model it was built with. To change the model, index your documents into a new collection.

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

`reranker` takes any reranker that fastembed supports. `"jinaai/jina-reranker-v1-turbo-en"` is small and English, and `"jinaai/jina-reranker-v2-base-multilingual"` covers other languages.

### Where models are stored

Models download on first use and are reused after that. They go into `cache_dir` if you pass it, otherwise into `$BESTRAG_CACHE_DIR`, otherwise into a `bestrag` folder in your user cache folder. If `$HF_HOME` is set, fastembed uses it instead.
