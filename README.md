# BestRAG

![BestRAG: hybrid search for RAG. Rust inside, Python outside.](https://raw.githubusercontent.com/samadpls/BestRAG/main/assets/hero.png)

[![PyPI](https://img.shields.io/pypi/v/bestrag)](https://pypi.org/project/bestrag/)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](https://github.com/samadpls/BestRAG/blob/main/LICENSE)
[![Test](https://github.com/samadpls/BestRAG/actions/workflows/test.yml/badge.svg)](https://github.com/samadpls/BestRAG/actions/workflows/test.yml)

BestRAG stores your documents in [Qdrant](https://qdrant.tech/) and searches them in two ways at once: by meaning (dense vectors) and by exact words (BM25). It merges the two result lists and then reranks the best results with a cross-encoder. All of this runs in Rust, and you use it from Python.

## ✨ What it does

- **Hybrid search.** Dense vectors find passages that mean the same thing as your question. BM25 finds exact words, names and codes like `XJ-9042`, which dense search often misses. Qdrant merges both lists with reciprocal rank fusion (RRF).
- **Reranking.** A small cross-encoder reads your question together with each candidate and puts the best one first. You can turn it off or choose another model.
- **PDFs and plain text.** PDFs are read page by page, and every result keeps its page number. Indexing a name again replaces the old version, so you never get duplicates.
- **Rust underneath.** You call Python, and Rust does the PDF reading, chunking, embeddings and Qdrant calls. Python's lock is released during this work, so your other threads keep running.
- **Keyword search in any language.** BM25 keeps Urdu, Arabic and accented text as it is. The default dense model and reranker are English models, and you can choose multilingual ones.

## 🚀 Install

```bash
pip install bestrag
```

Wheels are built for Linux (x86_64 and arm64), macOS on Apple Silicon (13.4 or newer) and Windows (x64), for Python 3.10 and newer. On other systems, pip builds BestRAG from source, which needs [Rust](https://rustup.rs/).

You also need a Qdrant server. You can run one locally with Docker, or use the free tier of [Qdrant Cloud](https://cloud.qdrant.io/):

```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

## Usage

```python
from bestrag import BestRAG

rag = BestRAG("http://localhost:6334", "my-docs")

rag.add_pdf("handbook.pdf")
rag.add_text("Part XJ-9042 replaces the filter in the 2019 pump.", "parts", {"team": "support"})

for hit in rag.search("which filter fits the old pump?", limit=5):
    print(f"{hit['score']:.2f}  {hit['source']}  page {hit['page']}  {hit['text'][:60]}")

rag.delete("parts")
```

Each hit is a dict with `text`, `score`, `source`, `page` and `metadata`, with the best hit first. The first `BestRAG(...)` call downloads two models, about 290 MB in total, and later calls reuse them.

BestRAG talks to Qdrant over gRPC, so use port **6334**, not the REST port 6333. For Qdrant Cloud, pass your cluster URL with `:6334` and `api_key="..."`.

To index many documents, use `add_texts`. It embeds them in batches, so it is much faster than calling `add_text` in a loop:

```python
rag.add_texts([
    {"name": "faq-1", "text": "How do I reset my password? ..."},
    {"name": "faq-2", "text": "Can I change my plan later? ...", "metadata": {"lang": "en"}},
])
```

<details>
<summary><b>All options</b></summary>

`BestRAG(url, collection, *, api_key=None, dense_model="BAAI/bge-small-en-v1.5", reranker="jinaai/jina-reranker-v1-turbo-en", cache_dir=None)`

| Option | What it does |
| --- | --- |
| `url` | Qdrant's gRPC address, usually ending in `:6334` |
| `collection` | The Qdrant collection. BestRAG creates it if it does not exist. |
| `api_key` | The Qdrant API key, for Qdrant Cloud or a secured server. It never appears in `repr()`. |
| `dense_model` | Any text embedding model that [fastembed](https://github.com/Anush008/fastembed-rs) supports. A collection keeps the model it was created with. |
| `reranker` | Any fastembed reranker, for example `"BAAI/bge-reranker-base"` or `"jinaai/jina-reranker-v2-base-multilingual"`. Use `None` to turn reranking off. |
| `cache_dir` | Where models are downloaded. The default is `$BESTRAG_CACHE_DIR`, then your user cache folder. If `$HF_HOME` is set, fastembed uses it instead. |

| Method | What it does |
| --- | --- |
| `add_pdf(path, name=None, metadata=None)` | Indexes a PDF page by page. `name` defaults to the file name. |
| `add_text(text, name, metadata=None)` | Indexes plain text. |
| `add_texts(documents)` | Indexes a list of `{"name", "text", "metadata"}` dicts in batches. |
| `search(query, limit=10, mode="hybrid")` | Returns up to `limit` hits. `mode` can be `"hybrid"`, `"dense"` or `"sparse"` (BM25 only). |
| `delete(name)` | Removes everything stored under `name`. |
| `count(name=None)` | Counts stored chunks, for one name or for the whole collection. |

Errors from BestRAG raise `bestrag.BestRAGError`.

</details>

## 📊 How well it works

We measured BestRAG on [BEIR SciFact](https://huggingface.co/datasets/BeIR/scifact): 5,183 scientific abstracts and 300 test questions, where experts marked the abstracts that answer each question. Each search asked for 10 results.

| Setup | nDCG@10 | Recall@10 | Median search time |
| --- | --- | --- | --- |
| BM25 only | 0.655 | 0.764 | 0.3 ms |
| Dense only | 0.715 | 0.821 | 8 ms |
| Hybrid (dense + BM25, RRF) | 0.731 | 0.859 | 7 ms |
| **Hybrid + reranker (the default)** | **0.754** | **0.862** | 1.0 s |

- **Hybrid search finds more of the right abstracts** than either search alone. Recall@10 rises from 0.821 with dense only to 0.859.
- **The reranker puts the right abstracts higher.** nDCG@10 rises from 0.731 to 0.754. But it makes each search about one second slower on a laptop CPU. If speed matters more than these points, pass `reranker=None`.
- **The dense-only score is a sanity check.** 0.715 is close to the 0.713 that the model's authors report for SciFact on its [model card](https://huggingface.co/BAAI/bge-small-en-v1.5). Dense-only is also what version 0.3 did in practice, because its BM25 vectors were never searched.
- **Indexing** all 5,183 abstracts (7,084 chunks) took about 6 minutes with `add_texts`.

<details>
<summary><b>How we measured</b></summary>

- **Machine:** MacBook Pro (M3 Pro), CPU only, with Qdrant 1.20.2 running locally.
- **Models:** the defaults, `bge-small-en-v1.5` for dense vectors and `jina-reranker-v1-turbo-en` for reranking.
- **Metrics:** nDCG@10 rewards putting the right abstracts near the top. Recall@10 is the share of right abstracts found in the top 10. When one abstract fills two chunks in the top 10, it counts once.
- **Search time:** the median time per question, measured from Python.
- **Reproduce:** run `eval/scifact.py`, as described in [CONTRIBUTING.md](https://github.com/samadpls/BestRAG/blob/main/CONTRIBUTING.md).

</details>

<details>
<summary><b>How it works</b></summary>

**Indexing.** Text is split into chunks of 200 words with a 40-word overlap. A short tail joins the last chunk, so no chunk is tiny. Each chunk gets two vectors:

1. a dense vector from the embedding model, stored as `dense`;
2. a BM25 vector, stored as `bm25`. Qdrant's IDF modifier adds the inverse document frequency on the server.

Indexing a name first deletes what was stored under it before, so re-indexing a file never creates duplicates.

**Search.** Qdrant runs the dense search and the BM25 search together and merges them with RRF. The reranker then scores the top `max(3 × limit, 30)` results again, and BestRAG returns the best `limit`.

</details>

## Upgrading from 0.3

Version 0.4 is a rewrite with a new API. It cannot read collections made by 0.3, so please index your documents again. The old pure-Python code is kept at the [`v0.3.3` tag](https://github.com/samadpls/BestRAG/tree/v0.3.3), and it is no longer maintained.

## 🤝 Contributing

Contributions are welcome. [CONTRIBUTING.md](https://github.com/samadpls/BestRAG/blob/main/CONTRIBUTING.md) explains how to build the project, run the tests and run the evaluation.

## License

[MIT](https://github.com/samadpls/BestRAG/blob/main/LICENSE). Made by [Abdul Samad](https://samadpls.com).
