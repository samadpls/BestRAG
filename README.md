# BestRAG

![BestRAG](https://raw.githubusercontent.com/samadpls/BestRAG/main/assets/hero.png)

[![PyPI](https://img.shields.io/pypi/v/bestrag)](https://pypi.org/project/bestrag/)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](https://github.com/samadpls/BestRAG/blob/main/LICENSE)
[![Test](https://github.com/samadpls/BestRAG/actions/workflows/test.yml/badge.svg?event=pull_request)](https://github.com/samadpls/BestRAG/actions/workflows/test.yml)
[![Works with PostgreSQL and pgvector](https://img.shields.io/badge/works%20with-PostgreSQL%20%2B%20pgvector-336791?logo=postgresql&logoColor=white)](https://bestrag.samadpls.com/databases/)
[![Works with Qdrant](https://img.shields.io/badge/works%20with-Qdrant-24376c?logo=qdrant&logoColor=white)](https://bestrag.samadpls.com/databases/)

BestRAG stores your documents in Postgres with [pgvector](https://github.com/pgvector/pgvector) or in [Qdrant](https://qdrant.tech/), and searches them in two ways at once: by meaning (dense vectors) and by exact words (BM25). It merges the two result lists, and it can rerank the best results with a cross-encoder.

Built and maintained by [samadpls](https://samadpls.com). Documentation: [bestrag.samadpls.com](https://bestrag.samadpls.com).

## ✨ What it does

- **Hybrid search.** Dense vectors find passages that mean the same thing as your question. BM25 finds exact words, names and codes like `XJ-9042`. The two lists are merged with reciprocal rank fusion (RRF).
- **Postgres or Qdrant, wherever they run.** Use any <img src="https://cdn.simpleicons.org/postgresql" height="14" alt=""> Postgres with pgvector, such as AWS RDS, <img src="https://cdn.simpleicons.org/googlecloud" height="14" alt=""> Google Cloud SQL, Azure, <img src="https://cdn.simpleicons.org/supabase" height="14" alt=""> Supabase or <img src="https://cdn.simpleicons.org/neon" height="14" alt=""> Neon, or use <img src="https://cdn.simpleicons.org/qdrant" height="14" alt=""> Qdrant on your machine or on Qdrant Cloud. The URL decides which.
- **PDFs and plain text.** Every result keeps its page number, and indexing a name again replaces the old version.
- **Optional reranking.** A cross-encoder can put the best result first. It is off by default, because it makes each search much slower.
- **Fast and private.** A hybrid search takes milliseconds. Embedding and reranking run on your machine, and your documents go only to your own database.
- **Any language.** PDFs are read in reading order, including right-to-left scripts, and long chunks are split to fit the model. Text written without spaces, such as Chinese or Thai, can still be cut off for the dense model.

## 🚀 Install

With [uv](https://docs.astral.sh/uv/) (recommended):

```bash
uv add bestrag            # in a uv project
uv pip install bestrag    # in any environment
```

With pip:

```bash
pip install bestrag
```

You also need a database. To try BestRAG on your machine, start one with Docker:

```bash
# Postgres with pgvector
docker run -p 5432:5432 -e POSTGRES_PASSWORD=postgres pgvector/pgvector:pg17

# or Qdrant
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

## Quick start

```python
from bestrag import BestRAG

rag = BestRAG("postgresql://postgres:postgres@localhost:5432/postgres", "my-docs")
# or Qdrant, on its gRPC port 6334: BestRAG("http://localhost:6334", "my-docs")

rag.add_pdf("handbook.pdf")
rag.add_text("Part XJ-9042 replaces the filter in the 2019 pump.", "parts", {"team": "support"})

for hit in rag.search("which filter fits the old pump?", limit=5):
    print(f"{hit['score']:.2f}  {hit['source']}  page {hit['page']}  {hit['text'][:60]}")
```

Each hit is a dict with `text`, `score`, `source`, `page` and `metadata`, with the best hit first. The first `BestRAG(...)` call downloads the embedding model, about 130 MB. The [Quickstart](https://bestrag.samadpls.com/quickstart/) explains each step.

## How well it works

![Search quality on BEIR SciFact: hybrid search beats dense or BM25 alone, and the optional reranker adds a little more](https://raw.githubusercontent.com/samadpls/BestRAG/main/assets/chart-quality.png)

On BEIR SciFact, hybrid search finds more of the right abstracts than dense or BM25 search alone, and the optional reranker ranks them a little higher. Postgres and Qdrant give the same quality, and [Benchmarks](https://bestrag.samadpls.com/benchmarks/) has the numbers for both.

## Documentation

The full documentation is at [bestrag.samadpls.com](https://bestrag.samadpls.com):

- [Quickstart](https://bestrag.samadpls.com/quickstart/): install BestRAG, start a database, then index and search a PDF.
- [Databases](https://bestrag.samadpls.com/databases/): Postgres and Qdrant, hosted databases, TLS and passwords.
- [Indexing](https://bestrag.samadpls.com/indexing/): PDFs, text, metadata, chunking and languages.
- [Search](https://bestrag.samadpls.com/search/): search modes, results, the reranker and choosing models.
- [API reference](https://bestrag.samadpls.com/api/): every class, method and argument.
- [Benchmarks](https://bestrag.samadpls.com/benchmarks/): search quality and speed, and how to measure them again.
- [llms.txt](https://bestrag.samadpls.com/llms.txt): these docs as Markdown, to give to your AI coding agent.

To build BestRAG yourself and run the tests, see [CONTRIBUTING.md](https://github.com/samadpls/BestRAG/blob/main/CONTRIBUTING.md).

## 🤗 Try it on Hugging Face Spaces

You can try BestRAG in your browser on [Hugging Face Spaces](https://huggingface.co/spaces/samadpls/BestRAG). Bring your own Postgres or Qdrant URL.

## Upgrading from 0.3

Version 1.0 is a rewrite with a new API, and it cannot read collections made by 0.3, so please index your documents again. The old pure-Python code is kept at the [`v0.3.3` tag](https://github.com/samadpls/BestRAG/tree/v0.3.3), and it is no longer maintained.

## Author

BestRAG is built and maintained by **[samadpls](https://github.com/samadpls)**.

- Website: [samadpls.com](https://samadpls.com)
- X: [@samadpls](https://x.com/samadpls)
- LinkedIn: [in/samadpls](https://www.linkedin.com/in/samadpls/)

## License

[MIT](https://github.com/samadpls/BestRAG/blob/main/LICENSE).
