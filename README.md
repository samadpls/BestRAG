<p align="center">
  <img alt="BestRAG: hybrid search for RAG on Postgres and Qdrant" src="https://raw.githubusercontent.com/samadpls/BestRAG/main/assets/hero.svg" width="100%">
</p>

<p align="center">
  <a href="https://bestrag.samadpls.com/databases/"><img alt="Works with PostgreSQL and pgvector" src="https://img.shields.io/badge/works%20with-PostgreSQL%20%2B%20pgvector-336791?logo=postgresql&logoColor=white"></a>
  <a href="https://bestrag.samadpls.com/databases/"><img alt="Works with Qdrant" src="https://img.shields.io/badge/works%20with-Qdrant-DC244C?logo=qdrant&logoColor=white"></a>
  <a href="https://bestrag.samadpls.com"><img alt="Docs" src="https://img.shields.io/badge/docs-bestrag.samadpls.com-24376c"></a>
  <a href="https://samadpls.com"><img alt="by samadpls.com" src="https://img.shields.io/badge/by-samadpls.com-24376c"></a>
  <a href="https://github.com/samadpls/BestRAG/blob/main/LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-24376c"></a>
  <a href="https://pypi.org/project/bestrag/"><img alt="PyPI" src="https://img.shields.io/pypi/v/bestrag?color=24376c&label=pypi"></a>
  <a href="https://pepy.tech/projects/bestrag"><img alt="Downloads" src="https://img.shields.io/pepy/dt/bestrag?label=downloads&color=24376c"></a>
  <img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=fff">
</p>

<p align="center">
  <b>Better RAG results from the database you already run.</b><br>
  BestRAG finds passages by meaning and by exact words, then merges the two lists. Use default local models or your provider's.
</p>

---

## ✨ What it does

- **Postgres or Qdrant, wherever they run.** Any <img src="https://cdn.simpleicons.org/postgresql" height="14" alt=""> Postgres with pgvector works, such as AWS RDS, <img src="https://cdn.simpleicons.org/googlecloud" height="14" alt=""> Google Cloud SQL, Azure, <img src="https://cdn.simpleicons.org/supabase" height="14" alt=""> Supabase or <img src="https://cdn.simpleicons.org/neon" height="14" alt=""> Neon. So does <img src="https://cdn.simpleicons.org/qdrant" height="14" alt=""> Qdrant, on your machine or on Qdrant Cloud.
- **Finds exact words too.** BM25 finds names and codes like `XJ-9042`, even when search by meaning misses them.
- **Your own embeddings, or the built-in local model.** Use OpenAI, Cohere, Gemini and more with no model download. By default, BestRAG runs a small model on your machine.
- **PDFs and text.** Every result keeps its page number. Indexing a name again replaces the old version.
- **Fast.** A hybrid search takes a few milliseconds, with no API calls by default.
- **Optional reranker.** A cross-encoder can put the best result first. It is off by default, because it is slower.
- **Many languages.** PDFs are read in reading order, including right-to-left scripts.

## 🚀 Install

```bash
uv add bestrag        # with uv
pip install bestrag   # with pip
```

You also need a database. To try one on your machine, start it with Docker:

```bash
docker run -p 5432:5432 -e POSTGRES_PASSWORD=postgres pgvector/pgvector:pg17   # Postgres with pgvector
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant                            # or Qdrant
```

## Quick start

```python
from bestrag import BestRAG

# Free and local: BestRAG runs a small embedding model on your machine.
rag = BestRAG("postgresql://postgres:postgres@localhost:5432/postgres", "my-docs")
# or Qdrant, on its gRPC port 6334: BestRAG("http://localhost:6334", "my-docs")

# To use your provider without a model download, comment out the local line and uncomment these:
# from openai import OpenAI
# client = OpenAI()
# def embed(texts, query):
#     return [d.embedding for d in client.embeddings.create(model="text-embedding-3-small", input=texts).data]
# rag = BestRAG("postgresql://postgres:postgres@localhost:5432/postgres", "my-docs", dense_model=embed)

rag.add_pdf("handbook.pdf")
rag.add_text("Part XJ-9042 replaces the filter in the 2019 pump.", "parts", {"team": "support"})

for hit in rag.search("which filter fits the old pump?", limit=5):
    print(f"{hit['score']:.2f}  {hit['source']}  page {hit['page']}  {hit['text'][:60]}")
```

Results come best first. Each hit is a dict with `text`, `score`, `source`, `page` and `metadata`. The first run with the local model downloads about 130 MB. [Embedding providers](https://bestrag.samadpls.com/embedding-providers/) has recipes for OpenAI, Azure OpenAI, Gemini, Cohere, Voyage, Mistral, Bedrock and Ollama.

The [documentation](https://bestrag.samadpls.com) covers every option and each cloud. Its [llms.txt](https://bestrag.samadpls.com/llms.txt) makes the docs available to your AI coding agent.

## How well it works

![Search quality on BEIR SciFact: hybrid search beats dense or BM25 alone, and the optional reranker adds a little more](https://raw.githubusercontent.com/samadpls/BestRAG/main/assets/chart-quality.png)

On BEIR SciFact, hybrid search finds more of the right abstracts than dense or BM25 search alone. Postgres and Qdrant give the same quality. [Benchmarks](https://bestrag.samadpls.com/benchmarks/) has the numbers for both.

## 🤗 Try it on Hugging Face Spaces

Try BestRAG in your browser on [Hugging Face Spaces](https://huggingface.co/spaces/samadpls/BestRAG), with your own Postgres or Qdrant URL.

## Upgrading from 0.3

Version 1.0 is a rewrite with a new API. It cannot read collections made by 0.3, so index your documents again. The old pure-Python code is kept at the [`v0.3.3` tag](https://github.com/samadpls/BestRAG/tree/v0.3.3). It is no longer maintained.

## Author

<p>
  <a href="https://samadpls.com"><img alt="samadpls.com" src="https://www.samadpls.com/apple-touch-icon.png" height="40"></a>&nbsp;
  <a href="https://github.com/samadpls"><img alt="GitHub" src="https://skillicons.dev/icons?i=github" height="40"></a>&nbsp;
  <a href="https://x.com/samadpls"><img alt="X" src="https://raw.githubusercontent.com/samadpls/BestRAG/main/assets/icons/x.svg" height="40"></a>&nbsp;
  <a href="https://www.linkedin.com/in/samadpls/"><img alt="LinkedIn" src="https://skillicons.dev/icons?i=linkedin" height="40"></a>
</p>

## License

[MIT](https://github.com/samadpls/BestRAG/blob/main/LICENSE)
