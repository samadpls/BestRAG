# BestRAG { .sr-only }

![](assets/hero.svg){ .hero }

BestRAG is an open-source Python library for hybrid search in RAG (retrieval-augmented generation). It stores your documents in Postgres with [pgvector](https://github.com/pgvector/pgvector) or in [Qdrant](https://qdrant.tech/). It searches by meaning with dense vectors and by exact words with BM25, then merges the results. An optional cross-encoder can rerank the best results.

The built-in PDF reader, text splitter, embedding models, database calls and reranker run in compiled code. Python's lock is released during this work, so other threads can keep running.

## A first example

```python
from bestrag import BestRAG

url = "postgresql://postgres:postgres@localhost:5432/postgres"
# or Qdrant, on its gRPC port: url = "http://localhost:6334"

# Free and local: BestRAG runs a small embedding model on your machine.
rag = BestRAG(url, "my-docs")

# To use your provider without a model download, comment out the local line and uncomment these:
# from openai import OpenAI
# client = OpenAI()
# def embed(texts, query):
#     return [d.embedding for d in client.embeddings.create(model="text-embedding-3-small", input=texts).data]
# rag = BestRAG(url, "my-docs", dense_model=embed)

rag.add_pdf("handbook.pdf")
rag.add_text(
    "Part XJ-9042 replaces the filter in the 2019 pump.",
    "parts",
    {"team": "support"},
)

for hit in rag.search("which filter fits the old pump?", limit=5):
    print(f"{hit['score']:.2f}  {hit['source']}  page {hit['page']}")
    print(hit["text"][:80])
```

This example uses Postgres on your machine. The comment shows the Qdrant URL. The [Quickstart](quickstart.md) shows how to start either database.

Results come best first. Each hit is a dict with `text`, `score`, `source`, `page` and `metadata`. The first `BestRAG(...)` call with the local model downloads about 130 MB. Later calls reuse it. For provider examples, see [Embedding providers](embedding-providers.md).

## What it does

- **Hybrid search.** Dense vectors find passages that mean the same thing as your question. BM25 finds exact words, names and codes like `XJ-9042`, which dense search often misses. The two lists are merged with reciprocal rank fusion (RRF).
- **Postgres or Qdrant, wherever it runs.** Use any :ai-postgresql: Postgres with pgvector, such as :ai-aws: AWS RDS and Aurora, :ai-googlecloud: Google Cloud SQL and AlloyDB, :ai-azure: Azure, :ai-supabase: Supabase, :ai-neon: Neon or your own server. You can also use :ai-qdrant: Qdrant locally, self-hosted or on Qdrant Cloud. The URL selects the database.
- **Optional reranking.** A cross-encoder can read your question together with each candidate and put the best one first. It is off by default, because it makes each search much slower.
- **PDFs and plain text.** PDFs are read page by page. Every result keeps its page number. Indexing a name again replaces the old version, so you never get duplicates.
- **Fast.** A hybrid search takes milliseconds. `add_texts` embeds many documents in batches.
- **Any language.** PDFs are read in reading order, including right-to-left scripts. With local models, every chunk is checked against the model's token limit.
- **Your own embeddings, or the built-in local model.** Use :ai-openai: OpenAI, :ai-azure: Azure OpenAI, :ai-gemini: Gemini, :ai-cohere: Cohere, :ai-voyage: Voyage, :ai-mistral: Mistral, :ai-bedrock: Amazon Bedrock or :ai-ollama: Ollama through a small function, with no model download. Or let BestRAG run a small model on your machine.
- **Private by default.** With local models, your documents go only to your own database. Embedding and reranking run on your machine without API calls. With an API provider, your documents also go to that provider.

## Next steps

To let your AI coding agent read these docs, give it <https://bestrag.samadpls.com/llms.txt>.

<div class="grid cards" markdown>

-   **[Quickstart](quickstart.md)**

    Install BestRAG, start a database and search your first PDF.

-   **[Databases](databases.md)**

    Connect to Qdrant or Postgres, on your machine or in the cloud.

-   **[Indexing](indexing.md)**

    Add, replace and delete documents, and see how text is split.

-   **[Search](search.md)**

    Choose a search mode, read the results and turn on the reranker.

-   **[Embedding providers](embedding-providers.md)**

    Use OpenAI, Gemini, Cohere and others, with no model download.

-   **[API reference](api.md)**

    Every class, method and argument, with examples.

-   **[Benchmarks](benchmarks.md)**

    Search quality and speed on BEIR SciFact.

</div>
