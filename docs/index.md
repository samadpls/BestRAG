# BestRAG { .sr-only }

![](assets/hero.png){ .hero }

BestRAG is an open-source Python library for hybrid search in RAG (retrieval-augmented generation). It stores your documents in Postgres with [pgvector](https://github.com/pgvector/pgvector) or in [Qdrant](https://qdrant.tech/), and it searches them in two ways at once: by meaning, with dense vectors, and by exact words, with BM25. It merges the two result lists, and it can rerank the best results with a cross-encoder.

The work runs in compiled code: reading PDFs, splitting text into chunks, making embeddings, talking to the database and reranking. Python's lock is released while it runs, so your other threads keep going.

## A first example

```python
from bestrag import BestRAG

url = "postgresql://postgres:postgres@localhost:5432/postgres"
rag = BestRAG(url, "my-docs")
# or Qdrant, on its gRPC port: BestRAG("http://localhost:6334", "my-docs")

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

This example uses a Postgres on your own machine, and the comment shows the Qdrant URL. The [Quickstart](quickstart.md) shows how to start either database.

Each hit is a dict with `text`, `score`, `source`, `page` and `metadata`, with the best hit first. The first `BestRAG(...)` call downloads the embedding model, about 130 MB, and later calls reuse it.

## What it does

- **Hybrid search.** Dense vectors find passages that mean the same thing as your question. BM25 finds exact words, names and codes like `XJ-9042`, which dense search often misses. BestRAG merges the two lists with reciprocal rank fusion (RRF).
- **Postgres or Qdrant, wherever it runs.** Use any :simple-postgresql: Postgres with pgvector, such as AWS RDS and Aurora, :simple-googlecloud: Google Cloud SQL and AlloyDB, Azure, :simple-supabase: Supabase, :brands-neon: Neon or your own server, or use :brands-qdrant: Qdrant locally, self-hosted or on Qdrant Cloud. The URL decides which one.
- **Optional reranking.** A cross-encoder can read your question together with each candidate and put the best one first. It is off by default, because it makes each search much slower.
- **PDFs and plain text.** PDFs are read page by page, and every result keeps its page number. Indexing a name again replaces the old version, so you never get duplicates.
- **Fast.** A hybrid search takes milliseconds, and `add_texts` embeds many documents in batches.
- **Any language.** PDFs are read in reading order, including right-to-left scripts, and every chunk is checked against the model's token limit.
- **Private.** Your documents go only to your own database. Embedding and reranking run on your machine, with no calls to any API.

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

-   **[API reference](api.md)**

    Every class, method and argument, with examples.

-   **[Benchmarks](benchmarks.md)**

    Search quality and speed on BEIR SciFact.

</div>
