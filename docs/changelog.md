# Changelog

## 1.2.0 (unreleased)

You can now pass Python functions for embeddings and reranking. Code written for 1.1 works unchanged.

- **`dense_model`** accepts `embed(texts, query)`, a function that returns one vector per text. Use it with OpenAI, Azure OpenAI, Gemini, Cohere, Voyage, Mistral, Amazon Bedrock or your own Ollama server. BestRAG does not download or load a model for the function. `query` is `True` for search questions and `False` for documents. Each call receives at most 64 texts. BestRAG checks for one vector per text, all the same size, with finite numbers. See [Embedding providers](embedding-providers.md).
- **`reranker`** accepts `rerank(query, texts)`, a function that returns one score per text. Use it with services such as Cohere Rerank or Voyage rerank.
- **Errors keep their cause.** When your function raises an exception, BestRAG raises `BestRAGError` with that exception as its `__cause__`.
- **Large vectors on Postgres.** New collections with more than 2,000 dimensions use `halfvec` with a `halfvec_cosine_ops` index. This includes the 3,072 dimensions of OpenAI's `text-embedding-3-large` or Gemini. It needs pgvector 0.7 or later. pgvector cannot index more than 4,000 dimensions, so larger vectors raise a clear error.
- **`repr()`** shows only the name of a function.

## 1.1.0 (2026-11-28)

BestRAG now works with Postgres too. Code written for 1.0 works unchanged.

- **Postgres with pgvector.** Point BestRAG at any Postgres with the pgvector extension: AWS RDS and Aurora, Google Cloud SQL and AlloyDB, Azure, Supabase, Neon or your own server. `postgresql://...` selects Postgres, and `https://...:6334` selects Qdrant. On Postgres, BM25 runs in plain SQL over a terms table. The two result lists are merged with the same formula as Qdrant. No extension is needed beyond pgvector. See [Databases](databases.md).
- **`backend=`** lets you set the database explicitly. It must match the URL. A misspelled name raises an error that lists the valid names.
- **`drop()`** deletes a whole collection.
- **TLS for Postgres.** A connection to a Postgres on another machine uses TLS by default, with the certificate and host name checked. BestRAG never turns TLS off on its own and never falls back to plain text. Inside a private network, such as Docker Compose, `sslmode=disable` turns TLS off. Add `sslrootcert=` for a provider's CA certificate, such as AWS RDS's bundle. The password can come from `$PGPASSWORD`, and it never appears in `repr()`, logs or errors.

## 1.0.0 (2026-11-27)

Version 1.0 is a full rewrite in Rust with a new Python API.

- **Hybrid search.** Dense and BM25 results are merged with reciprocal rank fusion (RRF). Version 0.3.3 did dense search only in practice, because its BM25 vectors were never searched.
- **Optional reranking.** A cross-encoder can rerank the best results. It is off by default, because it makes each search much slower.
- **PDFs and plain text.** PDFs are read page by page, and every result keeps its page number. All of the text is indexed, instead of only the start of each page.
- **Any language.** PDFs are read in reading order, including right-to-left scripts, and text in any script is kept. Every chunk is checked against the model's token limit.
- **`add_texts`** indexes many documents in batches.
- **`read_pdf(path)`** shows the text BestRAG reads from each page, so you can check a PDF before indexing it.
- **Replacing documents.** Indexing a name again replaces it, so you never get duplicates.
- **Fast.** Search is faster than in 0.3.3, and Python's lock is released while BestRAG works. See [Benchmarks](benchmarks.md).
- **The API key never appears in `repr()`.**
- **Wheels** for Linux (x86_64 and arm64), macOS on Apple Silicon (macOS 14 or newer) and Windows (x64), for Python 3.10 and newer.

**Breaking changes:**

- The API is new: `BestRAG(url, collection)`, with `add_pdf`, `add_text`, `add_texts`, `search`, `delete` and `count`.
- BestRAG talks to Qdrant over gRPC, so use port 6334.
- Collections made by 0.3 cannot be read, so please index your documents again.
- There is no wheel for Intel Macs, because ONNX Runtime ships no prebuilt library for them.

## 0.3.3 (2024-12-10)

This is the last version of the old pure-Python BestRAG. Its code is kept at the [`v0.3.3` tag](https://github.com/samadpls/BestRAG/tree/v0.3.3). It is no longer maintained.
