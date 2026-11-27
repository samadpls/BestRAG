# Contributing to BestRAG

Thank you for helping. BestRAG is a Rust library with a Python interface on top. The Rust code does all the work: reading PDFs, splitting text into chunks, making embeddings, talking to Postgres or Qdrant, and reranking. The Python package is built from the same crate with [maturin](https://www.maturin.rs/) and [pyo3](https://pyo3.rs/).

## How the code is organised

| Path | What it holds |
| --- | --- |
| `src/pdf.rs` | PDF text per page in reading order, including right-to-left scripts like Arabic |
| `src/text.rs` | Splitting text into overlapping chunks |
| `src/bm25.rs` | BM25 sparse vectors with Unicode and Arabic normalisation (the IDF part is added at search time) |
| `src/store.rs` | The `BestRag` type: models, chunking, indexing, search and reranking |
| `src/backend.rs` | Picking the database from the URL, and hiding passwords in messages |
| `src/qdrant.rs` | Qdrant storage and search |
| `src/postgres.rs` | Postgres storage with pgvector, BM25 in SQL, TLS rules |
| `src/python.rs` | The Python class, built only with the `python` feature |
| `python/bestrag/` | The Python package and its type stubs |
| `tests/` | End-to-end tests through Python, against real Qdrant and Postgres servers |
| `eval/` | The retrieval evaluation on BEIR SciFact |

## Set up

You need three things:

1. **Rust**, from [rustup.rs](https://rustup.rs/).
2. **uv**, from [docs.astral.sh/uv](https://docs.astral.sh/uv/).
3. **The databases.** The end-to-end tests run against Qdrant and against Postgres with pgvector, and skip whichever is not running. The easiest way is Docker:

   ```bash
   docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant:v1.20.2
   docker run -p 5433:5432 -e POSTGRES_USER=bestrag -e POSTGRES_DB=bestrag \
     -e POSTGRES_HOST_AUTH_METHOD=trust pgvector/pgvector:0.8.7-pg17
   ```

   You can also download the `qdrant` binary from [its releases page](https://github.com/qdrant/qdrant/releases), and build Postgres with [pgvector](https://github.com/pgvector/pgvector) from source.

Then build the package into a local virtual environment:

```bash
uv sync
```

Run `uv sync --reinstall-package bestrag` again after you change Rust code, so Python picks up the new build.

The first time BestRAG starts, it downloads the embedding model, about 130 MB. The optional reranker, which the tests also use, adds about 150 MB. They go into `$BESTRAG_CACHE_DIR` if you set it, or into your user cache folder. If `$HF_HOME` is set, they go there instead, because fastembed gives that variable priority.

## Run the checks

```bash
cargo fmt --check
cargo clippy --all-features --all-targets -- -D warnings
cargo test --lib          # fast unit tests, no database needed
uv run pytest             # end-to-end tests, against the databases above
```

Each end-to-end test runs once per database, and a database that is not running is skipped. The tests create their own collection and drop it at the end. To use other servers, set `BESTRAG_TEST_QDRANT` (gRPC, port 6334), `BESTRAG_TEST_QDRANT_REST` (REST, port 6333) and `BESTRAG_TEST_POSTGRES` (a `postgresql://` URL).

Please do not replace these tests with mocks. Version 0.3.3 had tests that mocked the embedding models, and they passed while the search itself was broken.

## Run the evaluation

The evaluation measures retrieval quality on [BEIR SciFact](https://huggingface.co/datasets/BeIR/scifact): 5,183 scientific abstracts and 300 test questions with expert labels.

```bash
mkdir -p data/eval/scifact && cd data/eval/scifact
curl -LO https://huggingface.co/datasets/BeIR/scifact/resolve/main/corpus/corpus-00000-of-00001.parquet
curl -LO https://huggingface.co/datasets/BeIR/scifact/resolve/main/queries/queries-00000-of-00001.parquet
curl -L -o qrels-test.tsv https://huggingface.co/datasets/BeIR/scifact-qrels/resolve/main/test.tsv
cd ../../..
uv run --group eval python eval/scifact.py
```

It indexes the abstracts into a `scifact` collection once, then reports nDCG@10, recall@10 and the median search time for each search mode. It uses the local Qdrant by default; add `--url postgresql://bestrag@localhost:5433/bestrag` to run it on Postgres.

## Pull requests

- Keep one change per pull request, and explain what it changes and why.
- Run the checks above before you open it.
- Library code should never print. Return an error in Rust; Python then raises `BestRAGError`.
- Never log or print an API key.
- If you change search quality, run the evaluation and put the numbers in the pull request.

## Releases (maintainers)

Pushing a tag like `v1.1.0` (or starting it by hand in the Actions tab) runs `.github/workflows/wheels.yml`. It builds wheels for Linux (x86_64 and arm64), macOS (Apple Silicon) and Windows (x64), plus a source distribution. Download the artifacts and upload them to PyPI with `uv publish`. There is no Intel Mac wheel, because ONNX Runtime ships no prebuilt library for it.
