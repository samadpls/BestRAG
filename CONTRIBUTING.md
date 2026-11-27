# Contributing to BestRAG

Thank you for helping. BestRAG is a Rust library with a Python interface on top. The Rust code does all the work: reading PDFs, splitting text into chunks, making embeddings, talking to Qdrant, and reranking. The Python package is built from the same crate with [maturin](https://www.maturin.rs/) and [pyo3](https://pyo3.rs/).

## How the code is organised

| Path | What it holds |
| --- | --- |
| `src/text.rs` | PDF text per page, and splitting text into overlapping chunks |
| `src/bm25.rs` | BM25 sparse vectors (Qdrant adds the IDF part) |
| `src/store.rs` | The `BestRag` type: collection setup, indexing, hybrid search and reranking |
| `src/python.rs` | The Python class, built only with the `python` feature |
| `python/bestrag/` | The Python package and its type stubs |
| `tests/` | End-to-end tests through Python, against a real Qdrant |
| `eval/` | The retrieval evaluation on BEIR SciFact |

## Set up

You need three things:

1. **Rust**, from [rustup.rs](https://rustup.rs/).
2. **uv**, from [docs.astral.sh/uv](https://docs.astral.sh/uv/).
3. **A Qdrant server.** The easiest way is Docker:

   ```bash
   docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant:v1.20.2
   ```

   You can also download the `qdrant` binary from [its releases page](https://github.com/qdrant/qdrant/releases) and run it.

Then build the package into a local virtual environment:

```bash
uv sync
```

Run `uv sync --reinstall-package bestrag` again after you change Rust code, so Python picks up the new build.

The first time BestRAG starts, it downloads two models, about 290 MB in total: the dense model and the reranker. They go into `$BESTRAG_CACHE_DIR` if you set it, or into your user cache folder. If `$HF_HOME` is set, they go there instead, because fastembed gives that variable priority.

## Run the checks

```bash
cargo fmt --check
cargo clippy --all-features --all-targets -- -D warnings
cargo test --lib          # fast unit tests, no Qdrant needed
uv run pytest             # end-to-end tests, needs Qdrant on localhost
```

The end-to-end tests are skipped when no Qdrant server is running. They create their own collection and delete it at the end. To use another server, set `BESTRAG_TEST_QDRANT` (gRPC, port 6334) and `BESTRAG_TEST_QDRANT_REST` (REST, port 6333).

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

It indexes the abstracts into a `scifact` collection once, then reports nDCG@10, recall@10 and the median search time for each search mode.

## Pull requests

- Keep one change per pull request, and explain what it changes and why.
- Run the checks above before you open it.
- Library code should never print. Return an error in Rust; Python then raises `BestRAGError`.
- Never log or print an API key.
- If you change search quality, run the evaluation and put the numbers in the pull request.

## Releases (maintainers)

Pushing a tag like `v0.4.0` runs `.github/workflows/wheels.yml`. It builds wheels for Linux (x86_64 and arm64), macOS (Apple Silicon) and Windows (x64), plus a source distribution. Download the artifacts and upload them to PyPI with `uv publish`. There is no Intel Mac wheel, because ONNX Runtime ships no prebuilt library for it.
