# Benchmarks

This page holds all of BestRAG's measured results. They come from the evaluation script in the repository, so you can run it yourself.

## Search quality on BEIR SciFact

[BEIR SciFact](https://huggingface.co/datasets/BeIR/scifact) has 5,183 scientific abstracts and 300 test questions. For each question, experts marked the abstracts that answer it. Each search asked for 10 results.

### Qdrant

| Setup | nDCG@10 | Recall@10 | Median search time |
| --- | --- | --- | --- |
| BM25 only | 0.655 | 0.764 | 0.2 ms |
| Dense only | 0.715 | 0.821 | 8 ms |
| **Hybrid (dense + BM25, RRF), the default** | **0.734** | **0.860** | **7 ms** |
| Hybrid + reranker (optional) | 0.754 | 0.864 | 813 ms |

### Postgres with pgvector

| Setup | nDCG@10 | Recall@10 | Median search time |
| --- | --- | --- | --- |
| BM25 only | 0.655 | 0.764 | 29 ms |
| Dense only | 0.715 | 0.821 | 7 ms |
| **Hybrid (dense + BM25, RRF), the default** | **0.733** | **0.856** | **36 ms** |
| Hybrid + reranker (optional) | 0.754 | 0.862 | 807 ms |

### What the numbers show

- **Hybrid search finds more of the right abstracts** than either search alone. Recall@10 rises from 0.821 with dense only to 0.860 with hybrid search on Qdrant, and to 0.856 on Postgres.
- **The optional reranker puts the right abstracts a little higher.** nDCG@10 rises from 0.734 to 0.754, but each search takes about 0.8 seconds instead of 7 ms. So it is off by default.
- **Qdrant and Postgres give the same quality.** The BM25 and dense scores are identical on both databases, and the hybrid scores differ by at most 0.004.
- **Hybrid search is slower on Postgres:** about 36 ms against 7 ms on Qdrant, because BM25 runs as SQL over the terms table. That is still fast for RAG.
- **The dense-only score is a sanity check.** 0.715 is close to the 0.713 that the model's authors report for SciFact on its [model card](https://huggingface.co/BAAI/bge-small-en-v1.5). Dense-only is also what version 0.3 did in practice, because its BM25 vectors were never searched.
- **Indexing** all 5,183 abstracts (7,088 chunks) with `add_texts` took about 5 minutes, on Qdrant and on Postgres.

## Compared with version 0.3.3

On the same 40-page PDF with the default settings, both versions using Qdrant:

| | 0.3.3 | 1.0 |
| --- | --- | --- |
| Median search time | 16 ms | **5.6 ms** (2.8 times faster) |
| Time to index the PDF | 5.7 s | 5.8 s |
| Share of the text it indexes | 49% (each page was cut at 512 tokens) | **100%** |
| Peak memory | 688 MB | 864 MB |

Version 1.0 indexes twice as much text in the same time. It uses a little more memory, mostly because it loads the full-precision embedding model and embeds in batches.

## How it was measured

- **Machine:** a MacBook Pro, CPU only, with Postgres 17.12 and pgvector 0.8.7, or Qdrant 1.20.2, running on the same machine.
- **Models:** `bge-small-en-v1.5` for dense vectors (the default) and `jina-reranker-v1-turbo-en` for the reranker rows.
- **Metrics:** nDCG@10 rewards putting the right abstracts near the top. Recall@10 is the share of the right abstracts found in the top 10. When one abstract fills two chunks in the top 10, it counts once.
- **Search time:** the median time per question, measured from Python.
- **Indexing:** the script sends the abstracts to `add_texts` in batches of 256.

## Run it yourself

The evaluation is `eval/scifact.py` in the [repository](https://github.com/samadpls/BestRAG). Run it from a clone that is set up as [CONTRIBUTING.md](https://github.com/samadpls/BestRAG/blob/main/CONTRIBUTING.md) describes, with a database running, such as the one from the [Quickstart](quickstart.md).

First, download the data:

```bash
mkdir -p data/eval/scifact && cd data/eval/scifact
BASE=https://huggingface.co/datasets/BeIR
curl -LO $BASE/scifact/resolve/main/corpus/corpus-00000-of-00001.parquet
curl -LO $BASE/scifact/resolve/main/queries/queries-00000-of-00001.parquet
curl -L -o qrels-test.tsv $BASE/scifact-qrels/resolve/main/test.tsv
cd ../../..
```

Then run the script against your database:

=== "Postgres"

    ```bash
    uv run --group eval python eval/scifact.py \
      --url postgresql://postgres:postgres@localhost:5432/postgres
    ```

=== "Qdrant"

    ```bash
    uv run --group eval python eval/scifact.py
    ```

The script indexes the abstracts into a `scifact` collection, then reports nDCG@10, Recall@10 and the median search time for each setup. If the collection already holds documents, it skips indexing and only searches. Use `--collection` to choose another name. The results are also written to `data/eval/scifact-results.json`.
