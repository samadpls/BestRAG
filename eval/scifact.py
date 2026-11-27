"""Measure BestRAG's retrieval on BEIR SciFact (5,183 abstracts, 300 test queries).

Download the data into data/eval/scifact first (see CONTRIBUTING.md), start
Qdrant, then run:

    uv run --group eval python eval/scifact.py
"""

import argparse
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from bestrag import BestRAG

DATA = Path("data/eval/scifact")
SETUPS = [
    ("BM25 only", "sparse", False),
    ("Dense only", "dense", False),
    ("Hybrid (dense + BM25, RRF), the default", "hybrid", False),
    ("Hybrid + reranker", "hybrid", True),
]


def load():
    corpus = pq.read_table(DATA / "corpus-00000-of-00001.parquet").to_pylist()
    queries = {q["_id"]: q["text"] for q in pq.read_table(DATA / "queries-00000-of-00001.parquet").to_pylist()}
    qrels = defaultdict(dict)
    for line in (DATA / "qrels-test.tsv").read_text().splitlines()[1:]:
        query_id, doc_id, score = line.split("\t")
        qrels[query_id][doc_id] = int(score)
    return corpus, {q: queries[q] for q in qrels}, qrels


def ndcg_at_10(ranked, relevant):
    dcg = sum(relevant.get(doc, 0) / math.log2(i + 2) for i, doc in enumerate(ranked[:10]))
    ideal = sorted(relevant.values(), reverse=True)[:10]
    return dcg / sum(gain / math.log2(i + 2) for i, gain in enumerate(ideal))


def recall_at_10(ranked, relevant):
    return len(set(ranked[:10]) & set(relevant)) / len(relevant)


def unique_sources(hits):
    seen = []
    for hit in hits:
        if hit["source"] not in seen:
            seen.append(hit["source"])
    return seen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:6334")
    parser.add_argument("--collection", default="scifact")
    args = parser.parse_args()

    corpus, queries, qrels = load()
    plain = BestRAG(args.url, args.collection)

    index_seconds = None
    if plain.count() == 0:
        documents = [{"name": d["_id"], "text": f"{d['title']}. {d['text']}"} for d in corpus]
        start = time.perf_counter()
        for i in range(0, len(documents), 256):
            plain.add_texts(documents[i : i + 256])
        index_seconds = time.perf_counter() - start
        print(f"indexed {len(corpus)} documents ({plain.count()} chunks) in {index_seconds:.0f} s")

    reranking = BestRAG(args.url, args.collection, reranker="jinaai/jina-reranker-v1-turbo-en")
    results = []
    for label, mode, rerank in SETUPS:
        rag = reranking if rerank else plain
        ndcg, recall, seconds = [], [], []
        for query_id, query in queries.items():
            start = time.perf_counter()
            # Ten chunks, as a user would ask for. A long abstract can fill two
            # of them, so ranks are counted per distinct document.
            hits = rag.search(query, limit=10, mode=mode)
            seconds.append(time.perf_counter() - start)
            ranked = unique_sources(hits)
            ndcg.append(ndcg_at_10(ranked, qrels[query_id]))
            recall.append(recall_at_10(ranked, qrels[query_id]))
        row = {
            "setup": label,
            "ndcg@10": statistics.mean(ndcg),
            "recall@10": statistics.mean(recall),
            "median_ms": statistics.median(seconds) * 1000,
        }
        results.append(row)
        print(f"| {label} | {row['ndcg@10']:.3f} | {row['recall@10']:.3f} | {row['median_ms']:.0f} ms |")

    out = DATA.parent / "scifact-results.json"
    out.write_text(json.dumps({"queries": len(queries), "documents": len(corpus), "index_seconds": index_seconds, "results": results}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
