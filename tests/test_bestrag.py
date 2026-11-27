"""End-to-end tests through the Python interface against a real Qdrant.

Start Qdrant first (see CONTRIBUTING.md). The first run downloads the models.
"""

import os
import urllib.request
import uuid
from pathlib import Path

import pytest

from bestrag import BestRAG, BestRAGError, read_pdf

GRPC = os.environ.get("BESTRAG_TEST_QDRANT", "http://localhost:6334")
REST = os.environ.get("BESTRAG_TEST_QDRANT_REST", "http://localhost:6333")
PDF = Path(__file__).parent / "fixtures" / "two-pages.pdf"
ARABIC_PDF = Path(__file__).parent / "fixtures" / "arabic.pdf"

DOCS = {
    "qdrant": "Qdrant is a vector database written in Rust. It stores dense and sparse vectors.",
    "lahore": "Lahore is a large city in Pakistan, known for its food, its gardens and the Badshahi Mosque.",
    "parts": "Part number XJ-9042 is the replacement filter for the 2019 garden pump.",
    "urdu": "لاہور پاکستان کا ایک بڑا شہر ہے اور اپنے کھانوں کے لیے مشہور ہے۔",
}


def qdrant_is_up() -> bool:
    try:
        with urllib.request.urlopen(f"{REST}/readyz", timeout=2):
            return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(not qdrant_is_up(), reason="needs a running Qdrant server")


def drop(collection: str) -> None:
    request = urllib.request.Request(f"{REST}/collections/{collection}", method="DELETE")
    urllib.request.urlopen(request, timeout=10).close()


@pytest.fixture(scope="module")
def collection():
    name = f"test-{uuid.uuid4().hex[:8]}"
    yield name
    drop(name)


@pytest.fixture(scope="module")
def rag(collection):
    rag = BestRAG(GRPC, collection)
    for name, text in DOCS.items():
        assert rag.add_text(text, name, {"topic": name}) == 1
    return rag


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("which database is written in rust?", "qdrant"),
        ("a famous mosque in pakistan", "lahore"),
        ("replacement filter for my pump", "parts"),
    ],
)
def test_hybrid_search_ranks_the_right_document_first(rag, query, expected):
    assert rag.search(query, limit=3)[0]["source"] == expected


def test_sparse_search_matches_exact_codes(rag):
    hits = rag.search("XJ-9042", mode="sparse")
    assert [h["source"] for h in hits] == ["parts"]


def test_dense_search_works_without_shared_words(rag):
    assert rag.search("famous mosque", limit=1, mode="dense")[0]["source"] == "lahore"


def test_non_ascii_text_is_stored_and_found(rag):
    hit = rag.search("لاہور", limit=1, mode="sparse")[0]
    assert hit["source"] == "urdu"
    assert hit["text"] == DOCS["urdu"]


def test_hits_carry_metadata(rag):
    hit = rag.search("vector database", limit=1)[0]
    assert hit["metadata"] == {"topic": "qdrant"}
    assert hit["page"] is None
    assert set(hit) == {"text", "score", "source", "page", "metadata"}


def test_adding_a_name_again_replaces_it(rag):
    rag.add_text(DOCS["parts"], "parts")
    assert rag.count("parts") == 1


def test_add_texts_indexes_many_documents_and_replaces_them(rag):
    documents = [{"name": f"batch-{i}", "text": f"Batch note {i} about lighthouses."} for i in range(5)]
    assert rag.add_texts(documents) == 5
    documents[0]["metadata"] = {"edited": True}
    assert rag.add_texts(documents) == 5
    assert rag.count("batch-0") == 1
    hit = rag.search("Batch note 0", limit=1, mode="sparse")[0]
    assert hit["metadata"] == {"edited": True}
    for doc in documents:
        rag.delete(doc["name"])


def test_add_texts_rejects_bad_batches(rag):
    with pytest.raises(BestRAGError, match="more than once"):
        rag.add_texts([{"name": "same", "text": "a"}, {"name": "same", "text": "b"}])
    with pytest.raises(ValueError, match="list of dicts"):
        rag.add_texts([("no", "tuples")])


def test_limit_above_twenty_is_honoured(rag):
    long_text = " ".join(f"Sentence {i} mentions the lighthouse keeper." for i in range(700))
    chunks = rag.add_text(long_text, "long")
    assert chunks > 25
    assert len(rag.search("lighthouse keeper", limit=25)) == 25
    rag.delete("long")


def test_delete_removes_only_that_name(rag):
    rag.add_text("A short note about tea.", "tea")
    before = rag.count()
    rag.delete("tea")
    assert rag.count("tea") == 0
    assert rag.count() == before - 1


def test_pdf_pages_are_kept(rag):
    assert rag.add_pdf(PDF) == 2
    hit = rag.search("zebrafish genetics", limit=1, mode="sparse")[0]
    assert (hit["source"], hit["page"]) == ("two-pages.pdf", 2)
    assert rag.search("aardvark", limit=1, mode="sparse")[0]["page"] == 1
    rag.delete("two-pages.pdf")


def test_arabic_pdf_text_comes_back_in_reading_order(rag):
    rag.add_pdf(ARABIC_PDF)
    text = rag.search("الإجازة", limit=1, mode="sparse")[0]["text"]
    for sentence in [
        "يحق للموظف الحصول على الإجازة السنوية بعد إكمال سنة كاملة من الخدمة.",
        "لا يجوز للموظف العمل لدى جهة أخرى أثناء الإجازة.",
        "تُصرف مكافأة نهاية الخدمة خلال 14 يوماً من تاريخ انتهاء العقد.",
        "نستخدم Qdrant للبحث في الوثائق.",
        "The word الإجازة means annual leave.",
    ]:
        assert sentence in text
    rag.delete("arabic.pdf")


def test_read_pdf_returns_each_page():
    pages = read_pdf(PDF)
    assert len(pages) == 2
    assert "aardvark" in pages[0] and "zebrafish" in pages[1]


def test_arabic_search_ignores_hamza_and_short_vowels(rag):
    rag.add_pdf(ARABIC_PDF)
    for query in ["الاجازة", "تصرف مكافاة"]:
        assert rag.search(query, limit=1, mode="sparse")[0]["source"] == "arabic.pdf"
    rag.delete("arabic.pdf")


def test_long_arabic_text_is_split_to_fit_the_model(rag):
    # 240 words is one chunk by word count, but far past bge-small's 512 tokens.
    chunks = rag.add_text(" ".join(["المكافأة السنوية للموظفين"] * 80), "arabic-long")
    assert chunks >= 3
    rag.delete("arabic-long")


def test_bad_input_raises(rag):
    with pytest.raises(BestRAGError, match="empty"):
        rag.add_text("   ", "blank")
    with pytest.raises(BestRAGError, match="metadata"):
        rag.add_text("text", "bad-metadata", [1, 2])
    with pytest.raises(BestRAGError):
        rag.add_pdf("missing.pdf")
    with pytest.raises(ValueError, match="mode"):
        rag.search("x", mode="fuzzy")


def test_the_optional_reranker_reorders_hits(collection, rag):
    reranking = BestRAG(GRPC, collection, reranker="jinaai/jina-reranker-v1-turbo-en")
    hits = reranking.search("which database is written in rust?", limit=3)
    assert hits[0]["source"] == "qdrant"
    # Cross-encoder scores are logits, not the small RRF scores of plain hybrid search.
    assert hits[0]["score"] != rag.search("which database is written in rust?", limit=1)[0]["score"]


def test_repr_never_shows_the_api_key(collection):
    rag = BestRAG(GRPC, collection, api_key="secret-key")
    assert "secret-key" not in repr(rag)
    assert "secret-key" not in str(rag)


def test_a_model_with_other_dimensions_is_refused(collection):
    with pytest.raises(BestRAGError, match="dimension"):
        BestRAG(GRPC, collection, dense_model="BAAI/bge-base-en-v1.5")
