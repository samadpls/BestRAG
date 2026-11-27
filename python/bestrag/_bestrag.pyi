from collections.abc import Callable, Sequence
from os import PathLike
from typing import Any, Literal, TypedDict

__version__: str

def read_pdf(path: str | PathLike[str]) -> list[str]: ...

class Hit(TypedDict):
    text: str
    score: float
    source: str
    page: int | None
    metadata: dict[str, Any] | None

class Document(TypedDict, total=False):
    name: str  # required
    text: str  # required
    metadata: dict[str, Any]

class BestRAGError(Exception): ...

class BestRAG:
    def __init__(
        self,
        url: str,
        collection: str,
        *,
        backend: Literal["qdrant", "postgres"] | None = None,
        api_key: str | None = None,
        dense_model: str | Callable[[list[str], bool], Sequence[Sequence[float]]] = "BAAI/bge-small-en-v1.5",
        reranker: str | Callable[[str, list[str]], Sequence[float]] | None = None,
        cache_dir: str | PathLike[str] | None = None,
    ) -> None: ...
    def add_pdf(
        self,
        path: str | PathLike[str],
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int: ...
    def add_text(self, text: str, name: str, metadata: dict[str, Any] | None = None) -> int: ...
    def add_texts(self, documents: list[Document]) -> int: ...
    def search(
        self,
        query: str,
        limit: int = 10,
        mode: Literal["hybrid", "dense", "sparse"] = "hybrid",
    ) -> list[Hit]: ...
    def delete(self, name: str) -> None: ...
    def count(self, name: str | None = None) -> int: ...
    def drop(self) -> None: ...
