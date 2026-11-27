# API reference

Import the public API from the `bestrag` package:

```python
from bestrag import BestRAG, BestRAGError, read_pdf
```

## BestRAG

```python
BestRAG(
    url: str,
    collection: str,
    *,
    backend: Literal["qdrant", "postgres"] | None = None,
    api_key: str | None = None,
    dense_model: str | EmbedFunction = "BAAI/bge-small-en-v1.5",
    reranker: str | RerankFunction | None = None,
    cache_dir: str | PathLike[str] | None = None,
)

# embed(texts, query) returns one vector per text.
EmbedFunction = Callable[[list[str], bool], Sequence[Sequence[float]]]
# rerank(query, texts) returns one score per text. Higher is better.
RerankFunction = Callable[[str, list[str]], Sequence[float]]
```

Connects to the database and creates the collection if needed. For local models, BestRAG opens the collection before downloading or loading them. This means a wrong URL fails before a large download. Models download on first use. When `dense_model` is a function, BestRAG first calls it once with one short text to find the vector size. Every argument after `collection` is keyword-only.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `url` | `str` | required | A `postgresql://` URL, or Qdrant's gRPC address, usually ending in `:6334`. The scheme picks the database. See [Databases](databases.md). |
| `collection` | `str` | required | The collection, or Postgres table, to use. BestRAG creates it if it does not exist. On Postgres, the name may use letters, digits, `_` and `-`, up to 48 characters. |
| `backend` | `"qdrant"`, `"postgres"` or `None` | `None` | Sets the database explicitly. The URL selects it by default. If given, this name must match the URL. |
| `api_key` | `str` or `None` | `None` | The Qdrant API key, for Qdrant Cloud or a secured server. Not allowed with Postgres, which takes its password from the URL or `$PGPASSWORD`. |
| `dense_model` | `str` or a function | `"BAAI/bge-small-en-v1.5"` | Any text embedding model that fastembed supports, or your own function `embed(texts, query)` that returns one vector per text. `query` is `True` for search questions and `False` for documents. With a function, nothing is downloaded. See [Embedding providers](embedding-providers.md). A collection keeps the model it was created with. |
| `reranker` | `str`, a function or `None` | `None` | Off by default. Name any fastembed reranker to turn reranking on, such as `"jinaai/jina-reranker-v1-turbo-en"`, or pass your own function `rerank(query, texts)` that returns one score per text. Higher scores are better. |
| `cache_dir` | `str`, `PathLike` or `None` | `None` | Where models are downloaded. The default is `$BESTRAG_CACHE_DIR`, then a `bestrag` folder in your user cache folder. If `$HF_HOME` is set, fastembed uses it instead. Not used for functions. |

**Raises** `BestRAGError` when:

- `dense_model` or `reranker` is not a model that fastembed supports;
- the URL scheme is not one BestRAG knows, or `backend` is unknown or does not match the URL;
- `api_key` is given with a Postgres URL;
- the Postgres URL is not valid or has an unknown `sslmode`;
- the `sslrootcert` file cannot be read or holds no certificates;
- the collection name breaks the Postgres naming rules;
- BestRAG cannot connect, for example because the server has no TLS (the message then suggests `sslmode=disable`), or cannot create the collection, for example because pgvector is not enabled;
- the existing collection stores vectors of a different size from those returned by `dense_model`, or was not created by BestRAG;
- a model cannot be downloaded or loaded;
- the `dense_model` function raises an exception, which becomes the error's `__cause__`, or returns something other than one vector of numbers for its text;
- on Postgres, a new collection's vectors have more than 4,000 dimensions, or more than 2,000 with pgvector older than 0.7.

**Raises** `TypeError` when `dense_model` or `reranker` is neither a string nor a function.

`repr()` and `str()` show the URL without its password. They never show the API key. Functions appear only by name, such as `dense_model=embed`.

**Example**

```python
from bestrag import BestRAG

# Free and local: BestRAG runs a small embedding model on your machine.
rag = BestRAG("http://localhost:6334", "my-docs")
print(rag)  # BestRAG(url="http://localhost:6334/", collection="my-docs")

# To use your provider without a model download, comment out the local example and uncomment below (see Embedding providers):
# rag = BestRAG("http://localhost:6334", "my-docs", dense_model=embed)
# print(rag)  # BestRAG(url="http://localhost:6334/", collection="my-docs", dense_model=embed)
```

### add_pdf

```python
add_pdf(
    path: str | PathLike[str],
    name: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> int
```

Indexes a PDF page by page. Indexing the same name again replaces it.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `path` | `str` or `PathLike` | required | The PDF file. |
| `name` | `str` or `None` | `None` | The name to store it under. The default is the file name, such as `handbook.pdf`. |
| `metadata` | `dict` or `None` | `None` | Stored with every chunk and returned with each hit. |

**Returns** the number of chunks stored.

**Raises**

- `BestRAGError` if the file cannot be opened as a PDF, needs a password, or holds no text at all (a scanned PDF needs OCR first), or if `metadata` is not a dict.
- `ValueError` if `metadata` holds values that JSON cannot hold.

**Example**

```python
chunks = rag.add_pdf("handbook.pdf", metadata={"team": "hr"})
```

### add_text

```python
add_text(text: str, name: str, metadata: dict[str, Any] | None = None) -> int
```

Indexes plain text under `name`. Indexing the same name again replaces it.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `text` | `str` | required | The text to index. |
| `name` | `str` | required | The name to store it under. |
| `metadata` | `dict` or `None` | `None` | Stored with every chunk and returned with each hit. |

**Returns** the number of chunks stored.

**Raises**

- `BestRAGError` if `text` is empty, or if `metadata` is not a dict.
- `ValueError` if `metadata` holds values that JSON cannot hold.

**Example**

```python
rag.add_text(
    "Part XJ-9042 replaces the filter in the 2019 pump.",
    "parts",
    {"team": "support"},
)
```

### add_texts

```python
add_texts(documents: list[Document]) -> int
```

Indexes many documents at once. It embeds chunks in batches, so it is much faster than calling `add_text` in a loop. Each document replaces the content already stored under its name.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `documents` | `list[Document]` | required | A list of dicts with `name`, `text` and an optional `metadata`. See [Document](#document). |

**Returns** the total number of chunks stored.

**Raises**

- `ValueError` if `documents` is not a list of dicts with `"name"` and `"text"`.
- `BestRAGError` if a text is empty, a `metadata` is not a dict, or a name appears more than once in the same call. BestRAG checks the whole list before it stores anything.

**Example**

```python
rag.add_texts([
    {"name": "faq-1", "text": "How do I reset my password? ..."},
    {
        "name": "faq-2",
        "text": "Can I change my plan later? ...",
        "metadata": {"lang": "en"},
    },
])
```

### search

```python
search(
    query: str,
    limit: int = 10,
    mode: Literal["hybrid", "dense", "sparse"] = "hybrid",
) -> list[Hit]
```

Searches the collection and returns the best chunks in order of score.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `query` | `str` | required | The question or the words to search for. |
| `limit` | `int` | `10` | The largest number of hits to return. |
| `mode` | `"hybrid"`, `"dense"` or `"sparse"` | `"hybrid"` | `"hybrid"` merges a dense and a BM25 search. `"dense"` and `"sparse"` (BM25) run one search only. See [Search modes](search.md#search-modes). |

**Returns** a list of up to `limit` [hits](#hit). A `limit` of 0 returns an empty list. A sparse search also returns an empty list if the query has no letters or digits.

**Raises**

- `ValueError` if `mode` is not one of the three names.
- `BestRAGError` if the database cannot be reached or the search fails for another reason. This includes a `dense_model` or `reranker` function that raises an exception or returns invalid output. Your exception is the error's `__cause__`. Sparse searches never call the `dense_model` function.

**Example**

```python
hits = rag.search("which filter fits the old pump?", limit=5, mode="hybrid")
for hit in hits:
    print(hit["score"], hit["source"], hit["page"])
```

### delete

```python
delete(name: str) -> None
```

Removes every chunk stored under `name`. If nothing is stored under that name, it does nothing.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `name` | `str` | required | The name the document was indexed under. |

**Returns** `None`.

**Raises** `BestRAGError` if the database call fails.

**Example**

```python
rag.delete("parts")
```

### count

```python
count(name: str | None = None) -> int
```

Counts stored chunks for one name or the whole collection.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `name` | `str` or `None` | `None` | Count only the chunks stored under this name. Leave it out to count the whole collection. |

**Returns** the number of chunks. This counts chunks, not documents.

**Raises** `BestRAGError` if the database call fails.

**Example**

```python
print(rag.count(), rag.count("handbook.pdf"))
```

### drop

```python
drop() -> None
```

Deletes the collection and all its contents. On Postgres, it drops both tables for the collection. This cannot be undone. To create the collection again, create a new `BestRAG` object. To keep a copy first, use your database's backup: `pg_dump` or your provider's point-in-time restore for Postgres, or a snapshot for Qdrant.

**Returns** `None`.

**Raises** `BestRAGError` if the database call fails.

**Example**

```python
rag.drop()
```

## read_pdf

```python
read_pdf(path: str | PathLike[str]) -> list[str]
```

Returns the text from each PDF page in page order. It needs no database or model. Use it before indexing to check for pages that need OCR. Unreadable pages return an empty string, so the list always has one entry per page.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `path` | `str` or `PathLike` | required | The PDF file. |

**Returns** a list of strings.

**Raises** `BestRAGError` if the file cannot be opened as a PDF or needs a password.

**Example**

```python
from bestrag import read_pdf

pages = read_pdf("handbook.pdf")
print(len(pages), pages[0][:200])
```

## BestRAGError

```python
class BestRAGError(Exception): ...
```

BestRAG raises `BestRAGError` for invalid settings, connection and database problems, and documents it cannot index. The message explains the problem and often how to fix it. URLs in messages never include the password.

In any method, exceptions from your `dense_model` or `reranker` function become the `__cause__` of a `BestRAGError`. This lets you handle provider errors, such as rate limits. `KeyboardInterrupt` and `SystemExit` are not changed.

Python raises `ValueError` for an unknown `mode`, a `documents` list in the wrong format, or `metadata` that JSON cannot hold. An argument of the wrong type raises `TypeError`.

**Example**

```python
from bestrag import BestRAG, BestRAGError

url = "postgresql://app_user@db.example.com/app"
try:
    rag = BestRAG(url, "docs", backend="qdrant")
except BestRAGError as error:
    print(f"BestRAG could not start: {error}")
```

## Hit

`search` returns a list of `Hit` dicts:

```python
class Hit(TypedDict):
    text: str
    score: float
    source: str
    page: int | None
    metadata: dict[str, Any] | None
```

| Key | Description |
| --- | --- |
| `text` | The text of the chunk. |
| `score` | Higher is better. The scale depends on the mode and on the reranker, so compare scores only within one search. |
| `source` | The name the document was indexed under. |
| `page` | The PDF page, starting at 1, or `None` for plain text. |
| `metadata` | The metadata given when the document was indexed, or `None`. |

## Document

`add_texts` takes a list of `Document` dicts:

```python
class Document(TypedDict, total=False):
    name: str  # required
    text: str  # required
    metadata: dict[str, Any]
```

| Key | Description |
| --- | --- |
| `name` | Required. The name to store the document under. |
| `text` | Required. The text to index. |
| `metadata` | Optional. A dict stored with every chunk and returned with each hit. |

`Hit` and `Document` are defined in the package's type stubs for type checkers and editors. At run time, they are plain dicts. They cannot be imported.

## \_\_version\_\_

`bestrag.__version__` is the installed version, as a string.

```python
import bestrag

print(bestrag.__version__)
```
