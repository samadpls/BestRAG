# API reference

Everything on this page is imported from the `bestrag` package:

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
    dense_model: str = "BAAI/bge-small-en-v1.5",
    reranker: str | None = None,
    cache_dir: str | PathLike[str] | None = None,
)
```

Connects to the database, creates the collection if it does not exist, and loads the models, downloading them on first use. BestRAG opens the collection before it loads the models, so a wrong URL fails fast, before a large download. Every argument after `collection` is keyword-only.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `url` | `str` | required | A `postgresql://` URL, or Qdrant's gRPC address, usually ending in `:6334`. The scheme picks the database. See [Databases](databases.md). |
| `collection` | `str` | required | The collection, or Postgres table, to use. BestRAG creates it if it does not exist. On Postgres, the name may use letters, digits, `_` and `-`, up to 48 characters. |
| `backend` | `"qdrant"`, `"postgres"` or `None` | `None` | Names the database outright. Optional: the URL decides, and a given name must agree with it. |
| `api_key` | `str` or `None` | `None` | The Qdrant API key, for Qdrant Cloud or a secured server. Not allowed with Postgres, which takes its password from the URL or `$PGPASSWORD`. |
| `dense_model` | `str` | `"BAAI/bge-small-en-v1.5"` | Any text embedding model that fastembed supports. A collection keeps the model it was created with. |
| `reranker` | `str` or `None` | `None` | Off by default. Name any fastembed reranker to turn reranking on, such as `"jinaai/jina-reranker-v1-turbo-en"`. |
| `cache_dir` | `str`, `PathLike` or `None` | `None` | Where models are downloaded. The default is `$BESTRAG_CACHE_DIR`, then a `bestrag` folder in your user cache folder. If `$HF_HOME` is set, fastembed uses it instead. |

**Raises** `BestRAGError` when:

- `dense_model` or `reranker` is not a model that fastembed supports;
- the URL scheme is not one BestRAG knows, or `backend` is unknown or does not match the URL;
- `api_key` is given with a Postgres URL;
- the Postgres URL is not valid or has an unknown `sslmode`;
- the `sslrootcert` file cannot be read or holds no certificates;
- the collection name breaks the Postgres naming rules;
- BestRAG cannot connect, for example because the server has no TLS (the message then suggests `sslmode=disable`), or cannot create the collection, for example because pgvector is not enabled;
- the collection exists but stores vectors of another size than `dense_model` makes, or was not created by BestRAG;
- a model cannot be downloaded or loaded.

`repr()` and `str()` show the URL without its password, and never the API key.

**Example**

```python
from bestrag import BestRAG

rag = BestRAG("http://localhost:6334", "my-docs")
print(rag)  # BestRAG(url="http://localhost:6334/", collection="my-docs")
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

Indexes many documents at once. It embeds the chunks in batches, so it is much faster than calling `add_text` in a loop. Each document replaces anything stored under its name before.

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

Searches the collection and returns the best chunks, best first.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `query` | `str` | required | The question or the words to search for. |
| `limit` | `int` | `10` | The largest number of hits to return. |
| `mode` | `"hybrid"`, `"dense"` or `"sparse"` | `"hybrid"` | `"hybrid"` merges a dense and a BM25 search. `"dense"` and `"sparse"` (BM25) run one search only. See [Search modes](search.md#search-modes). |

**Returns** a list of up to `limit` [hits](#hit). A `limit` of 0 returns an empty list, and so does a sparse search for a query with no letters or digits.

**Raises**

- `ValueError` if `mode` is not one of the three names.
- `BestRAGError` if the search fails, for example when the database cannot be reached.

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

Counts the stored chunks, for one name or for the whole collection.

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

Deletes the whole collection, with everything in it. On Postgres, it drops both of the collection's tables. This cannot be undone. To use the collection again, create a new `BestRAG` object, which creates it again.

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

Returns the text that BestRAG reads from each page of a PDF, in page order. It needs no database and no model. Use it to check a PDF before you index it, for example to find pages that need OCR. A page that cannot be read comes back as an empty string, so the list always has one entry per page.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `path` | `str` or `PathLike` | required | The PDF file. |

**Returns** a list with the text of each page.

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

BestRAG raises `BestRAGError` for its own errors: wrong settings, connection and database problems, and documents it cannot index. The message says what went wrong, and often how to fix it. URLs in messages never include the password.

Python's usual errors are raised for arguments of the wrong form: `ValueError` for an unknown `mode`, a `documents` list in the wrong shape, or `metadata` that JSON cannot hold, and `TypeError` for an argument of the wrong type.

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

`Hit` and `Document` are defined in the package's type stubs, so type checkers and editors know them. At run time they are plain dicts, and they cannot be imported.

## \_\_version\_\_

`bestrag.__version__` is the installed version, as a string.

```python
import bestrag

print(bestrag.__version__)
```
