# Indexing

Indexing turns your documents into chunks, makes two vectors for each chunk, and stores them in your collection. This page shows how to add, replace and remove documents, and how BestRAG splits text into chunks.

## Add a PDF

```python
chunks = rag.add_pdf("handbook.pdf")
```

`add_pdf` reads the PDF page by page and returns the number of chunks it stored. Each chunk keeps its page number, so a search result can show where its text came from. The document is stored under its file name, here `handbook.pdf`, unless you pass a `name`:

```python
rag.add_pdf(
    "reports/2026-q3.pdf",
    name="q3-report",
    metadata={"year": 2026, "team": "finance"},
)
```

## Add text

```python
rag.add_text(
    "Part XJ-9042 replaces the filter in the 2019 pump.",
    "parts",
    {"team": "support"},
)
```

`add_text(text, name, metadata=None)` stores plain text under `name` and returns the number of chunks. Plain text has no pages, so its hits have `page` set to `None`.

## Add many documents at once

`add_texts` takes a list of dicts with `name`, `text` and an optional `metadata`. It embeds the chunks in batches, so it is much faster than calling `add_text` in a loop:

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

It returns the total number of chunks stored. BestRAG checks the whole list before it stores anything: no text may be empty, every `metadata` must be a dict, and a name may appear only once in the same call.

A call keeps all of its chunks and vectors in memory until they are stored. For a large collection, send a few hundred documents per call, as the [evaluation](benchmarks.md) does:

```python
for start in range(0, len(documents), 256):
    rag.add_texts(documents[start:start + 256])
```

## Metadata

Metadata is a dict that BestRAG stores with every chunk of a document and returns with each search hit. Use it for things like a team, a language, a link or a date. Its values must be types that JSON can hold: strings, numbers, booleans, `None`, lists and dicts. Turn other values, such as dates, into strings first.

`search` does not filter by metadata. It returns the metadata with each hit, so you can filter or group the hits in your own code.

## Replace a document

Indexing a name again replaces it. BestRAG first deletes everything stored under that name, then stores the new chunks, so you never get duplicates. To update a document, index it again under the same name:

```python
rag.add_pdf("handbook.pdf")  # the first version
rag.add_pdf("handbook.pdf")  # a new version replaces it
```

On Postgres, the old chunks are deleted and the new ones stored in one transaction, so a failed upload never leaves half a document behind.

## Delete, count and drop

```python
rag.delete("parts")        # remove everything stored under "parts"
rag.count("handbook.pdf")  # the number of chunks stored under one name
rag.count()                # the number of chunks in the whole collection
rag.drop()                 # delete the whole collection
```

- `delete(name)` removes every chunk stored under `name`. If nothing is stored under that name, it does nothing.
- `count(name=None)` counts chunks, not documents. Without a name, it counts the whole collection.
- `drop()` deletes the whole collection, with everything in it. On Postgres, it drops both of the collection's tables. To use the collection again, create a new `BestRAG` object, which creates the collection again.

!!! warning "`drop()` cannot be undone"
    It deletes every document in the collection, including documents that other programs stored there.

## Check a PDF first

`read_pdf` returns the text that BestRAG reads from each page, in page order. It needs no database and no model, so it is a quick way to check a PDF before you index it, for example to find pages that need OCR:

```python
from bestrag import read_pdf

pages = read_pdf("handbook.pdf")
empty = [n for n, text in enumerate(pages, start=1) if not text.strip()]
print(f"{len(pages)} pages, pages with no text: {empty}")
```

## How chunking works

1. BestRAG reads the text. It reads a PDF one page at a time, so a chunk never spans two pages.
2. It splits the text into chunks of 200 words, and each chunk overlaps the one before it by 40 words. A tail of 40 words or fewer joins the last chunk, so no chunk is tiny.
3. It checks every chunk with the dense model's own tokenizer. If a chunk is longer than the model can read, BestRAG splits it in half, again and again if needed, until every part fits. This matters for languages that need more tokens per word than English, such as Arabic.
4. Each chunk gets two vectors: a dense vector from the embedding model, and a BM25 vector of its words.

!!! note "Scripts without spaces"
    Words are counted by the spaces between them. Text written without spaces, such as Chinese, Japanese or Thai, can form one very long "word" that BestRAG cannot split. Then the dense model reads only its start, so the rest of that text is cut off for dense search.

## Languages

BestRAG keeps text in any script. When it reads a PDF, it rebuilds each line from the positions of its glyphs, so right-to-left scripts such as Arabic come out in reading order, with ligatures and punctuation in the right place.

For BM25, text is normalised with Unicode NFKC and with the rules of Lucene's Arabic normaliser, so Arabic spelling variants with and without hamza or short vowels match each other. For example, الإجازة matches الاجازة. The tests include an Arabic PDF that checks these cases.

The default models are English models. For other languages, choose multilingual ones, as [Choosing models](search.md#choosing-models) shows.

## Scanned PDFs need OCR

BestRAG reads the text that a PDF stores. It does not read text from images.

- A scanned PDF holds only images of its pages. `add_pdf` finds no text in it and raises `BestRAGError`. Run OCR on it first, then index the result.
- Some PDFs use fonts with no map from glyphs to letters (no ToUnicode map). They hold only shapes, so their pages need OCR too.
- If only some pages are scanned, `add_pdf` indexes the other pages and skips the empty ones without an error. Use `read_pdf` to find them.
