# Embedding providers

Pass your own Python function as `dense_model` to use an embedding API such as OpenAI, Cohere or Gemini. The `reranker` can be a function too. BestRAG does not download or load a model for either function.

By default, BestRAG runs `BAAI/bge-small-en-v1.5` on your machine. This small embedding model downloads once and is about 130 MB.

## Your embedding function

```python
def embed(texts: list[str], query: bool) -> list[list[float]]: ...
```

- The function receives a list of texts. It returns one vector (a list of numbers) per text, in the same order. A NumPy array works too.
- `query` is `True` for a search question and `False` for documents. Cohere, Voyage and Gemini embed the two differently. Other providers ignore it.
- When you create `BestRAG`, it calls the function once with one short text to find the vector size.
- The function receives at most 64 texts per call.
- BestRAG checks that the output has one vector per text, all the same size, with no NaN or infinite numbers. Invalid output raises `BestRAGError` with a message explaining the problem.
- If your function raises an exception, BestRAG raises `BestRAGError` with your exception as its `__cause__`. `KeyboardInterrupt` and `SystemExit` are not changed.
- Sparse (BM25) searches do not call it.
- It runs with Python's GIL held. The rest of BestRAG's work runs without the GIL, as before.
- Text is split into chunks of 200 words with a 40-word overlap, as with the local model. The last chunk can have up to 240 words. BestRAG has no tokenizer for your provider's model, so it cannot check the token limit. 240 English words are about 300 to 350 tokens. This is far below the input limits of the models on this page.
- A collection keeps its original vector size. BestRAG rejects a model with a different size, but it cannot tell apart models with the same size. When you change the model or provider, index your documents into a new collection.

To handle a provider's error, look at `__cause__`:

```python
import openai

from bestrag import BestRAGError

try:
    rag.add_texts(documents)
except BestRAGError as error:
    if isinstance(error.__cause__, openai.RateLimitError):
        ...  # wait, then try again
    raise
```

## Embedding recipes

Each recipe uses a provider function to create `BestRAG`. To switch to the free local model, comment out that line and uncomment the line below it.

### OpenAI

```bash
pip install openai
```

```python
from openai import OpenAI

from bestrag import BestRAG

client = OpenAI()  # reads OPENAI_API_KEY

def embed(texts, query):
    response = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [item.embedding for item in response.data]

url = "postgresql://postgres:postgres@localhost:5432/postgres"
rag = BestRAG(url, "my-docs", dense_model=embed)
# To use the free local model on your machine, comment out the line above and uncomment this:
# rag = BestRAG(url, "my-docs")
```

- `text-embedding-3-small` returns vectors of 1,536 numbers. `text-embedding-3-large` returns 3,072. On Postgres, BestRAG stores vectors with more than 2,000 numbers as `halfvec` (see [Postgres and large vectors](#postgres-and-large-vectors)).
- `dimensions=` makes shorter vectors, for example `dimensions=1024`.
- OpenAI embeds questions and documents the same way, so `query` is not used.
- Each request can contain up to 2,048 texts and 300,000 tokens. Each text can contain up to 8,192 tokens.

### Azure OpenAI

```bash
pip install openai
```

```python
import os

from openai import OpenAI

from bestrag import BestRAG

client = OpenAI(
    base_url="https://YOUR-RESOURCE-NAME.openai.azure.com/openai/v1/",
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
)

def embed(texts, query):
    # model is the name of your deployment, not the name of the model.
    response = client.embeddings.create(model="YOUR-DEPLOYMENT", input=texts)
    return [item.embedding for item in response.data]

url = "postgresql://postgres:postgres@localhost:5432/postgres"
rag = BestRAG(url, "my-docs", dense_model=embed)
# To use the free local model on your machine, comment out the line above and uncomment this:
# rag = BestRAG(url, "my-docs")
```

- Azure's v1 API uses the standard `OpenAI` client. It needs no `api_version`.
- Azure offers `text-embedding-3-small` (1,536), `text-embedding-3-large` (3,072) and `text-embedding-ada-002` (1,536). The limits are the same as OpenAI's: 2,048 texts and 300,000 tokens per request.
- To sign in with Microsoft Entra ID instead of a key, install `azure-identity` and pass `api_key=get_bearer_token_provider(DefaultAzureCredential(), "https://ai.azure.com/.default")`.

### Google Gemini

```bash
pip install google-genai
```

```python
from google import genai
from google.genai import types

from bestrag import BestRAG

client = genai.Client()  # reads GEMINI_API_KEY

def embed(texts, query):
    prefix = "task: search result | query: " if query else "title: none | text: "
    # One Content per text: a plain list of texts gives one combined vector.
    contents = [types.Content(parts=[types.Part.from_text(text=prefix + text)]) for text in texts]
    result = client.models.embed_content(model="gemini-embedding-2", contents=contents)
    return [embedding.values for embedding in result.embeddings]

url = "postgresql://postgres:postgres@localhost:5432/postgres"
rag = BestRAG(url, "my-docs", dense_model=embed)
# To use the free local model on your machine, comment out the line above and uncomment this:
# rag = BestRAG(url, "my-docs")
```

- `gemini-embedding-2` returns vectors of 3,072 numbers by default. Postgres stores them as `halfvec`. To use fewer dimensions, pass `config=types.EmbedContentConfig(output_dimensionality=768)`. The range is 128 to 3,072. Google recommends 768, 1,536 or 3,072.
- `gemini-embedding-2` has no `task_type`. The prefixes above are Google's way to tell questions from documents.
- The older `gemini-embedding-001` takes the plain list, `contents=texts`, with `task_type="RETRIEVAL_QUERY"` or `"RETRIEVAL_DOCUMENT"` in the config. Its vectors cannot be mixed with those of `gemini-embedding-2`.
- `gemini-embedding-2` reads up to 8,192 tokens. `gemini-embedding-001` reads up to 2,048. Google's docs give no limit on the number of texts per request.
- `GOOGLE_API_KEY` also works. It takes priority if both keys are set.

### Cohere

```bash
pip install cohere
```

```python
import cohere

from bestrag import BestRAG

co = cohere.ClientV2()  # reads CO_API_KEY

def embed(texts, query):
    response = co.embed(
        texts=texts,
        model="embed-v4.0",
        input_type="search_query" if query else "search_document",
        embedding_types=["float"],
    )
    return response.embeddings.float_

url = "postgresql://postgres:postgres@localhost:5432/postgres"
rag = BestRAG(url, "my-docs", dense_model=embed)
# To use the free local model on your machine, comment out the line above and uncomment this:
# rag = BestRAG(url, "my-docs")
```

- In Python the field is `float_`, with an underscore, because `float` is a Python name. `.float` raises `AttributeError`.
- `embed-v4.0` returns vectors of 1,536 numbers. Set `output_dimension=` to 256, 512, 1,024 or 1,536 to choose the size. The newer `embed-v5.0-pro` and `embed-v5.0-fast` return 2,048 by default.
- Each request can contain up to 96 texts.

### Voyage AI

```bash
pip install voyageai
```

```python
import voyageai

from bestrag import BestRAG

vo = voyageai.Client()  # reads VOYAGE_API_KEY

def embed(texts, query):
    result = vo.embed(texts, model="voyage-4", input_type="query" if query else "document")
    return result.embeddings

url = "postgresql://postgres:postgres@localhost:5432/postgres"
rag = BestRAG(url, "my-docs", dense_model=embed)
# To use the free local model on your machine, comment out the line above and uncomment this:
# rag = BestRAG(url, "my-docs")
```

- `voyage-4` returns vectors of 1,024 numbers. Set `output_dimension=` to 256, 512, 1,024 or 2,048 to choose the size. `voyage-4-large` gives the best results, and `voyage-4-lite` costs less. Vectors from the voyage-4 models work together.
- Each request can contain up to 1,000 texts. The token limits are 320,000 for `voyage-4`, 120,000 for `voyage-4-large` and 1,000,000 for `voyage-4-lite`.

### Mistral AI

```bash
pip install mistralai
```

```python
import os

from mistralai.client import Mistral

from bestrag import BestRAG

client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

def embed(texts, query):
    vectors = []
    # Mistral's own examples keep each request under about 16,000 tokens.
    for start in range(0, len(texts), 16):
        response = client.embeddings.create(model="mistral-embed", inputs=texts[start : start + 16])
        vectors += [item.embedding for item in response.data]
    return vectors

url = "postgresql://postgres:postgres@localhost:5432/postgres"
rag = BestRAG(url, "my-docs", dense_model=embed)
# To use the free local model on your machine, comment out the line above and uncomment this:
# rag = BestRAG(url, "my-docs")
```

- Since mistralai 2.0, the import is `from mistralai.client import Mistral`.
- `mistral-embed` returns vectors of 1,024 numbers and reads up to 8,000 tokens per text. For code, `codestral-embed` returns 1,536 by default. Use `output_dimension=` for up to 3,072.
- Mistral embeds questions and documents the same way.

### Amazon Bedrock

```bash
pip install boto3
```

```python
import json

import boto3

from bestrag import BestRAG

bedrock = boto3.client("bedrock-runtime")  # uses your AWS credentials and region

def embed(texts, query):
    vectors = []
    for text in texts:  # Titan takes one text per call
        response = bedrock.invoke_model(
            modelId="amazon.titan-embed-text-v2:0",
            body=json.dumps({"inputText": text}),
        )
        vectors.append(json.loads(response["body"].read())["embedding"])
    return vectors

url = "postgresql://postgres:postgres@localhost:5432/postgres"
rag = BestRAG(url, "my-docs", dense_model=embed)
# To use the free local model on your machine, comment out the line above and uncomment this:
# rag = BestRAG(url, "my-docs")
```

- Amazon Titan Text Embeddings V2 returns vectors of 1,024 numbers by default. Add `"dimensions": 512` or `256` to the body for fewer dimensions. It reads up to 8,192 tokens or 50,000 characters per text. It has no setting to distinguish questions from documents.
- One call per text is slow for large collections. Bedrock also offers Cohere Embed, such as `cohere.embed-v4:0`. It takes up to 96 texts per call, with `texts` and `input_type` in the body.
- A Bedrock API key in `$AWS_BEARER_TOKEN_BEDROCK` works instead of AWS credentials.

### Ollama

```bash
pip install ollama
ollama pull embeddinggemma
```

```python
import ollama

from bestrag import BestRAG

def embed(texts, query):
    # embeddinggemma's prefixes for questions and documents
    prefix = "task: search result | query: " if query else "title: none | text: "
    return ollama.embed(model="embeddinggemma", input=[prefix + text for text in texts]).embeddings

url = "postgresql://postgres:postgres@localhost:5432/postgres"
rag = BestRAG(url, "my-docs", dense_model=embed)
# To use the free local model on your machine, comment out the line above and uncomment this:
# rag = BestRAG(url, "my-docs")
```

- Ollama runs the model on your own server at `$OLLAMA_HOST` (default `http://127.0.0.1:11434`). `ollama pull` downloads the model once. Run it when you build the server, not at run time.
- `embeddinggemma` returns vectors of 768 numbers and reads up to 2,048 tokens. Other models need different prefixes. `nomic-embed-text` uses `"search_query: "` and `"search_document: "`. `mxbai-embed-large` uses `"Represent this sentence for searching relevant passages: "` for questions only.
- By default, Ollama cuts inputs that are too long for the model.

## Reranker recipes

```python
def rerank(query: str, texts: list[str]) -> list[float]: ...
```

- The function returns one score per text, in the same order as `texts`. Higher scores are better. `search` sorts the hits by these scores.
- The function receives all candidates in one call: `max(3 * limit, 30)` texts.
- BestRAG checks the scores and handles errors as it does for the embedding function.

Each recipe below uses `embed` from the same provider's embedding recipe above.

### Cohere Rerank

```python
def rerank(query, texts):
    response = co.rerank(model="rerank-v5.0-pro", query=query, documents=texts)
    scores = [0.0] * len(texts)
    for result in response.results:  # sorted by score, so put each back in its place
        scores[result.index] = result.relevance_score
    return scores

rag = BestRAG(url, "my-docs", dense_model=embed, reranker=rerank)
# To use the free local 150 MB reranker, comment out the line above and uncomment this:
# rag = BestRAG(url, "my-docs", dense_model=embed, reranker="jinaai/jina-reranker-v1-turbo-en")
```

- Cohere recommends `rerank-v5.0-pro` for new projects. `rerank-v5.0-fast` is faster. The scores are from 0 to 1.
- Cohere cuts each text at `max_tokens_per_doc`, which is 4,096 tokens by default. It suggests at most 1,000 texts per request.

### Voyage AI Rerank

```python
def rerank(query, texts):
    result = vo.rerank(query, texts, model="rerank-3")
    scores = [0.0] * len(texts)
    for item in result.results:  # sorted by score, so put each back in its place
        scores[item.index] = item.relevance_score
    return scores

rag = BestRAG(url, "my-docs", dense_model=embed, reranker=rerank)
# To use the free local 150 MB reranker, comment out the line above and uncomment this:
# rag = BestRAG(url, "my-docs", dense_model=embed, reranker="jinaai/jina-reranker-v1-turbo-en")
```

- `rerank-3` gives the best results, and `rerank-3-lite` is faster. Each request can contain up to 1,000 texts.

## No model downloads on your servers

There are two ways to run BestRAG where servers must not download models.

1. Use an API provider. Set `dense_model=embed` and leave `reranker` off or set it to a function. BestRAG does not use `cache_dir` in this setup. The tests check that the cache folder and `$HF_HOME` stay empty.
2. Put the local model into your Docker image. Use this when your documents must stay inside your network. BestRAG looks in `cache_dir` first and downloads only missing models. Fill a `models` folder once where downloads are allowed, then copy it into the image at build time. BestRAG opens the collection before loading the model, so this step needs a database. You can use one from the [Quickstart](quickstart.md):

    ```python
    from bestrag import BestRAG

    BestRAG("http://localhost:6334", "warm-up", cache_dir="models").drop()
    ```

    ```dockerfile
    FROM python:3.12-slim
    RUN pip install bestrag
    COPY models/ /models/
    ENV BESTRAG_CACHE_DIR=/models
    ```

    - `$BESTRAG_CACHE_DIR` sets the default `cache_dir`. You can also pass `cache_dir="/models"` in your code.
    - Do not set `$HF_HOME` when you fill the folder or in the image: fastembed then uses it instead of `cache_dir`.
    - To use the local reranker, add `reranker="jinaai/jina-reranker-v1-turbo-en"` to the warm-up call, so its model goes into the same folder.

## Privacy and speed

- An API provider receives every chunk during indexing and every question during search. Read the provider's terms for your data. To keep documents inside your network, use the local model or a model you run yourself, such as Ollama.
- Each search calls the provider once to embed the question. This takes roughly 100 to 300 ms before the database search. Sparse searches make no call. Indexing makes one call per 64 chunks.
- The API key stays in your function, and BestRAG never sees it. `repr(rag)` shows only the function's name, such as `dense_model=embed`.

## Postgres and large vectors

pgvector's HNSW index supports up to 2,000 dimensions for `vector` and 4,000 for `halfvec`. A new collection with more than 2,000 dimensions uses `halfvec` with a `halfvec_cosine_ops` index. Examples include the 3,072 dimensions of `text-embedding-3-large` or Gemini. `halfvec` stores each number in 16 bits and needs pgvector 0.7 or later. Older versions raise `BestRAGError`.

Vectors above 4,000 dimensions also raise `BestRAGError`. Ask the provider for fewer dimensions with its dimensions option. Collections with 2,000 dimensions or fewer use `vector`, as before.

Qdrant needs nothing special. The tests store 3,072-dimension vectors on both databases.
