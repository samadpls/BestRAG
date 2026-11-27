# Databases

BestRAG stores documents in Postgres with the pgvector extension or in Qdrant. Both use the same API and hybrid search. This page explains how to connect and how BestRAG protects your data.

## How the URL picks the database

Pass the database URL as the first argument of `BestRAG`. Its scheme selects the database:

| The URL starts with | Database |
| --- | --- |
| `postgresql://` or `postgres://` | Postgres with pgvector |
| `http://` or `https://` | Qdrant, on its gRPC port (usually 6334) |

```python
from bestrag import BestRAG

# Postgres with pgvector, wherever it runs
rag = BestRAG("postgresql://app_user:password@db.example.com:5432/app", "docs")

# Qdrant: local, self-hosted or Qdrant Cloud, on the gRPC port 6334
rag = BestRAG("https://xyz.cloud.qdrant.io:6334", "docs", api_key="...")
```

Other URL schemes raise `BestRAGError` with a list of supported schemes.

You can also set `backend="postgres"` or `backend="qdrant"`. This is optional and must match the URL. A mismatch raises a clear error before BestRAG tries to connect. A misspelled name raises an error that lists the valid names.

```python
rag = BestRAG(
    "postgresql://app_user@db.example.com:5432/app",
    "docs",
    backend="postgres",
)
```

## Postgres with pgvector

BestRAG works with any Postgres that has the pgvector extension, wherever it runs: AWS RDS and Aurora, Google Cloud SQL and AlloyDB, Azure, Supabase, Neon or your own server. The URL is a normal Postgres connection URL:

```text
postgresql://user:password@host:5432/database
```

### The password

Put the password in the URL or set `$PGPASSWORD`. BestRAG reads this variable when the URL has no password, so you can keep it out of your code:

```bash
export PGPASSWORD='...'
```

```python
rag = BestRAG("postgresql://app_user@db.example.com:5432/app", "docs")
```

Percent-encode characters with a special meaning in URLs, such as `/`, `?` or `#`. You can also use `$PGPASSWORD` to avoid encoding them.

`api_key` is only for Qdrant. If you pass it with a Postgres URL, BestRAG raises `BestRAGError`.

### Hosted Postgres

BestRAG must trust the authority that signed the server's certificate. Some providers use their own certificate authority. Download that authority's certificate and add `sslrootcert=` to the URL:

| Where your Postgres runs | What to add to the URL |
| --- | --- |
| AWS RDS or Aurora | `?sslrootcert=/path/to/global-bundle.pem`, the CA bundle AWS publishes |
| :simple-googlecloud: Google Cloud SQL or AlloyDB | Nothing, if you connect through Google's Auth Proxy on `127.0.0.1`. See [Google Cloud SQL and AlloyDB](#google-cloud-sql-and-alloydb). |
| :simple-supabase: Supabase | `?sslrootcert=/path/to/supabase-ca.crt`, the CA certificate from your project's database settings |
| :brands-neon: Neon, Azure | Nothing. Their certificates are trusted by your system. |
| Your own server on `localhost` | Nothing. No TLS is needed on the same machine. |

For example, for AWS RDS or Aurora, download the bundle once:

```bash
curl -O https://truststore.pki.rds.amazonaws.com/global/global-bundle.pem
```

Then add its path to the URL:

```python
rag = BestRAG(
    "postgresql://app_user@my-db.abc123.eu-west-1.rds.amazonaws.com:5432/app"
    "?sslrootcert=/path/to/global-bundle.pem",
    "docs",
)
```

With `sslrootcert=<file>`, BestRAG trusts only the certificates in that file. Without it, or with `sslrootcert=system`, it trusts the certificates your system trusts.

### Google Cloud SQL and AlloyDB

A simple way to connect is through the [Cloud SQL Auth Proxy](https://docs.cloud.google.com/sql/docs/postgres/connect-auth-proxy) or the [AlloyDB Auth Proxy](https://docs.cloud.google.com/alloydb/docs/auth-proxy/connect). The proxy runs next to your app and listens on `127.0.0.1`. It encrypts the connection to Google Cloud. BestRAG connects to the proxy as a local server:

```bash
# Cloud SQL
cloud-sql-proxy --port 5432 PROJECT:REGION:INSTANCE

# AlloyDB, which listens on 127.0.0.1:5432 by default
alloydb-auth-proxy \
  "projects/PROJECT/locations/REGION/clusters/CLUSTER/instances/INSTANCE"
```

```python
rag = BestRAG("postgresql://app_user@127.0.0.1:5432/app", "docs")
```

For a direct Cloud SQL connection, download `server-ca.pem` from the Google Cloud console. Pass this server CA certificate with `?sslrootcert=/path/to/server-ca.pem`. BestRAG checks the host name against the server's certificate. You may need the instance's DNS name instead of its IP address. Direct AlloyDB connections follow the same rules.

### TLS and `sslmode`

For Postgres on another machine, BestRAG uses TLS by default and checks the server's certificate and host name. These checks also apply with `sslmode=prefer` or `sslmode=require`. BestRAG never turns TLS off on its own or falls back to plain text if TLS fails. If the server does not support TLS, the error message suggests `sslmode=disable`.

On the same machine (`localhost`, `127.0.0.1`, `::1` or a Unix socket), no data crosses a network, so BestRAG connects without TLS unless the URL asks for it.

| `sslmode` in the URL | Another machine | The same machine |
| --- | --- | --- |
| Not set | TLS, with certificate and host name checks | No TLS |
| `disable` | No TLS. Use it only inside a private network. | No TLS |
| `allow`, `prefer`, `require`, `verify-ca` or `verify-full` | TLS, with certificate and host name checks | TLS, with certificate and host name checks |

Any other `sslmode` value raises `BestRAGError`.

### Inside a private network

In a private network without TLS, such as Docker Compose or Kubernetes, add `sslmode=disable` to the URL. This works for any host, as in other Postgres clients. The example connects the app to a pgvector service named `postgres`:

```yaml
services:
  postgres:
    image: pgvector/pgvector:pg17
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
      POSTGRES_DB: db
  app:
    build: .
    depends_on: [postgres]
    environment:
      DATABASE_URL: postgresql://user:password@postgres:5432/db?sslmode=disable
```

```python
import os

from bestrag import BestRAG

rag = BestRAG(os.environ["DATABASE_URL"], "docs")
```

The connection is not encrypted, so use this setting only inside a private network.

### The pgvector extension

The database needs the pgvector extension. When BestRAG creates a collection, it runs `CREATE EXTENSION IF NOT EXISTS vector` for you. If your database user is not allowed to do that, ask an admin to run this once:

```sql
CREATE EXTENSION vector;
```

Some managed services require you to allow the extension in their settings first.

### Collection names and tables

Each collection becomes two tables:

- `<collection>` stores each chunk's text, page, metadata and dense vector. The vector has an HNSW index for cosine distance. This index supports up to 2,000 dimensions for `vector`. Larger vectors use `halfvec` (16-bit numbers), which needs pgvector 0.7 or later. Examples include the 3,072 dimensions of OpenAI's `text-embedding-3-large` or Gemini. Above 4,000 dimensions, BestRAG raises `BestRAGError`. Ask your provider for fewer dimensions.
- `<collection>_terms` holds the BM25 weight of each word in each chunk.

Postgres collection names may contain only letters, digits, `_` and `-`, up to 48 characters. Other names raise `BestRAGError`.

BM25 runs in plain SQL over the terms table. It counts word statistics from your collection's documents and needs no extension beyond pgvector. BestRAG merges dense and BM25 results with the same reciprocal rank fusion formula as Qdrant.

When you index a name again, BestRAG deletes the old chunks and stores the new ones in one transaction. A failed upload never leaves a partly stored document. Managed databases often close idle connections. BestRAG detects this and reconnects on the next call.

## Qdrant

### On your own machine

Start Qdrant with Docker and connect to its gRPC port:

```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

```python
rag = BestRAG("http://localhost:6334", "docs")
```

!!! note "Use port 6334"
    BestRAG talks to Qdrant over gRPC, so use port **6334**, not the REST port 6333.

### Qdrant Cloud or a secured server

Qdrant Cloud has a free tier. It and other secured servers need a key passed with `api_key`. Keep the key out of your code by reading it from an environment variable:

```python
import os

from bestrag import BestRAG

rag = BestRAG(
    "https://xyz.cloud.qdrant.io:6334",
    "docs",
    api_key=os.environ["QDRANT_API_KEY"],
)
```

For Qdrant, `https://` encrypts the connection and checks the server's certificate against those your system trusts. `http://` is not encrypted. Use `https://` for Qdrant on another machine.

### What BestRAG creates in Qdrant

Each chunk is a point with two named vectors. `dense` uses cosine similarity, and `bm25` is a sparse vector. The `bm25` index uses Qdrant's IDF modifier to add inverse document frequency on the server. BestRAG also indexes `source`, the field that stores each document's name. Qdrant merges dense and BM25 results with reciprocal rank fusion.

## Keeping your data safe

- **Secrets.** Database passwords and API keys come only from the URL, `$PGPASSWORD` or `api_key`. They never appear in `repr()`, logs or error messages. The URL in `repr()` has no password:

    ```pycon
    >>> BestRAG("postgresql://app_user:s3cret@localhost:5433/app", "docs")
    BestRAG(url="postgresql://app_user@localhost:5433/app", collection="docs")
    ```

- **Documents.** With local models, documents go only to your own database. Embedding and reranking run on your machine without API calls. Only the models are downloaded, once. If your embedding or reranker function calls an API, your documents also go to that provider. See [Embedding providers](embedding-providers.md#privacy-and-speed).
