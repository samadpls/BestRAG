# Databases

BestRAG stores your documents in Postgres with the pgvector extension, or in Qdrant. The API and the hybrid search are the same on both. This page shows how to connect to each one, and how BestRAG keeps your data safe.

## How the URL picks the database

The first argument of `BestRAG` is a URL, and its scheme decides which database BestRAG uses:

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

If the URL starts with anything else, BestRAG raises `BestRAGError` and says which schemes it knows.

You can also name the database with `backend="postgres"` or `backend="qdrant"`. This is optional, because the URL already decides. If you pass it, it must agree with the URL. If it does not, BestRAG stops with a clear message instead of a confusing connection error later, and a misspelled name gets a message that lists the valid names.

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

Put the password in the URL, or leave it out and set `$PGPASSWORD`. BestRAG reads `$PGPASSWORD` when the URL has no password, so the password stays out of your code:

```bash
export PGPASSWORD='...'
```

```python
rag = BestRAG("postgresql://app_user@db.example.com:5432/app", "docs")
```

If the password holds characters that have a meaning in URLs, such as `/`, `?` or `#`, percent-encode them in the URL, or use `$PGPASSWORD` instead.

`api_key` is only for Qdrant. If you pass it with a Postgres URL, BestRAG raises `BestRAGError`.

### Hosted Postgres

BestRAG checks the server's certificate, so it must trust the authority that signed it. Some providers use their own certificate authority. For them, download its certificate and add `sslrootcert=` to the URL:

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

The simple way is Google's Auth Proxy: the [Cloud SQL Auth Proxy](https://docs.cloud.google.com/sql/docs/postgres/connect-auth-proxy) or the [AlloyDB Auth Proxy](https://docs.cloud.google.com/alloydb/docs/auth-proxy/connect). It runs next to your app, listens on `127.0.0.1` and encrypts the connection to Google Cloud. BestRAG then connects to it like a local server:

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

You can also connect to Cloud SQL directly. Then pass its server CA certificate, `server-ca.pem` from the Google Cloud console, with `?sslrootcert=/path/to/server-ca.pem`. BestRAG checks the host name against the server's certificate, so you may need to connect with the instance's DNS name instead of its IP address. A direct connection to AlloyDB follows the same rules.

### TLS and `sslmode`

For a Postgres on another machine, BestRAG uses TLS by default and checks the server's certificate and host name. It does this even when the URL asks for less, such as `sslmode=prefer` or `sslmode=require`. BestRAG never turns TLS off on its own, and it never falls back to plain text when TLS fails. If the server does not support TLS, the error message suggests `sslmode=disable`.

On the same machine (`localhost`, `127.0.0.1`, `::1` or a Unix socket), no data crosses a network, so BestRAG connects without TLS unless the URL asks for it.

| `sslmode` in the URL | Another machine | The same machine |
| --- | --- | --- |
| Not set | TLS, with certificate and host name checks | No TLS |
| `disable` | No TLS. Use it only inside a private network. | No TLS |
| `allow`, `prefer`, `require`, `verify-ca` or `verify-full` | TLS, with certificate and host name checks | TLS, with certificate and host name checks |

Any other `sslmode` value raises `BestRAGError`.

### Inside a private network

Inside a private network without TLS, such as Docker Compose or Kubernetes, you can turn TLS off by adding `sslmode=disable` to the URL. This works for any host, as in other Postgres clients. Here, the app reaches a pgvector service named `postgres`:

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

Use `sslmode=disable` only inside a private network, because the connection is not encrypted.

### The pgvector extension

The database needs the pgvector extension. When BestRAG creates a collection, it runs `CREATE EXTENSION IF NOT EXISTS vector` for you. If your database user is not allowed to do that, ask an admin to run this once:

```sql
CREATE EXTENSION vector;
```

On some managed services, you first allow the extension in the service's settings.

### Collection names and tables

Each collection becomes two tables:

- `<collection>` holds the chunks: their text, page, metadata and dense vector. The vector has an HNSW index for cosine distance.
- `<collection>_terms` holds the BM25 weight of each word in each chunk.

So on Postgres, a collection name may use only letters, digits, `_` and `-`, up to 48 characters. Other names raise `BestRAGError`.

BM25 runs as plain SQL over the terms table, with word statistics counted from the documents in your collection, so it needs no extension beyond pgvector. BestRAG merges the dense and BM25 results itself, with the same reciprocal rank fusion formula that Qdrant uses.

When you index a name again, BestRAG deletes the old chunks and stores the new ones in one transaction, so a failed upload never leaves half a document behind. Managed databases often close idle connections; BestRAG notices this and reconnects on the next call.

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

Qdrant Cloud has a free tier. Qdrant Cloud and secured servers need an API key, which you pass with `api_key`. It is a good idea to keep the key out of your code. In this example, your code reads it from an environment variable and passes it on:

```python
import os

from bestrag import BestRAG

rag = BestRAG(
    "https://xyz.cloud.qdrant.io:6334",
    "docs",
    api_key=os.environ["QDRANT_API_KEY"],
)
```

As with Postgres, the URL decides: `https://` is encrypted and `http://` is not. So a Qdrant on another machine needs an `https://` URL. With `https://`, the server's certificate is checked against the certificates your system trusts.

### What BestRAG creates in Qdrant

Each chunk is a point with two named vectors: `dense`, compared by cosine similarity, and `bm25`, a sparse vector. The `bm25` index uses Qdrant's IDF modifier, so Qdrant adds the inverse document frequency on the server. BestRAG also adds an index on the `source` field, which holds the name of each document. Qdrant merges the dense and BM25 results itself, with reciprocal rank fusion.

## Keeping your data safe

- **Encrypted connections.** A connection to a Postgres on another machine uses TLS by default, with the certificate and host name checked. BestRAG never turns TLS off on its own; only `sslmode=disable` in the URL does, for private networks. A Qdrant on another machine needs an `https://` URL, because `http://` is not encrypted.
- **No secrets in output.** Passwords and API keys come only from the URL, `$PGPASSWORD` or `api_key`. They never appear in `repr()`, logs or error messages. `repr()` shows the URL without its password:

    ```pycon
    >>> BestRAG("postgresql://app_user:s3cret@localhost:5433/app", "docs")
    BestRAG(url="postgresql://app_user@localhost:5433/app", collection="docs")
    ```

- **Your documents stay with you.** Your documents go only to your own database. Embedding and reranking run on your machine, with no calls to any API. The only download is the models, once.
