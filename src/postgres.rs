//! Postgres with pgvector: one table of chunks with a dense vector column, and
//! one table of BM25 terms. Keyword search is plain SQL over the terms table,
//! with IDF counted from the stored documents, so it needs no extension beyond
//! pgvector and works on any managed Postgres. BestRAG fuses the two searches
//! itself, with the same reciprocal rank fusion formula Qdrant uses.

use crate::backend::Record;
use crate::store::{Hit, Mode};
use anyhow::{Context, Result, bail};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{Mutex, MutexGuard};
use tokio_postgres::config::{Host, SslMode};
use tokio_postgres::{Client, Config, NoTls};
use tokio_postgres_rustls::MakeRustlsConnect;
use uuid::Uuid;

const INSERT_BATCH: usize = 500;
// Qdrant's default RRF: each list adds 1 / (position + 2), positions from 0.
const RRF_K: f32 = 2.0;
// pgvector's HNSW search returns at most `hnsw.ef_search` rows and checks
// that many candidates. Its default of 40 cost 0.007 nDCG@10 on SciFact
// against Qdrant; 100 is closer to exact search for a small extra cost.
const MIN_EF_SEARCH: u64 = 100;
const MAX_EF_SEARCH: u64 = 1000;

pub struct PgStore {
    config: Config,
    tls: Option<MakeRustlsConnect>,
    client: Mutex<Client>,
    name: String,
    chunks: String,
    terms: String,
}

impl PgStore {
    pub async fn connect(url: &str, collection: &str) -> Result<Self> {
        let name = table_name(collection)?;
        let (config, tls) = target(url)?;
        let tls = tls.map(MakeRustlsConnect::new);
        let client = open(&config, &tls)
            .await
            .context("could not connect to Postgres")?;
        Ok(Self {
            chunks: quote(&name),
            terms: quote(&format!("{name}_terms")),
            name,
            config,
            tls,
            client: Mutex::new(client),
        })
    }

    async fn client(&self) -> Result<MutexGuard<'_, Client>> {
        let mut client = self.client.lock().await;
        // Managed databases close idle connections, so reconnect when needed.
        if client.is_closed() {
            *client = open(&self.config, &self.tls)
                .await
                .context("could not reconnect to Postgres")?;
        }
        Ok(client)
    }

    pub async fn ensure(&self, dim: u64) -> Result<()> {
        let client = self.client().await?;
        let exists: bool = client
            .query_one("SELECT to_regclass($1::text) IS NOT NULL", &[&self.chunks])
            .await?
            .get(0);
        if exists {
            let stored = client
                .query_opt(
                    "SELECT format_type(atttypid, atttypmod) FROM pg_attribute \
                     WHERE attrelid = to_regclass($1::text) AND attname = 'dense' \
                     AND NOT attisdropped",
                    &[&self.chunks],
                )
                .await?
                .and_then(|row| {
                    let kind: String = row.get(0);
                    kind.strip_prefix("vector(")?
                        .strip_suffix(')')?
                        .parse::<u64>()
                        .ok()
                });
            return match stored {
                Some(size) if size == dim => Ok(()),
                Some(size) => bail!(
                    "the collection stores {size}-dimension vectors, but the dense model \
                     makes {dim}; use the model it was built with, or a new collection"
                ),
                None => bail!("the table was not created by BestRAG (no dense vector column)"),
            };
        }

        client
            .batch_execute("CREATE EXTENSION IF NOT EXISTS vector")
            .await
            .context(
                "pgvector is not enabled in this database; ask an admin to run \
                 CREATE EXTENSION vector",
            )?;
        let (chunks, terms, name) = (&self.chunks, &self.terms, &self.name);
        client
            .batch_execute(&format!(
                "CREATE TABLE IF NOT EXISTS {chunks} (
                    id uuid PRIMARY KEY,
                    source text NOT NULL,
                    page integer,
                    chunk integer NOT NULL,
                    text text NOT NULL,
                    metadata jsonb,
                    dense vector({dim}) NOT NULL
                 );
                 CREATE INDEX IF NOT EXISTS {source_idx} ON {chunks} (source);
                 CREATE INDEX IF NOT EXISTS {dense_idx} ON {chunks}
                    USING hnsw (dense vector_cosine_ops);
                 CREATE TABLE IF NOT EXISTS {terms} (
                    id uuid NOT NULL REFERENCES {chunks} (id) ON DELETE CASCADE,
                    term bigint NOT NULL,
                    weight real NOT NULL
                 );
                 CREATE INDEX IF NOT EXISTS {term_idx} ON {terms} (term);
                 CREATE INDEX IF NOT EXISTS {term_id_idx} ON {terms} (id);",
                source_idx = quote(&format!("{name}_source")),
                dense_idx = quote(&format!("{name}_dense")),
                term_idx = quote(&format!("{name}_terms_term")),
                term_id_idx = quote(&format!("{name}_terms_id")),
            ))
            .await?;
        Ok(())
    }

    /// Deletes everything stored under `sources` and stores `records`, in one
    /// transaction, so a failed upload never leaves half a document behind.
    pub async fn replace(&self, sources: Vec<String>, records: Vec<Record>) -> Result<()> {
        let mut client = self.client().await?;
        let tx = client.transaction().await?;
        tx.execute(
            &format!("DELETE FROM {} WHERE source = ANY($1)", self.chunks),
            &[&sources],
        )
        .await?;
        for batch in records.chunks(INSERT_BATCH) {
            let ids: Vec<Uuid> = batch.iter().map(|r| r.id).collect();
            let sources: Vec<&str> = batch.iter().map(|r| r.source.as_str()).collect();
            let pages: Vec<Option<i32>> = batch
                .iter()
                .map(|r| r.page.and_then(|p| i32::try_from(p).ok()))
                .collect();
            let chunks: Vec<i32> = batch
                .iter()
                .map(|r| i32::try_from(r.chunk).unwrap_or(i32::MAX))
                .collect();
            let texts: Vec<&str> = batch.iter().map(|r| r.text.as_str()).collect();
            let metadata: Vec<&serde_json::Value> = batch.iter().map(|r| &r.metadata).collect();
            let dense: Vec<String> = batch.iter().map(|r| vector_text(&r.dense)).collect();
            tx.execute(
                &format!(
                    "INSERT INTO {} (id, source, page, chunk, text, metadata, dense)
                     SELECT id, source, page, chunk, text, metadata, dense::vector
                     FROM UNNEST($1::uuid[], $2::text[], $3::int4[], $4::int4[],
                                 $5::text[], $6::jsonb[], $7::text[])
                          AS u(id, source, page, chunk, text, metadata, dense)",
                    self.chunks
                ),
                &[&ids, &sources, &pages, &chunks, &texts, &metadata, &dense],
            )
            .await?;

            let (mut term_ids, mut terms, mut weights) = (Vec::new(), Vec::new(), Vec::new());
            for record in batch {
                for (term, weight) in record.sparse.0.iter().zip(&record.sparse.1) {
                    term_ids.push(record.id);
                    terms.push(i64::from(*term));
                    weights.push(*weight);
                }
            }
            tx.execute(
                &format!(
                    "INSERT INTO {} (id, term, weight)
                     SELECT * FROM UNNEST($1::uuid[], $2::int8[], $3::float4[])",
                    self.terms
                ),
                &[&term_ids, &terms, &weights],
            )
            .await?;
        }
        tx.commit().await?;
        Ok(())
    }

    pub async fn search(
        &self,
        dense: Option<Vec<f32>>,
        sparse: Vec<(u32, f32)>,
        mode: Mode,
        pool: u64,
    ) -> Result<Vec<Hit>> {
        let client = self.client().await?;
        let limit = i64::try_from(pool).unwrap_or(i64::MAX);
        let dense_ranked = match (&dense, mode) {
            (Some(vector), Mode::Dense | Mode::Hybrid) => {
                client
                    .batch_execute(&format!(
                        "SET hnsw.ef_search = {}",
                        pool.clamp(MIN_EF_SEARCH, MAX_EF_SEARCH)
                    ))
                    .await?;
                let rows = client
                    .query(
                        &format!(
                            "SELECT id, (1 - (dense <=> $1::text::vector))::float4 FROM {}
                             ORDER BY dense <=> $1::text::vector LIMIT $2",
                            self.chunks
                        ),
                        &[&vector_text(vector), &limit],
                    )
                    .await?;
                rows.iter().map(|r| (r.get(0), r.get(1))).collect()
            }
            _ => Vec::new(),
        };
        let sparse_ranked = if mode == Mode::Dense || sparse.is_empty() {
            Vec::new()
        } else {
            let terms: Vec<i64> = sparse.iter().map(|(term, _)| i64::from(*term)).collect();
            // BM25: each matching term adds its stored weight times its IDF,
            // counted here from the documents in this collection.
            // shortcut: N is a full count(*) per query; keep a running count
            // if collections grow past a few million chunks.
            let rows = client
                .query(
                    &format!(
                        "WITH q AS (SELECT DISTINCT unnest($1::int8[]) AS term),
                              n AS (SELECT count(*)::float8 AS n FROM {chunks}),
                              df AS (SELECT t.term, count(*)::float8 AS df
                                     FROM {terms} t JOIN q USING (term) GROUP BY t.term)
                         SELECT t.id,
                                sum(t.weight * ln(1 + (n.n - df.df + 0.5) / (df.df + 0.5)))::float4
                                    AS score
                         FROM {terms} t JOIN df USING (term) CROSS JOIN n
                         GROUP BY t.id ORDER BY score DESC LIMIT $2",
                        chunks = self.chunks,
                        terms = self.terms
                    ),
                    &[&terms, &limit],
                )
                .await?;
            rows.iter().map(|r| (r.get(0), r.get(1))).collect()
        };
        let ranked: Vec<(Uuid, f32)> = match mode {
            Mode::Dense => dense_ranked,
            Mode::Sparse => sparse_ranked,
            Mode::Hybrid => rrf(&[dense_ranked, sparse_ranked], pool as usize),
        };

        let ids: Vec<Uuid> = ranked.iter().map(|(id, _)| *id).collect();
        let rows = client
            .query(
                &format!(
                    "SELECT id, source, page, text, metadata FROM {} WHERE id = ANY($1)",
                    self.chunks
                ),
                &[&ids],
            )
            .await?;
        let mut found: HashMap<Uuid, tokio_postgres::Row> =
            rows.into_iter().map(|row| (row.get(0), row)).collect();
        Ok(ranked
            .into_iter()
            .filter_map(|(id, score)| {
                let row = found.remove(&id)?;
                Some(Hit {
                    source: row.get(1),
                    page: row
                        .get::<_, Option<i32>>(2)
                        .and_then(|p| u32::try_from(p).ok()),
                    text: row.get(3),
                    metadata: row
                        .get::<_, Option<serde_json::Value>>(4)
                        .unwrap_or(serde_json::Value::Null),
                    score,
                })
            })
            .collect())
    }

    pub async fn delete(&self, sources: Vec<String>) -> Result<()> {
        self.client()
            .await?
            .execute(
                &format!("DELETE FROM {} WHERE source = ANY($1)", self.chunks),
                &[&sources],
            )
            .await?;
        Ok(())
    }

    pub async fn count(&self, source: Option<&str>) -> Result<u64> {
        let client = self.client().await?;
        let row = match source {
            Some(source) => {
                client
                    .query_one(
                        &format!("SELECT count(*) FROM {} WHERE source = $1", self.chunks),
                        &[&source],
                    )
                    .await?
            }
            None => {
                client
                    .query_one(&format!("SELECT count(*) FROM {}", self.chunks), &[])
                    .await?
            }
        };
        Ok(u64::try_from(row.get::<_, i64>(0)).unwrap_or(0))
    }

    pub async fn drop_collection(&self) -> Result<()> {
        self.client()
            .await?
            .batch_execute(&format!(
                "DROP TABLE IF EXISTS {}, {}",
                self.terms, self.chunks
            ))
            .await?;
        Ok(())
    }
}

async fn open(config: &Config, tls: &Option<MakeRustlsConnect>) -> Result<Client> {
    Ok(match tls {
        Some(tls) => {
            let (client, connection) = config.connect(tls.clone()).await.map_err(|err| {
                // tokio-postgres has no public error kind for this case.
                let no_tls = std::error::Error::source(&err)
                    .is_some_and(|source| source.to_string() == "server does not support TLS");
                let err = anyhow::Error::new(err);
                if no_tls {
                    err.context(
                        "the server has no TLS; inside a private network, such as \
                         Docker Compose, add sslmode=disable to the URL",
                    )
                } else {
                    err
                }
            })?;
            tokio::spawn(async move {
                let _ = connection.await;
            });
            client
        }
        None => {
            let (client, connection) = config.connect(NoTls).await?;
            tokio::spawn(async move {
                let _ = connection.await;
            });
            client
        }
    })
}

/// Parses a libpq-style URL and decides on TLS. Connections to other machines
/// are encrypted with full certificate and host-name checks, even if the URL
/// asks for a weaker mode such as `prefer`. Only an explicit `sslmode=disable`
/// turns TLS off, for private networks such as Docker Compose.
/// `sslrootcert=<file>` adds a cloud CA, such as AWS RDS's.
fn target(url: &str) -> Result<(Config, Option<rustls::ClientConfig>)> {
    let mut parsed = url::Url::parse(url).context("the Postgres URL is not valid")?;
    let (mut sslmode, mut rootcert) = (None, None);
    // tokio-postgres knows neither sslrootcert nor the verify-* modes, so
    // BestRAG reads them itself and passes the rest through.
    let kept: Vec<(String, String)> = parsed
        .query_pairs()
        .filter_map(|(key, value)| match &*key {
            "sslmode" => {
                sslmode = Some(value.into_owned());
                None
            }
            "sslrootcert" => {
                rootcert = Some(value.into_owned());
                None
            }
            _ => Some((key.into_owned(), value.into_owned())),
        })
        .collect();
    parsed.set_query(None);
    if !kept.is_empty() {
        parsed.query_pairs_mut().extend_pairs(&kept);
    }
    let mut config: Config = parsed
        .as_str()
        .parse()
        .context("the Postgres URL is not valid")?;
    if config.get_password().is_none()
        && let Ok(password) = std::env::var("PGPASSWORD")
    {
        config.password(password);
    }

    let local = config.get_hosts().iter().all(|host| match host {
        Host::Tcp(name) => matches!(name.as_str(), "localhost" | "127.0.0.1" | "::1"),
        Host::Unix(_) => true,
    });
    let tls = match (sslmode.as_deref(), local) {
        (Some("disable"), _) | (None, true) => None,
        (None | Some("allow" | "prefer" | "require" | "verify-ca" | "verify-full"), _) => {
            Some(tls_config(rootcert.as_deref())?)
        }
        (Some(other), _) => bail!("unknown sslmode {other:?}"),
    };
    config.ssl_mode(if tls.is_some() {
        SslMode::Require
    } else {
        SslMode::Disable
    });
    Ok((config, tls))
}

fn tls_config(rootcert: Option<&str>) -> Result<rustls::ClientConfig> {
    let mut roots = rustls::RootCertStore::empty();
    match rootcert {
        Some(path) if path != "system" => {
            let pem = std::fs::read(path)
                .with_context(|| format!("could not read sslrootcert {path}"))?;
            for cert in rustls_pemfile::certs(&mut pem.as_slice()) {
                roots.add(cert.context("sslrootcert holds an invalid certificate")?)?;
            }
            if roots.is_empty() {
                bail!("sslrootcert {path} holds no certificates");
            }
        }
        _ => {
            for cert in rustls_native_certs::load_native_certs().certs {
                // One unreadable system certificate should not block the rest.
                let _ = roots.add(cert);
            }
            if roots.is_empty() {
                bail!("no trusted certificates found on this system; pass sslrootcert=<file>");
            }
        }
    }
    Ok(rustls::ClientConfig::builder_with_provider(Arc::new(
        rustls::crypto::ring::default_provider(),
    ))
    .with_safe_default_protocol_versions()?
    .with_root_certificates(roots)
    .with_no_client_auth())
}

/// Collection names become table names, so they stay within plain characters
/// and leave room for the suffixes of the terms table and the indexes.
fn table_name(collection: &str) -> Result<String> {
    let valid = !collection.is_empty()
        && collection.len() <= 48
        && collection
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-');
    if !valid {
        bail!("a Postgres collection name may use letters, digits, _ and -, up to 48 characters");
    }
    Ok(collection.to_owned())
}

// Safe only for names checked by `table_name`, which cannot hold a quote.
fn quote(identifier: &str) -> String {
    format!("\"{identifier}\"")
}

fn vector_text(vector: &[f32]) -> String {
    let values: Vec<String> = vector.iter().map(f32::to_string).collect();
    format!("[{}]", values.join(","))
}

/// Reciprocal rank fusion, the same formula as Qdrant's default.
fn rrf(lists: &[Vec<(Uuid, f32)>], limit: usize) -> Vec<(Uuid, f32)> {
    let mut scores: HashMap<Uuid, f32> = HashMap::new();
    for list in lists {
        for (position, (id, _)) in list.iter().enumerate() {
            *scores.entry(*id).or_default() += 1.0 / (position as f32 + RRF_K);
        }
    }
    let mut fused: Vec<(Uuid, f32)> = scores.into_iter().collect();
    fused.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    fused.truncate(limit);
    fused
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remote_hosts_get_verified_tls_by_default() {
        let (config, tls) = target("postgresql://u:p@db.example.com/app").unwrap();
        assert!(tls.is_some());
        assert_eq!(config.get_ssl_mode(), SslMode::Require);
        // Weaker modes are upgraded, not trusted.
        assert!(
            target("postgresql://u@db.example.com/app?sslmode=prefer")
                .unwrap()
                .1
                .is_some()
        );
        assert!(
            target("postgresql://u@db.example.com/app?sslmode=verify-full")
                .unwrap()
                .1
                .is_some()
        );
    }

    #[test]
    fn only_an_explicit_sslmode_disable_turns_tls_off() {
        let (config, tls) = target("postgresql://u:p@postgres:5432/app?sslmode=disable").unwrap();
        assert!(tls.is_none());
        assert_eq!(config.get_ssl_mode(), SslMode::Disable);
    }

    #[test]
    fn local_servers_need_no_tls_unless_asked() {
        assert!(
            target("postgresql://u@localhost:5433/app")
                .unwrap()
                .1
                .is_none()
        );
        assert!(target("postgresql://u@127.0.0.1/app").unwrap().1.is_none());
        assert!(
            target("postgresql://u@localhost/app?sslmode=require")
                .unwrap()
                .1
                .is_some()
        );
    }

    #[test]
    fn bad_urls_and_files_fail_clearly() {
        assert!(target("postgresql://u@h/app?sslmode=sometimes").is_err());
        let err = target("postgresql://u@db.example.com/app?sslrootcert=/no/such.pem")
            .unwrap_err()
            .to_string();
        assert!(err.contains("could not read sslrootcert"));
    }

    #[test]
    fn collection_names_are_safe_table_names() {
        assert!(table_name("hr-docs_2026").is_ok());
        for bad in [
            "",
            "docs; DROP TABLE x",
            "a\"b",
            "with space",
            &"x".repeat(49),
        ] {
            assert!(table_name(bad).is_err(), "{bad:?} should be refused");
        }
    }

    #[test]
    fn rrf_matches_qdrants_formula() {
        let (a, b, c) = (Uuid::from_u128(1), Uuid::from_u128(2), Uuid::from_u128(3));
        let fused = rrf(&[vec![(a, 0.9), (b, 0.8)], vec![(b, 5.0), (c, 1.0)]], 10);
        // b: 1/3 + 1/2, a: 1/2, c: 1/3
        assert_eq!(
            fused.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
            [b, a, c]
        );
        assert!((fused[0].1 - (1.0 / 3.0 + 0.5)).abs() < 1e-6);
        assert_eq!(rrf(&[vec![(a, 1.0)], vec![]], 0), vec![]);
    }

    #[test]
    fn vectors_are_sent_in_pgvector_text_form() {
        assert_eq!(vector_text(&[0.5, -1.0, 2.25]), "[0.5,-1,2.25]");
    }
}
