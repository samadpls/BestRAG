//! Choosing the vector database behind a collection, and the small interface
//! the store uses to talk to it.

use crate::postgres::PgStore;
use crate::qdrant::QdrantStore;
use crate::store::{Hit, Mode};
use anyhow::{Result, anyhow, bail};
use serde_json::Value;

/// One chunk, ready to store.
pub struct Record {
    pub id: uuid::Uuid,
    pub text: String,
    pub source: String,
    pub page: Option<u32>,
    pub chunk: usize,
    pub metadata: Value,
    pub dense: Vec<f32>,
    /// BM25 term ids and their weights.
    pub sparse: (Vec<u32>, Vec<f32>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Qdrant,
    Postgres,
}

impl Kind {
    fn name(self) -> &'static str {
        match self {
            Kind::Qdrant => "Qdrant",
            Kind::Postgres => "Postgres",
        }
    }
}

/// The database a URL points at. `backend` is optional; when given, it must
/// agree with the URL, so a wrong name fails with a clear message instead of
/// a confusing connection error later.
pub fn kind(url: &str, backend: Option<&str>) -> Result<Kind> {
    let scheme = url
        .split_once("://")
        .map(|(scheme, _)| scheme.to_ascii_lowercase());
    let from_url = match scheme.as_deref() {
        Some("postgres" | "postgresql") => Some(Kind::Postgres),
        Some("http" | "https") => Some(Kind::Qdrant),
        _ => None,
    };
    let named = backend
        .map(|name| match name.trim().to_ascii_lowercase().as_str() {
            "qdrant" => Ok(Kind::Qdrant),
            "postgres" | "postgresql" | "pgvector" => Ok(Kind::Postgres),
            other => Err(anyhow!(
                "unknown backend {other:?}; use \"qdrant\" or \"postgres\""
            )),
        })
        .transpose()?;
    match (named, from_url) {
        (Some(named), Some(url_kind)) if named != url_kind => bail!(
            "backend {:?} does not match the URL, which is a {} URL ({})",
            backend.unwrap_or_default(),
            url_kind.name(),
            redact(url)
        ),
        (Some(kind), _) | (None, Some(kind)) => Ok(kind),
        (None, None) => bail!(
            "cannot tell the database from {}; start the URL with https:// for Qdrant \
             or postgresql:// for Postgres",
            redact(url)
        ),
    }
}

/// The URL without its password, safe to show in messages, logs and `repr`.
pub fn redact(url: &str) -> String {
    match url::Url::parse(url) {
        Ok(mut parsed) if parsed.password().is_some() => {
            let _ = parsed.set_password(None);
            parsed.to_string()
        }
        Ok(parsed) => parsed.to_string(),
        // Never echo a URL we cannot parse: it might hold a secret we
        // failed to find.
        Err(_) => "<unreadable URL>".into(),
    }
}

pub enum Backend {
    Qdrant(QdrantStore),
    // Boxed: the Postgres store is several times larger than the Qdrant one.
    Postgres(Box<PgStore>),
}

impl Backend {
    pub async fn ensure(&self, dim: u64) -> Result<()> {
        match self {
            Backend::Qdrant(store) => store.ensure(dim).await,
            Backend::Postgres(store) => store.ensure(dim).await,
        }
    }

    pub async fn replace(&self, sources: Vec<String>, records: Vec<Record>) -> Result<()> {
        match self {
            Backend::Qdrant(store) => store.replace(sources, records).await,
            Backend::Postgres(store) => store.replace(sources, records).await,
        }
    }

    pub async fn search(
        &self,
        dense: Option<Vec<f32>>,
        sparse: Vec<(u32, f32)>,
        mode: Mode,
        pool: u64,
    ) -> Result<Vec<Hit>> {
        match self {
            Backend::Qdrant(store) => store.search(dense, sparse, mode, pool).await,
            Backend::Postgres(store) => store.search(dense, sparse, mode, pool).await,
        }
    }

    pub async fn delete(&self, sources: Vec<String>) -> Result<()> {
        match self {
            Backend::Qdrant(store) => store.delete(sources).await,
            Backend::Postgres(store) => store.delete(sources).await,
        }
    }

    pub async fn count(&self, source: Option<&str>) -> Result<u64> {
        match self {
            Backend::Qdrant(store) => store.count(source).await,
            Backend::Postgres(store) => store.count(source).await,
        }
    }

    pub async fn drop_collection(&self) -> Result<()> {
        match self {
            Backend::Qdrant(store) => store.drop_collection().await,
            Backend::Postgres(store) => store.drop_collection().await,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_url_scheme_picks_the_database() {
        assert_eq!(
            kind("https://x.cloud.qdrant.io:6334", None).unwrap(),
            Kind::Qdrant
        );
        assert_eq!(kind("http://localhost:6334", None).unwrap(), Kind::Qdrant);
        assert_eq!(kind("postgresql://u@h/db", None).unwrap(), Kind::Postgres);
        assert_eq!(kind("POSTGRES://u@h/db", None).unwrap(), Kind::Postgres);
    }

    #[test]
    fn an_explicit_backend_must_agree_with_the_url() {
        assert_eq!(
            kind("postgresql://h/db", Some("pgvector")).unwrap(),
            Kind::Postgres
        );
        assert_eq!(
            kind("http://10.0.0.5:6334", Some(" Qdrant ")).unwrap(),
            Kind::Qdrant
        );
        let err = kind("postgresql://u:secret@h/db", Some("qdrant"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("does not match") && !err.contains("secret"));
        let err = kind("http://h:6334", Some("pgvectr"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("unknown backend") && err.contains("\"postgres\""));
        assert!(kind("localhost:6334", None).is_err());
    }

    #[test]
    fn redact_removes_only_the_password() {
        assert_eq!(
            redact("postgresql://bob:s3cret@db.example.com:5432/app?sslmode=require"),
            "postgresql://bob@db.example.com:5432/app?sslmode=require"
        );
        assert_eq!(redact("http://localhost:6334"), "http://localhost:6334/");
        assert_eq!(redact("not a url with secret"), "<unreadable URL>");
    }
}
