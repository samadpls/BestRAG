//! BestRAG: hybrid retrieval over Qdrant. Dense and BM25 search, fused with
//! reciprocal rank fusion and reranked by a cross-encoder. The Python package
//! is built from the same crate with the `python` feature.

mod bm25;
#[cfg(feature = "python")]
mod python;
mod store;
mod text;

pub use store::{BestRag, DEFAULT_DENSE_MODEL, DEFAULT_RERANKER, Document, Hit, Mode, Options};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
