//! BestRAG: hybrid retrieval over Qdrant. Dense and BM25 search, fused with
//! reciprocal rank fusion and reranked by a cross-encoder. The Python package
//! is built from the same crate with the `python` feature.

mod bm25;
mod pdf;
#[cfg(feature = "python")]
mod python;
mod store;
mod text;

pub use store::{BestRag, DEFAULT_DENSE_MODEL, DEFAULT_RERANKER, Document, Hit, Mode, Options};

/// The text BestRAG reads from each page of a PDF, in page order. Useful to
/// check a PDF before indexing it, for example for pages that need OCR.
pub use pdf::pages as read_pdf;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
