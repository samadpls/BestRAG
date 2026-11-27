//! BestRAG: hybrid retrieval over Qdrant or Postgres with pgvector. Dense and
//! BM25 search are fused with reciprocal rank fusion and can be reranked by a
//! cross-encoder. The dense model and the reranker run locally, or are your own
//! functions, for example calls to an embedding API. The Python package is
//! built from the same crate with the `python` feature.

mod backend;
mod bm25;
mod pdf;
mod postgres;
#[cfg(feature = "python")]
mod python;
mod qdrant;
mod store;
mod text;

pub use store::{
    BestRag, DEFAULT_DENSE_MODEL, DenseModel, Document, EmbedFn, Hit, Mode, Options,
    RECOMMENDED_RERANKER, RerankFn, Reranker,
};

/// The text BestRAG reads from each page of a PDF, in page order. Useful to
/// check a PDF before indexing it, for example for pages that need OCR.
pub use pdf::pages as read_pdf;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
