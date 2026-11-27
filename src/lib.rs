//! BestRAG: hybrid retrieval over Qdrant. Dense and sparse search, fused and
//! reranked, with a Python interface behind the `python` feature.

#[cfg(feature = "python")]
mod python;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
