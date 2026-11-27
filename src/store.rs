//! The Qdrant-backed store: indexing, hybrid search and reranking.

use crate::{bm25, pdf, text};
use anyhow::{Context, Result, anyhow, bail};
use fastembed::{
    AutoDevicePolicy, EmbeddingModel, RerankInitOptions, RerankerModel, TextEmbedding,
    TextInitOptions, TextRerank,
};
use qdrant_client::qdrant::{
    Condition, CountPointsBuilder, CreateCollectionBuilder, CreateFieldIndexCollectionBuilder,
    DeletePointsBuilder, Distance, FieldType, Filter, Fusion, Modifier, NamedVectors, PointStruct,
    PrefetchQueryBuilder, QueryPointsBuilder, ScoredPoint, SparseVectorParamsBuilder,
    SparseVectorsConfigBuilder, UpsertPointsBuilder, Vector, VectorParamsBuilder,
    VectorsConfigBuilder, vectors_config,
};
use qdrant_client::{Payload, Qdrant};
use serde_json::{Map, Value, json};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use tokio::runtime::Runtime;

pub const DEFAULT_DENSE_MODEL: &str = "BAAI/bge-small-en-v1.5";
pub const DEFAULT_RERANKER: &str = "jinaai/jina-reranker-v1-turbo-en";

const DENSE: &str = "dense";
const SPARSE: &str = "bm25";
const EMBED_BATCH: usize = 32;
const UPSERT_BATCH: usize = 64;
// The reranker reorders a wider pool than it returns, so good chunks that the
// first stage ranked low still get a chance.
const MIN_RERANK_POOL: usize = 30;

pub struct Options {
    pub api_key: Option<String>,
    /// A fastembed model code, for example `BAAI/bge-small-en-v1.5`.
    pub dense_model: String,
    /// A fastembed reranker code, or `None` to keep the first-stage order.
    pub reranker: Option<String>,
    /// Where models are downloaded. Defaults to `$BESTRAG_CACHE_DIR`, then the
    /// user cache folder. fastembed lets `$HF_HOME` override this when it is set.
    pub cache_dir: Option<PathBuf>,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            api_key: None,
            dense_model: DEFAULT_DENSE_MODEL.into(),
            reranker: Some(DEFAULT_RERANKER.into()),
            cache_dir: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Dense and BM25 results, merged with reciprocal rank fusion.
    Hybrid,
    Dense,
    Sparse,
}

/// A document for [`BestRag::add_texts`].
#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct Document {
    pub name: String,
    pub text: String,
    #[serde(default)]
    pub metadata: Value,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Hit {
    pub text: String,
    /// Higher is better. The scale depends on the mode and on the reranker.
    pub score: f32,
    pub source: String,
    pub page: Option<u32>,
    pub metadata: Value,
}

pub struct BestRag {
    runtime: Runtime,
    client: Qdrant,
    collection: String,
    dense: Mutex<TextEmbedding>,
    reranker: Option<Mutex<TextRerank>>,
}

impl BestRag {
    /// Connects to Qdrant over gRPC (port 6334), creates the collection if it
    /// does not exist, and loads the models, downloading them on first use.
    pub fn new(url: &str, collection: &str, options: Options) -> Result<Self> {
        let (model, dim) = dense_model(&options.dense_model)?;
        let reranker = options
            .reranker
            .as_deref()
            .map(reranker_model)
            .transpose()?;

        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()?;
        let client = {
            let _guard = runtime.enter();
            Qdrant::from_url(url).api_key(options.api_key).build()?
        };
        // Checked before loading models, so a wrong URL fails fast instead of
        // after a large download.
        runtime
            .block_on(ensure_collection(&client, collection, dim))
            .with_context(|| format!("could not open collection {collection:?} at {url}"))?;

        let cache = options.cache_dir.unwrap_or_else(default_cache_dir);
        // ort's default device policy hands the model to CoreML on macOS, which
        // rebuilds it for every new input shape: one batch of 16 chunks took
        // 1.9 s and 5.5 GB that way, and 0.4 s and 0.37 GB on the CPU.
        let dense = TextEmbedding::try_new(
            TextInitOptions::new(model)
                .with_cache_dir(cache.clone())
                .with_auto_device(AutoDevicePolicy::PreferCPU)
                .with_show_download_progress(false),
        )?;
        let reranker = reranker
            .map(|model| {
                TextRerank::try_new(
                    RerankInitOptions::new(model)
                        .with_cache_dir(cache.clone())
                        .with_auto_device(AutoDevicePolicy::PreferCPU)
                        .with_show_download_progress(false),
                )
                .map(Mutex::new)
            })
            .transpose()?;

        Ok(Self {
            runtime,
            client,
            collection: collection.into(),
            dense: Mutex::new(dense),
            reranker,
        })
    }

    /// Indexes plain text under `name`, replacing anything stored under that
    /// name before. Returns the number of chunks stored.
    pub fn add_text(&self, text: &str, name: &str, metadata: Value) -> Result<usize> {
        self.add_texts(&[Document {
            name: name.into(),
            text: text.into(),
            metadata,
        }])
    }

    /// Indexes many documents in one go. Much faster than `add_text` in a
    /// loop: chunks are embedded in batches and Qdrant gets fewer requests.
    pub fn add_texts(&self, documents: &[Document]) -> Result<usize> {
        let entries = documents
            .iter()
            .map(|doc| {
                let chunks = text::chunk_words(&doc.text, None);
                if chunks.is_empty() {
                    bail!("text for {:?} is empty", doc.name);
                }
                Ok((doc.name.as_str(), chunks, &doc.metadata))
            })
            .collect::<Result<Vec<_>>>()?;
        self.store(&entries)
    }

    /// Indexes a PDF page by page. `name` defaults to the file name.
    pub fn add_pdf(&self, path: &Path, name: Option<&str>, metadata: Value) -> Result<usize> {
        let chunks: Vec<text::Chunk> = pdf::pages(path)?
            .iter()
            .zip(1..)
            .flat_map(|(page, number)| text::chunk_words(page, Some(number)))
            .collect();
        if chunks.is_empty() {
            bail!(
                "no text found in {}; scanned PDFs need OCR first",
                path.display()
            );
        }
        let name = match name {
            Some(name) => name.to_owned(),
            None => path
                .file_name()
                .context("the PDF path has no file name")?
                .to_string_lossy()
                .into_owned(),
        };
        self.store(&[(name.as_str(), chunks, &metadata)])
    }

    fn store(&self, entries: &[(&str, Vec<text::Chunk>, &Value)]) -> Result<usize> {
        let entries = entries
            .iter()
            .map(|(name, chunks, metadata)| Ok((*name, self.fit_to_model(chunks)?, *metadata)))
            .collect::<Result<Vec<_>>>()?;
        let mut names = HashSet::new();
        for (name, _, metadata) in &entries {
            if !(metadata.is_object() || metadata.is_null()) {
                bail!("metadata for {name:?} must be a dict (a JSON object)");
            }
            if !names.insert(*name) {
                bail!("{name:?} appears more than once in the same call");
            }
        }
        let texts: Vec<&str> = entries
            .iter()
            .flat_map(|(_, chunks, _)| chunks.iter().map(|c| c.text.as_str()))
            .collect();
        let mut vectors = lock(&self.dense)?
            .embed(&texts, Some(EMBED_BATCH))?
            .into_iter();

        let mut points = Vec::with_capacity(texts.len());
        for (source, chunks, metadata) in &entries {
            for (index, chunk) in chunks.iter().enumerate() {
                let dense = vectors
                    .next()
                    .context("the dense model returned too few vectors")?;
                let (indices, values) = bm25::document_vector(&chunk.text);
                let payload = json!({
                    "text": chunk.text,
                    "source": source,
                    "page": chunk.page,
                    "chunk": index,
                    "metadata": metadata,
                });
                points.push(PointStruct::new(
                    uuid::Uuid::new_v4().to_string(),
                    NamedVectors::default()
                        .add_vector(DENSE, dense)
                        .add_vector(SPARSE, Vector::new_sparse(indices, values)),
                    Payload::try_from(payload)?,
                ));
            }
        }
        let sources: Vec<String> = names.into_iter().map(str::to_owned).collect();

        self.runtime.block_on(async {
            // Replacing rather than appending keeps re-indexing a file idempotent.
            self.delete_sources(sources).await?;
            for batch in points.chunks(UPSERT_BATCH) {
                self.client
                    .upsert_points(
                        UpsertPointsBuilder::new(&self.collection, batch.to_vec()).wait(true),
                    )
                    .await?;
            }
            Ok::<_, anyhow::Error>(())
        })?;
        Ok(points.len())
    }

    pub fn search(&self, query: &str, limit: usize, mode: Mode) -> Result<Vec<Hit>> {
        let sparse = bm25::query_vector(query);
        // A query with no letters or digits has no BM25 terms; fall back to dense.
        let mode = match (mode, sparse.is_empty()) {
            (Mode::Sparse, true) => return Ok(Vec::new()),
            (Mode::Hybrid, true) => Mode::Dense,
            (mode, _) => mode,
        };
        if limit == 0 {
            return Ok(Vec::new());
        }
        let pool = match self.reranker {
            Some(_) => (limit * 3).max(MIN_RERANK_POOL),
            None => limit,
        } as u64;

        let request = QueryPointsBuilder::new(&self.collection)
            .limit(pool)
            .with_payload(true);
        let request = match mode {
            Mode::Dense => request.query(self.embed_query(query)?).using(DENSE),
            Mode::Sparse => request.query(sparse).using(SPARSE),
            Mode::Hybrid => request
                .add_prefetch(
                    PrefetchQueryBuilder::default()
                        .query(self.embed_query(query)?)
                        .using(DENSE)
                        .limit(pool),
                )
                .add_prefetch(
                    PrefetchQueryBuilder::default()
                        .query(sparse)
                        .using(SPARSE)
                        .limit(pool),
                )
                .query(Fusion::Rrf),
        };
        let points = self.runtime.block_on(self.client.query(request))?.result;
        let mut hits: Vec<Hit> = points.into_iter().map(hit_from_point).collect();

        if let Some(reranker) = &self.reranker
            && !hits.is_empty()
        {
            let texts: Vec<&str> = hits.iter().map(|h| h.text.as_str()).collect();
            let ranked = lock(reranker)?.rerank(query, texts, false, None)?;
            let mut reranked: Vec<Hit> = ranked
                .into_iter()
                .map(|r| Hit {
                    score: r.score,
                    ..hits[r.index].clone()
                })
                .collect();
            reranked.sort_by(|a, b| b.score.total_cmp(&a.score));
            hits = reranked;
        }
        hits.truncate(limit);
        Ok(hits)
    }

    /// Removes every chunk stored under `name`.
    pub fn delete(&self, name: &str) -> Result<()> {
        self.runtime
            .block_on(self.delete_sources(vec![name.to_owned()]))
    }

    /// The number of stored chunks, for one name or for the whole collection.
    pub fn count(&self, name: Option<&str>) -> Result<u64> {
        let mut request = CountPointsBuilder::new(&self.collection).exact(true);
        if let Some(name) = name {
            request = request.filter(source_filter(vec![name.to_owned()]));
        }
        let response = self.runtime.block_on(self.client.count(request))?;
        Ok(response.result.map_or(0, |r| r.count))
    }

    async fn delete_sources(&self, sources: Vec<String>) -> Result<()> {
        self.client
            .delete_points(
                DeletePointsBuilder::new(&self.collection)
                    .points(source_filter(sources))
                    .wait(true),
            )
            .await?;
        Ok(())
    }

    /// Halves any chunk the dense model would cut off, so no text goes unseen.
    /// Word counts are only a guess at length: Arabic, for example, needs
    /// several tokens per word in an English model's vocabulary.
    fn fit_to_model(&self, chunks: &[text::Chunk]) -> Result<Vec<text::Chunk>> {
        let model = lock(&self.dense)?;
        let mut pending: Vec<text::Chunk> = chunks.iter().rev().cloned().collect();
        let mut fitted = Vec::with_capacity(chunks.len());
        while let Some(chunk) = pending.pop() {
            let encoding = model
                .tokenizer
                .encode(chunk.text.as_str(), true)
                .map_err(|e| anyhow!("could not tokenize a chunk: {e}"))?;
            let words: Vec<&str> = chunk.text.split(' ').collect();
            if encoding.get_overflowing().is_empty() || words.len() < 2 {
                fitted.push(chunk);
                continue;
            }
            let (left, right) = words.split_at(words.len() / 2);
            pending.push(text::Chunk {
                text: right.join(" "),
                page: chunk.page,
            });
            pending.push(text::Chunk {
                text: left.join(" "),
                page: chunk.page,
            });
        }
        Ok(fitted)
    }

    fn embed_query(&self, query: &str) -> Result<Vec<f32>> {
        lock(&self.dense)?
            .embed([query], None)?
            .pop()
            .context("the dense model returned no vector")
    }
}

async fn ensure_collection(client: &Qdrant, name: &str, dim: u64) -> Result<()> {
    if client.collection_exists(name).await? {
        let info = client.collection_info(name).await?;
        let stored = info
            .result
            .and_then(|i| i.config)
            .and_then(|c| c.params)
            .and_then(|p| p.vectors_config)
            .and_then(|v| v.config)
            .and_then(|config| match config {
                vectors_config::Config::ParamsMap(map) => map.map.get(DENSE).map(|p| p.size),
                vectors_config::Config::Params(_) => None,
            });
        return match stored {
            Some(size) if size == dim => Ok(()),
            Some(size) => bail!(
                "the collection stores {size}-dimension vectors, but the dense model makes \
                 {dim}; use the model it was built with, or a new collection"
            ),
            None => bail!("the collection was not created by BestRAG (no {DENSE:?} vector)"),
        };
    }

    let mut vectors = VectorsConfigBuilder::default();
    vectors.add_named_vector_params(DENSE, VectorParamsBuilder::new(dim, Distance::Cosine));
    let mut sparse = SparseVectorsConfigBuilder::default();
    sparse.add_named_vector_params(
        SPARSE,
        SparseVectorParamsBuilder::default().modifier(Modifier::Idf),
    );
    client
        .create_collection(
            CreateCollectionBuilder::new(name)
                .vectors_config(vectors)
                .sparse_vectors_config(sparse),
        )
        .await?;
    client
        .create_field_index(
            CreateFieldIndexCollectionBuilder::new(name, "source", FieldType::Keyword).wait(true),
        )
        .await?;
    Ok(())
}

/// Finds a fastembed model by its code ("Xenova/bge-small-en-v1.5"), its enum
/// name, or the original Hugging Face name ("BAAI/bge-small-en-v1.5"), which
/// only shares the part after the slash with the ONNX copy fastembed uses.
fn find_model<T: Clone>(name: &str, models: &[(String, String, T)]) -> Option<T> {
    let repo = |code: &str| code.rsplit('/').next().unwrap_or(code).to_ascii_lowercase();
    models
        .iter()
        .find(|(code, enum_name, _)| {
            code.eq_ignore_ascii_case(name) || enum_name.eq_ignore_ascii_case(name)
        })
        .or_else(|| models.iter().find(|(code, _, _)| repo(code) == repo(name)))
        .map(|(_, _, model)| model.clone())
}

fn dense_model(name: &str) -> Result<(EmbeddingModel, u64)> {
    let models: Vec<_> = TextEmbedding::list_supported_models()
        .into_iter()
        .map(|i| {
            (
                i.model_code,
                format!("{:?}", i.model),
                (i.model, i.dim as u64),
            )
        })
        .collect();
    find_model(name, &models)
        .ok_or_else(|| anyhow!("unknown dense model {name:?}; pick one fastembed supports"))
}

fn reranker_model(name: &str) -> Result<RerankerModel> {
    let models: Vec<_> = TextRerank::list_supported_models()
        .into_iter()
        .map(|i| (i.model_code, format!("{:?}", i.model), i.model))
        .collect();
    find_model(name, &models)
        .ok_or_else(|| anyhow!("unknown reranker {name:?}; pick one fastembed supports"))
}

/// Matches chunks whose `source` is any of `sources`.
fn source_filter(sources: Vec<String>) -> Filter {
    Filter::must([Condition::matches("source", sources)])
}

fn hit_from_point(point: ScoredPoint) -> Hit {
    let mut payload: Map<String, Value> = point
        .payload
        .into_iter()
        .map(|(key, value)| (key, value.into_json()))
        .collect();
    let mut take_str = |key: &str| match payload.remove(key) {
        Some(Value::String(s)) => s,
        _ => String::new(),
    };
    let (text, source) = (take_str("text"), take_str("source"));
    Hit {
        text,
        source,
        score: point.score,
        page: payload
            .get("page")
            .and_then(Value::as_u64)
            .and_then(|p| u32::try_from(p).ok()),
        metadata: payload.remove("metadata").unwrap_or(Value::Null),
    }
}

fn default_cache_dir() -> PathBuf {
    std::env::var_os("BESTRAG_CACHE_DIR")
        .map(PathBuf::from)
        .or_else(|| dirs::cache_dir().map(|dir| dir.join("bestrag")))
        .unwrap_or_else(|| PathBuf::from(".bestrag_cache"))
}

fn lock<T>(mutex: &Mutex<T>) -> Result<std::sync::MutexGuard<'_, T>> {
    mutex
        .lock()
        .map_err(|_| anyhow!("a model panicked earlier; create a new BestRAG"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn models_resolve_from_their_hugging_face_names() {
        assert_eq!(
            reranker_model("BAAI/bge-reranker-v2-m3").unwrap(),
            RerankerModel::BGERerankerV2M3
        );
        assert_eq!(
            reranker_model(DEFAULT_RERANKER).unwrap(),
            RerankerModel::JINARerankerV1TurboEn
        );
        assert_eq!(dense_model(DEFAULT_DENSE_MODEL).unwrap().1, 384);
        assert_eq!(dense_model("BAAI/bge-m3").unwrap().1, 1024);
        assert!(dense_model("no/such-model").is_err());
        assert!(reranker_model("no/such-model").is_err());
    }
}
