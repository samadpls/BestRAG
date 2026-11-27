//! The store: models, chunking, indexing, hybrid search and reranking, over
//! Qdrant or Postgres with pgvector (see `backend`).

use crate::backend::{self, Backend, Kind, Record};
use crate::postgres::PgStore;
use crate::qdrant::QdrantStore;
use crate::{bm25, pdf, text};
use anyhow::{Context, Result, anyhow, bail};
use fastembed::{
    AutoDevicePolicy, EmbeddingModel, RerankInitOptions, RerankerModel, TextEmbedding,
    TextInitOptions, TextRerank,
};
use serde_json::Value;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use tokio::runtime::Runtime;

pub const DEFAULT_DENSE_MODEL: &str = "BAAI/bge-small-en-v1.5";
/// A small English reranker (151 MB). Reranking is off by default: on
/// SciFact it added 0.02 nDCG@10 but made each search about 45 times slower.
pub const RECOMMENDED_RERANKER: &str = "jinaai/jina-reranker-v1-turbo-en";

const EMBED_BATCH: usize = 32;
// Embedding functions get at most this many texts per call: Cohere takes up to
// 96 texts per request, and Gemini up to 100.
const FUNCTION_BATCH: usize = 64;
// The reranker reorders a wider pool than it returns, so good chunks that the
// first stage ranked low still get a chance.
const MIN_RERANK_POOL: usize = 30;

/// Your own embedding function, usually a call to an embedding API. It gets up
/// to 64 texts and `query`, which is true for a search question and false for
/// documents, and returns one vector per text.
pub type EmbedFn = Box<dyn Fn(&[&str], bool) -> Result<Vec<Vec<f32>>> + Send + Sync>;
/// Your own reranker: it gets the question and the candidate texts, and
/// returns one score per text. Higher is better.
pub type RerankFn = Box<dyn Fn(&str, &[&str]) -> Result<Vec<f32>> + Send + Sync>;

pub enum DenseModel {
    /// A fastembed model code, for example `BAAI/bge-small-en-v1.5`. It is
    /// downloaded on first use.
    Name(String),
    /// Nothing is downloaded or loaded.
    Function(EmbedFn),
}

pub enum Reranker {
    /// A fastembed reranker code, such as [`RECOMMENDED_RERANKER`].
    Name(String),
    /// Nothing is downloaded or loaded.
    Function(RerankFn),
}

pub struct Options {
    /// "qdrant" or "postgres". Normally left out: the URL scheme decides, and
    /// a given name must agree with it.
    pub backend: Option<String>,
    /// Qdrant's API key. Postgres takes its password from the URL or
    /// `$PGPASSWORD`.
    pub api_key: Option<String>,
    pub dense_model: DenseModel,
    /// `None` (the default) keeps the fused order.
    pub reranker: Option<Reranker>,
    /// Where models are downloaded. Defaults to `$BESTRAG_CACHE_DIR`, then the
    /// user cache folder. fastembed lets `$HF_HOME` override this when it is set.
    pub cache_dir: Option<PathBuf>,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            backend: None,
            api_key: None,
            dense_model: DenseModel::Name(DEFAULT_DENSE_MODEL.into()),
            reranker: None,
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
    backend: Backend,
    dense: Dense,
    reranker: Option<Rerank>,
}

// The local models are boxed: they are far larger than a function.
enum Dense {
    Local(Box<Mutex<TextEmbedding>>),
    /// The function and the vector size it makes.
    Function(EmbedFn, usize),
}

enum Rerank {
    Local(Box<Mutex<TextRerank>>),
    Function(RerankFn),
}

impl BestRag {
    /// Connects to the database (Qdrant over gRPC, usually port 6334, or
    /// Postgres with pgvector), creates the collection if it does not exist,
    /// and loads the models, downloading them on first use.
    pub fn new(url: &str, collection: &str, options: Options) -> Result<Self> {
        let kind = backend::kind(url, options.backend.as_deref())?;
        let dim = match &options.dense_model {
            DenseModel::Name(name) => dense_model(name)?.1,
            // One call with a short text tells the vector size, so the
            // collection's size check works for functions too.
            DenseModel::Function(embed) => {
                call_embed(embed, &["dimension check"], false, None)?[0].len() as u64
            }
        };
        if let Some(Reranker::Name(name)) = &options.reranker {
            reranker_model(name)?;
        }

        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()?;
        let mut backend = match kind {
            Kind::Qdrant => {
                Backend::Qdrant(QdrantStore::connect(url, options.api_key, collection)?)
            }
            Kind::Postgres => {
                if options.api_key.is_some() {
                    bail!(
                        "api_key is for Qdrant; give Postgres its password in the URL \
                         or in $PGPASSWORD"
                    );
                }
                Backend::Postgres(Box::new(
                    runtime.block_on(PgStore::connect(url, collection))?,
                ))
            }
        };
        // Checked before loading models, so a wrong URL fails fast instead of
        // after a large download.
        runtime.block_on(backend.ensure(dim)).with_context(|| {
            format!(
                "could not open collection {collection:?} at {}",
                backend::redact(url)
            )
        })?;

        let cache = options.cache_dir.unwrap_or_else(default_cache_dir);
        // ort's default device policy hands the model to CoreML on macOS, which
        // rebuilds it for every new input shape: one batch of 16 chunks took
        // 1.9 s and 5.5 GB that way, and 0.4 s and 0.37 GB on the CPU.
        let dense = match options.dense_model {
            DenseModel::Name(name) => {
                let model = TextEmbedding::try_new(
                    TextInitOptions::new(dense_model(&name)?.0)
                        .with_cache_dir(cache.clone())
                        .with_auto_device(AutoDevicePolicy::PreferCPU)
                        .with_show_download_progress(false),
                )?;
                Dense::Local(Box::new(Mutex::new(model)))
            }
            DenseModel::Function(embed) => Dense::Function(embed, dim as usize),
        };
        let reranker = match options.reranker {
            None => None,
            Some(Reranker::Name(name)) => {
                let model = TextRerank::try_new(
                    RerankInitOptions::new(reranker_model(&name)?)
                        .with_cache_dir(cache)
                        .with_auto_device(AutoDevicePolicy::PreferCPU)
                        .with_show_download_progress(false),
                )?;
                Some(Rerank::Local(Box::new(Mutex::new(model))))
            }
            Some(Reranker::Function(rerank)) => Some(Rerank::Function(rerank)),
        };

        Ok(Self {
            runtime,
            backend,
            dense,
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
    /// loop: chunks are embedded in batches.
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
        let mut vectors = self.embed(&texts, false)?.into_iter();

        let mut records = Vec::with_capacity(texts.len());
        for (source, chunks, metadata) in &entries {
            for (index, chunk) in chunks.iter().enumerate() {
                let dense = vectors
                    .next()
                    .context("the dense model returned too few vectors")?;
                records.push(Record {
                    id: uuid::Uuid::new_v4(),
                    text: chunk.text.clone(),
                    source: (*source).to_owned(),
                    page: chunk.page,
                    chunk: index,
                    metadata: (*metadata).clone(),
                    dense,
                    sparse: bm25::document_vector(&chunk.text),
                });
            }
        }
        let stored = records.len();
        let sources: Vec<String> = names.into_iter().map(str::to_owned).collect();
        // Replacing rather than appending keeps re-indexing a file idempotent.
        self.runtime
            .block_on(self.backend.replace(sources, records))?;
        Ok(stored)
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

        let dense = match mode {
            Mode::Sparse => None,
            Mode::Dense | Mode::Hybrid => Some(self.embed_query(query)?),
        };
        let mut hits = self
            .runtime
            .block_on(self.backend.search(dense, sparse, mode, pool))?;

        if let Some(reranker) = &self.reranker
            && !hits.is_empty()
        {
            let texts: Vec<&str> = hits.iter().map(|h| h.text.as_str()).collect();
            let ranked: Vec<(usize, f32)> = match reranker {
                Rerank::Local(model) => lock(model)?
                    .rerank(query, &texts, false, None)?
                    .into_iter()
                    .map(|r| (r.index, r.score))
                    .collect(),
                Rerank::Function(rerank) => {
                    let scores = rerank(query, &texts)?;
                    check_scores(&scores, texts.len())?;
                    scores.into_iter().enumerate().collect()
                }
            };
            let mut reranked: Vec<Hit> = ranked
                .into_iter()
                .map(|(index, score)| Hit {
                    score,
                    ..hits[index].clone()
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
            .block_on(self.backend.delete(vec![name.to_owned()]))
    }

    /// The number of stored chunks, for one name or for the whole collection.
    pub fn count(&self, name: Option<&str>) -> Result<u64> {
        self.runtime.block_on(self.backend.count(name))
    }

    /// Deletes the whole collection, with everything in it.
    pub fn drop_collection(&self) -> Result<()> {
        self.runtime.block_on(self.backend.drop_collection())
    }

    /// Halves any chunk the dense model would cut off, so no text goes unseen.
    /// Word counts are only a guess at length: Arabic, for example, needs
    /// several tokens per word in an English model's vocabulary.
    fn fit_to_model(&self, chunks: &[text::Chunk]) -> Result<Vec<text::Chunk>> {
        // shortcut: chunks for an embedding function are not checked, because
        // there is no tokenizer for the provider's model. 240 words fit the
        // hosted models we know (512 tokens or more), but a long chunk in a
        // script that needs many tokens per word could be cut off; check
        // against the provider's token counts if that matters.
        let Dense::Local(model) = &self.dense else {
            return Ok(chunks.to_vec());
        };
        let model = lock(model)?;
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
        self.embed(&[query], true)?
            .pop()
            .context("the dense model returned no vector")
    }

    fn embed(&self, texts: &[&str], query: bool) -> Result<Vec<Vec<f32>>> {
        match &self.dense {
            Dense::Local(model) => Ok(lock(model)?.embed(texts, Some(EMBED_BATCH))?),
            Dense::Function(embed, dim) => call_embed(embed, texts, query, Some(*dim)),
        }
    }
}

/// Calls an embedding function in batches, and checks its output, which
/// usually comes from a remote API. `dim` is `None` only for the first call.
fn call_embed(
    embed: &EmbedFn,
    texts: &[&str],
    query: bool,
    dim: Option<usize>,
) -> Result<Vec<Vec<f32>>> {
    let mut vectors = Vec::with_capacity(texts.len());
    for batch in texts.chunks(FUNCTION_BATCH) {
        let output = embed(batch, query)?;
        check_vectors(&output, batch.len(), dim)?;
        vectors.extend(output);
    }
    Ok(vectors)
}

fn check_vectors(vectors: &[Vec<f32>], texts: usize, dim: Option<usize>) -> Result<()> {
    if vectors.len() != texts {
        bail!(
            "the embedding function must return one vector per text, but it got {texts} \
             and returned {}",
            vectors.len()
        );
    }
    let dim = dim.or(vectors.first().map(Vec::len)).unwrap_or(0);
    for vector in vectors {
        if vector.is_empty() {
            bail!("the embedding function returned an empty vector");
        }
        if vector.len() != dim {
            bail!(
                "the embedding function returned a vector of {} numbers, but its vectors \
                 have {dim}; every vector must have the same size",
                vector.len()
            );
        }
        if !vector.iter().all(|x| x.is_finite()) {
            bail!("the embedding function returned a vector with NaN or infinite numbers");
        }
    }
    Ok(())
}

fn check_scores(scores: &[f32], texts: usize) -> Result<()> {
    if scores.len() != texts {
        bail!(
            "the reranker function must return one score per text, but it got {texts} \
             and returned {}",
            scores.len()
        );
    }
    if !scores.iter().all(|x| x.is_finite()) {
        bail!("the reranker function returned a NaN or infinite score");
    }
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
            reranker_model(RECOMMENDED_RERANKER).unwrap(),
            RerankerModel::JINARerankerV1TurboEn
        );
        assert_eq!(dense_model(DEFAULT_DENSE_MODEL).unwrap().1, 384);
        assert_eq!(dense_model("BAAI/bge-m3").unwrap().1, 1024);
        assert!(dense_model("no/such-model").is_err());
        assert!(reranker_model("no/such-model").is_err());
    }

    #[test]
    fn embedding_functions_get_batches_of_at_most_64_texts() {
        let calls = std::sync::Arc::new(Mutex::new(Vec::new()));
        let seen = calls.clone();
        let embed: EmbedFn = Box::new(move |texts, query| {
            seen.lock().unwrap().push((texts.len(), query));
            Ok(vec![vec![0.5; 3]; texts.len()])
        });
        let texts = vec!["a"; 130];
        assert_eq!(
            call_embed(&embed, &texts, true, Some(3)).unwrap().len(),
            130
        );
        assert_eq!(*calls.lock().unwrap(), [(64, true), (64, true), (2, true)]);
    }

    #[test]
    fn bad_function_output_is_refused() {
        let err =
            |vectors: &[Vec<f32>], dim| check_vectors(vectors, 2, dim).unwrap_err().to_string();
        assert!(err(&[vec![1.0]], None).contains("one vector per text"));
        assert!(err(&[vec![], vec![]], None).contains("empty vector"));
        assert!(err(&[vec![1.0], vec![1.0, 2.0]], None).contains("same size"));
        assert!(err(&[vec![1.0], vec![1.0]], Some(2)).contains("same size"));
        assert!(err(&[vec![1.0], vec![f32::NAN]], None).contains("NaN"));
        assert!(err(&[vec![1.0], vec![f32::INFINITY]], None).contains("infinite"));
        assert!(check_vectors(&[vec![1.0, 0.0], vec![0.0, 1.0]], 2, Some(2)).is_ok());

        assert!(
            check_scores(&[1.0], 2)
                .unwrap_err()
                .to_string()
                .contains("one score per text")
        );
        assert!(check_scores(&[1.0, f32::NAN], 2).is_err());
        assert!(check_scores(&[-3.5, 0.0], 2).is_ok());
    }
}
