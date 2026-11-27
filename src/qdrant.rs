//! Qdrant storage: one collection with a dense vector and a BM25 sparse vector
//! per chunk. Qdrant fuses the two searches itself with reciprocal rank fusion.

use crate::backend::Record;
use crate::store::{Hit, Mode};
use anyhow::{Result, bail};
use qdrant_client::qdrant::{
    Condition, CountPointsBuilder, CreateCollectionBuilder, CreateFieldIndexCollectionBuilder,
    DeletePointsBuilder, Distance, FieldType, Filter, Fusion, Modifier, NamedVectors, PointStruct,
    PrefetchQueryBuilder, QueryPointsBuilder, ScoredPoint, SparseVectorParamsBuilder,
    SparseVectorsConfigBuilder, UpsertPointsBuilder, Vector, VectorParamsBuilder,
    VectorsConfigBuilder, vectors_config,
};
use qdrant_client::{Payload, Qdrant};
use serde_json::{Map, Value, json};

const DENSE: &str = "dense";
const SPARSE: &str = "bm25";
const UPSERT_BATCH: usize = 64;

pub struct QdrantStore {
    client: Qdrant,
    collection: String,
}

impl QdrantStore {
    /// Builds the gRPC client.
    pub fn connect(url: &str, api_key: Option<String>, collection: &str) -> Result<Self> {
        Ok(Self {
            client: Qdrant::from_url(url)
                .api_key(api_key)
                // qdrant-client prints its version check to stdout; library code never prints.
                .skip_compatibility_check()
                .build()?,
            collection: collection.into(),
        })
    }

    pub async fn ensure(&self, dim: u64) -> Result<()> {
        let name = &self.collection;
        if self.client.collection_exists(name).await? {
            let info = self.client.collection_info(name).await?;
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
                    "the collection stores {size}-dimension vectors, but the dense model \
                     makes {dim}; use the model it was built with, or a new collection"
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
        self.client
            .create_collection(
                CreateCollectionBuilder::new(name)
                    .vectors_config(vectors)
                    .sparse_vectors_config(sparse),
            )
            .await?;
        self.client
            .create_field_index(
                CreateFieldIndexCollectionBuilder::new(name, "source", FieldType::Keyword)
                    .wait(true),
            )
            .await?;
        Ok(())
    }

    /// Deletes everything stored under `sources`, then stores `records`.
    pub async fn replace(&self, sources: Vec<String>, records: Vec<Record>) -> Result<()> {
        self.delete(sources).await?;
        let points = records
            .into_iter()
            .map(|r| {
                let payload = json!({
                    "text": r.text,
                    "source": r.source,
                    "page": r.page,
                    "chunk": r.chunk,
                    "metadata": r.metadata,
                });
                Ok(PointStruct::new(
                    r.id.to_string(),
                    NamedVectors::default()
                        .add_vector(DENSE, r.dense)
                        .add_vector(SPARSE, Vector::new_sparse(r.sparse.0, r.sparse.1)),
                    Payload::try_from(payload)?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        for batch in points.chunks(UPSERT_BATCH) {
            self.client
                .upsert_points(
                    UpsertPointsBuilder::new(&self.collection, batch.to_vec()).wait(true),
                )
                .await?;
        }
        Ok(())
    }

    pub async fn search(
        &self,
        dense: Option<Vec<f32>>,
        sparse: Vec<(u32, f32)>,
        mode: Mode,
        pool: u64,
    ) -> Result<Vec<Hit>> {
        let request = QueryPointsBuilder::new(&self.collection)
            .limit(pool)
            .with_payload(true);
        let request = match (mode, dense) {
            (Mode::Sparse, _) => request.query(sparse).using(SPARSE),
            (Mode::Dense, Some(dense)) => request.query(dense).using(DENSE),
            (Mode::Hybrid, Some(dense)) => request
                .add_prefetch(
                    PrefetchQueryBuilder::default()
                        .query(dense)
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
            (_, None) => bail!("a dense search needs a query vector"),
        };
        let points = self.client.query(request).await?.result;
        Ok(points.into_iter().map(hit_from_point).collect())
    }

    pub async fn delete(&self, sources: Vec<String>) -> Result<()> {
        self.client
            .delete_points(
                DeletePointsBuilder::new(&self.collection)
                    .points(source_filter(sources))
                    .wait(true),
            )
            .await?;
        Ok(())
    }

    pub async fn count(&self, source: Option<&str>) -> Result<u64> {
        let mut request = CountPointsBuilder::new(&self.collection).exact(true);
        if let Some(source) = source {
            request = request.filter(source_filter(vec![source.to_owned()]));
        }
        Ok(self
            .client
            .count(request)
            .await?
            .result
            .map_or(0, |r| r.count))
    }

    pub async fn drop_collection(&self) -> Result<()> {
        if self.client.collection_exists(&self.collection).await? {
            self.client.delete_collection(&self.collection).await?;
        }
        Ok(())
    }
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
