use crate::{
    BestRag, DEFAULT_DENSE_MODEL, DenseModel, Document, EmbedFn, Hit, Mode, Options, RerankFn,
    Reranker,
};
use anyhow::Context;
use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pythonize::{depythonize, pythonize};
use serde_json::{Value, json};
use std::path::PathBuf;

create_exception!(bestrag, BestRAGError, PyException);

fn to_py(err: anyhow::Error) -> PyErr {
    let error = BestRAGError::new_err(format!("{err:#}"));
    // An exception from the user's own function stays as __cause__, so callers
    // can still catch, for example, a provider's RateLimitError.
    let Some(cause) = err.downcast_ref::<PyErr>() else {
        return error;
    };
    Python::attach(|py| {
        // Ctrl+C (KeyboardInterrupt) and SystemExit must still stop the program.
        if !cause.is_instance_of::<PyException>(py) {
            return cause.clone_ref(py);
        }
        error.set_cause(py, Some(cause.clone_ref(py)));
        error
    })
}

/// `dense_model` or `reranker`: a fastembed model name, or the user's function.
#[derive(FromPyObject)]
enum ModelArg {
    Name(String),
    Function(Py<PyAny>),
}

/// Checks that a model argument that is not a name can be called, and adds
/// its name to `shown`, for `repr`.
fn callable(
    py: Python<'_>,
    function: Py<PyAny>,
    arg: &str,
    shown: &mut String,
) -> PyResult<Py<PyAny>> {
    let bound = function.bind(py);
    if !bound.is_callable() {
        return Err(PyTypeError::new_err(format!(
            "{arg} must be a model name or a function"
        )));
    }
    // Only the name: the repr of a callable can show a secret, such as the API
    // key in a functools.partial.
    let name = match bound.getattr("__qualname__") {
        Ok(name) => name.to_string(),
        Err(_) => bound.get_type().qualname()?.to_string(),
    };
    shown.push_str(&format!(", {arg}={name}"));
    Ok(function)
}

/// Wraps a Python `embed(texts, query)`. It runs with the GIL held, while the
/// work around it keeps the GIL released.
fn embed_fn(function: Py<PyAny>) -> EmbedFn {
    Box::new(move |texts, query| {
        Python::attach(|py| {
            let output = function
                .bind(py)
                .call1((texts, query))
                .context("the embedding function raised an error")?;
            output
                .try_iter()
                .and_then(|rows| rows.map(|row| floats(&row?)).collect())
                .context("the embedding function must return one list of numbers per text")
        })
    })
}

/// Wraps a Python `rerank(query, texts)`, like `embed_fn`.
fn rerank_fn(function: Py<PyAny>) -> RerankFn {
    Box::new(move |query, texts| {
        Python::attach(|py| {
            let output = function
                .bind(py)
                .call1((query, texts))
                .context("the reranker function raised an error")?;
            floats(&output).context("the reranker function must return one number per text")
        })
    })
}

/// Any iterable of numbers, such as a list or a NumPy array.
fn floats(values: &Bound<'_, PyAny>) -> PyResult<Vec<f32>> {
    values.try_iter()?.map(|value| value?.extract()).collect()
}

fn to_json(metadata: Option<&Bound<'_, PyAny>>) -> PyResult<Value> {
    metadata.map_or(Ok(Value::Null), |m| {
        depythonize(m).map_err(|e| PyValueError::new_err(format!("metadata: {e}")))
    })
}

fn hit_to_json(hit: Hit) -> Value {
    json!({
        "text": hit.text,
        "score": hit.score,
        "source": hit.source,
        "page": hit.page,
        "metadata": hit.metadata,
    })
}

/// Hybrid search over a Qdrant collection or a Postgres table with pgvector.
/// Dense and BM25 results are merged with reciprocal rank fusion, and can be
/// reranked by a cross-encoder.
///
/// `url` picks the database: `https://...:6334` or `http://...:6334` for
/// Qdrant's gRPC port, `postgresql://user:password@host/db` for Postgres.
/// Pass `backend="qdrant"` or `backend="postgres"` only to be explicit; it must
/// agree with the URL.
///
/// `dense_model` is a fastembed model name, or your own function
/// `embed(texts, query)` that returns one vector per text, for example from an
/// embedding API. `query` is True for search questions and False for
/// documents. With a function, nothing is downloaded.
///
/// Reranking is off by default. Pass a reranker name, such as
/// "jinaai/jina-reranker-v1-turbo-en", or your own function
/// `rerank(query, texts)` that returns one score per text, higher is better.
///
/// Models download into `cache_dir` (default: `$BESTRAG_CACHE_DIR`, then the
/// user cache folder), unless `$HF_HOME` is set.
#[pyclass(name = "BestRAG", module = "bestrag", frozen)]
struct PyBestRag {
    inner: BestRag,
    url: String,
    collection: String,
    /// `, dense_model=<name>` and `, reranker=<name>` for arguments that are
    /// functions, for `repr`.
    functions: String,
}

#[pymethods]
impl PyBestRag {
    #[new]
    // One parameter per Python keyword argument.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        url,
        collection,
        *,
        backend = None,
        api_key = None,
        dense_model = ModelArg::Name(DEFAULT_DENSE_MODEL.to_string()),
        reranker = None,
        cache_dir = None,
    ))]
    fn new(
        py: Python<'_>,
        url: String,
        collection: String,
        backend: Option<String>,
        api_key: Option<String>,
        dense_model: ModelArg,
        reranker: Option<ModelArg>,
        cache_dir: Option<PathBuf>,
    ) -> PyResult<Self> {
        let mut functions = String::new();
        let dense_model = match dense_model {
            ModelArg::Name(name) => DenseModel::Name(name),
            ModelArg::Function(f) => {
                DenseModel::Function(embed_fn(callable(py, f, "dense_model", &mut functions)?))
            }
        };
        let reranker = match reranker {
            None => None,
            Some(ModelArg::Name(name)) => Some(Reranker::Name(name)),
            Some(ModelArg::Function(f)) => Some(Reranker::Function(rerank_fn(callable(
                py,
                f,
                "reranker",
                &mut functions,
            )?))),
        };
        let options = Options {
            backend,
            api_key,
            dense_model,
            reranker,
            cache_dir,
        };
        let inner = py
            .detach(|| BestRag::new(&url, &collection, options))
            .map_err(to_py)?;
        Ok(Self {
            inner,
            url: crate::backend::redact(&url),
            collection,
            functions,
        })
    }

    /// Index a PDF page by page and return the number of chunks stored.
    /// `name` defaults to the file name. Indexing the same name again replaces it.
    #[pyo3(signature = (path, name = None, metadata = None))]
    fn add_pdf(
        &self,
        py: Python<'_>,
        path: PathBuf,
        name: Option<String>,
        metadata: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<usize> {
        let metadata = to_json(metadata)?;
        py.detach(|| self.inner.add_pdf(&path, name.as_deref(), metadata))
            .map_err(to_py)
    }

    /// Index plain text under `name` and return the number of chunks stored.
    /// Indexing the same name again replaces it.
    #[pyo3(signature = (text, name, metadata = None))]
    fn add_text(
        &self,
        py: Python<'_>,
        text: String,
        name: String,
        metadata: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<usize> {
        let metadata = to_json(metadata)?;
        py.detach(|| self.inner.add_text(&text, &name, metadata))
            .map_err(to_py)
    }

    /// Index many documents at once and return the number of chunks stored.
    /// Each document is a dict with `name`, `text` and an optional `metadata`
    /// dict. Much faster than calling `add_text` in a loop.
    fn add_texts(&self, py: Python<'_>, documents: &Bound<'_, PyAny>) -> PyResult<usize> {
        let documents: Vec<Document> = depythonize(documents).map_err(|e| {
            PyValueError::new_err(format!(
                "documents must be a list of dicts with \"name\" and \"text\": {e}"
            ))
        })?;
        py.detach(|| self.inner.add_texts(&documents))
            .map_err(to_py)
    }

    /// Return up to `limit` chunks as dicts with `text`, `score`, `source`,
    /// `page` and `metadata`, best first. `mode` is "hybrid", "dense" or "sparse".
    #[pyo3(signature = (query, limit = 10, mode = "hybrid"))]
    fn search<'py>(
        &self,
        py: Python<'py>,
        query: String,
        limit: usize,
        mode: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        let mode = match mode {
            "hybrid" => Mode::Hybrid,
            "dense" => Mode::Dense,
            "sparse" => Mode::Sparse,
            other => {
                return Err(PyValueError::new_err(format!(
                    "mode must be \"hybrid\", \"dense\" or \"sparse\", not {other:?}"
                )));
            }
        };
        let hits = py
            .detach(|| self.inner.search(&query, limit, mode))
            .map_err(to_py)?;
        let rows: Vec<Value> = hits.into_iter().map(hit_to_json).collect();
        Ok(pythonize(py, &rows)?)
    }

    /// Remove every chunk stored under `name`.
    fn delete(&self, py: Python<'_>, name: String) -> PyResult<()> {
        py.detach(|| self.inner.delete(&name)).map_err(to_py)
    }

    /// Delete the whole collection, with everything in it.
    fn drop(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.inner.drop_collection()).map_err(to_py)
    }

    /// The number of stored chunks, for one `name` or for the whole collection.
    #[pyo3(signature = (name = None))]
    fn count(&self, py: Python<'_>, name: Option<String>) -> PyResult<u64> {
        py.detach(|| self.inner.count(name.as_deref()))
            .map_err(to_py)
    }

    // Secrets are left out on purpose: the API key is never shown, the URL is
    // stored without its password, and functions only by name, because reprs
    // end up in logs.
    fn __repr__(&self) -> String {
        format!(
            "BestRAG(url={:?}, collection={:?}{})",
            self.url, self.collection, self.functions
        )
    }
}

/// The text BestRAG reads from each page of a PDF, in page order. Use it to
/// check a PDF before indexing it, for example to find pages that need OCR.
#[pyfunction]
fn read_pdf(py: Python<'_>, path: PathBuf) -> PyResult<Vec<String>> {
    py.detach(|| crate::read_pdf(&path)).map_err(to_py)
}

#[pymodule]
fn _bestrag(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", crate::VERSION)?;
    m.add_function(wrap_pyfunction!(read_pdf, m)?)?;
    m.add_class::<PyBestRag>()?;
    m.add("BestRAGError", m.py().get_type::<BestRAGError>())?;
    Ok(())
}
