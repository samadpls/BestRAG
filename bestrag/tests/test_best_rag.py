import pytest
from unittest.mock import patch, MagicMock
from bestrag.best_rag import BestRAG

url = "http://localhost:6333"
api_key = "dummy_api_key"
collection_name = "test_collection"


@pytest.fixture
@patch("bestrag.best_rag.QdrantClient")
def best_rag_instance(mock_qdrant_client):
    """Fixture to create a BestRAG instance for testing with mocked QdrantClient."""
    mock_qdrant_client_instance = mock_qdrant_client.return_value
    mock_qdrant_client_instance.get_collections.return_value = MagicMock(
        collections=[])
    mock_qdrant_client_instance.create_collection.return_value = None
    mock_qdrant_client_instance.upsert.return_value = None
    mock_qdrant_client_instance.query_points.return_value = [
        {"payload": {"text": "Mock document"}}]

    return BestRAG(url, api_key, collection_name)


def test_create_or_use_collection(best_rag_instance):
    """Test collection creation or selection."""
    with patch.object(best_rag_instance.client, 'get_collections',
                      return_value=MagicMock(collections=[])) as mock_get_collections, \
            patch.object(best_rag_instance.client, 'create_collection',
                         return_value=None) as mock_create_collection:

        best_rag_instance._create_or_use_collection()
        mock_get_collections.assert_called_once()
        mock_create_collection.assert_called_once()


def test_clean_text(best_rag_instance):
    """Test text cleaning functionality."""
    raw_text = "This is _an example_ text.\nNew line with    multiple spaces."
    expected_cleaned_text = "This is an example text. New line with multiple spaces."
    assert best_rag_instance._clean_text(raw_text) == expected_cleaned_text


def test_get_dense_embedding(best_rag_instance):
    """Test dense embedding generation."""
    with patch.object(best_rag_instance.dense_model, 'embed',
                      return_value=[[0.1, 0.2, 0.3]]):
        result = best_rag_instance._get_dense_embedding("Test text")
        assert isinstance(result, list)
        assert len(result) == 3


def test_get_late_interaction_embedding(best_rag_instance):
    """Test late interaction embedding generation."""
    with patch.object(best_rag_instance.late_interaction_model, 'embed',
                      return_value=[[0.4, 0.5, 0.6]]):
        result = best_rag_instance._get_late_interaction_embedding("Test text")
        assert isinstance(result, list)
        assert len(result) == 3


def test_get_sparse_embedding(best_rag_instance):
    """Test sparse embedding generation."""
    mock_sparse_embedding = MagicMock(indices=[1, 2], values=[0.1, 0.2])
    with patch.object(best_rag_instance.sparse_model, 'embed',
                      return_value=iter([mock_sparse_embedding])):
        result = best_rag_instance._get_sparse_embedding("Test text")
        assert hasattr(result, 'indices')
        assert hasattr(result, 'values')
        assert result.indices == [1, 2]
        assert result.values == [0.1, 0.2]


def test_extract_pdf_text_per_page(best_rag_instance, tmp_path):
    """Test PDF text extraction per page."""
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b'%PDF-1.4...')

    with patch("PyPDF2.PdfReader") as mock_pdf_reader:
        mock_reader_instance = MagicMock()
        mock_pdf_reader.return_value = mock_reader_instance
        mock_reader_instance.pages = [MagicMock(extract_text=lambda: "Page 1 text"),
                                      MagicMock(extract_text=lambda: "Page 2 text")]

        extracted_text = best_rag_instance._extract_pdf_text_per_page(
            str(pdf_path))
        assert extracted_text == ["Page 1 text", "Page 2 text"]


def test_store_pdf_embeddings(best_rag_instance, tmp_path):
    """Test storing PDF embeddings in Qdrant."""
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b'%PDF-1.4...')

    with patch.object(best_rag_instance, '_extract_pdf_text_per_page',
                      return_value=["Page 1 text", "Page 2 text"]), \
            patch.object(best_rag_instance, '_get_dense_embedding',
                         return_value=[0.1, 0.2, 0.3]), \
            patch.object(best_rag_instance, '_get_late_interaction_embedding',
                         return_value=[0.4, 0.5, 0.6]), \
            patch.object(best_rag_instance, '_get_sparse_embedding',
                         return_value=MagicMock(indices=[1, 2], values=[0.1, 0.2])), \
            patch.object(best_rag_instance.client, 'upsert',
                         return_value=None) as mock_upsert:

        best_rag_instance.store_pdf_embeddings(str(pdf_path), "sample.pdf")
        assert mock_upsert.call_count == 2


def test_search(best_rag_instance):
    """Test search functionality."""
    with patch.object(best_rag_instance, '_get_dense_embedding',
                      return_value=[0.1, 0.2, 0.3]), \
            patch.object(best_rag_instance, '_get_late_interaction_embedding',
                         return_value=[0.4, 0.5, 0.6]), \
            patch.object(best_rag_instance, '_get_sparse_embedding',
                         return_value=MagicMock(indices=[1, 2], values=[0.1, 0.2])), \
            patch.object(best_rag_instance.client, 'query_points',
                         return_value=[{"payload": {"text": "Relevant document"}}]) as mock_query:

        results = best_rag_instance.search("sample query", limit=10)
        mock_query.assert_called_once()
        assert isinstance(results, list)
        assert results[0]["payload"]["text"] == "Relevant document"
