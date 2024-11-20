"""BestRAG"""
# Authors: Abdul Samad Siddiqui <abdulsamadsid1@gmail.com>

import re
import uuid
from typing import List, Optional
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance
from fastembed import TextEmbedding
from fastembed.sparse.bm25 import Bm25
import PyPDF2


class BestRAG:
    """
    BestRAG (Best Retrieval Augmented by Qdrant) is a class that provides
    functionality for storing and searching document embeddings in a Qdrant
    vector database.

    It supports both dense and sparse embeddings, as well as a late interaction
    model for improved retrieval performance.

    Args:
        url (str): The URL of the Qdrant server.
        api_key (str): The API key for the Qdrant server.
        collection_name (str): The name of the Qdrant collection to use.
        late_interaction_model_name (Optional[str]): The name of the late
            interaction model to use. Defaults to "BAAI/bge-small-en-v1.5".
    """

    def __init__(self,
                 url: str,
                 api_key: str,
                 collection_name: str,
                 late_interaction_model_name: Optional[str] = "BAAI/bge-small-en-v1.5"
                 ):
        self.collection_name = collection_name
        self.api_key = api_key
        self.url = url
        self.client = QdrantClient(url=url, api_key=api_key)

        self.dense_model = TextEmbedding()
        self.late_interaction_model = TextEmbedding(
            late_interaction_model_name)
        self.sparse_model = Bm25("Qdrant/bm25")

        self._create_or_use_collection()

    def _create_or_use_collection(self):
        """
        Create a new Qdrant collection if it doesn't exist, or use an existing one.
        """
        collections = self.client.get_collections()
        collection_names = [
            collection.name for collection in collections.collections]

        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense-vector": models.VectorParams(
                        size=384,
                        distance=Distance.COSINE
                    ),
                    "output-token-embeddings": models.VectorParams(
                        size=384,
                        distance=Distance.COSINE,
                        multivector_config=models.MultiVectorConfig(
                            comparator=models.MultiVectorComparator.MAX_SIM
                        )
                    ),
                },
                sparse_vectors_config={"sparse": models.SparseVectorParams()}
            )
            print(f"Created collection: {self.collection_name}")
        else:
            print(f"Using existing collection: {self.collection_name}")

    def _clean_text(self, text: str) -> str:
        """
        Clean the input text by removing special characters and formatting.

        Args:
            text (str): The text to be cleaned.

        Returns:
            str: The cleaned text.
        """
        text = re.sub(r'_+', '', text)
        text = text.replace('\n', ' ')
        text = re.sub(r'\s{2,}', ' ', text)
        text = re.sub(r'(\d+\.)\s', r'\n\1 ', text)
        return text.strip()

    def _get_dense_embedding(self, text: str):
        """
        Get the dense embedding for the given text.

        Args:
            text (str): The text to be embedded.

        Returns:
            List[float]: The dense embedding vector.
        """
        return list(self.dense_model.embed([text]))[0]

    def _get_late_interaction_embedding(self, text: str):
        """
        Get the late interaction embedding for the given text.

        Args:
            text (str): The text to be embedded.

        Returns:
            List[float]: The late interaction embedding vector.
        """
        return list(self.late_interaction_model.embed([text]))[0]

    def _get_sparse_embedding(self, text: str):
        """
        Get the sparse embedding for the given text.

        Args:
            text (str): The text to be embedded.

        Returns:
            models.SparseVector: The sparse embedding vector.
        """
        return next(self.sparse_model.embed(text))

    def _extract_pdf_text_per_page(self, pdf_path: str) -> List[str]:
        """
        Load a PDF file and extract the text from each page.

        Args:
            pdf_path (str): The path to the PDF file.

        Returns:
            List[str]: The text from each page of the PDF.
        """
        with open(pdf_path, "rb") as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            return [page.extract_text() for page in reader.pages]

    def store_pdf_embeddings(self, pdf_path: str):
        """
        Store the embeddings for each page of a PDF file in the Qdrant collection.

        Args:
            pdf_path (str): The path to the PDF file.
        """
        texts = self._extract_pdf_text_per_page(pdf_path)

        for page_num, text in enumerate(texts):
            clean_text = self._clean_text(text)
            dense_embedding = self._get_dense_embedding(clean_text)
            late_interaction_embedding = self._get_late_interaction_embedding(
                clean_text)
            sparse_embedding = self._get_sparse_embedding(clean_text)

            hybrid_vector = {
                "dense-vector": dense_embedding,
                "output-token-embeddings": late_interaction_embedding,
                "sparse": models.SparseVector(
                    indices=sparse_embedding.indices,
                    values=sparse_embedding.values,
                )
            }

            payload = {
                "text": clean_text,
                "page_number": page_num + 1
            }

            point = models.PointStruct(
                id=str(uuid.uuid4()),
                vector=hybrid_vector,
                payload=payload
            )

            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )

            print(
                f"Stored embedding for page {page_num + 1} in collection '{self.collection_name}'.")

    def search(self, query: str, limit: int = 10):
        """
        Search the Qdrant collection for documents that match the given query.

        Args:
            query (str): The search query.
            limit (int): The maximum number of results to return. Defaults to 10.

        Returns:
            List[models.SearchResult]: The search results.
        """
        clean_query = self._clean_text(query)
        dense_query = self._get_dense_embedding(clean_query)
        late_interaction_query = self._get_late_interaction_embedding(
            clean_query)
        sparse_query = self._get_sparse_embedding(clean_query)

        query_vector = {
            "dense-vector": dense_query,
            "output-token-embeddings": late_interaction_query,
            "sparse": {
                "indices": sparse_query.indices,
                "values": sparse_query.values,
            }
        }

        results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(
                    query=query_vector["dense-vector"],
                    using="dense-vector",
                    limit=20,
                )
            ],
            query=query_vector["output-token-embeddings"],
            using="output-token-embeddings",
            limit=limit,
        )

        return results

    def __str__(self):
        """
        Return a string representation of the BestRAG object, including its parameters.
        """
        info = (
            "**************************************************\n"
            "* BestRAG Object Information                     *\n"
            "**************************************************\n"
            f"* URL: {self.url}\n"
            f"* API Key: {self.api_key}\n"
            f"* Collection Name: {self.collection_name}\n"
            "**************************************************"
        )
        return info
