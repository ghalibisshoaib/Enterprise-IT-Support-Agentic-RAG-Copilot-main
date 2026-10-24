import unittest
from unittest.mock import Mock, patch

from app.rag.vectorstore import ensure_index, get_embedding_dimension


class EmbeddingDimensionTests(unittest.TestCase):
    def test_known_embedding_dimensions(self):
        self.assertEqual(get_embedding_dimension("text-embedding-3-small"), 1536)
        self.assertEqual(get_embedding_dimension("text-embedding-3-large"), 3072)
        self.assertEqual(get_embedding_dimension("all-MiniLM-L6-v2"), 384)

    def test_unknown_embedding_model_is_rejected(self):
        with self.assertRaises(ValueError):
            get_embedding_dimension("unknown-model")


class EnsureIndexTests(unittest.TestCase):
    @patch("app.rag.vectorstore.Pinecone")
    @patch("app.rag.vectorstore.settings")
    def test_dimension_mismatch_preserves_existing_index(self, settings, pinecone):
        settings.pinecone_api_key = "test-key"
        settings.pinecone_index_name = "existing-index"
        settings.embedding_model = "text-embedding-3-small"

        client = Mock()
        client.list_indexes.return_value = [{"name": "existing-index"}]
        client.describe_index.return_value = {"dimension": 3072}
        pinecone.return_value = client

        with self.assertRaisesRegex(RuntimeError, "Existing index data was not changed"):
            ensure_index()

        client.delete_index.assert_not_called()
        client.create_index.assert_not_called()

    @patch("app.rag.vectorstore.Pinecone")
    @patch("app.rag.vectorstore.settings")
    def test_creates_index_when_missing(self, settings, pinecone):
        settings.pinecone_api_key = "test-key"
        settings.pinecone_index_name = "new-index"
        settings.embedding_model = "text-embedding-3-small"

        client = Mock()
        client.list_indexes.return_value = []
        client.describe_index.return_value = Mock(status={"ready": True})
        pinecone.return_value = client

        ensure_index()

        client.create_index.assert_called_once()
        client.Index.assert_called_once_with("new-index")


if __name__ == "__main__":
    unittest.main()
