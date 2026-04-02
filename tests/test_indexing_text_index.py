import unittest
from unittest.mock import patch

from llama_index.core import Document

from complex_document_rag.indexing.text_index import build_text_index_from_documents


class TextIndexingTests(unittest.TestCase):
    def test_build_text_index_from_documents_uses_vectorstore_pipeline(self):
        documents = [
            Document(text="第一页内容", metadata={"doc_id": "doc_demo", "page_no": 1}),
            Document(text="第二页内容", metadata={"doc_id": "doc_demo", "page_no": 2}),
        ]

        with patch("complex_document_rag.indexing.text_index.StorageContext.from_defaults") as mock_storage_context:
            mock_storage_context.return_value = "storage-context"
            with patch("complex_document_rag.indexing.text_index.VectorStoreIndex.from_documents") as mock_from_documents:
                mock_from_documents.return_value = "vector-index"

                index = build_text_index_from_documents(documents)

        self.assertEqual(index, "vector-index")
        mock_storage_context.assert_called_once()
        mock_from_documents.assert_called_once_with(
            documents,
            storage_context="storage-context",
            embed_model=unittest.mock.ANY,
            show_progress=True,
        )


if __name__ == "__main__":
    unittest.main()
