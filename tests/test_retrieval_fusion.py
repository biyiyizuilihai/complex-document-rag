import unittest

from complex_document_rag.retrieval.fusion import build_query_fusion_retriever


class _FakeRetriever:
    def retrieve(self, query):  # pragma: no cover - behavior irrelevant for this test
        return []

    async def aretrieve(self, query):  # pragma: no cover - behavior irrelevant for this test
        return []


class QueryUtilsTests(unittest.TestCase):
    def test_build_query_fusion_retriever_disables_async(self):
        retriever = build_query_fusion_retriever(
            retrievers=[_FakeRetriever(), _FakeRetriever()],
            similarity_top_k=8,
        )

        self.assertFalse(retriever.use_async)
        self.assertEqual(retriever.similarity_top_k, 8)


if __name__ == "__main__":
    unittest.main()
