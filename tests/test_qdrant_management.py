import unittest

from complex_document_rag.qdrant_management import (
    count_doc_vectors,
    delete_doc_vectors,
    managed_collection_names,
)


class _FakeClient:
    def __init__(self):
        self.deleted_collections = []
        self.deleted_points = []
        self.count_calls = []

    def delete_collection(self, collection_name):
        self.deleted_collections.append(collection_name)

    def delete(self, collection_name, points_selector, wait=True):
        self.deleted_points.append((collection_name, points_selector, wait))

    def count(self, collection_name, count_filter, exact=True):
        self.count_calls.append((collection_name, count_filter, exact))

        class _Response:
            count = 12

        return _Response()


class QdrantManagementTests(unittest.TestCase):
    def test_managed_collection_names_include_tables(self):
        names = managed_collection_names()
        self.assertIn("text_chunks", names)
        self.assertIn("image_descriptions", names)
        self.assertIn("table_blocks", names)

    def test_delete_doc_vectors_deletes_from_all_managed_collections(self):
        client = _FakeClient()

        delete_doc_vectors(client=client, doc_id="doc_demo", source_path="/tmp/demo.pdf")

        self.assertEqual(
            [item[0] for item in client.deleted_points],
            managed_collection_names(),
        )
        selector = client.deleted_points[0][1]
        conditions = selector.filter.should
        self.assertEqual(len(conditions), 4)
        self.assertEqual(conditions[0].key, "source_doc_id")
        self.assertEqual(conditions[0].match.value, "doc_demo")
        self.assertEqual(conditions[1].key, "source_path")
        self.assertEqual(conditions[1].match.value, "/tmp/demo.pdf")
        self.assertEqual(conditions[2].key, "source_document_path")
        self.assertEqual(conditions[2].match.value, "/tmp/demo.pdf")
        self.assertEqual(conditions[3].key, "source_document_path")
        self.assertEqual(conditions[3].match.value, "demo.pdf")

    def test_count_doc_vectors_uses_same_doc_filters(self):
        client = _FakeClient()

        count = count_doc_vectors(
            client=client,
            collection_name="image_descriptions",
            doc_id="doc_demo",
            source_path="/tmp/demo.pdf",
        )

        self.assertEqual(count, 12)
        collection_name, count_filter, exact = client.count_calls[0]
        self.assertEqual(collection_name, "image_descriptions")
        self.assertTrue(exact)
        conditions = count_filter.should
        self.assertEqual(len(conditions), 4)
        self.assertEqual(conditions[0].key, "source_doc_id")
        self.assertEqual(conditions[0].match.value, "doc_demo")
        self.assertEqual(conditions[1].key, "source_path")
        self.assertEqual(conditions[1].match.value, "/tmp/demo.pdf")
        self.assertEqual(conditions[2].key, "source_document_path")
        self.assertEqual(conditions[2].match.value, "/tmp/demo.pdf")
        self.assertEqual(conditions[3].key, "source_document_path")
        self.assertEqual(conditions[3].match.value, "demo.pdf")


if __name__ == "__main__":
    unittest.main()
