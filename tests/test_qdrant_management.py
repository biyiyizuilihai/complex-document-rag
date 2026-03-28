import unittest

from complex_document_rag.qdrant_management import (
    delete_doc_vectors,
    managed_collection_names,
)


class _FakeClient:
    def __init__(self):
        self.deleted_collections = []
        self.deleted_points = []

    def delete_collection(self, collection_name):
        self.deleted_collections.append(collection_name)

    def delete(self, collection_name, points_selector, wait=True):
        self.deleted_points.append((collection_name, points_selector, wait))


class QdrantManagementTests(unittest.TestCase):
    def test_managed_collection_names_include_tables(self):
        names = managed_collection_names()
        self.assertIn("text_chunks", names)
        self.assertIn("image_descriptions", names)
        self.assertIn("table_blocks", names)

    def test_delete_doc_vectors_deletes_from_all_managed_collections(self):
        client = _FakeClient()

        delete_doc_vectors(client=client, doc_id="doc_demo")

        self.assertEqual(
            [item[0] for item in client.deleted_points],
            managed_collection_names(),
        )


if __name__ == "__main__":
    unittest.main()
