import unittest

from complex_document_rag.ingestion.common import build_page_aware_markdown, sanitize_doc_id

class IngestionCommonTests(unittest.TestCase):
    def test_build_page_aware_markdown_keeps_page_boundaries(self):
        markdown = build_page_aware_markdown(
            [
                {"page_no": 1, "content": "第一页内容", "page_label": "i"},
                {"page_no": 2, "content": "第二页内容", "page_label": "1"},
            ]
        )

        self.assertIn("<!-- page: 1 label: i -->", markdown)
        self.assertIn("## 第 i 页", markdown)
        self.assertIn("<!-- page: 2 label: 1 -->", markdown)
        self.assertIn("## 第 1 页", markdown)

    def test_sanitize_doc_id_keeps_non_ascii_stems_distinct(self):
        doc_id = sanitize_doc_id("/tmp/客户投诉流程图.docx")
        self.assertTrue(doc_id.startswith("doc_"))
        self.assertNotEqual(doc_id, "doc_input")

if __name__ == "__main__":
    unittest.main()
