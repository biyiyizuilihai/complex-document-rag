import os
import tempfile
import unittest

try:
    from types import SimpleNamespace
    from complex_document_rag.ingestion.pipeline import build_llama_documents, ingest_document
except Exception:  # pragma: no cover - 允许在缺少运行依赖时跳过
    SimpleNamespace = None
    build_llama_documents = None
    ingest_document = None

class IngestionPipelineTests(unittest.TestCase):
    def test_build_llama_documents_preserves_source_doc_id(self):
        if build_llama_documents is None:
            self.skipTest("ingestion pipeline dependencies unavailable")

        documents = build_llama_documents(
            [
                {
                    "content": "demo text",
                    "doc_id": "doc_demo",
                    "source_doc_id": "doc_demo",
                    "source_path": "/tmp/demo.pdf",
                    "page_no": 1,
                    "page_label": "1",
                    "block_type": "text",
                    "origin": "pdf_ocr",
                    "block_id": "doc_demo_text_0001",
                }
            ]
        )

        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].metadata["doc_id"], "doc_demo")
        self.assertEqual(documents[0].metadata["source_doc_id"], "doc_demo")

    def test_ingest_rejects_docx_input_in_pdf_only_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            docx_path = os.path.join(tmpdir, "sample.docx")
            open(docx_path, "w").close()

            args = SimpleNamespace(
                input=docx_path,
                output_dir=os.path.join(tmpdir, "output"),
                ocr_model="qwen3.5-plus",
                workers=1,
                dpi=150,
                visual_image_threshold=5,
                visual_component_threshold=12,
                diagram_threshold=0.18,
            )

            with self.assertRaises(ValueError):
                ingest_document(args)

if __name__ == "__main__":
    unittest.main()
