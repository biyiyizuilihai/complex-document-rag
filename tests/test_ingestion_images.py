import os
import tempfile
import unittest

from complex_document_rag.ingestion.images import build_pdf_ocr_image_descriptions

class IngestionImageTests(unittest.TestCase):
    def test_build_pdf_ocr_image_descriptions_preserves_absolute_source_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = tmpdir
            ocr_doc_dir = os.path.join(tmpdir, "demo")
            images_dir = os.path.join(ocr_doc_dir, "images")
            os.makedirs(images_dir)

            with open(os.path.join(ocr_doc_dir, "page_0001.md"), "w", encoding="utf-8") as f:
                f.write("流程说明\n\n![img_p01_001](images/img_p01_001.png)\n")
            with open(os.path.join(ocr_doc_dir, "page_0001_raw.md"), "w", encoding="utf-8") as f:
                f.write(
                    "流程说明\n\n"
                    "```json\n"
                    '{"regions":[{"id":"img_001","caption":"登录流程图","type":"diagram"}]}'
                    "\n```"
                )
            image_path = os.path.join(images_dir, "img_p01_001.png")
            open(image_path, "wb").close()

            descriptions = build_pdf_ocr_image_descriptions(
                ocr_doc_dir=ocr_doc_dir,
                project_root=project_root,
                doc_id="doc_demo",
                source_path="relative/demo.pdf",
                page_labels={1: "1"},
                images_dir=images_dir,
            )

        payload = descriptions["doc_demo_img_p01_001"]
        self.assertEqual(payload["source_doc_id"], "doc_demo")
        self.assertTrue(payload["source_path"].endswith(os.path.join("relative", "demo.pdf")))
        self.assertTrue(payload["source_document_path"].endswith(os.path.join("relative", "demo.pdf")))

    def test_build_pdf_ocr_image_descriptions_uses_raw_region_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = tmpdir
            ocr_doc_dir = os.path.join(tmpdir, "demo")
            images_dir = os.path.join(ocr_doc_dir, "images")
            os.makedirs(images_dir)

            with open(os.path.join(ocr_doc_dir, "page_0001.md"), "w", encoding="utf-8") as f:
                f.write("流程说明\n\n![img_p01_001](images/img_p01_001.png)\n")
            with open(os.path.join(ocr_doc_dir, "page_0001_raw.md"), "w", encoding="utf-8") as f:
                f.write(
                    "流程说明\n\n"
                    "```json\n"
                    '{"regions":[{"id":"img_001","caption":"登录流程图","type":"diagram"}]}'
                    "\n```"
                )
            image_path = os.path.join(images_dir, "img_p01_001.png")
            open(image_path, "wb").close()

            descriptions = build_pdf_ocr_image_descriptions(
                ocr_doc_dir=ocr_doc_dir,
                project_root=project_root,
                doc_id="doc_demo",
                source_path="/tmp/demo.pdf",
                page_labels={1: "1"},
            )

        payload = descriptions["doc_demo_img_p01_001"]
        self.assertEqual(payload["summary"], "登录流程图")
        self.assertEqual(payload["origin"], "pdf_ocr")
        self.assertEqual(payload["page_no"], 1)
        self.assertEqual(payload["page_label"], "1")
        self.assertEqual(payload["block_type"], "image")
        self.assertEqual(payload["source_image_path"], os.path.join("demo", "images", "img_p01_001.png"))

if __name__ == "__main__":
    unittest.main()
