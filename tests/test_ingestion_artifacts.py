import json
import os
import tempfile
import unittest

from complex_document_rag.ingestion.artifacts import (
    collect_folder_ocr_output,
    collect_pdf_ocr_output,
    should_run_visual_parse,
    write_manifest,
)

class IngestionArtifactTests(unittest.TestCase):
    def test_collect_pdf_ocr_output_reads_pages_and_images(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_dir = os.path.join(tmpdir, "demo")
            images_dir = os.path.join(pdf_dir, "images")
            os.makedirs(images_dir)

            with open(os.path.join(pdf_dir, "page_0001.md"), "w", encoding="utf-8") as f:
                f.write("第一页内容\n\n```json\n{\n  \"regions\": []\n}\n")
            with open(os.path.join(pdf_dir, "page_0001_raw.md"), "w", encoding="utf-8") as f:
                f.write("原始bbox内容")
            with open(os.path.join(pdf_dir, "page_0002.md"), "w", encoding="utf-8") as f:
                f.write("第二页内容")
            open(os.path.join(images_dir, "img_p01_001.png"), "wb").close()
            open(os.path.join(images_dir, "img_p02_001.png"), "wb").close()

            merged_text, text_blocks, image_blocks, table_blocks = collect_pdf_ocr_output(
                ocr_doc_dir=pdf_dir,
                doc_id="doc_demo",
                source_path="/tmp/demo.pdf",
            )

        self.assertIn("<!-- page: 1 label: 1 -->", merged_text)
        self.assertIn("## 第 1 页", merged_text)
        self.assertIn("第一页内容", merged_text)
        self.assertNotIn("原始bbox内容", merged_text)
        self.assertNotIn("```json", merged_text)
        self.assertNotIn('"regions"', merged_text)
        self.assertIn("<!-- page: 2 label: 2 -->", merged_text)
        self.assertEqual([block["page_no"] for block in text_blocks], [1, 2])
        self.assertEqual([block["origin"] for block in text_blocks], ["pdf_ocr", "pdf_ocr"])
        self.assertEqual([block["page_no"] for block in image_blocks], [1, 2])
        self.assertEqual(table_blocks, [])
        self.assertTrue(image_blocks[0]["image_path"].endswith("img_p01_001.png"))

    def test_collect_folder_ocr_output_reads_selected_pages(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            page_dir = os.path.join(tmpdir, "page_0003")
            images_dir = os.path.join(page_dir, "images")
            os.makedirs(images_dir)
            with open(os.path.join(page_dir, "page_0003.md"), "w", encoding="utf-8") as f:
                f.write("第三页视觉解析")
            open(os.path.join(images_dir, "img_001.png"), "wb").close()

            merged_text, text_blocks, image_blocks = collect_folder_ocr_output(
                ocr_output_dir=tmpdir,
                doc_id="doc_demo",
                source_path="/tmp/demo.docx",
            )

        self.assertEqual(merged_text, "第三页视觉解析")
        self.assertEqual(text_blocks[0]["page_no"], 3)
        self.assertEqual(text_blocks[0]["origin"], "visual_page")
        self.assertEqual(image_blocks[0]["page_no"], 3)

    def test_should_run_visual_parse_respects_each_trigger(self):
        self.assertTrue(
            should_run_visual_parse(
                inline_image_count=6,
                visual_component_count=1,
                diagram_score=0.01,
            )
        )
        self.assertTrue(
            should_run_visual_parse(
                inline_image_count=0,
                visual_component_count=14,
                diagram_score=0.01,
            )
        )
        self.assertTrue(
            should_run_visual_parse(
                inline_image_count=0,
                visual_component_count=1,
                diagram_score=0.3,
            )
        )
        self.assertFalse(
            should_run_visual_parse(
                inline_image_count=1,
                visual_component_count=4,
                diagram_score=0.05,
            )
        )

    def test_write_manifest_persists_text_and_image_blocks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = os.path.join(tmpdir, "manifest.json")
            text_blocks = [{"block_id": "text_001", "content": "正文"}]
            image_blocks = [{"block_id": "image_001", "image_path": "/tmp/a.png"}]
            table_blocks = [{"block_id": "table_001", "table_id": "table_p01_001"}]

            manifest = write_manifest(
                manifest_path=manifest_path,
                doc_id="doc_demo",
                source_path="/tmp/demo.docx",
                document_markdown_path="/tmp/document.md",
                text_blocks=text_blocks,
                image_blocks=image_blocks,
                table_blocks=table_blocks,
            )

            with open(manifest_path, "r", encoding="utf-8") as f:
                on_disk = json.load(f)

        self.assertEqual(manifest["doc_id"], "doc_demo")
        self.assertEqual(on_disk["text_blocks"][0]["block_id"], "text_001")
        self.assertEqual(on_disk["image_blocks"][0]["block_id"], "image_001")
        self.assertEqual(on_disk["table_blocks"][0]["table_id"], "table_p01_001")

if __name__ == "__main__":
    unittest.main()
