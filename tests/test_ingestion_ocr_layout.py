import os
import tempfile
import unittest

from PIL import Image

from complex_document_rag.ingestion.ocr_layout import (
    extract_tables_from_raw_markdown,
    materialize_missing_pdf_region_images,
)

class IngestionOcrLayoutTests(unittest.TestCase):
    def test_extract_tables_from_raw_markdown_normalizes_ids(self):
        raw_markdown = (
            "正文\n\n"
            "```json\n"
            '{"regions":[],"tables":[{"id":"table_001","caption":"表A.1","type":"simple","headers":["图形","含义"],'
            '"semantic_summary":"该表说明图形含义和颜色规则。","continued_from_prev":false,"continues_to_next":true,"bbox_normalized":[0.1,0.2,0.8,0.6]}]}\n'
            "```"
        )

        tables = extract_tables_from_raw_markdown(raw_markdown, page_no=6)

        self.assertEqual(len(tables), 1)
        self.assertEqual(tables[0]["id"], "table_p06_001")
        self.assertEqual(tables[0]["caption"], "表A.1")
        self.assertEqual(tables[0]["semantic_summary"], "该表说明图形含义和颜色规则。")
        self.assertEqual(tables[0]["headers"], ["图形", "含义"])
        self.assertTrue(tables[0]["continues_to_next"])

    def test_materialize_missing_pdf_region_images_backfills_bbox_crops(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ocr_doc_dir = os.path.join(tmpdir, "demo")
            page_images_dir = os.path.join(ocr_doc_dir, "page_images")
            images_dir = os.path.join(ocr_doc_dir, "images")
            os.makedirs(page_images_dir)

            image = Image.new("RGB", (100, 100), "white")
            image.putpixel((30, 30), (255, 0, 0))
            image.save(os.path.join(page_images_dir, "page_0019.png"))

            with open(os.path.join(ocr_doc_dir, "page_0019_raw.md"), "w", encoding="utf-8") as f:
                f.write(
                    "```json\n"
                    '{"regions":[{"id":"img_001","caption":"当心中毒","type":"icon","bbox_normalized":[0.2,0.2,0.4,0.4]}]}\n'
                    "```"
                )

            created = materialize_missing_pdf_region_images(
                ocr_doc_dir=ocr_doc_dir,
                target_images_dir=images_dir,
                padding=0.0,
            )

            output_path = os.path.join(images_dir, "img_p19_001.png")
            self.assertEqual(created, [output_path])
            self.assertTrue(os.path.exists(output_path))
            with Image.open(output_path) as cropped:
                self.assertEqual(cropped.size, (20, 20))

if __name__ == "__main__":
    unittest.main()
