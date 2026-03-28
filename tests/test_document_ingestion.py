import json
import os
import tempfile
import unittest

from PIL import Image

from complex_document_rag.document_ingestion import (
    build_page_aware_markdown,
    build_pdf_ocr_image_descriptions,
    build_pdf_ocr_table_blocks,
    collect_folder_ocr_output,
    collect_pdf_ocr_output,
    extract_tables_from_raw_markdown,
    materialize_missing_pdf_region_images,
    markdown_table_from_rows,
    sanitize_doc_id,
    should_run_visual_parse,
    write_manifest,
)
from complex_document_rag.table_summary import summarize_table_blocks

try:
    from types import SimpleNamespace

    from complex_document_rag.step0_document_ingestion import ingest_document
except Exception:  # pragma: no cover - 允许在缺少运行依赖时跳过
    SimpleNamespace = None
    ingest_document = None


class DocumentIngestionTests(unittest.TestCase):
    def test_markdown_table_from_rows_escapes_cell_content(self):
        markdown = markdown_table_from_rows(
            [
                ["Name", "Note"],
                ["Alice", "A|B"],
                ["Bob", "line1\nline2"],
            ]
        )

        self.assertEqual(
            markdown,
            "\n".join(
                [
                    "| Name | Note |",
                    "| --- | --- |",
                    "| Alice | A\\|B |",
                    "| Bob | line1<br>line2 |",
                ]
            ),
        )

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

    def test_build_pdf_ocr_table_blocks_preserves_raw_and_normalized_forms(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ocr_doc_dir = os.path.join(tmpdir, "demo")
            os.makedirs(ocr_doc_dir)

            with open(os.path.join(ocr_doc_dir, "page_0001.md"), "w", encoding="utf-8") as f:
                f.write(
                    "表格说明\n\n"
                    "<!-- table: table_001 -->\n"
                    "| 图形 | 含义 |\n"
                    "| --- | --- |\n"
                    "| 圆环加斜线 | 禁止 |\n"
                )
            with open(os.path.join(ocr_doc_dir, "page_0001_raw.md"), "w", encoding="utf-8") as f:
                f.write(
                    "表格说明\n\n"
                    "```json\n"
                    '{"regions":[],"tables":[{"id":"table_001","caption":"表A.1 基本图形","type":"simple","headers":["图形","含义"],'
                    '"semantic_summary":"该表列出了基本图形、对应含义及配色要求。","continued_from_prev":false,"continues_to_next":false,"bbox_normalized":[0.1,0.2,0.9,0.7]}]}\n'
                    "```"
                )

            table_blocks = build_pdf_ocr_table_blocks(
                ocr_doc_dir=ocr_doc_dir,
                doc_id="doc_demo",
                source_path="/tmp/demo.pdf",
                page_labels={1: "1"},
            )

        self.assertEqual(len(table_blocks), 1)
        payload = table_blocks[0]
        self.assertEqual(payload["table_id"], "table_p01_001")
        self.assertEqual(payload["block_type"], "table")
        self.assertEqual(payload["raw_format"], "markdown")
        self.assertIn("| 图形 | 含义 |", payload["raw_table"])
        self.assertIn("列=图形|含义", payload["normalized_table_text"])
        self.assertIn("行1=圆环加斜线|禁止", payload["normalized_table_text"])
        self.assertEqual(payload["caption"], "表A.1 基本图形")
        self.assertEqual(payload["semantic_summary"], "该表列出了基本图形、对应含义及配色要求。")
        self.assertEqual(payload["summary"], "该表列出了基本图形、对应含义及配色要求。")

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

    def test_summarize_table_blocks_prefers_existing_semantic_summary(self):
        table_blocks = [
            {
                "table_id": "table_p12_001",
                "caption": "表 B.1 禁止标识",
                "semantic_summary": "该表概括了禁止类职业病危害标识及其适用场所。",
                "normalized_table_text": "列=编号|名称及图形符号|标识种类|设置范围和地点；行1=1|禁止入内|H|作业场所入口处",
                "page_label": "12",
            }
        ]

        summarized = summarize_table_blocks(table_blocks)

        payload = summarized[0]
        self.assertEqual(payload["caption"], "表 B.1 禁止标识")
        self.assertEqual(payload["semantic_summary"], "该表概括了禁止类职业病危害标识及其适用场所。")
        self.assertEqual(payload["summary"], "该表概括了禁止类职业病危害标识及其适用场所。")

    def test_summarize_table_blocks_falls_back_to_caption_when_builder_is_blank(self):
        table_blocks = [
            {
                "table_id": "table_p17_001",
                "caption": "表 C.1 基本警示语句",
                "normalized_table_text": "列=编号|语句内容；行1=1|禁止入内",
                "page_label": "17",
            }
        ]

        summarized = summarize_table_blocks(table_blocks)

        payload = summarized[0]
        self.assertEqual(payload["semantic_summary"], "表 C.1 基本警示语句")
        self.assertEqual(payload["summary"], "表 C.1 基本警示语句")

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

    def test_sanitize_doc_id_keeps_non_ascii_stems_distinct(self):
        doc_id = sanitize_doc_id("/tmp/客户投诉流程图.docx")
        self.assertTrue(doc_id.startswith("doc_"))
        self.assertNotEqual(doc_id, "doc_input")

    @unittest.skipIf(ingest_document is None, "step0 运行依赖不可用")
    def test_step0_rejects_docx_input_in_pdf_only_mode(self):
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
