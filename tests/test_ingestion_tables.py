import os
import tempfile
import unittest

from complex_document_rag.ingestion.tables import build_pdf_ocr_table_blocks

class IngestionTableTests(unittest.TestCase):
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

    def test_build_pdf_ocr_table_blocks_merges_multipage_continued_tables(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ocr_doc_dir = os.path.join(tmpdir, "demo")
            os.makedirs(ocr_doc_dir)

            with open(os.path.join(ocr_doc_dir, "page_0001.md"), "w", encoding="utf-8") as f:
                f.write(
                    "<!-- table: table_001 -->\n"
                    "| 名称 | 设置范围 |\n"
                    "| --- | --- |\n"
                    "| 当心中毒 | 使用有毒物品作业场所 |\n"
                )
            with open(os.path.join(ocr_doc_dir, "page_0001_raw.md"), "w", encoding="utf-8") as f:
                f.write(
                    "```json\n"
                    '{"regions":[],"tables":[{"id":"table_001","caption":"表 B.2 警告标识","type":"simple","headers":["名称","设置范围"],'
                    '"semantic_summary":"该表列出警告标识及设置范围。","continued_from_prev":false,"continues_to_next":true,"bbox_normalized":[0.1,0.2,0.9,0.7]}]}\n'
                    "```"
                )

            with open(os.path.join(ocr_doc_dir, "page_0002.md"), "w", encoding="utf-8") as f:
                f.write(
                    "<!-- table: table_001 -->\n"
                    "| 名称 | 设置范围 |\n"
                    "| --- | --- |\n"
                    "| 当心腐蚀 | 存在腐蚀性物质的作业场所 |\n"
                )
            with open(os.path.join(ocr_doc_dir, "page_0002_raw.md"), "w", encoding="utf-8") as f:
                f.write(
                    "```json\n"
                    '{"regions":[],"tables":[{"id":"table_001","caption":"表 B.2 警告标识（续）","type":"simple","headers":["名称","设置范围"],'
                    '"semantic_summary":"","continued_from_prev":true,"continues_to_next":false,"bbox_normalized":[0.1,0.2,0.9,0.7]}]}\n'
                    "```"
                )

            table_blocks = build_pdf_ocr_table_blocks(
                ocr_doc_dir=ocr_doc_dir,
                doc_id="doc_demo",
                source_path="/tmp/demo.pdf",
                page_labels={1: "12", 2: "13"},
            )

        self.assertEqual(len(table_blocks), 1)
        payload = table_blocks[0]
        self.assertEqual(payload["table_id"], "table_p01_001")
        self.assertEqual(payload["logical_table_id"], "table_p01_001")
        self.assertEqual(payload["page_no"], 1)
        self.assertEqual(payload["page_label"], "12-13")
        self.assertEqual(payload["logical_page_numbers"], [1, 2])
        self.assertEqual(payload["logical_page_labels"], ["12", "13"])
        self.assertEqual(payload["fragment_table_ids"], ["table_p01_001", "table_p02_001"])
        self.assertEqual(payload["fragment_count"], 2)
        self.assertFalse(payload["continued_from_prev"])
        self.assertFalse(payload["continues_to_next"])
        self.assertNotIn("logical-table-page", payload["raw_table"])
        self.assertEqual(payload["logical_table_sections"][0]["page_label"], "12")
        self.assertEqual(payload["logical_table_sections"][1]["page_label"], "13")
        self.assertIn("行1=当心中毒|使用有毒物品作业场所", payload["normalized_table_text"])
        self.assertIn("行2=当心腐蚀|存在腐蚀性物质的作业场所", payload["normalized_table_text"])

    def test_build_pdf_ocr_table_blocks_promotes_heading_context_into_display_title(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ocr_doc_dir = os.path.join(tmpdir, "demo")
            os.makedirs(ocr_doc_dir)

            with open(os.path.join(ocr_doc_dir, "page_0001.md"), "w", encoding="utf-8") as f:
                f.write(
                    "## Appendix 20 - Quality Escalation Matrix\n\n"
                    "<!-- table: table_001 -->\n"
                    "| Level | Condition |\n"
                    "| --- | --- |\n"
                    "| L1 | Minor issue |\n"
                )
            with open(os.path.join(ocr_doc_dir, "page_0001_raw.md"), "w", encoding="utf-8") as f:
                f.write(
                    "```json\n"
                    '{"regions":[],"tables":[{"id":"table_001","caption":"ASSEMBLY FOL [ FRONT OF LINE ] IN AP LINE","type":"simple","headers":["Level","Condition"],'
                    '"semantic_summary":"该表描述质量升级矩阵在前段工序中的触发条件。","continued_from_prev":false,"continues_to_next":false,"bbox_normalized":[0.1,0.2,0.9,0.7]}]}\n'
                    "```"
                )

            table_blocks = build_pdf_ocr_table_blocks(
                ocr_doc_dir=ocr_doc_dir,
                doc_id="doc_demo",
                source_path="/tmp/demo.pdf",
                page_labels={1: "70"},
            )

        self.assertEqual(len(table_blocks), 1)
        payload = table_blocks[0]
        self.assertEqual(payload["section_title"], "Appendix 20 - Quality Escalation Matrix")
        self.assertIn("Quality Escalation Matrix", payload["display_title"])
        self.assertIn("标题：Appendix 20 - Quality Escalation Matrix", payload["normalized_table_text"])

    def test_build_pdf_ocr_table_blocks_logical_sections_exclude_raw_table(self):
        """逻辑表的 logical_table_sections 记录应只保存轻量字段，不重复存储 raw_table。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            ocr_doc_dir = os.path.join(tmpdir, "demo")
            os.makedirs(ocr_doc_dir)

            with open(os.path.join(ocr_doc_dir, "page_0001.md"), "w", encoding="utf-8") as f:
                f.write(
                    "多页表格\n\n"
                    "<!-- table: table_001 -->\n"
                    "| A | B |\n"
                    "| --- | --- |\n"
                    "| 1 | 2 |\n"
                )
            with open(os.path.join(ocr_doc_dir, "page_0001_raw.md"), "w", encoding="utf-8") as f:
                f.write(
                    "```json\n"
                    '{"regions":[],"tables":[{"id":"table_001","caption":"多页表格","type":"simple","headers":["A","B"],'
                    '"semantic_summary":"","continued_from_prev":false,"continues_to_next":true,"bbox_normalized":[0.1,0.1,0.9,0.5]}]}\n'
                    "```"
                )

            with open(os.path.join(ocr_doc_dir, "page_0002.md"), "w", encoding="utf-8") as f:
                f.write(
                    "<!-- table: table_001 -->\n"
                    "| A | B |\n"
                    "| --- | --- |\n"
                    "| 3 | 4 |\n"
                )
            with open(os.path.join(ocr_doc_dir, "page_0002_raw.md"), "w", encoding="utf-8") as f:
                f.write(
                    "```json\n"
                    '{"regions":[],"tables":[{"id":"table_001","caption":"","type":"simple","headers":["A","B"],'
                    '"semantic_summary":"","continued_from_prev":true,"continues_to_next":false,"bbox_normalized":[0.1,0.1,0.9,0.5]}]}\n'
                    "```"
                )

            tables = build_pdf_ocr_table_blocks(
                ocr_doc_dir, doc_id="doc_test", source_path="test.pdf"
            )

        self.assertEqual(len(tables), 1, "两页碎片应合并为一个逻辑表格")
        sections = tables[0].get("logical_table_sections", [])
        self.assertEqual(len(sections), 2, "应有两个页段记录")
        for section in sections:
            self.assertNotIn("raw_table", section, "sections 中不应重复存储 raw_table")
            self.assertIn("table_id", section)
            self.assertIn("page_label", section)
            self.assertIn("page_no", section)
            self.assertIn("row_count", section)

if __name__ == "__main__":
    unittest.main()
