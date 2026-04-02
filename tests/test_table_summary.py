import unittest

from complex_document_rag.ingestion.table_summary import summarize_table_blocks

class TableSummaryTests(unittest.TestCase):
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

if __name__ == "__main__":
    unittest.main()
