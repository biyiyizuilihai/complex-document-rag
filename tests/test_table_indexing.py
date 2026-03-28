import unittest
from llama_index.core.schema import MetadataMode

from complex_document_rag.step3_table_indexing import index_single_table


class TableIndexingTests(unittest.TestCase):
    def test_index_single_table_keeps_semantic_summary_in_metadata_and_text(self):
        node = index_single_table(
            {
                "table_id": "table_p12_001",
                "doc_id": "doc_demo",
                "caption": "表 B.1 禁止标识",
                "semantic_summary": "该表概括了禁止类标识及其设置场景。",
                "normalized_table_text": "列=编号|名称及图形符号|标识种类|设置范围和地点；行1=1|禁止入内|H|作业场所入口处",
                "raw_table": "| 编号 | 名称 |\n| --- | --- |\n| 1 | 禁止入内 |",
            }
        )

        self.assertEqual(node.metadata["semantic_summary"], "该表概括了禁止类标识及其设置场景。")
        self.assertIn("该表概括了禁止类标识及其设置场景。", node.text)
        self.assertIn("列=编号|名称及图形符号", node.text)

    def test_index_single_table_truncates_overlong_embedding_text(self):
        long_text = "列=缺陷|说明；" + "行1=PROCESS DEFECT|" + ("X" * 9000)
        node = index_single_table(
            {
                "table_id": "table_p114_001",
                "doc_id": "doc_demo",
                "caption": "PROCESS DEFECT / FAILURE DISPOSITION MATRIX",
                "semantic_summary": "该表概括了缺陷类型、判定标准及处置方式。",
                "normalized_table_text": long_text,
                "raw_table": "| Defect | Description |\n| --- | --- |\n| PROCESS DEFECT | ... |",
            }
        )

        embed_content = node.get_content(metadata_mode=MetadataMode.EMBED)

        self.assertLessEqual(len(node.text.encode("utf-8")), 8192)
        self.assertLessEqual(len(embed_content.encode("utf-8")), 8192)
        self.assertIn("该表概括了缺陷类型、判定标准及处置方式。", node.text)
        self.assertIn("[内容已截断]", node.text)
        self.assertEqual(node.metadata["normalized_table_text"], long_text)
        self.assertIn("raw_table", node.excluded_embed_metadata_keys)
        self.assertIn("normalized_table_text", node.excluded_embed_metadata_keys)


if __name__ == "__main__":
    unittest.main()
