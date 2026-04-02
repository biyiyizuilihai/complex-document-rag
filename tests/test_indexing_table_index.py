import unittest
from llama_index.core.schema import MetadataMode

from complex_document_rag.indexing.table_index import index_single_table


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

    def test_index_single_table_preserves_heading_context_metadata(self):
        node = index_single_table(
            {
                "table_id": "table_p70_001",
                "doc_id": "doc_demo",
                "caption": "ASSEMBLY FOL [ FRONT OF LINE ] IN AP LINE",
                "display_title": "Appendix 20 - Quality Escalation Matrix · ASSEMBLY FOL [ FRONT OF LINE ] IN AP LINE",
                "section_title": "Appendix 20 - Quality Escalation Matrix",
                "semantic_summary": "该表描述质量升级矩阵在前段工序中的触发条件。",
                "normalized_table_text": (
                    "标题：Appendix 20 - Quality Escalation Matrix · ASSEMBLY FOL [ FRONT OF LINE ] IN AP LINE；"
                    "页码：70；列=Level|Condition；行1=L1|Minor issue"
                ),
                "raw_table": "| Level | Condition |\n| --- | --- |\n| L1 | Minor issue |",
            }
        )

        self.assertEqual(node.metadata["display_title"], "Appendix 20 - Quality Escalation Matrix · ASSEMBLY FOL [ FRONT OF LINE ] IN AP LINE")
        self.assertEqual(node.metadata["section_title"], "Appendix 20 - Quality Escalation Matrix")
        self.assertIn("Quality Escalation Matrix", node.text)

    def test_index_single_table_keeps_logical_table_metadata(self):
        node = index_single_table(
            {
                "table_id": "table_p12_001",
                "logical_table_id": "table_p12_001",
                "logical_page_numbers": [12, 13],
                "logical_page_labels": ["12", "13"],
                "fragment_table_ids": ["table_p12_001", "table_p13_001"],
                "fragment_count": 2,
                "doc_id": "doc_demo",
                "caption": "表 B.2 警告标识",
                "semantic_summary": "该表概括了警告类标识及其设置场景。",
                "normalized_table_text": (
                    "页码：12-13；列=名称|设置范围；"
                    "行1=当心中毒|使用有毒物品作业场所；"
                    "行2=当心腐蚀|存在腐蚀性物质的作业场所"
                ),
                "raw_table": "| 名称 | 设置范围 |\n| --- | --- |\n| 当心中毒 | 使用有毒物品作业场所 |",
            }
        )

        self.assertEqual(node.metadata["logical_table_id"], "table_p12_001")
        self.assertEqual(node.metadata["logical_page_numbers"], [12, 13])
        self.assertEqual(node.metadata["logical_page_labels"], ["12", "13"])
        self.assertEqual(node.metadata["fragment_count"], 2)

    def test_index_single_table_has_no_redundant_summary_field(self):
        """表格入库模块不应再向 metadata 写入 summary，只保留 semantic_summary。"""
        node = index_single_table(
            {
                "table_id": "table_p5_001",
                "doc_id": "doc_demo",
                "caption": "变更通知流程",
                "semantic_summary": "该表列出了各变更类型的通知流程和审批要求。",
                "normalized_table_text": "列=变更类型|审批人；行1=材料变更|质量总监",
                "raw_table": "| 变更类型 | 审批人 |\n| --- | --- |\n| 材料变更 | 质量总监 |",
            }
        )
        self.assertNotIn("summary", node.metadata)
        self.assertEqual(node.metadata["semantic_summary"], "该表列出了各变更类型的通知流程和审批要求。")

    def test_index_single_table_embedding_text_starts_with_semantic_summary(self):
        """在 embedding 文本中，semantic_summary 应出现在 normalized_table_text 之前。"""
        node = index_single_table(
            {
                "table_id": "table_p20_001",
                "doc_id": "doc_demo",
                "caption": "PFAS 管控要求",
                "semantic_summary": "该表概括了PFAS物质的管控阈值和替换时间表。",
                "normalized_table_text": "列=物质|阈值|截止日期；行1=PFOA|100ppb|2026-12",
                "raw_table": "| 物质 | 阈值 |\n| --- | --- |\n| PFOA | 100ppb |",
            }
        )
        summary_pos = node.text.find("该表概括了PFAS物质的管控阈值")
        table_pos = node.text.find("列=物质|阈值")
        self.assertGreater(summary_pos, -1, "semantic_summary 应在 embedding text 中")
        self.assertGreater(table_pos, -1, "normalized_table_text 应在 embedding text 中")
        self.assertLess(summary_pos, table_pos, "semantic_summary 应排在表格正文之前")

    def test_index_single_table_no_caption_does_not_duplicate_semantic_summary(self):
        """无 caption 时 display_title 回退到 semantic_summary，embedding text 中该文本只应出现一次。"""
        node = index_single_table(
            {
                "table_id": "table_p5_002",
                "doc_id": "doc_demo",
                "semantic_summary": "该表概括了某些内容。",
                "normalized_table_text": "列=A|B；行1=x|y",
            }
        )
        occurrences = node.text.count("该表概括了某些内容。")
        self.assertEqual(occurrences, 1, "semantic_summary 不应重复出现")


if __name__ == "__main__":
    unittest.main()
