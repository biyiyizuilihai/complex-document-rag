import unittest

from tests.web_test_support import (
    ARTIFACTS_ROOT,
    MagicMock,
    Path,
    QueryBackend,
    TestClient,
    _CapturingIndex,
    _CapturingJudgeLLM,
    _CapturingMultimodalLLM,
    _CapturingRetriever,
    _CapturingTextLLM,
    _FakeBackend,
    _FakeEmbedModel,
    _FakeNode,
    _FakeStreamChunk,
    app,
    build_artifact_url,
    config_module,
    copy,
    create_app,
    json,
    normalize_table_asset_paths,
    patch,
    render_answer_markdown_html,
    serialize_scored_node,
    tempfile,
    web_backend_module,
    web_jobs_module,
    web_query_stream_module,
    web_settings_module,
)

class QueryBackendTests(unittest.TestCase):
    def test_build_query_variants_adds_bilingual_hints_for_cross_lingual_query(self):
        backend = QueryBackend.__new__(QueryBackend)

        variants = QueryBackend.build_query_variants(backend, "MRB触发时机")

        self.assertEqual(variants[0], "MRB触发时机")
        self.assertGreaterEqual(len(variants), 2)
        self.assertIn("Material Review Board", variants[1])
        self.assertIn("trigger criteria", variants[1])

    def test_filter_retrieval_enforces_branch_thresholds_and_relative_margin(self):
        backend = QueryBackend.__new__(QueryBackend)
        backend.reranker = object()
        retrieval = {
            "text_results": [
                _FakeNode(score=0.91, metadata={"block_id": "text_1"}),
                _FakeNode(score=0.79, metadata={"block_id": "text_2"}),
                _FakeNode(score=0.58, metadata={"block_id": "text_3"}),
            ],
            "image_results": [
                _FakeNode(score=0.84, metadata={"type": "image_description", "image_id": "img_1"}),
                _FakeNode(score=0.73, metadata={"type": "image_description", "image_id": "img_2"}),
                _FakeNode(score=0.61, metadata={"type": "image_description", "image_id": "img_3"}),
            ],
            "table_results": [
                _FakeNode(score=0.88, metadata={"type": "table_block", "table_id": "table_1"}),
                _FakeNode(score=0.81, metadata={"type": "table_block", "table_id": "table_2"}),
                _FakeNode(score=0.66, metadata={"type": "table_block", "table_id": "table_3"}),
            ],
        }

        with (
            patch.object(web_backend_module, "TEXT_RETRIEVAL_SCORE_THRESHOLD", 0.6),
            patch.object(web_backend_module, "IMAGE_RETRIEVAL_SCORE_THRESHOLD", 0.7),
            patch.object(web_backend_module, "TABLE_RETRIEVAL_SCORE_THRESHOLD", 0.7),
            patch.object(web_backend_module, "TEXT_RETRIEVAL_SCORE_MARGIN", 0.1),
            patch.object(web_backend_module, "IMAGE_RETRIEVAL_SCORE_MARGIN", 0.08),
            patch.object(web_backend_module, "TABLE_RETRIEVAL_SCORE_MARGIN", 0.08),
        ):
            filtered = QueryBackend.filter_retrieval(backend, retrieval)

        self.assertEqual(
            [node.metadata.get("block_id") for node in filtered["text_results"]],
            ["text_1"],
        )
        self.assertEqual(
            [node.metadata.get("image_id") for node in filtered["image_results"]],
            ["img_1"],
        )
        self.assertEqual(
            [node.metadata.get("table_id") for node in filtered["table_results"]],
            ["table_1", "table_2"],
        )

    def test_filter_retrieval_drops_low_confidence_branch_results_entirely(self):
        backend = QueryBackend.__new__(QueryBackend)
        backend.reranker = object()
        retrieval = {
            "text_results": [
                _FakeNode(score=0.43, metadata={"block_id": "text_1"}),
                _FakeNode(score=0.39, metadata={"block_id": "text_2"}),
            ],
            "image_results": [],
            "table_results": [],
        }

        with (
            patch.object(web_backend_module, "TEXT_RETRIEVAL_SCORE_THRESHOLD", 0.55),
            patch.object(web_backend_module, "TEXT_RETRIEVAL_SCORE_MARGIN", 0.12),
        ):
            filtered = QueryBackend.filter_retrieval(backend, retrieval)

        self.assertEqual(filtered["text_results"], [])

    def test_filter_retrieval_without_reranker_does_not_apply_absolute_score_cutoff(self):
        backend = QueryBackend.__new__(QueryBackend)
        backend.reranker = None
        retrieval = {
            "text_results": [],
            "image_results": [
                _FakeNode(score=0.5427, metadata={"type": "image_description", "image_id": "img_1"}),
                _FakeNode(score=0.5404, metadata={"type": "image_description", "image_id": "img_2"}),
                _FakeNode(score=0.51, metadata={"type": "image_description", "image_id": "img_3"}),
            ],
            "table_results": [
                _FakeNode(score=0.5623, metadata={"type": "table_block", "table_id": "table_1"}),
                _FakeNode(score=0.5396, metadata={"type": "table_block", "table_id": "table_2"}),
            ],
        }

        with (
            patch.object(web_backend_module, "IMAGE_RETRIEVAL_SCORE_THRESHOLD", 0.7),
            patch.object(web_backend_module, "TABLE_RETRIEVAL_SCORE_THRESHOLD", 0.7),
            patch.object(web_backend_module, "IMAGE_RETRIEVAL_SCORE_MARGIN", 0.08),
            patch.object(web_backend_module, "TABLE_RETRIEVAL_SCORE_MARGIN", 0.08),
        ):
            filtered = QueryBackend.filter_retrieval(backend, retrieval)

        self.assertEqual(
            [node.metadata.get("image_id") for node in filtered["image_results"]],
            ["img_1", "img_2", "img_3"],
        )
        self.assertEqual(
            [node.metadata.get("table_id") for node in filtered["table_results"]],
            ["table_1", "table_2"],
        )

    def test_filter_retrieval_applies_absolute_threshold_only_to_reranked_branches(self):
        backend = QueryBackend.__new__(QueryBackend)
        backend.reranker = object()
        retrieval = {
            "text_results": [
                _FakeNode(score=0.43, metadata={"block_id": "text_1"}),
                _FakeNode(score=0.39, metadata={"block_id": "text_2"}),
            ],
            "image_results": [
                _FakeNode(score=0.66, metadata={"type": "image_description", "image_id": "img_1"}),
                _FakeNode(score=0.63, metadata={"type": "image_description", "image_id": "img_2"}),
            ],
            "table_results": [
                _FakeNode(score=0.64, metadata={"type": "table_block", "table_id": "table_1"}),
                _FakeNode(score=0.60, metadata={"type": "table_block", "table_id": "table_2"}),
            ],
        }

        with (
            patch.object(web_backend_module, "TEXT_RETRIEVAL_SCORE_THRESHOLD", 0.55),
            patch.object(web_backend_module, "IMAGE_RETRIEVAL_SCORE_THRESHOLD", 0.70),
            patch.object(web_backend_module, "TABLE_RETRIEVAL_SCORE_THRESHOLD", 0.70),
            patch.object(web_backend_module, "TEXT_RETRIEVAL_SCORE_MARGIN", 0.12),
            patch.object(web_backend_module, "IMAGE_RETRIEVAL_SCORE_MARGIN", 0.08),
            patch.object(web_backend_module, "TABLE_RETRIEVAL_SCORE_MARGIN", 0.08),
        ):
            filtered = QueryBackend.filter_retrieval(
                backend,
                retrieval,
                reranked_branches={"text_results"},
            )

        self.assertEqual(filtered["text_results"], [])
        self.assertEqual(
            [node.metadata.get("image_id") for node in filtered["image_results"]],
            ["img_1", "img_2"],
        )
        self.assertEqual(
            [node.metadata.get("table_id") for node in filtered["table_results"]],
            ["table_1", "table_2"],
        )

    def test_filter_retrieval_falls_back_to_raw_branch_when_reranker_scores_are_unreliable(self):
        backend = QueryBackend.__new__(QueryBackend)
        backend.reranker = object()
        reranked = {
            "text_results": [
                _FakeNode(score=0.12, metadata={"block_id": "text_reranked_1"}),
                _FakeNode(score=0.02, metadata={"block_id": "text_reranked_2"}),
            ],
            "image_results": [],
            "table_results": [
                _FakeNode(score=0.04, metadata={"type": "table_block", "table_id": "table_reranked_1"}),
                _FakeNode(score=0.01, metadata={"type": "table_block", "table_id": "table_reranked_2"}),
            ],
        }
        raw = {
            "text_results": [
                _FakeNode(score=0.63, metadata={"block_id": "text_raw_1"}),
                _FakeNode(score=0.60, metadata={"block_id": "text_raw_2"}),
            ],
            "image_results": [],
            "table_results": [
                _FakeNode(score=0.62, metadata={"type": "table_block", "table_id": "table_raw_1"}),
                _FakeNode(score=0.58, metadata={"type": "table_block", "table_id": "table_raw_2"}),
            ],
        }

        with (
            patch.object(web_backend_module, "TEXT_RETRIEVAL_SCORE_THRESHOLD", 0.55),
            patch.object(web_backend_module, "TABLE_RETRIEVAL_SCORE_THRESHOLD", 0.70),
            patch.object(web_backend_module, "TEXT_RETRIEVAL_SCORE_MARGIN", 0.12),
            patch.object(web_backend_module, "TABLE_RETRIEVAL_SCORE_MARGIN", 0.08),
            patch.object(web_backend_module, "RERANK_CONFIDENCE_FLOOR", 0.20),
        ):
            filtered = QueryBackend.filter_retrieval(backend, reranked, raw_retrieval=raw)

        self.assertEqual(
            [node.metadata.get("block_id") for node in filtered["text_results"]],
            ["text_raw_1", "text_raw_2"],
        )
        self.assertEqual(
            [node.metadata.get("table_id") for node in filtered["table_results"]],
            ["table_raw_1", "table_raw_2"],
        )

    def test_select_answer_assets_enforces_threshold_and_topk(self):
        backend = QueryBackend.__new__(QueryBackend)
        retrieval = {
            "text_results": [],
            "image_results": [
                _FakeNode(score=0.93, metadata={"type": "image_description", "image_id": "img_1"}),
                _FakeNode(score=0.86, metadata={"type": "image_description", "image_id": "img_2"}),
                _FakeNode(score=0.81, metadata={"type": "image_description", "image_id": "img_3"}),
                _FakeNode(score=0.74, metadata={"type": "image_description", "image_id": "img_4"}),
                _FakeNode(score=0.69, metadata={"type": "image_description", "image_id": "img_5"}),
            ],
            "table_results": [
                _FakeNode(score=0.97, metadata={"type": "table_block", "table_id": "table_1"}),
                _FakeNode(score=0.88, metadata={"type": "table_block", "table_id": "table_2"}),
            ],
        }

        selected = QueryBackend.select_answer_assets(backend, retrieval)

        self.assertEqual(len(selected), 5)
        self.assertTrue(all(node.score >= 0.7 for node in selected))
        selected_ids = [
            node.metadata.get("image_id") or node.metadata.get("table_id")
            for node in selected
        ]
        self.assertEqual(selected_ids, ["table_1", "table_2", "img_1", "img_2", "img_3"])

    def test_select_answer_assets_omits_images_already_rendered_inside_selected_tables(self):
        backend = QueryBackend.__new__(QueryBackend)
        retrieval = {
            "text_results": [],
            "image_results": [
                _FakeNode(score=0.96, metadata={"type": "image_description", "image_id": "doc_demo_img_p06_003"}),
                _FakeNode(score=0.85, metadata={"type": "image_description", "image_id": "doc_demo_img_p06_004"}),
                _FakeNode(score=0.79, metadata={"type": "image_description", "image_id": "doc_demo_img_p09_001"}),
            ],
            "table_results": [
                _FakeNode(
                    score=0.94,
                    metadata={
                        "type": "table_block",
                        "table_id": "table_p06_001",
                        "raw_table": (
                            "| 图形 | 含义 |\n"
                            "| --- | --- |\n"
                            "| ![img_p06_003](images/img_p06_003.png) | 警告 |\n"
                            "| ![img_p06_004](images/img_p06_004.png) | 提示 |\n"
                        ),
                    },
                )
            ],
        }

        selected = QueryBackend.select_answer_assets(backend, retrieval)

        selected_ids = [
            node.metadata.get("image_id") or node.metadata.get("table_id")
            for node in selected
        ]
        self.assertEqual(selected_ids, ["table_p06_001", "doc_demo_img_p09_001"])

    def test_select_answer_assets_orders_selected_items_by_document_flow(self):
        backend = QueryBackend.__new__(QueryBackend)
        retrieval = {
            "text_results": [],
            "image_results": [
                _FakeNode(
                    score=0.88,
                    metadata={
                        "type": "image_description",
                        "image_id": "doc_demo_img_p19_001",
                        "summary": "当心中毒警告标志",
                        "page_no": 19,
                    },
                )
            ],
            "table_results": [
                _FakeNode(
                    score=0.94,
                    metadata={
                        "type": "table_block",
                        "table_id": "table_p13_001",
                        "caption": "表 B.2 警告标识",
                        "page_no": 13,
                    },
                ),
                _FakeNode(
                    score=0.82,
                    metadata={
                        "type": "table_block",
                        "table_id": "table_p12_001",
                        "caption": "表 B.1 禁止标识",
                        "page_no": 12,
                    },
                ),
            ],
        }

        selected = QueryBackend.select_answer_assets(backend, retrieval)
        selected_ids = [node.metadata.get("image_id") or node.metadata.get("table_id") for node in selected]

        self.assertEqual(selected_ids, ["table_p12_001", "table_p13_001", "doc_demo_img_p19_001"])

    def test_select_answer_assets_places_tables_before_images_even_when_images_are_earlier(self):
        backend = QueryBackend.__new__(QueryBackend)
        retrieval = {
            "text_results": [],
            "image_results": [
                _FakeNode(
                    score=0.96,
                    metadata={
                        "type": "image_description",
                        "image_id": "doc_demo_img_p05_001",
                        "summary": "第 5 页图片",
                        "page_no": 5,
                    },
                )
            ],
            "table_results": [
                _FakeNode(
                    score=0.90,
                    metadata={
                        "type": "table_block",
                        "table_id": "table_p12_001",
                        "caption": "表 B.1 禁止标识",
                        "page_no": 12,
                    },
                ),
            ],
        }

        selected = QueryBackend.select_answer_assets(backend, retrieval)
        selected_ids = [node.metadata.get("image_id") or node.metadata.get("table_id") for node in selected]

        self.assertEqual(selected_ids, ["table_p12_001", "doc_demo_img_p05_001"])

    def test_select_answer_assets_uses_llm_judge_to_filter_cross_modal_noise(self):
        backend = QueryBackend.__new__(QueryBackend)
        backend.asset_judge_llm = _CapturingJudgeLLM('{"selected_ids":["doc_demo_img_p48_001"]}')
        retrieval = {
            "text_results": [],
            "image_results": [
                _FakeNode(
                    score=0.6721,
                    metadata={
                        "type": "image_description",
                        "image_id": "doc_demo_img_p48_001",
                        "summary": "关键不良处理流程图",
                        "detailed_description": "关键不良发生后，先停线并通知工程，再做原因分析与 MRB 处理。",
                        "page_no": 48,
                    },
                )
            ],
            "table_results": [
                _FakeNode(
                    score=0.6496,
                    metadata={
                        "type": "table_block",
                        "table_id": "table_p80_001",
                        "caption": "关键失效项目矩阵",
                        "semantic_summary": "该表列出了多个工艺段的关键失效项目。",
                        "headers": ["Process", "Critical failure items"],
                        "normalized_table_text": (
                            "页码：80；列=Process|Critical failure items；"
                            "行1=Process control|Wrong material；"
                            "行2=D/B|Chip crack"
                        ),
                        "page_no": 80,
                    },
                )
            ],
        }

        selected = QueryBackend.select_answer_assets(
            backend,
            retrieval,
            query="关键不良处理流程（flow图）",
        )

        self.assertEqual(
            [node.metadata.get("image_id") or node.metadata.get("table_id") for node in selected],
            ["doc_demo_img_p48_001"],
        )
        self.assertIn("关键不良处理流程（flow图）", backend.asset_judge_llm.prompts[0])
        self.assertIn("关键失效项目矩阵", backend.asset_judge_llm.prompts[0])

    def test_select_answer_assets_prefers_consistent_focus_document_assets(self):
        backend = QueryBackend.__new__(QueryBackend)
        backend.asset_judge_llm = None
        retrieval = {
            "text_results": [
                _FakeNode(
                    score=0.84,
                    metadata={
                        "block_id": "text_doc_a",
                        "doc_id": "doc_a",
                        "page_no": 48,
                    },
                )
            ],
            "image_results": [
                _FakeNode(
                    score=0.66,
                    metadata={
                        "type": "image_description",
                        "image_id": "doc_a_img_p48_001",
                        "summary": "General Flow for Control of Non-conforming Material/Products (Critical Defect) Flowchart",
                        "doc_id": "doc_a",
                        "page_no": 48,
                    },
                )
            ],
            "table_results": [
                _FakeNode(
                    score=0.65,
                    metadata={
                        "type": "table_block",
                        "table_id": "table_p99_001",
                        "caption": "Containment Lot Management Flow (RBG)",
                        "semantic_summary": "该表详细列出了 RBG 围堵批次管理的各个流程步骤。",
                        "doc_id": "doc_b",
                        "page_no": 99,
                    },
                )
            ],
        }

        selected = QueryBackend.select_answer_assets(
            backend,
            retrieval,
            query="关键不良处理流程（flow图）",
        )

        self.assertEqual(
            [node.metadata.get("image_id") or node.metadata.get("table_id") for node in selected],
            ["doc_a_img_p48_001"],
        )

    def test_select_answer_assets_dedupes_same_logical_table_fragments(self):
        backend = QueryBackend.__new__(QueryBackend)
        retrieval = {
            "text_results": [],
            "image_results": [],
            "table_results": [
                _FakeNode(
                    score=0.91,
                    metadata={
                        "type": "table_block",
                        "table_id": "table_p12_001",
                        "logical_table_id": "logical_table_b2",
                        "caption": "表 B.2 警告标识",
                        "page_no": 12,
                    },
                ),
                _FakeNode(
                    score=0.86,
                    metadata={
                        "type": "table_block",
                        "table_id": "table_p13_001",
                        "logical_table_id": "logical_table_b2",
                        "caption": "表 B.2 警告标识（续）",
                        "page_no": 13,
                    },
                ),
            ],
        }

        selected = QueryBackend.select_answer_assets(backend, retrieval)

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].metadata.get("logical_table_id"), "logical_table_b2")

    def test_build_answer_prompt_uses_human_readable_asset_names(self):
        backend = QueryBackend.__new__(QueryBackend)
        retrieval = {
            "text_results": [],
            "image_results": [
                _FakeNode(
                    score=0.81,
                    metadata={
                        "type": "image_description",
                        "image_id": "doc_demo_img_p19_001",
                        "summary": "当心中毒警告标志",
                        "page_no": 19,
                        "page_label": "19",
                    },
                )
            ],
            "table_results": [
                _FakeNode(
                    score=0.83,
                    metadata={
                        "type": "table_block",
                        "table_id": "table_p13_001",
                        "caption": "表 B.2 警告标识",
                        "semantic_summary": "该表列出警告标识及设置范围。",
                        "raw_table": "| 名称 | 说明 |\n| --- | --- |\n| 当心中毒 | 使用有毒物品作业场所 |\n",
                        "page_no": 13,
                        "page_label": "13",
                    },
                )
            ],
        }

        prompt = QueryBackend.build_answer_prompt(
            backend,
            query="使用有毒物品作业场所应设置哪些警示标识？",
            retrieval=retrieval,
            answer_assets=retrieval["table_results"] + retrieval["image_results"],
        )

        self.assertIn("表格:表 B.2 警告标识", prompt)
        self.assertIn("图片:当心中毒警告标志", prompt)
        self.assertIn("标题=表 B.2 警告标识", prompt)
        self.assertIn("名称=当心中毒警告标志", prompt)
        self.assertNotIn("table_p13_001", prompt)
        self.assertNotIn("doc_demo_img_p19_001", prompt)
        self.assertIn("如果输出思考过程，也请全程使用中文", prompt)

    def test_build_answer_prompt_prefers_display_title_when_available(self):
        backend = QueryBackend.__new__(QueryBackend)
        table_node = _FakeNode(
            score=0.83,
            metadata={
                "type": "table_block",
                "table_id": "table_p70_001",
                "caption": "ASSEMBLY FOL [ FRONT OF LINE ] IN AP LINE",
                "display_title": "Appendix 20 - Quality Escalation Matrix · ASSEMBLY FOL [ FRONT OF LINE ] IN AP LINE",
                "section_title": "Appendix 20 - Quality Escalation Matrix",
                "semantic_summary": "该表描述质量升级矩阵在前段工序中的触发条件。",
                "headers": ["Level", "Condition"],
                "normalized_table_text": "标题：Appendix 20 - Quality Escalation Matrix · ASSEMBLY FOL [ FRONT OF LINE ] IN AP LINE；页码：70；列=Level|Condition；行1=L1|Minor issue",
                "page_no": 70,
                "page_label": "70",
            },
        )
        retrieval = {
            "text_results": [],
            "image_results": [],
            "table_results": [table_node],
        }

        prompt = QueryBackend.build_answer_prompt(
            backend,
            query="质量升级矩阵（表格）",
            retrieval=retrieval,
            answer_assets=[table_node],
        )

        self.assertIn("Appendix 20 - Quality Escalation Matrix", prompt)
        self.assertNotIn("标题=ASSEMBLY FOL [ FRONT OF LINE ] IN AP LINE", prompt)

    def test_build_answer_prompt_only_includes_judged_image_and_table_context(self):
        backend = QueryBackend.__new__(QueryBackend)
        image_node = _FakeNode(
            score=0.82,
            metadata={
                "type": "image_description",
                "image_id": "doc_demo_img_p48_001",
                "summary": "关键不良处理流程图",
                "page_no": 48,
                "page_label": "48",
            },
        )
        table_node = _FakeNode(
            score=0.79,
            metadata={
                "type": "table_block",
                "table_id": "table_p80_001",
                "caption": "关键失效项目矩阵",
                "semantic_summary": "该表列出了多个工艺段的关键失效项目。",
                "headers": ["Process", "Critical failure items"],
                "normalized_table_text": (
                    "页码：80；列=Process|Critical failure items；"
                    "行1=Process control|Wrong material"
                ),
                "page_no": 80,
                "page_label": "80",
            },
        )
        retrieval = {
            "text_results": [],
            "image_results": [image_node],
            "table_results": [table_node],
        }

        prompt = QueryBackend.build_answer_prompt(
            backend,
            query="关键不良处理流程（flow图）",
            retrieval=retrieval,
            answer_assets=[image_node],
        )

        self.assertIn("名称=关键不良处理流程图", prompt)
        self.assertNotIn("标题=关键失效项目矩阵", prompt)
        self.assertNotIn("该表列出了多个工艺段的关键失效项目。", prompt)

    def test_build_answer_prompt_uses_compact_table_context_instead_of_full_raw_table(self):
        backend = QueryBackend.__new__(QueryBackend)
        raw_table = (
            "| 名称 | 设置范围 |\n"
            "| --- | --- |\n"
            "| 当心中毒 | 使用有毒物品作业场所 |\n"
            "| 当心腐蚀 | 存在腐蚀性物质的作业场所 |\n"
            "| 当心感染 | 存在生物性职业病危害因素的作业场所 |\n"
        )
        retrieval = {
            "text_results": [],
            "image_results": [],
            "table_results": [
                _FakeNode(
                    score=0.88,
                    metadata={
                        "type": "table_block",
                        "table_id": "table_p13_001",
                        "caption": "表 B.2 警告标识",
                        "semantic_summary": "该表列出警告标识及对应作业场所。",
                        "headers": ["名称", "设置范围"],
                        "normalized_table_text": (
                            "页码：13；列=名称|设置范围；"
                            "行1=当心中毒|使用有毒物品作业场所；"
                            "行2=当心腐蚀|存在腐蚀性物质的作业场所；"
                            "行3=当心感染|存在生物性职业病危害因素的作业场所"
                        ),
                        "raw_table": raw_table,
                        "page_no": 13,
                        "page_label": "13",
                    },
                )
            ],
        }

        prompt = QueryBackend.build_answer_prompt(
            backend,
            query="使用有毒物品作业场所应设置哪些警示标识？",
            retrieval=retrieval,
            answer_assets=retrieval["table_results"],
        )

        self.assertIn("摘要=该表列出警告标识及对应作业场所。", prompt)
        self.assertIn("列名=名称|设置范围", prompt)
        self.assertIn("示例行1=当心中毒|使用有毒物品作业场所", prompt)
        self.assertNotIn(raw_table, prompt)

    def test_build_answer_prompt_requests_mermaid_for_flowchart_queries(self):
        backend = QueryBackend.__new__(QueryBackend)
        image_node = _FakeNode(
            score=0.91,
            metadata={
                "type": "image_description",
                "image_id": "doc_demo_img_p59_001",
                "summary": "低良率不合格处理流程图",
                "page_no": 59,
                "page_label": "59",
            },
        )
        retrieval = {
            "text_results": [],
            "image_results": [image_node],
            "table_results": [],
        }

        prompt = QueryBackend.build_answer_prompt(
            backend,
            query="低良率不合格处理流程单（flow图）",
            retrieval=retrieval,
            answer_assets=[image_node],
        )

        self.assertIn("如果问题涉及流程图", prompt)
        self.assertIn("```mermaid", prompt)
        self.assertIn("先输出文字说明", prompt)

    def test_build_answer_prompt_requests_structured_bulleted_answers(self):
        backend = QueryBackend.__new__(QueryBackend)
        retrieval = {
            "text_results": [
                _FakeNode(
                    text="使用有毒物品作业场所应设置警告标识、指令标识、提示标识和禁止标识。",
                    score=0.82,
                    metadata={"page_no": 12, "page_label": "12"},
                )
            ],
            "image_results": [],
            "table_results": [],
        }

        prompt = QueryBackend.build_answer_prompt(
            backend,
            query="使用有毒物品作业场所应设置哪些警示标识？",
            retrieval=retrieval,
            answer_assets=[],
        )

        self.assertIn("请默认使用分点回答", prompt)
        self.assertIn("先用一句话给出结论", prompt)
        self.assertIn("再用 1. 2. 3. 列出关键点", prompt)

    def test_build_answer_prompt_includes_recent_conversation_history(self):
        backend = QueryBackend.__new__(QueryBackend)
        retrieval = {
            "text_results": [
                _FakeNode(
                    text="TI 客户需要参考附录中的 8D 周期要求。",
                    score=0.83,
                    metadata={"page_no": 51, "page_label": "51"},
                )
            ],
            "image_results": [],
            "table_results": [],
        }

        prompt = QueryBackend.build_answer_prompt(
            backend,
            query="那 TI 呢？",
            retrieval=retrieval,
            answer_assets=[],
            history=[
                {
                    "query": "ST 客户的 8D 要求是什么？",
                    "answer": "ST 客户要求 8D CT 为 10 个日历天。",
                }
            ],
        )

        self.assertIn("对话历史", prompt)
        self.assertIn("上一轮用户：ST 客户的 8D 要求是什么？", prompt)
        self.assertIn("上一轮助手：ST 客户要求 8D CT 为 10 个日历天。", prompt)
        self.assertIn("当前问题：那 TI 呢？", prompt)

    def test_build_answer_prompt_merges_logical_table_pages_into_single_context(self):
        backend = QueryBackend.__new__(QueryBackend)
        table_node = _FakeNode(
            score=0.88,
            metadata={
                "type": "table_block",
                "table_id": "table_p12_001",
                "logical_table_id": "logical_table_b2",
                "caption": "表 B.2 警告标识",
                "semantic_summary": "该逻辑表列出警告类标识及设置范围。",
                "headers": ["名称", "设置范围"],
                "normalized_table_text": (
                    "页码：12-13；列=名称|设置范围；"
                    "行1=当心中毒|使用有毒物品作业场所；"
                    "行2=当心腐蚀|存在腐蚀性物质的作业场所"
                ),
                "logical_page_labels": ["12", "13"],
                "fragment_count": 2,
                "page_no": 12,
                "page_label": "12-13",
            },
        )
        retrieval = {
            "text_results": [],
            "image_results": [],
            "table_results": [table_node],
        }

        prompt = QueryBackend.build_answer_prompt(
            backend,
            query="详细解释一下表 B.2",
            retrieval=retrieval,
            answer_assets=[table_node],
        )

        self.assertIn("标题=表 B.2 警告标识", prompt)
        self.assertIn("页码=12-13", prompt)
        self.assertIn("示例行2=当心腐蚀|存在腐蚀性物质的作业场所", prompt)

    def test_build_retrieval_query_calls_llm_with_history_context(self):
        """`_build_retrieval_query` 会调用改写 LLM，并返回其输出。"""
        backend = QueryBackend.__new__(QueryBackend)
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "使用有毒物品作业场所警示标识 表B.2 警告标识"
        mock_llm.complete.return_value = mock_response
        backend.query_rewrite_llm = mock_llm

        retrieval_query = QueryBackend._build_retrieval_query(
            backend,
            query="能帮我详细的解释一下B.2吗？",
            history=[
                {
                    "query": "使用有毒物品作业场所应设置哪些警示标识？",
                    "answer": "警告标识见相关表格（表 B.2 警告标识），指令标识包括注意通风。",
                }
            ],
        )

        self.assertEqual(retrieval_query, "使用有毒物品作业场所警示标识 表B.2 警告标识")
        mock_llm.complete.assert_called_once()
        prompt_arg = mock_llm.complete.call_args[0][0]
        self.assertIn("使用有毒物品作业场所应设置哪些警示标识？", prompt_arg)
        self.assertIn("能帮我详细的解释一下B.2吗？", prompt_arg)

    def test_build_retrieval_query_returns_original_when_no_llm(self):
        """当 `query_rewrite_llm` 为 `None` 时，`_build_retrieval_query` 应回退到原始查询。"""
        backend = QueryBackend.__new__(QueryBackend)
        backend.query_rewrite_llm = None

        retrieval_query = QueryBackend._build_retrieval_query(
            backend,
            query="能帮我详细的解释一下B.2吗？",
            history=[{"query": "使用有毒物品作业场所应设置哪些警示标识？", "answer": "..."}],
        )

        self.assertEqual(retrieval_query, "能帮我详细的解释一下B.2吗？")

    def test_build_retrieval_query_falls_back_on_llm_error(self):
        """如果 LLM 抛异常，`_build_retrieval_query` 应返回原始查询。"""
        backend = QueryBackend.__new__(QueryBackend)
        mock_llm = MagicMock()
        mock_llm.complete.side_effect = RuntimeError("LLM timeout")
        backend.query_rewrite_llm = mock_llm

        retrieval_query = QueryBackend._build_retrieval_query(
            backend,
            query="能帮我详细的解释一下B.2吗？",
            history=[{"query": "使用有毒物品作业场所应设置哪些警示标识？", "answer": "..."}],
        )

        self.assertEqual(retrieval_query, "能帮我详细的解释一下B.2吗？")

    def test_build_retrieval_query_skips_llm_when_no_history(self):
        """当历史对话为空时，`_build_retrieval_query` 应原样返回查询。"""
        backend = QueryBackend.__new__(QueryBackend)
        mock_llm = MagicMock()
        backend.query_rewrite_llm = mock_llm

        result = QueryBackend._build_retrieval_query(backend, query="供应商变更通知流程", history=[])

        self.assertEqual(result, "供应商变更通知流程")
        mock_llm.complete.assert_not_called()

    def test_retrieve_once_reuses_one_query_embedding_for_all_branches(self):
        backend = QueryBackend.__new__(QueryBackend)
        embed_model = _FakeEmbedModel()
        text_retriever = _CapturingRetriever([_FakeNode(score=0.8, metadata={"block_id": "text_1"})])
        image_retriever = _CapturingRetriever([_FakeNode(score=0.7, metadata={"image_id": "img_1"})])
        table_retriever = _CapturingRetriever([_FakeNode(score=0.6, metadata={"table_id": "table_1"})])
        backend.text_index = _CapturingIndex(text_retriever, embed_model=embed_model)
        backend.image_index = _CapturingIndex(image_retriever, embed_model=embed_model)
        backend.table_index = _CapturingIndex(table_retriever, embed_model=embed_model)
        backend.embed_model = embed_model
        backend.reranker = None

        retrieval = QueryBackend._retrieve_once(backend, "query text")

        self.assertEqual(embed_model.calls, 1)
        self.assertEqual(len(retrieval["text_results"]), 1)
        self.assertEqual(len(retrieval["image_results"]), 1)
        self.assertEqual(len(retrieval["table_results"]), 1)
        for retriever in (text_retriever, image_retriever, table_retriever):
            self.assertEqual(len(retriever.queries), 1)
            self.assertEqual(retriever.queries[0].query_str, "query text")
            self.assertEqual(retriever.queries[0].embedding, [0.1, 0.2, 0.3])

    def test_retrieve_once_preserves_raw_scores_when_reranker_mutates_nodes_in_place(self):
        backend = QueryBackend.__new__(QueryBackend)
        embed_model = _FakeEmbedModel()
        raw_text_node = _FakeNode(score=0.63, metadata={"block_id": "text_raw_1"})
        raw_table_node = _FakeNode(score=0.62, metadata={"type": "table_block", "table_id": "table_raw_1"})
        backend.text_index = _CapturingIndex(_CapturingRetriever([raw_text_node]), embed_model=embed_model)
        backend.image_index = None
        backend.table_index = _CapturingIndex(_CapturingRetriever([raw_table_node]), embed_model=embed_model)
        backend.embed_model = embed_model

        class _EnabledReranker:
            enabled = True

        backend.reranker = _EnabledReranker()

        def _mutating_rerank(*args, **kwargs):
            retrieval = kwargs["retrieval"]
            for node in retrieval["text_results"]:
                node.score = 0.12
            for node in retrieval["table_results"]:
                node.score = 0.04
            return retrieval

        with patch.object(web_backend_module, "rerank_retrieval_bundle", side_effect=_mutating_rerank):
            retrieval = QueryBackend._retrieve_once(backend, "MRB触发时机")

        self.assertEqual([round(node.score, 4) for node in retrieval["text_results"]], [0.63])
        self.assertEqual([round(node.score, 4) for node in retrieval["table_results"]], [0.62])

    def test_retrieve_once_only_reranks_text_branch(self):
        backend = QueryBackend.__new__(QueryBackend)
        embed_model = _FakeEmbedModel()
        raw_text_node = _FakeNode(score=0.81, metadata={"block_id": "text_raw_1"})
        raw_image_node = _FakeNode(score=0.66, metadata={"type": "image_description", "image_id": "img_raw_1"})
        raw_table_node = _FakeNode(score=0.64, metadata={"type": "table_block", "table_id": "table_raw_1"})
        backend.text_index = _CapturingIndex(_CapturingRetriever([raw_text_node]), embed_model=embed_model)
        backend.image_index = _CapturingIndex(_CapturingRetriever([raw_image_node]), embed_model=embed_model)
        backend.table_index = _CapturingIndex(_CapturingRetriever([raw_table_node]), embed_model=embed_model)
        backend.embed_model = embed_model

        class _EnabledReranker:
            enabled = True

        backend.reranker = _EnabledReranker()

        def _mutating_rerank(*args, **kwargs):
            retrieval = kwargs["retrieval"]
            for branch in retrieval.values():
                for node in branch:
                    node.score = 0.01
            return retrieval

        with (
            patch.object(web_backend_module, "rerank_retrieval_bundle", side_effect=_mutating_rerank),
            patch.object(web_backend_module, "IMAGE_RETRIEVAL_SCORE_THRESHOLD", 0.70),
            patch.object(web_backend_module, "TABLE_RETRIEVAL_SCORE_THRESHOLD", 0.70),
            patch.object(web_backend_module, "IMAGE_RETRIEVAL_SCORE_MARGIN", 0.08),
            patch.object(web_backend_module, "TABLE_RETRIEVAL_SCORE_MARGIN", 0.08),
        ):
            retrieval = QueryBackend._retrieve_once(backend, "使用有毒物品作业场所应设置哪些警示标识？")

        self.assertEqual([round(node.score, 4) for node in retrieval["text_results"]], [0.81])
        self.assertEqual([round(node.score, 4) for node in retrieval["image_results"]], [0.66])
        self.assertEqual([round(node.score, 4) for node in retrieval["table_results"]], [0.64])

    def test_retrieve_once_merges_results_from_bilingual_query_variants(self):
        class _QueryAwareRetriever:
            def __init__(self, mapping=None):
                self.mapping = mapping or {}
                self.queries = []

            def retrieve(self, query):
                self.queries.append(query.query_str)
                return list(self.mapping.get(query.query_str, []))

        backend = QueryBackend.__new__(QueryBackend)
        embed_model = _FakeEmbedModel()
        expanded_query = "MRB触发时机 Material Review Board trigger criteria when to initiate"
        text_retriever = _QueryAwareRetriever(
            {
                "MRB触发时机": [
                    _FakeNode(score=0.51, metadata={"block_id": "text_original"}),
                ],
                expanded_query: [
                    _FakeNode(score=0.62, metadata={"block_id": "text_expanded"}),
                ],
            }
        )
        backend.text_index = _CapturingIndex(text_retriever, embed_model=embed_model)
        backend.image_index = None
        backend.table_index = None
        backend.embed_model = embed_model
        backend.reranker = None

        with patch.object(QueryBackend, "build_query_variants", return_value=["MRB触发时机", expanded_query]):
            retrieval = QueryBackend._retrieve_once(backend, "MRB触发时机")

        self.assertEqual(embed_model.calls, 2)
        self.assertEqual(text_retriever.queries, ["MRB触发时机", expanded_query])
        self.assertEqual(
            [node.metadata.get("block_id") for node in retrieval["text_results"]],
            ["text_expanded", "text_original"],
        )

    def test_stream_answer_uses_multimodal_llm_for_selected_image_assets(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "flow.png"
            image_path.write_bytes(b"fake-image-bytes")

            backend = QueryBackend.__new__(QueryBackend)
            backend.llm = _CapturingTextLLM(text="文本路径")
            backend.multimodal_llm = _CapturingMultimodalLLM(text="图片路径")
            backend.asset_judge_llm = None
            backend._embedding_cache = {}
            image_node = _FakeNode(
                score=0.92,
                metadata={
                    "type": "image_description",
                    "image_id": "doc_demo_img_p93_001",
                    "summary": "Non-Conformance Lot Management Flowchart",
                    "page_no": 93,
                    "page_label": "93",
                    "image_path": str(image_path),
                },
            )

            result = QueryBackend.stream_answer(
                backend,
                query="Non-Conformance Lot Management Flowchart能帮我具体讲解一下这个图吗？",
                retrieval={
                    "text_results": [],
                    "image_results": [image_node],
                    "table_results": [],
                },
            )

            chunks = list(result["stream"])

        self.assertEqual("".join(chunk.text for chunk in chunks), "图片路径")
        self.assertEqual(len(backend.multimodal_llm.stream_calls), 1)
        self.assertEqual(
            backend.multimodal_llm.stream_calls[0]["image_paths"],
            [str(image_path)],
        )
        self.assertEqual(backend.llm.stream_prompts, [])

    def test_answer_keeps_text_only_path_for_table_assets(self):
        backend = QueryBackend.__new__(QueryBackend)
        backend.llm = _CapturingTextLLM(text="表格回答")
        backend.multimodal_llm = _CapturingMultimodalLLM(text="不该走图片")
        backend.asset_judge_llm = None
        backend._embedding_cache = {}
        table_node = _FakeNode(
            score=0.88,
            metadata={
                "type": "table_block",
                "table_id": "table_p80_001",
                "caption": "质量升级矩阵",
                "semantic_summary": "该表描述了质量升级矩阵的触发条件。",
                "headers": ["Level", "Condition"],
                "normalized_table_text": "页码：80；列=Level|Condition；行1=L1|Minor issue",
                "page_no": 80,
                "page_label": "80",
            },
        )

        result = QueryBackend.answer(
            backend,
            query="质量升级矩阵（表格）",
            retrieval={
                "text_results": [],
                "image_results": [],
                "table_results": [table_node],
            },
        )

        self.assertEqual(result["answer"], "表格回答")
        self.assertEqual(len(backend.multimodal_llm.complete_calls), 0)
        self.assertEqual(len(backend.llm.complete_prompts), 1)

    def test_query_backend_uses_dedicated_web_answer_model(self):
        with patch("complex_document_rag.retrieval.query_console.load_indexes", return_value=(None, None, None)):
            with patch("complex_document_rag.web.backend.create_text_llm", return_value=object()) as mock_create_text_llm:
                backend = QueryBackend()

        self.assertIsNotNone(backend)
        self.assertEqual(mock_create_text_llm.call_count, 3)
        self.assertEqual(
            mock_create_text_llm.call_args_list[0].kwargs["model_name"],
            config_module.WEB_ANSWER_LLM_MODEL,
        )
        self.assertFalse(mock_create_text_llm.call_args_list[0].kwargs["disable_thinking"])
        self.assertEqual(
            mock_create_text_llm.call_args_list[1].kwargs["model_name"],
            web_settings_module.ASSET_JUDGE_MODEL,
        )
        self.assertTrue(mock_create_text_llm.call_args_list[1].kwargs["disable_thinking"])
        self.assertEqual(
            mock_create_text_llm.call_args_list[2].kwargs["model_name"],
            web_settings_module.QUERY_REWRITE_MODEL,
        )
        self.assertTrue(mock_create_text_llm.call_args_list[2].kwargs["disable_thinking"])


class EvidenceFallbackTests(unittest.TestCase):
    def _make_backend(self):
        backend = QueryBackend.__new__(QueryBackend)
        backend.asset_judge_llm = None
        backend._embedding_cache = {}
        return backend

    def test_apply_evidence_fallback_returns_original_when_text_present(self):
        """当文本检索结果非空时，不触发 fallback，直接返回原始 answer_assets。"""
        backend = self._make_backend()
        retrieval = {
            "text_results": [_FakeNode(score=0.6, metadata={"type": "text"})],
            "image_results": [],
            "table_results": [
                _FakeNode(score=0.75, metadata={"type": "table_block", "table_id": "t1"})
            ],
        }
        original_assets = []
        result = backend._apply_evidence_fallback(retrieval, original_assets)
        self.assertEqual(result, original_assets)

    def test_apply_evidence_fallback_returns_original_when_assets_present(self):
        """当 answer_assets 已非空时，不触发 fallback。"""
        backend = self._make_backend()
        retrieval = {
            "text_results": [],
            "image_results": [],
            "table_results": [],
        }
        asset = _FakeNode(score=0.8, metadata={"type": "table_block", "table_id": "t1"})
        result = backend._apply_evidence_fallback(retrieval, [asset])
        self.assertEqual(result, [asset])

    def test_apply_evidence_fallback_uses_filtered_table_results_when_all_empty(self):
        """当文本结果和素材都为空时，fallback 应取过滤后 table_results 里最高分的节点。"""
        backend = self._make_backend()
        tbl_high = _FakeNode(score=0.72, metadata={"type": "table_block", "table_id": "t_high"})
        tbl_low  = _FakeNode(score=0.65, metadata={"type": "table_block", "table_id": "t_low"})
        img      = _FakeNode(score=0.68, metadata={"type": "image_description", "image_id": "img1"})
        retrieval = {
            "text_results": [],
            "image_results": [img],
            "table_results": [tbl_high, tbl_low],
        }
        result = backend._apply_evidence_fallback(retrieval, [])
        self.assertEqual(len(result), 2)
        # 最高分节点排第一
        self.assertEqual(result[0].metadata["table_id"], "t_high")

    def test_apply_evidence_fallback_returns_empty_when_retrieval_also_empty(self):
        """当 retrieval 中也没有任何节点时，fallback 应返回空列表且不崩溃。"""
        backend = self._make_backend()
        retrieval = {"text_results": [], "image_results": [], "table_results": []}
        result = backend._apply_evidence_fallback(retrieval, [])
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
