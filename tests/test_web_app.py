import unittest
import tempfile
import copy
from unittest.mock import MagicMock, patch
from pathlib import Path

from fastapi.testclient import TestClient

import complex_document_rag.web_app as web_app_module
from complex_document_rag.web_app import QueryBackend, app, create_app
from complex_document_rag.web_helpers import (
    build_artifact_url,
    normalize_table_asset_paths,
    render_answer_markdown_html,
    serialize_scored_node,
)


class _FakeNode:
    def __init__(self, text="", score=0.0, metadata=None):
        self.text = text
        self.score = score
        self.metadata = metadata or {}


class _FakeResponse:
    def __init__(self, response, source_nodes=None):
        self.response = response
        self.source_nodes = source_nodes or []


class _FakeStreamChunk:
    def __init__(self, text: str = "", reasoning: str = ""):
        self.text = text
        self.delta = text or None
        self.additional_kwargs = {"reasoning_delta": reasoning} if reasoning else {}


class _FakeCompletion:
    def __init__(self, text: str):
        self.text = text


class _CapturingJudgeLLM:
    def __init__(self, text: str):
        self.text = text
        self.prompts = []

    def complete(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return _FakeCompletion(self.text)


class _CapturingTextLLM:
    def __init__(self, text: str = "文本回答"):
        self.text = text
        self.complete_prompts = []
        self.stream_prompts = []

    def complete(self, prompt, **kwargs):
        self.complete_prompts.append(prompt)
        return _FakeCompletion(self.text)

    def stream_complete(self, prompt, **kwargs):
        self.stream_prompts.append(prompt)
        return iter([_FakeStreamChunk(text=self.text)])


class _CapturingMultimodalLLM:
    def __init__(self, text: str = "多模态回答"):
        self.text = text
        self.complete_calls = []
        self.stream_calls = []

    def complete(self, prompt, **kwargs):
        self.complete_calls.append({"prompt": prompt, **kwargs})
        return _FakeCompletion(self.text)

    def stream_complete(self, prompt, **kwargs):
        self.stream_calls.append({"prompt": prompt, **kwargs})
        return iter([_FakeStreamChunk(text=self.text)])


class _FakeEmbedModel:
    def __init__(self):
        self.calls = 0

    def get_query_embedding(self, query):
        self.calls += 1
        return [0.1, 0.2, 0.3]


class _CapturingRetriever:
    def __init__(self, nodes=None):
        self.nodes = nodes or []
        self.queries = []

    def retrieve(self, query):
        self.queries.append(query)
        return list(self.nodes)


class _CapturingIndex:
    def __init__(self, retriever, embed_model=None):
        self._retriever = retriever
        self._embed_model = embed_model
        self.top_ks = []

    def as_retriever(self, similarity_top_k):
        self.top_ks.append(similarity_top_k)
        return self._retriever


class _FakeBackend:
    def __init__(self, fail_answer=False, fail_retrieve=False):
        self.fail_answer = fail_answer
        self.fail_retrieve = fail_retrieve

    def retrieve(self, query):
        if self.fail_retrieve:
            raise RuntimeError("Embedding SSL EOF")

        return {
            "text_results": [
                _FakeNode(
                    text="附录 B 规定了警示标识设置要求。",
                    score=0.81,
                    metadata={
                        "doc_id": "doc_demo",
                        "page_no": 12,
                        "page_label": "12",
                        "source_path": "/tmp/demo.pdf",
                        "block_type": "text",
                    },
                )
            ],
            "image_results": [
                _FakeNode(
                    text="",
                    score=0.73,
                    metadata={
                        "type": "image_description",
                        "image_id": "doc_demo_img_p12_001",
                        "summary": "紧急出口标识",
                        "image_path": "/workspace/complex_document_rag/ingestion_output/doc_demo/images/img_p12_001.png",
                        "doc_id": "doc_demo",
                        "page_no": 12,
                        "page_label": "12",
                    },
                )
            ],
            "table_results": [
                _FakeNode(
                    text="",
                    score=0.69,
                    metadata={
                        "type": "table_block",
                        "table_id": "table_p12_001",
                        "caption": "表 B.1 禁止标识",
                        "semantic_summary": "该表列出禁止入内、禁止停留、禁止启动等标识及设置范围。",
                        "raw_table": "| 名称 | 数值 |\n| --- | --- |\n| 苯 | 6 |",
                        "raw_format": "markdown",
                        "doc_id": "doc_demo",
                        "page_no": 12,
                        "page_label": "12",
                    },
                )
            ],
        }

    def answer(self, query, retrieval=None):
        if self.fail_answer:
            raise RuntimeError("LLM unavailable")

        return {
            "answer": "苯相关作业场所需要设置相应警示标识，并参照附录 B。",
            "answer_sources": [
                _FakeNode(
                    text="附录 B 规定了警示标识设置要求。",
                    score=0.81,
                    metadata={
                        "block_type": "text",
                        "doc_id": "doc_demo",
                        "page_no": 12,
                        "page_label": "12",
                        "source_path": "/tmp/demo.pdf",
                    },
                ),
                _FakeNode(
                    text="",
                    score=0.73,
                    metadata={
                        "type": "image_description",
                        "image_id": "doc_demo_img_p12_001",
                        "summary": "紧急出口标识",
                        "image_path": "/workspace/complex_document_rag/ingestion_output/doc_demo/images/img_p12_001.png",
                        "doc_id": "doc_demo",
                        "page_no": 12,
                        "page_label": "12",
                    },
                ),
            ],
            "answer_assets": [
                _FakeNode(
                    text="",
                    score=0.73,
                    metadata={
                        "type": "image_description",
                        "image_id": "doc_demo_img_p12_001",
                        "summary": "紧急出口标识",
                        "image_path": "/workspace/complex_document_rag/ingestion_output/doc_demo/images/img_p12_001.png",
                        "doc_id": "doc_demo",
                        "page_no": 12,
                        "page_label": "12",
                    },
                ),
                _FakeNode(
                    text="",
                    score=0.69,
                    metadata={
                        "type": "table_block",
                        "table_id": "table_p12_001",
                        "caption": "表 B.1 禁止标识",
                        "semantic_summary": "该表列出禁止入内、禁止停留、禁止启动等标识及设置范围。",
                        "raw_table": "| 名称 | 数值 |\n| --- | --- |\n| 苯 | 6 |",
                        "raw_format": "markdown",
                        "doc_id": "doc_demo",
                        "page_no": 12,
                        "page_label": "12",
                    },
                ),
            ],
        }

    def stream_answer(self, query, retrieval=None):
        for delta in ["苯相关作业场所", "需要设置相应警示标识", "，并参照附录 B。"]:
            yield delta


class WebHelpersTests(unittest.TestCase):
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
            patch.object(web_app_module, "TEXT_RETRIEVAL_SCORE_THRESHOLD", 0.6),
            patch.object(web_app_module, "IMAGE_RETRIEVAL_SCORE_THRESHOLD", 0.7),
            patch.object(web_app_module, "TABLE_RETRIEVAL_SCORE_THRESHOLD", 0.7),
            patch.object(web_app_module, "TEXT_RETRIEVAL_SCORE_MARGIN", 0.1),
            patch.object(web_app_module, "IMAGE_RETRIEVAL_SCORE_MARGIN", 0.08),
            patch.object(web_app_module, "TABLE_RETRIEVAL_SCORE_MARGIN", 0.08),
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
            patch.object(web_app_module, "TEXT_RETRIEVAL_SCORE_THRESHOLD", 0.55),
            patch.object(web_app_module, "TEXT_RETRIEVAL_SCORE_MARGIN", 0.12),
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
            patch.object(web_app_module, "IMAGE_RETRIEVAL_SCORE_THRESHOLD", 0.7),
            patch.object(web_app_module, "TABLE_RETRIEVAL_SCORE_THRESHOLD", 0.7),
            patch.object(web_app_module, "IMAGE_RETRIEVAL_SCORE_MARGIN", 0.08),
            patch.object(web_app_module, "TABLE_RETRIEVAL_SCORE_MARGIN", 0.08),
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
            patch.object(web_app_module, "TEXT_RETRIEVAL_SCORE_THRESHOLD", 0.55),
            patch.object(web_app_module, "IMAGE_RETRIEVAL_SCORE_THRESHOLD", 0.70),
            patch.object(web_app_module, "TABLE_RETRIEVAL_SCORE_THRESHOLD", 0.70),
            patch.object(web_app_module, "TEXT_RETRIEVAL_SCORE_MARGIN", 0.12),
            patch.object(web_app_module, "IMAGE_RETRIEVAL_SCORE_MARGIN", 0.08),
            patch.object(web_app_module, "TABLE_RETRIEVAL_SCORE_MARGIN", 0.08),
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
            patch.object(web_app_module, "TEXT_RETRIEVAL_SCORE_THRESHOLD", 0.55),
            patch.object(web_app_module, "TABLE_RETRIEVAL_SCORE_THRESHOLD", 0.70),
            patch.object(web_app_module, "TEXT_RETRIEVAL_SCORE_MARGIN", 0.12),
            patch.object(web_app_module, "TABLE_RETRIEVAL_SCORE_MARGIN", 0.08),
            patch.object(web_app_module, "RERANK_CONFIDENCE_FLOOR", 0.20),
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

    def test_build_artifact_url_maps_files_inside_ingestion_output(self):
        url = build_artifact_url(
            "/workspace/complex_document_rag/ingestion_output/doc_demo/images/img_p12_001.png",
            artifacts_root="/workspace/complex_document_rag/ingestion_output",
        )

        self.assertEqual(url, "/artifacts/doc_demo/images/img_p12_001.png")

    def test_build_artifact_url_rejects_files_outside_ingestion_output(self):
        url = build_artifact_url(
            "/workspace/other/file.png",
            artifacts_root="/workspace/complex_document_rag/ingestion_output",
        )

        self.assertEqual(url, "")

    def test_normalize_table_asset_paths_rewrites_markdown_and_html_assets(self):
        markdown = "![img](images/img_p12_001.png)"
        html = '<img src="images/img_p13_001.png" alt="img_p13_001" />'

        normalized_markdown = normalize_table_asset_paths(markdown, "markdown", "doc_demo")
        normalized_html = normalize_table_asset_paths(html, "html", "doc_demo")

        self.assertIn("/artifacts/doc_demo/images/img_p12_001.png", normalized_markdown)
        self.assertIn('/artifacts/doc_demo/images/img_p13_001.png', normalized_html)

    def test_normalize_table_asset_paths_materializes_bbox_images_from_page_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts_root = Path(tmpdir) / "ingestion_output"
            doc_root = artifacts_root / "doc_demo"
            raw_ocr_dir = doc_root / "raw_pdf_ocr" / "demo"
            page_images_dir = raw_ocr_dir / "page_images"
            page_images_dir.mkdir(parents=True)

            from PIL import Image

            image = Image.new("RGB", (100, 100), "white")
            image.save(page_images_dir / "page_0019.png")

            (raw_ocr_dir / "page_0019_raw.md").write_text(
                "```json\n"
                '{"regions":[{"id":"img_001","caption":"当心中毒","type":"icon","bbox_normalized":[0.2,0.2,0.4,0.4]}]}\n'
                "```",
                encoding="utf-8",
            )

            html = '<img src="bbox://0.2,0.2,0.4,0.4" alt="img_p19_001" style="width: 80%;" />'

            normalized_html = normalize_table_asset_paths(
                html,
                "html",
                "doc_demo",
                page_no=19,
                artifacts_root=str(artifacts_root),
            )

            self.assertIn('/artifacts/doc_demo/images/img_p19_001.png', normalized_html)
            self.assertTrue((doc_root / "images" / "img_p19_001.png").exists())

    def test_serialize_scored_node_normalizes_table_html_assets(self):
        node = _FakeNode(
            metadata={
                "type": "table_block",
                "table_id": "table_p13_001",
                "caption": "表 B.2 警告标识",
                "semantic_summary": "该表概括了警告类标识及其典型作业场所。",
                "raw_table": '<table><tbody><tr><td><img src="images/img_p13_001.png" /></td></tr></tbody></table>',
                "raw_format": "html",
                "doc_id": "doc_demo",
            }
        )

        payload = serialize_scored_node(node, artifacts_root="/workspace/complex_document_rag/ingestion_output")
        self.assertIn("/artifacts/doc_demo/images/img_p13_001.png", payload["raw_table"])
        self.assertEqual(payload["semantic_summary"], "该表概括了警告类标识及其典型作业场所。")

    def test_serialize_scored_node_recovers_image_url_from_legacy_p0_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts_root = Path(tmpdir) / "complex_document_rag" / "ingestion_output"
            image_path = artifacts_root / "doc_demo" / "images" / "img_p15_005.png"
            image_path.parent.mkdir(parents=True)
            image_path.write_bytes(b"png")

            node = _FakeNode(
                metadata={
                    "type": "image_description",
                    "doc_id": "doc_demo",
                    "summary": "注意通风标志",
                    "source_image_path": "p0_basic_rag/ingestion_output/doc_demo/images/img_p15_005.png",
                }
            )

            payload = serialize_scored_node(node, artifacts_root=str(artifacts_root))

        self.assertEqual(payload["image_url"], "/artifacts/doc_demo/images/img_p15_005.png")

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

        with patch.object(web_app_module, "rerank_retrieval_bundle", side_effect=_mutating_rerank):
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
            patch.object(web_app_module, "rerank_retrieval_bundle", side_effect=_mutating_rerank),
            patch.object(web_app_module, "IMAGE_RETRIEVAL_SCORE_THRESHOLD", 0.70),
            patch.object(web_app_module, "TABLE_RETRIEVAL_SCORE_THRESHOLD", 0.70),
            patch.object(web_app_module, "IMAGE_RETRIEVAL_SCORE_MARGIN", 0.08),
            patch.object(web_app_module, "TABLE_RETRIEVAL_SCORE_MARGIN", 0.08),
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

    def test_render_answer_markdown_html_supports_bold_and_tables(self):
        markdown = (
            "根据**表 A.1**，警示图形基本几何图形见下表：\n\n"
            "| 图形 | 含义 |\n"
            "| --- | --- |\n"
            "| 圆环加斜线 | 禁止 |\n"
            "| 等边三角形 | 警告 |\n"
        )

        rendered = render_answer_markdown_html(markdown)

        self.assertIn("<strong>表 A.1</strong>", rendered)
        self.assertIn("<table>", rendered)
        self.assertIn("<td>圆环加斜线</td>", rendered)
        self.assertIn("<td>警告</td>", rendered)

    def test_render_answer_markdown_html_supports_mermaid_code_blocks(self):
        markdown = (
            "流程如下：\n\n"
            "```mermaid\n"
            "flowchart TD\n"
            "A[开始] --> B[复测]\n"
            "```\n"
        )

        rendered = render_answer_markdown_html(markdown)

        self.assertIn('<pre class="mermaid">', rendered)
        self.assertIn("flowchart TD", rendered)
        self.assertIn("A[开始] --&gt; B[复测]", rendered)


class WebAppTests(unittest.TestCase):
    def test_frontend_template_prioritizes_ai_answer(self):
        html = Path("complex_document_rag/web_static/index.html").read_text(encoding="utf-8")

        self.assertIn('id="answerBox"', html)
        self.assertIn("查看证据 / 来源", html)
        self.assertIn("回答附图 / 附表", html)
        self.assertIn("function renderAnswerMarkdown", html)
        self.assertIn("<details", html)
        self.assertIn("card-wide", html)
        self.assertIn("table-image", html)
        self.assertIn("semantic_summary", html)
        self.assertIn('id="imageLightbox"', html)
        self.assertIn('id="lightboxDownload"', html)
        self.assertIn("function openImageLightbox", html)
        self.assertIn("document.addEventListener(\"click\"", html)

    def test_frontend_template_signals_waiting_for_first_token_and_uses_chunk_html(self):
        html = Path("complex_document_rag/web_static/index.html").read_text(encoding="utf-8")

        self.assertIn("等待模型首个 token", html)
        self.assertIn("payload.answer_html || renderAnswerMarkdown(streamedAnswer)", html)

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

    def test_query_endpoint_returns_structured_results(self):
        with patch("complex_document_rag.web_app.get_query_backend", return_value=_FakeBackend()):
            client = TestClient(app)
            response = client.post(
                "/api/query",
                json={"query": "苯的职业病危害告知卡包含哪些信息？", "generate_answer": True},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()

        self.assertEqual(payload["query"], "苯的职业病危害告知卡包含哪些信息？")
        self.assertIn("附录 B", payload["answer"])
        self.assertEqual(len(payload["retrieval"]["text_results"]), 1)
        self.assertEqual(len(payload["retrieval"]["image_results"]), 1)
        self.assertEqual(len(payload["retrieval"]["table_results"]), 1)
        self.assertEqual(
            payload["retrieval"]["image_results"][0]["image_url"],
            "/artifacts/doc_demo/images/img_p12_001.png",
        )
        self.assertEqual(payload["answer_sources"][1]["kind"], "image")
        self.assertEqual(len(payload["answer_assets"]), 2)
        self.assertEqual(payload["answer_assets"][0]["kind"], "table")
        self.assertEqual(payload["answer_assets"][1]["kind"], "image")
        self.assertEqual(
            payload["answer_assets"][0]["semantic_summary"],
            "该表列出禁止入内、禁止停留、禁止启动等标识及设置范围。",
        )

    def test_query_endpoint_keeps_retrieval_when_answer_fails(self):
        with patch("complex_document_rag.web_app.get_query_backend", return_value=_FakeBackend(fail_answer=True)):
            client = TestClient(app)
            response = client.post(
                "/api/query",
                json={"query": "紧急出口标识有哪些方向类型？", "generate_answer": True},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["answer"], "")
        self.assertIn("LLM unavailable", payload["answer_error"])
        self.assertEqual(len(payload["retrieval"]["image_results"]), 1)

    def test_query_endpoint_keeps_http_200_when_retrieval_fails(self):
        with patch("complex_document_rag.web_app.get_query_backend", return_value=_FakeBackend(fail_retrieve=True)):
            client = TestClient(app)
            response = client.post(
                "/api/query",
                json={"query": "紧急出口标识有哪些方向类型？", "generate_answer": True},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["answer"], "")
        self.assertIn("Embedding SSL EOF", payload["retrieval_error"])
        self.assertEqual(payload["retrieval"]["text_results"], [])

    def test_query_endpoint_returns_rendered_answer_html(self):
        class _MarkdownBackend(_FakeBackend):
            def answer(self, query, retrieval=None):
                return {
                    "answer": (
                        "根据**表 A.1**，警示图形基本几何图形见下表：\n\n"
                        "| 图形 | 含义 |\n"
                        "| --- | --- |\n"
                        "| 圆环加斜线 | 禁止 |\n"
                    ),
                    "answer_sources": [],
                    "answer_assets": [],
                }

        with patch("complex_document_rag.web_app.get_query_backend", return_value=_MarkdownBackend()):
            client = TestClient(app)
            response = client.post(
                "/api/query",
                json={"query": "警示图形基本几何图形表", "generate_answer": True},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("<strong>表 A.1</strong>", payload["answer_html"])
        self.assertIn("<table>", payload["answer_html"])

    def test_stream_endpoint_streams_answer_chunks(self):
        with patch("complex_document_rag.web_app.get_query_backend", return_value=_FakeBackend()):
            client = TestClient(app)
            with client.stream(
                "POST",
                "/api/query/stream",
                json={"query": "苯的职业病危害告知卡包含哪些信息？", "generate_answer": True},
            ) as response:
                body = b"".join(response.iter_bytes()).decode("utf-8")

        self.assertEqual(response.status_code, 200)
        self.assertIn("event: retrieval", body)
        self.assertIn("event: chunk", body)
        self.assertIn("event: done", body)
        self.assertIn("苯相关作业场所", body)

    def test_stream_endpoint_emits_partial_answer_html_for_chunks(self):
        class _MarkdownStreamingBackend(_FakeBackend):
            def stream_answer(self, query, retrieval=None):
                return {
                    "stream": iter(
                        [
                            "根据**表 A.1**，警示图形见下表：\n\n",
                            "| 图形 | 含义 |\n| --- | --- |\n| 圆环加斜线 | 禁止 |\n",
                        ]
                    ),
                    "answer_sources": [],
                    "answer_assets": [],
                }

        with patch("complex_document_rag.web_app.get_query_backend", return_value=_MarkdownStreamingBackend()):
            client = TestClient(app)
            with client.stream(
                "POST",
                "/api/query/stream",
                json={"query": "警示图形基本几何图形表", "generate_answer": True},
            ) as response:
                body = b"".join(response.iter_bytes()).decode("utf-8")

        self.assertEqual(response.status_code, 200)
        self.assertIn("event: chunk", body)
        self.assertIn('"answer_html"', body)
        self.assertIn("<strong>表 A.1</strong>", body)
        self.assertIn("<table>", body)

    def test_stream_endpoint_emits_reasoning_before_visible_content(self):
        class _ReasoningBackend(_FakeBackend):
            def stream_answer(self, query, retrieval=None):
                return {
                    "stream": iter(
                        [
                            _FakeStreamChunk(reasoning="Thinking"),
                            _FakeStreamChunk(text="根据"),
                            _FakeStreamChunk(text="证据可知。"),
                        ]
                    ),
                    "answer_sources": [],
                    "answer_assets": [],
                }

        with patch("complex_document_rag.web_app.get_query_backend", return_value=_ReasoningBackend()):
            client = TestClient(app)
            with client.stream(
                "POST",
                "/api/query/stream",
                json={"query": "苯的职业病危害告知卡包含哪些信息？", "generate_answer": True},
            ) as response:
                body = b"".join(response.iter_bytes()).decode("utf-8")

        self.assertEqual(response.status_code, 200)
        self.assertIn("event: reasoning", body)
        self.assertIn('"delta": "Thinking"', body)
        self.assertLess(body.index("event: reasoning"), body.index("event: chunk"))

    def test_stream_endpoint_logs_first_stream_chunk_and_first_visible_content_separately(self):
        class _ReasoningBackend(_FakeBackend):
            def stream_answer(self, query, retrieval=None):
                return {
                    "stream": iter(
                        [
                            _FakeStreamChunk(reasoning="Thinking"),
                            _FakeStreamChunk(text="根据"),
                        ]
                    ),
                    "answer_sources": [],
                    "answer_assets": [],
                }

        fake_logger = MagicMock()
        with patch("complex_document_rag.web_app.LOGGER", fake_logger):
            with patch("complex_document_rag.web_app.get_query_backend", return_value=_ReasoningBackend()):
                client = TestClient(app)
                with client.stream(
                    "POST",
                    "/api/query/stream",
                    json={"query": "苯的职业病危害告知卡包含哪些信息？", "generate_answer": True},
                ) as response:
                    _ = b"".join(response.iter_bytes()).decode("utf-8")

        messages = [call.args[0] for call in fake_logger.info.call_args_list]
        self.assertTrue(any("[llm-any ]" in message for message in messages))
        self.assertTrue(any("[llm-gen ]" in message for message in messages))

    def test_app_warms_query_backend_on_startup(self):
        with patch("complex_document_rag.web_app.get_query_backend", return_value=object()) as mock_get_query_backend:
            warm_app = create_app()
            with TestClient(warm_app):
                pass

        self.assertGreaterEqual(mock_get_query_backend.call_count, 1)

    def test_ingest_page_route_returns_html(self):
        client = TestClient(app)
        response = client.get("/ingest")

        self.assertEqual(response.status_code, 200)
        self.assertIn("文档摄入", response.text)

    def test_ingest_options_endpoint_returns_defaults(self):
        client = TestClient(app)
        response = client.get("/api/ingest/options")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("ocr_models", payload)
        self.assertIn("default_ocr_model", payload)
        self.assertIn("default_workers", payload)

    def test_ingest_job_creation_requires_pdf(self):
        client = TestClient(app)
        response = client.post(
            "/api/ingest/jobs",
            files={"file": ("demo.txt", b"not-a-pdf", "text/plain")},
            data={"ocr_model": "qwen3.5-plus", "workers": "4"},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("仅支持 PDF", response.text)

    def test_ingest_job_creation_returns_accepted_job(self):
        web_app_module.INGEST_JOBS.clear()

        with patch("complex_document_rag.web_app._submit_ingest_job", return_value=None) as mock_submit:
            client = TestClient(app)
            response = client.post(
                "/api/ingest/jobs",
                files={"file": ("demo.pdf", b"%PDF-1.4\n%", "application/pdf")},
                data={"ocr_model": "qwen3.5-plus", "workers": "4"},
            )

        self.assertEqual(response.status_code, 202)
        payload = response.json()
        self.assertEqual(payload["status"], "queued")
        self.assertEqual(payload["filename"], "demo.pdf")
        mock_submit.assert_called_once_with(payload["job_id"])

    def test_ingest_job_status_endpoint_returns_snapshot(self):
        web_app_module.INGEST_JOBS.clear()
        web_app_module.INGEST_JOBS["job-demo"] = {
            "job_id": "job-demo",
            "status": "running",
            "filename": "demo.pdf",
            "ocr_model": "qwen3.5-plus",
            "workers": 4,
            "logs": ["11:00:00 start"],
            "error": "",
            "doc_id": "",
            "output_dir": "",
        }

        client = TestClient(app)
        response = client.get("/api/ingest/jobs/job-demo")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["job_id"], "job-demo")
        self.assertEqual(payload["status"], "running")
        self.assertEqual(payload["logs"], ["11:00:00 start"])

    def test_run_ingest_job_deletes_existing_vectors_and_refreshes_backend(self):
        web_app_module.INGEST_JOBS.clear()
        web_app_module.INGEST_JOBS["job-demo"] = {
            "job_id": "job-demo",
            "status": "queued",
            "filename": "demo.pdf",
            "ocr_model": "qwen3.5-plus",
            "workers": 4,
            "logs": [],
            "error": "",
            "doc_id": "",
            "output_dir": "",
            "upload_path": "/tmp/demo.pdf",
        }

        fake_cache_clear = MagicMock()
        fake_backend_loader = MagicMock()
        fake_backend_loader.cache_clear = fake_cache_clear
        fake_client = object()

        def _fake_ingest_document(args):
            print(f"ingesting {args.input}")

        with (
            patch("complex_document_rag.web_app.create_qdrant_client", return_value=fake_client),
            patch("complex_document_rag.web_app.delete_doc_vectors") as mock_delete_doc_vectors,
            patch("complex_document_rag.web_app.ingest_document", side_effect=_fake_ingest_document),
            patch("complex_document_rag.web_app.get_query_backend", fake_backend_loader),
        ):
            web_app_module._run_ingest_job("job-demo")

        job = web_app_module.INGEST_JOBS["job-demo"]
        self.assertEqual(job["status"], "succeeded")
        self.assertIn("demo", job["doc_id"])
        self.assertTrue(any("ingesting /tmp/demo.pdf" in line for line in job["logs"]))
        mock_delete_doc_vectors.assert_called_once()
        fake_cache_clear.assert_called_once()
        fake_backend_loader.assert_called_once()

    def test_query_backend_uses_dedicated_web_answer_model(self):
        with patch("complex_document_rag.step4_basic_query.load_indexes", return_value=(None, None, None)):
            with patch("complex_document_rag.web_app.create_text_llm", return_value=object()) as mock_create_text_llm:
                backend = QueryBackend()

        self.assertIsNotNone(backend)
        self.assertEqual(mock_create_text_llm.call_count, 2)
        self.assertEqual(
            mock_create_text_llm.call_args_list[0].kwargs["model_name"],
            web_app_module.WEB_ANSWER_LLM_MODEL,
        )
        self.assertFalse(mock_create_text_llm.call_args_list[0].kwargs["disable_thinking"])
        self.assertEqual(
            mock_create_text_llm.call_args_list[1].kwargs["model_name"],
            web_app_module.ASSET_JUDGE_MODEL,
        )
        self.assertTrue(mock_create_text_llm.call_args_list[1].kwargs["disable_thinking"])


if __name__ == "__main__":
    unittest.main()
