import unittest
import tempfile
import copy
import json
from unittest.mock import MagicMock, patch
from pathlib import Path

from fastapi.testclient import TestClient

import complex_document_rag.core.config as config_module
import complex_document_rag.web.backend as web_backend_module
import complex_document_rag.web.jobs as web_jobs_module
import complex_document_rag.web.query_stream as web_query_stream_module
import complex_document_rag.web.settings as web_settings_module
from complex_document_rag.web.backend import QueryBackend
from complex_document_rag.web.helpers import (
    ARTIFACTS_ROOT,
    build_artifact_url,
    normalize_table_asset_paths,
    render_answer_markdown_html,
    serialize_scored_node,
)
from complex_document_rag.web.routes import app, create_app


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

    def retrieve(self, query, history=None):
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

    def retrieve(self, query, history=None):
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

    def answer(self, query, retrieval=None, history=None):
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

    def stream_answer(self, query, retrieval=None, history=None):
        for delta in ["苯相关作业场所", "需要设置相应警示标识", "，并参照附录 B。"]:
            yield delta
