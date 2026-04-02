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

class WebQueryStreamTests(unittest.TestCase):
    def test_stream_endpoint_streams_answer_chunks(self):
        with patch("complex_document_rag.web.backend.get_query_backend", return_value=_FakeBackend()):
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
            def stream_answer(self, query, retrieval=None, history=None):
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

        with patch("complex_document_rag.web.backend.get_query_backend", return_value=_MarkdownStreamingBackend()):
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
            def stream_answer(self, query, retrieval=None, history=None):
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

        with patch("complex_document_rag.web.backend.get_query_backend", return_value=_ReasoningBackend()):
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
            def stream_answer(self, query, retrieval=None, history=None):
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
        with patch.object(web_query_stream_module, "LOGGER", fake_logger):
            with patch("complex_document_rag.web.backend.get_query_backend", return_value=_ReasoningBackend()):
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


if __name__ == "__main__":
    unittest.main()
