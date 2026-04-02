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

class WebQueryHttpTests(unittest.TestCase):
    def test_query_endpoint_returns_structured_results(self):
        artifact_path = Path(ARTIFACTS_ROOT) / "doc_demo" / "images" / "img_p12_001.png"
        created_artifact = False
        if not artifact_path.exists():
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path.write_bytes(b"png")
            created_artifact = True

        try:
            with patch("complex_document_rag.web.backend.get_query_backend", return_value=_FakeBackend()):
                client = TestClient(app)
                response = client.post(
                    "/api/query",
                    json={"query": "苯的职业病危害告知卡包含哪些信息？", "generate_answer": True},
                )
        finally:
            if created_artifact:
                artifact_path.unlink(missing_ok=True)
                for candidate in (artifact_path.parent, artifact_path.parent.parent):
                    try:
                        candidate.rmdir()
                    except OSError:
                        pass

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

    def test_query_endpoint_forwards_history_to_backend_answer(self):
        class _HistoryBackend(_FakeBackend):
            def __init__(self):
                super().__init__()
                self.answer_calls = []

            def answer(self, query, retrieval=None, history=None):
                self.answer_calls.append(
                    {
                        "query": query,
                        "history": copy.deepcopy(history),
                    }
                )
                return super().answer(query, retrieval=retrieval)

        backend = _HistoryBackend()
        with patch("complex_document_rag.web.backend.get_query_backend", return_value=backend):
            client = TestClient(app)
            response = client.post(
                "/api/query",
                json={
                    "query": "那 TI 呢？",
                    "generate_answer": True,
                    "history": [
                        {
                            "query": "ST 客户的 8D 要求是什么？",
                            "answer": "ST 客户要求 8D CT 为 10 个日历天。",
                        }
                    ],
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(backend.answer_calls), 1)
        self.assertEqual(
            backend.answer_calls[0]["history"],
            [
                {
                    "query": "ST 客户的 8D 要求是什么？",
                    "answer": "ST 客户要求 8D CT 为 10 个日历天。",
                }
            ],
        )

    def test_query_endpoint_keeps_retrieval_when_answer_fails(self):
        with patch("complex_document_rag.web.backend.get_query_backend", return_value=_FakeBackend(fail_answer=True)):
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
        with patch("complex_document_rag.web.backend.get_query_backend", return_value=_FakeBackend(fail_retrieve=True)):
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
            def answer(self, query, retrieval=None, history=None):
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

        with patch("complex_document_rag.web.backend.get_query_backend", return_value=_MarkdownBackend()):
            client = TestClient(app)
            response = client.post(
                "/api/query",
                json={"query": "警示图形基本几何图形表", "generate_answer": True},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("<strong>表 A.1</strong>", payload["answer_html"])
        self.assertIn("<table>", payload["answer_html"])

    def test_app_warms_query_backend_on_startup(self):
        with patch("complex_document_rag.web.backend.get_query_backend", return_value=object()) as mock_get_query_backend:
            warm_app = create_app()
            with TestClient(warm_app):
                pass

        self.assertGreaterEqual(mock_get_query_backend.call_count, 1)


if __name__ == "__main__":
    unittest.main()
