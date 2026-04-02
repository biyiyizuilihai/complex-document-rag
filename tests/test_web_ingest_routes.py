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

class WebIngestRouteTests(unittest.TestCase):
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
        self.assertEqual(payload["max_workers"], 20)

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
        web_jobs_module.INGEST_JOBS.clear()

        with patch("complex_document_rag.web.jobs._submit_ingest_job", return_value=None) as mock_submit:
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
        web_jobs_module.INGEST_JOBS.clear()
        web_jobs_module.INGEST_JOBS["job-demo"] = {
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


if __name__ == "__main__":
    unittest.main()
