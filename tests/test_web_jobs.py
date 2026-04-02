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

class WebJobTests(unittest.TestCase):
    def test_run_ingest_job_deletes_existing_vectors_and_refreshes_backend(self):
        web_jobs_module.INGEST_JOBS.clear()
        web_jobs_module.INGEST_JOBS["job-demo"] = {
            "job_id": "job-demo",
            "status": "queued",
            "filename": "demo.pdf",
            "ocr_model": "qwen3.5-plus",
            "workers": 4,
            "logs": [],
            "error": "",
            "doc_id": "",
            "output_dir": "",
            "index_status": "pending",
            "index_message": "",
            "artifact_warnings": [],
            "artifact_counts": {},
            "index_counts": {},
            "upload_path": "/tmp/demo.pdf",
        }

        fake_cache_clear = MagicMock()
        fake_backend_loader = MagicMock()
        fake_backend_loader.cache_clear = fake_cache_clear
        fake_client = object()

        def _fake_ingest_document(args):
            print(f"ingesting {args.input}")

        with (
            patch("complex_document_rag.web.jobs.create_qdrant_client", return_value=fake_client),
            patch("complex_document_rag.web.jobs.delete_doc_vectors") as mock_delete_doc_vectors,
            patch("complex_document_rag.web.jobs.ingest_document", side_effect=_fake_ingest_document),
            patch(
                "complex_document_rag.web.jobs._compute_ingest_index_status",
                return_value={
                    "index_status": "verified",
                    "index_message": "索引校验通过：文本块 1 -> 文本索引 2；图片描述 1 -> 图片索引 1；表格块 1 -> 表格索引 1",
                    "artifact_warnings": [],
                    "artifact_counts": {
                        "text_blocks": 1,
                        "image_files": 1,
                        "image_descriptions": 1,
                        "table_blocks": 1,
                    },
                    "index_counts": {
                        "text_chunks": 2,
                        "image_descriptions": 1,
                        "table_blocks": 1,
                    },
                },
            ),
            patch("complex_document_rag.web.backend.get_query_backend", fake_backend_loader),
        ):
            web_jobs_module._run_ingest_job("job-demo")

        job = web_jobs_module.INGEST_JOBS["job-demo"]
        self.assertEqual(job["status"], "succeeded")
        self.assertIn("demo", job["doc_id"])
        self.assertEqual(job["index_status"], "verified")
        self.assertEqual(job["index_counts"]["image_descriptions"], 1)
        self.assertIn("索引校验通过", job["index_message"])
        self.assertTrue(any("ingesting /tmp/demo.pdf" in line for line in job["logs"]))
        mock_delete_doc_vectors.assert_called_once_with(fake_client, job["doc_id"], source_path="/tmp/demo.pdf")
        fake_cache_clear.assert_called_once()
        fake_backend_loader.assert_called_once()

    def test_compute_ingest_index_status_reports_verified_with_artifact_warning(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "manifest.json").write_text(
                json.dumps(
                    {
                        "text_blocks": [{"id": "t1"}, {"id": "t2"}],
                        "image_blocks": [{"id": "i1"}, {"id": "i2"}, {"id": "i3"}],
                        "table_blocks": [{"id": "tb1"}],
                    }
                ),
                encoding="utf-8",
            )
            Path(tmpdir, "image_descriptions.json").write_text(
                json.dumps({"img_1": {"summary": "a"}, "img_2": {"summary": "b"}}),
                encoding="utf-8",
            )
            Path(tmpdir, "table_blocks.json").write_text(
                json.dumps([{"table_id": "table_1"}]),
                encoding="utf-8",
            )

            with patch(
                "complex_document_rag.web.jobs.count_doc_vectors",
                side_effect=[5, 2, 1],
            ):
                report = web_jobs_module._compute_ingest_index_status(
                    doc_id="doc_demo",
                    output_dir=tmpdir,
                    source_path="/tmp/demo.pdf",
                    client=object(),
                )

        self.assertEqual(report["index_status"], "verified")
        self.assertEqual(report["artifact_counts"]["image_files"], 3)
        self.assertEqual(report["artifact_counts"]["image_descriptions"], 2)
        self.assertEqual(report["index_counts"]["image_descriptions"], 2)
        self.assertEqual(len(report["artifact_warnings"]), 1)
        self.assertIn("未生成图片描述", report["artifact_warnings"][0])


if __name__ == "__main__":
    unittest.main()
