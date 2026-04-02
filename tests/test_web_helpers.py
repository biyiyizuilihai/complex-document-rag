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

class WebHelperTests(unittest.TestCase):
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


class SerializeScoredNodeImageUrlTests(unittest.TestCase):
    def test_serialize_image_node_returns_empty_url_when_file_missing(self):
        """图片文件不存在时，serialize_scored_node 应返回空 image_url。"""
        import tempfile, os
        with tempfile.TemporaryDirectory() as artifacts_root:
            node = _FakeNode(
                score=0.85,
                metadata={
                    "type": "image_description",
                    "image_id": "img_p109_001",
                    "image_path": os.path.join(artifacts_root, "doc_qcg", "images", "img_p109_001.png"),
                    "doc_id": "doc_qcg",
                    "page_no": 109,
                    "page_label": "109",
                    "source_path": "",
                    "summary": "test image",
                },
            )
            result = serialize_scored_node(node, artifacts_root=artifacts_root)
            self.assertEqual(result["image_url"], "")

    def test_serialize_image_node_returns_url_when_file_exists(self):
        """图片文件存在时，serialize_scored_node 应返回非空 image_url。"""
        import tempfile, os
        with tempfile.TemporaryDirectory() as artifacts_root:
            img_dir = os.path.join(artifacts_root, "doc_qcg", "images")
            os.makedirs(img_dir)
            img_path = os.path.join(img_dir, "img_p109_001.png")
            open(img_path, "wb").close()

            node = _FakeNode(
                score=0.85,
                metadata={
                    "type": "image_description",
                    "image_id": "img_p109_001",
                    "image_path": img_path,
                    "doc_id": "doc_qcg",
                    "page_no": 109,
                    "page_label": "109",
                    "source_path": "",
                    "summary": "test image",
                },
            )
            result = serialize_scored_node(node, artifacts_root=artifacts_root)
            self.assertTrue(result["image_url"].startswith("/artifacts/"))


if __name__ == "__main__":
    unittest.main()
