import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class WebStaticIngestFrontendTestCase(unittest.TestCase):
    def _load_html(self) -> str:
        return Path("complex_document_rag/web_static/ingest.html").read_text(encoding="utf-8")

    def test_ingest_page_exists_and_has_upload_controls(self):
        html = self._load_html()

        self.assertIn("文档摄入", html)
        self.assertIn('id="fileInput"', html)
        self.assertIn('id="ocrModelSelect"', html)
        self.assertIn('id="workersInput"', html)
        self.assertIn('id="startIngestButton"', html)
        self.assertIn('id="jobLogs"', html)
        self.assertIn('id="jobIndexStatus"', html)
        self.assertIn('id="jobIndexSummary"', html)

    def test_ingest_page_polls_job_status(self):
        html = self._load_html()

        self.assertIn('fetch("/api/ingest/options")', html)
        self.assertIn('fetch("/api/ingest/jobs"', html)
        self.assertIn("function pollJobStatus", html)
        self.assertIn('window.location.href = "/";', html)

    def test_inline_script_is_valid_javascript(self):
        node_path = shutil.which("node")
        if node_path is None:
            self.skipTest("node is required for frontend script syntax validation")

        html = self._load_html()
        script_match = re.search(r"<script>(.*)</script>\s*</body>", html, re.S)
        self.assertIsNotNone(script_match, "expected inline <script> in ingest template")

        with tempfile.NamedTemporaryFile("w", suffix=".js", encoding="utf-8") as handle:
            handle.write(script_match.group(1))
            handle.flush()
            result = subprocess.run(
                [node_path, "--check", handle.name],
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
