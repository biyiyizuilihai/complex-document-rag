import sys
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from batch_ocr import DEFAULT_PROMPT, clean_raw_output, strip_region_json_block  # noqa: E402


class BatchOcrCleanupTests(unittest.TestCase):
    def test_prompt_requires_table_registry(self):
        self.assertIn('"tables"', DEFAULT_PROMPT)
        self.assertIn("<!-- table: table_001 -->", DEFAULT_PROMPT)

    def test_clean_raw_output_preserves_inner_region_fence(self):
        raw_output = (
            "正文内容\n\n"
            "```json\n"
            '{\n  "regions": []\n}\n'
            "```"
        )

        cleaned = clean_raw_output(raw_output)

        self.assertEqual(cleaned, raw_output)

    def test_strip_region_json_block_handles_unterminated_fence(self):
        markdown = (
            "正文内容\n\n"
            "```json\n"
            '{\n  "regions": [{"id": "img_001"}]\n}\n'
        )

        stripped = strip_region_json_block(markdown)

        self.assertEqual(stripped, "正文内容")


if __name__ == "__main__":
    unittest.main()
