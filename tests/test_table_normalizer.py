import sys
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from table_normalizer import normalize_table_blocks  # noqa: E402


class TableNormalizerTests(unittest.TestCase):
    def test_normalize_table_blocks_repairs_unclosed_simple_table(self):
        markdown = (
            "标题\n\n"
            "<table>\n"
            "<thead><tr><th>列1</th><th>列2</th></tr></thead>\n"
            "<tbody>\n"
            "<tr><td>A</td><td>B</td></tr>\n"
            "<tr><td>C</td><td>D"
        )

        normalized = normalize_table_blocks(markdown)

        self.assertIn("| 列1 | 列2 |", normalized)
        self.assertIn("| A | B |", normalized)
        self.assertIn("<!-- TABLE_UNFINISHED -->", normalized)
        self.assertNotIn("Unclosed <table>", normalized)

    def test_normalize_table_blocks_keeps_closed_table_unchanged(self):
        markdown = (
            "标题\n\n"
            "<table>\n"
            "<thead><tr><th>列1</th></tr></thead>\n"
            "<tbody><tr><td>A</td></tr></tbody>\n"
            "</table>"
        )

        normalized = normalize_table_blocks(markdown)

        self.assertIn("| 列1 |", normalized)
        self.assertIn("| A |", normalized)
        self.assertNotIn("<!-- TABLE_UNFINISHED -->", normalized)


if __name__ == "__main__":
    unittest.main()
