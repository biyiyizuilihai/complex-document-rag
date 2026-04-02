import unittest

from complex_document_rag.ingestion.docx import markdown_table_from_rows

class IngestionDocxTests(unittest.TestCase):
    def test_markdown_table_from_rows_escapes_cell_content(self):
        markdown = markdown_table_from_rows(
            [
                ["Name", "Note"],
                ["Alice", "A|B"],
                ["Bob", "line1\nline2"],
            ]
        )

        self.assertEqual(
            markdown,
            "\n".join(
                [
                    "| Name | Note |",
                    "| --- | --- |",
                    "| Alice | A\\|B |",
                    "| Bob | line1<br>line2 |",
                ]
            ),
        )

if __name__ == "__main__":
    unittest.main()
