import unittest
from unittest.mock import patch


class CLITests(unittest.TestCase):
    def test_build_parser_exposes_expected_top_level_commands(self):
        from complex_document_rag.cli import build_parser

        parser = build_parser()
        subparsers_action = next(
            action for action in parser._actions if action.__class__.__name__ == "_SubParsersAction"
        )

        self.assertEqual(
            set(subparsers_action.choices),
            {"serve", "ingest", "index-text", "index-images", "index-tables", "query", "qdrant", "cleanup"},
        )

    def test_cleanup_upload_jobs_command_calls_cleanup_helper(self):
        from complex_document_rag.cli import main

        with patch("complex_document_rag.cli.cleanup_upload_jobs", return_value=["/tmp/job_1"]) as mock_cleanup:
            with patch("builtins.print") as mock_print:
                exit_code = main(["cleanup", "upload-jobs", "--keep", "2", "--dry-run", "--root", "/tmp/upload_jobs"])

        self.assertEqual(exit_code, 0)
        mock_cleanup.assert_called_once_with(
            root="/tmp/upload_jobs",
            keep=2,
            dry_run=True,
        )
        self.assertTrue(any("dry-run" in call.args[0] for call in mock_print.call_args_list))


if __name__ == "__main__":
    unittest.main()
