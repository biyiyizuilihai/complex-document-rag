import tempfile
import time
import unittest
from pathlib import Path

from complex_document_rag.core.cleanup import cleanup_upload_jobs


class RuntimeCleanupTests(unittest.TestCase):
    def test_cleanup_upload_jobs_removes_only_job_directories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            removable = root / "job_old"
            removable.mkdir()
            (removable / "demo.pdf").write_text("pdf", encoding="utf-8")
            keep_non_job = root / "manual_notes"
            keep_non_job.mkdir()

            removed = cleanup_upload_jobs(root=root)

            self.assertEqual([Path(path).name for path in removed], ["job_old"])
            self.assertFalse(removable.exists())
            self.assertTrue(keep_non_job.exists())

    def test_cleanup_upload_jobs_keeps_newest_directories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            old_dir = root / "job_old"
            mid_dir = root / "job_mid"
            new_dir = root / "job_new"
            for directory in (old_dir, mid_dir, new_dir):
                directory.mkdir()
                (directory / "demo.pdf").write_text(directory.name, encoding="utf-8")
                time.sleep(0.01)

            removed = cleanup_upload_jobs(root=root, keep=1)

            self.assertEqual(sorted(Path(path).name for path in removed), ["job_mid", "job_old"])
            self.assertFalse(old_dir.exists())
            self.assertFalse(mid_dir.exists())
            self.assertTrue(new_dir.exists())

    def test_cleanup_upload_jobs_supports_dry_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            job_dir = root / "job_demo"
            job_dir.mkdir()

            removed = cleanup_upload_jobs(root=root, dry_run=True)

            self.assertEqual([Path(path).name for path in removed], ["job_demo"])
            self.assertTrue(job_dir.exists())

    def test_cleanup_upload_jobs_rejects_negative_keep(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                cleanup_upload_jobs(root=tmpdir, keep=-1)


if __name__ == "__main__":
    unittest.main()
