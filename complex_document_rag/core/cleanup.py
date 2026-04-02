"""用于清理运行时遗留目录的辅助函数。

这些目录通常是摄入任务上传的 PDF 等运行产物。把清理逻辑集中到一个
模块中，便于测试，也能避免临时手动删除带来的行为不一致。
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from complex_document_rag.core.paths import project_root_from_file


PROJECT_ROOT = project_root_from_file(__file__)
DEFAULT_UPLOAD_JOBS_ROOT = Path(PROJECT_ROOT) / "complex_document_rag" / "upload_jobs"


def cleanup_upload_jobs(
    root: str | os.PathLike[str] | None = None,
    *,
    keep: int = 0,
    dry_run: bool = False,
) -> list[str]:
    """删除过期的 `job_*` 上传目录，同时保留最新的 N 个。"""
    if keep < 0:
        raise ValueError("keep must be >= 0")

    jobs_root = Path(root) if root is not None else DEFAULT_UPLOAD_JOBS_ROOT
    if not jobs_root.exists():
        return []

    candidates = [
        path
        for path in jobs_root.iterdir()
        if path.is_dir() and path.name.startswith("job_")
    ]
    # 按最后修改时间倒序排列，优先保留最近的任务目录。
    candidates.sort(key=lambda path: (path.stat().st_mtime, path.name), reverse=True)

    removable = candidates[keep:]
    removed_paths: list[str] = []
    for path in removable:
        removed_paths.append(str(path))
        if not dry_run:
            shutil.rmtree(path)

    return removed_paths


__all__ = ["DEFAULT_UPLOAD_JOBS_ROOT", "cleanup_upload_jobs"]
