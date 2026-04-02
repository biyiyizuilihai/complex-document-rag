"""供 ingestion、indexing 和 web 模块共用的路径辅助函数。

把根路径推导收敛到这个小模块里，可以避免代码库里散落大量
`os.path.dirname(..dirname(..))` 之类的手写路径逻辑。
"""

from __future__ import annotations

import os


def project_root_from_file(file_path: str) -> str:
    """从 `complex_document_rag/` 下任意文件推导仓库根目录。

    有些模块直接位于 `complex_document_rag/` 下，有些则位于
    `complex_document_rag/web/`、`complex_document_rag/indexing/` 这类子包中。
    这里会一路向上找到包目录本身，再返回它的父目录，确保所有调用方
    无论层级深浅都拿到同一个仓库根路径。
    """
    current = os.path.abspath(file_path)
    if os.path.isfile(current):
        current = os.path.dirname(current)

    while True:
        if os.path.basename(current) == "complex_document_rag":
            return os.path.dirname(current)
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent

    return os.path.dirname(os.path.dirname(os.path.abspath(file_path)))


__all__ = ["project_root_from_file"]
