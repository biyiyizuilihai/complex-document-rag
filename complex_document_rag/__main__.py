from __future__ import annotations

import sys


MIN_PYTHON = (3, 10)


def _ensure_supported_python() -> None:
    """在导入仅支持 Python 3.10+ 的依赖前，先用明确提示快速失败。"""
    if sys.version_info < MIN_PYTHON:
        version = ".".join(str(part) for part in sys.version_info[:3])
        required = ".".join(str(part) for part in MIN_PYTHON)
        raise SystemExit(
            f"complex_document_rag 需要 Python {required}+；当前解释器为 {version}。"
            "请使用 `conda activate test` 环境或其他受支持的 Python。"
        )


if __name__ == "__main__":
    _ensure_supported_python()
    from complex_document_rag.cli import main

    raise SystemExit(main())
