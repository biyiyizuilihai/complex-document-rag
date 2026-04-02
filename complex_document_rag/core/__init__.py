"""`core` 包的惰性导出入口。"""

from __future__ import annotations

from importlib import import_module


_MODULE_PATHS = (
    "complex_document_rag.core.config",
    "complex_document_rag.core.models",
)


def __getattr__(name: str):
    for module_path in _MODULE_PATHS:
        module = import_module(module_path)
        if hasattr(module, name):
            value = getattr(module, name)
            globals()[name] = value
            return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
