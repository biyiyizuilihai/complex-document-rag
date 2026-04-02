"""`retrieval` 包的惰性导出入口。"""

from __future__ import annotations

from importlib import import_module


_MODULE_PATHS = (
    "complex_document_rag.retrieval.fusion",
    "complex_document_rag.retrieval.query_console",
    "complex_document_rag.retrieval.reranking",
)

__all__ = [
    "RETRIEVAL_KEYS",
    "SiliconFlowRerankPostprocessor",
    "SiliconFlowReranker",
    "build_query_fusion_retriever",
    "build_reranker",
    "collection_exists",
    "create_basic_query_engine",
    "get_default_test_queries",
    "load_indexes",
    "rerank_retrieval_bundle",
    "test_full_query",
    "test_retrieval_only",
]


def __getattr__(name: str):
    for module_path in _MODULE_PATHS:
        module = import_module(module_path)
        if hasattr(module, name):
            value = getattr(module, name)
            globals()[name] = value
            return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
