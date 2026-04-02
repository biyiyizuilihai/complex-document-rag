"""`indexing` 包的惰性导出入口。"""

from __future__ import annotations

from importlib import import_module


_MODULE_PATHS = (
    "complex_document_rag.indexing.image_index",
    "complex_document_rag.indexing.qdrant",
    "complex_document_rag.indexing.table_index",
    "complex_document_rag.indexing.text_index",
)

__all__ = [
    "MAX_TABLE_EMBED_TEXT_LENGTH",
    "TRUNCATION_MARKER",
    "batch_index_images",
    "batch_index_tables",
    "build_text_index",
    "build_text_index_from_documents",
    "count_doc_vectors",
    "create_qdrant_client",
    "delete_doc_vectors",
    "drop_managed_collections",
    "index_single_image",
    "index_single_table",
    "load_existing_index",
    "load_image_index",
    "load_table_index",
    "managed_collection_names",
]


def __getattr__(name: str):
    for module_path in _MODULE_PATHS:
        module = import_module(module_path)
        if hasattr(module, name):
            value = getattr(module, name)
            globals()[name] = value
            return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
