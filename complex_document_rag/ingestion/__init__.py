"""`ingestion` 包的惰性导出入口。

这个包过去会急切导入所有 ingestion 子模块，导致像
`complex_document_rag.ingestion.image_records` 这样的简单导入也会把整条
step0 流水线拉起来，最终引发循环依赖。现在改成按模块惰性暴露同名公共接口。
"""

from __future__ import annotations

from importlib import import_module


_MODULE_PATHS = (
    "complex_document_rag.ingestion.artifacts",
    "complex_document_rag.ingestion.common",
    "complex_document_rag.ingestion.docx",
    "complex_document_rag.ingestion.image_description",
    "complex_document_rag.ingestion.images",
    "complex_document_rag.ingestion.ocr_layout",
    "complex_document_rag.ingestion.pipeline",
    "complex_document_rag.ingestion.tables",
)

__all__ = [
    "DESCRIBE_PROMPT",
    "batch_describe_images",
    "build_display_title",
    "build_document_markdown",
    "build_ingest_parser",
    "build_normalized_table_text",
    "build_page_aware_markdown",
    "build_pdf_ocr_image_descriptions",
    "build_pdf_ocr_table_blocks",
    "clean_page_text_for_context",
    "collect_folder_ocr_output",
    "collect_pdf_ocr_output",
    "copy_images_to_standard_dir",
    "describe_image",
    "escape_markdown_cell",
    "estimate_page_visual_metrics",
    "extract_docx_native_blocks",
    "extract_ocr_metadata_payload",
    "extract_page_heading_context",
    "extract_pdf_page_labels",
    "extract_regions_from_raw_markdown",
    "extract_table_blocks_from_markdown",
    "extract_tables_from_raw_markdown",
    "get_docx_inline_shape_count",
    "ingest_document",
    "iter_block_items",
    "markdown_table_from_rows",
    "materialize_missing_pdf_region_images",
    "merge_logical_table_blocks",
    "normalize_page_scoped_id",
    "parse_page_number",
    "render_docx_to_pdf",
    "render_pdf_to_images",
    "save_descriptions",
    "sanitize_doc_id",
    "should_run_visual_parse",
    "strip_table_markers",
    "strip_trailing_region_metadata",
    "write_manifest",
]


def __getattr__(name: str):
    for module_path in _MODULE_PATHS:
        module = import_module(module_path)
        if name == "build_ingest_parser" and hasattr(module, "build_parser"):
            value = getattr(module, "build_parser")
            globals()[name] = value
            return value
        if hasattr(module, name):
            value = getattr(module, name)
            globals()[name] = value
            return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
