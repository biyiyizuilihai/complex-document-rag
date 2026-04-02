"""基于 OCR 原始 markdown 的图片侧摄入辅助函数。"""

from __future__ import annotations

import os
import re
from pathlib import Path

from complex_document_rag.ingestion.common import (
    build_display_title,
    parse_page_number,
    strip_table_markers,
    strip_trailing_region_metadata,
)
from complex_document_rag.ingestion.ocr_layout import extract_regions_from_raw_markdown
from complex_document_rag.ingestion.tables import extract_page_heading_context



def clean_page_text_for_context(page_markdown: str) -> str:
    """去掉较重的标记内容，让页面附近文本能作为图片上下文使用。"""
    text = strip_trailing_region_metadata(page_markdown)
    text = strip_table_markers(text)
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", text)
    text = re.sub(r'<img\s+[^>]*src="[^"]+"[^>]*>', "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()



def build_pdf_ocr_image_descriptions(
    ocr_doc_dir: str,
    project_root: str,
    doc_id: str,
    source_path: str,
    page_labels: dict[int, str] | None = None,
    images_dir: str | None = None,
) -> dict[str, dict[str, object]]:
    """根据 OCR 原始 region 元数据构建图片描述记录。"""
    ocr_dir = Path(ocr_doc_dir)
    project_root = os.path.realpath(project_root)
    source_path = os.path.abspath(source_path)
    page_labels = page_labels or {}
    images_root = Path(images_dir) if images_dir else (ocr_dir / "images")
    descriptions: dict[str, dict[str, object]] = {}

    for raw_page_path in sorted(ocr_dir.glob("page_*_raw.md")):
        page_no = parse_page_number(raw_page_path.name)
        if page_no is None:
            continue

        page_md_path = ocr_dir / f"page_{page_no:04d}.md"
        page_markdown = (
            strip_trailing_region_metadata(page_md_path.read_text(encoding="utf-8"))
            if page_md_path.exists()
            else ""
        )
        page_context = clean_page_text_for_context(page_markdown)
        page_headings = extract_page_heading_context(page_markdown)
        section_title = " / ".join(page_headings[:2])
        page_label = page_labels.get(page_no, str(page_no))
        raw_markdown = raw_page_path.read_text(encoding="utf-8")

        for region in extract_regions_from_raw_markdown(raw_markdown, page_no):
            image_filename = f"{region['id']}.png"
            image_path = images_root / image_filename
            if not image_path.exists():
                continue

            caption = str(region.get("caption", "")).strip()
            region_type = str(region.get("type", "other")).strip() or "other"
            summary = caption or build_display_title(section_title, f"第 {page_label} 页{region_type}") or f"第 {page_label} 页{region_type}"
            detail_parts = [summary]
            if section_title:
                detail_parts.append(f"章节：{section_title}")
            if region_type != "other":
                detail_parts.append(f"类型：{region_type}")
            detail_parts.append(f"页码：{page_label}")
            if page_context:
                detail_parts.append(f"所在页内容：{page_context[:160]}")

            abs_image_path = Path(os.path.realpath(image_path))
            descriptions[f"{doc_id}_{region['id']}"] = {
                "summary": summary,
                "detailed_description": "；".join(part for part in detail_parts if part),
                "nodes": [],
                "external_references": [],
                "tags": [tag for tag in [region_type, f"page_{page_label}"] if tag],
                "source_image_path": os.path.relpath(str(abs_image_path), project_root),
                "source_image_filename": image_filename,
                "doc_id": doc_id,
                "source_doc_id": doc_id,
                "source_path": source_path,
                "source_document_path": source_path,
                "page_no": page_no,
                "page_label": page_label,
                "section_title": section_title,
                "display_title": build_display_title(section_title, summary),
                "origin": "pdf_ocr",
                "block_type": "image",
            }

    return descriptions


__all__ = [
    "build_pdf_ocr_image_descriptions",
    "clean_page_text_for_context",
]
