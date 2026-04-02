"""文档摄入流程共用的基础辅助函数。

这里统一放文件名与页码解析、markdown 清理和展示标题拼接逻辑，
让表格与图片采集器复用同一套规则。
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import fitz


PAGE_MD_PATTERN = re.compile(r"page_(\d{4})(?:_raw)?\.md$")
PAGE_IMAGE_DIR_PATTERN = re.compile(r"page_(\d{4})$")
IMAGE_PAGE_PATTERN = re.compile(r"_p(\d{2,4})_")
TABLE_MARKER_RE = re.compile(r"<!--\s*table:\s*([A-Za-z0-9_]+)\s*-->")


def sanitize_doc_id(source_path: str) -> str:
    """根据源文件名构造稳定的文档 id。"""
    stem = Path(source_path).stem.lower()
    normalized = re.sub(r"[^\w]+", "_", stem, flags=re.UNICODE).strip("_")
    if not normalized:
        digest = hashlib.sha1(stem.encode("utf-8")).hexdigest()[:8]
        normalized = f"input_{digest}"
    return f"doc_{normalized}"


def parse_page_number(name: str) -> int | None:
    """从 OCR markdown 文件名或产物文件名里提取页码。"""
    page_match = PAGE_MD_PATTERN.search(name)
    if page_match:
        return int(page_match.group(1))

    dir_match = PAGE_IMAGE_DIR_PATTERN.search(name)
    if dir_match:
        return int(dir_match.group(1))

    image_match = IMAGE_PAGE_PATTERN.search(name)
    if image_match:
        return int(image_match.group(1))

    return None


def extract_pdf_page_labels(pdf_path: str) -> dict[int, str]:
    """在可能的情况下，从源 PDF 中读取页码标签。"""
    document = fitz.open(pdf_path)
    labels: dict[int, str] = {}
    try:
        for index, page in enumerate(document, start=1):
            label = ""
            if hasattr(page, "get_label"):
                label = page.get_label() or ""
            labels[index] = label or str(index)
    finally:
        document.close()
    return labels


def build_page_aware_markdown(page_blocks: list[dict[str, object]]) -> str:
    """把逐页 OCR 文本拼成一个带明确页边界的 markdown 文档。"""
    sections: list[str] = []
    for block in sorted(page_blocks, key=lambda item: item.get("page_no") or 0):
        content = str(block.get("content", "")).strip()
        if not content:
            continue
        page_no = block.get("page_no")
        page_label = str(block.get("page_label") or page_no or "")
        sections.append(
            "\n\n".join(
                [
                    f"<!-- page: {page_no} label: {page_label} -->",
                    f"## 第 {page_label} 页",
                    content,
                ]
            ).strip()
        )
    return "\n\n---\n\n".join(sections).strip()


def strip_trailing_region_metadata(markdown: str) -> str:
    """移除追加在单页 markdown 末尾的 OCR JSON 载荷。"""
    patterns = (
        r"\n*```json\s*\{[\s\S]*?(?:\"regions\"|\"tables\")[\s\S]*?\}\s*```\s*$",
        r"\n*```json\s*\{[\s\S]*?(?:\"regions\"|\"tables\")[\s\S]*?\}\s*$",
    )
    stripped = markdown
    for pattern in patterns:
        candidate = re.sub(pattern, "", stripped, count=1)
        if candidate != stripped:
            return candidate.strip()
    return markdown.strip()


def strip_table_markers(markdown: str) -> str:
    """在表格元数据被消费后，去掉 HTML 表格标记。"""
    return TABLE_MARKER_RE.sub("", markdown).strip()


def build_display_title(section_title: str, title: str) -> str:
    """把章节标题与局部标题组合成更适合 UI 展示的名称。"""
    normalized_section = re.sub(r"\s+", " ", str(section_title or "")).strip()
    normalized_title = re.sub(r"\s+", " ", str(title or "")).strip()
    if normalized_section and normalized_title:
        if (
            normalized_title.casefold() in normalized_section.casefold()
            or normalized_section.casefold() in normalized_title.casefold()
        ):
            return normalized_title if len(normalized_title) >= len(normalized_section) else normalized_section
        return f"{normalized_section} · {normalized_title}"
    return normalized_title or normalized_section


__all__ = [
    "TABLE_MARKER_RE",
    "build_display_title",
    "build_page_aware_markdown",
    "extract_pdf_page_labels",
    "parse_page_number",
    "sanitize_doc_id",
    "strip_table_markers",
    "strip_trailing_region_metadata",
]
