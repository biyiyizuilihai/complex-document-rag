"""摄入流程中的表格抽取与逻辑表合并逻辑。"""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

from complex_document_rag.ingestion.common import (
    TABLE_MARKER_RE,
    build_display_title,
    parse_page_number,
    strip_trailing_region_metadata,
)
from complex_document_rag.ingestion.ocr_layout import (
    extract_tables_from_raw_markdown,
    normalize_page_scoped_id,
)


MARKDOWN_TABLE_ALIGN_RE = re.compile(r"^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*$")
HTML_TABLE_OPEN_RE = re.compile(r"<table\b[^>]*>", re.IGNORECASE)
HTML_TABLE_CLOSE_RE = re.compile(r"</table>", re.IGNORECASE)
HTML_ROW_RE = re.compile(r"<tr\b[^>]*>([\s\S]*?)</tr>", re.IGNORECASE)
HTML_CELL_RE = re.compile(r"<(th|td)\b[^>]*>([\s\S]*?)</\1>", re.IGNORECASE)
HEADING_MARKDOWN_RE = re.compile(r"^#{1,6}\s+(.+?)\s*$")



def _is_markdown_table_start(lines: list[str], index: int) -> bool:
    if index + 1 >= len(lines):
        return False
    header = lines[index].strip()
    align = lines[index + 1].strip()
    return "|" in header and bool(MARKDOWN_TABLE_ALIGN_RE.match(align))



def _capture_html_table(lines: list[str], start: int) -> tuple[str, int]:
    collected: list[str] = []
    depth = 0
    index = start
    while index < len(lines):
        line = lines[index]
        collected.append(line)
        depth += len(HTML_TABLE_OPEN_RE.findall(line))
        depth -= len(HTML_TABLE_CLOSE_RE.findall(line))
        index += 1
        if depth <= 0 and HTML_TABLE_CLOSE_RE.search(line):
            break
    return "\n".join(collected).strip(), index



def _capture_markdown_table(lines: list[str], start: int) -> tuple[str, int]:
    collected = [lines[start], lines[start + 1]]
    index = start + 2
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if not stripped or "|" not in stripped:
            break
        collected.append(line)
        index += 1
    return "\n".join(collected).strip(), index



def extract_table_blocks_from_markdown(page_markdown: str) -> list[dict[str, str]]:
    """在 OCR 渲染后的页面 markdown 中寻找 markdown/html 表格。"""
    lines = page_markdown.splitlines()
    blocks: list[dict[str, str]] = []
    pending_table_id: str | None = None
    index = 0

    while index < len(lines):
        stripped = lines[index].strip()
        marker_match = TABLE_MARKER_RE.fullmatch(stripped)
        if marker_match:
            pending_table_id = marker_match.group(1).strip()
            index += 1
            continue

        if not stripped:
            index += 1
            continue

        if stripped.lower().startswith("<table"):
            raw_table, next_index = _capture_html_table(lines, index)
            blocks.append(
                {
                    "table_id": pending_table_id or "",
                    "raw_table": raw_table,
                    "raw_format": "html",
                }
            )
            pending_table_id = None
            index = next_index
            continue

        if _is_markdown_table_start(lines, index):
            raw_table, next_index = _capture_markdown_table(lines, index)
            blocks.append(
                {
                    "table_id": pending_table_id or "",
                    "raw_table": raw_table,
                    "raw_format": "markdown",
                }
            )
            pending_table_id = None
            index = next_index
            continue

        pending_table_id = None
        index += 1

    return blocks



def _clean_table_cell_text(text: str) -> str:
    normalized = re.sub(r'<img\b[^>]*alt="([^"]*)"[^>]*>', r"[image:\1]", text, flags=re.IGNORECASE)
    normalized = re.sub(r"<br\s*/?>", " ", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"</?(thead|tbody|p|span|div)\b[^>]*>", "", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"<[^>]+>", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.replace("\\|", "|").strip()



def _parse_markdown_table_rows(raw_table: str) -> list[list[str]]:
    lines = [line.strip() for line in raw_table.splitlines() if line.strip()]
    if len(lines) < 2:
        return []

    rows: list[list[str]] = []
    for line in lines:
        if MARKDOWN_TABLE_ALIGN_RE.match(line):
            continue
        normalized = line
        if normalized.startswith("|"):
            normalized = normalized[1:]
        if normalized.endswith("|"):
            normalized = normalized[:-1]
        cells = [_clean_table_cell_text(cell) for cell in re.split(r"(?<!\\)\|", normalized)]
        rows.append(cells)
    return rows



def _parse_html_table_rows(raw_table: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for row_html in HTML_ROW_RE.findall(raw_table):
        cells = [_clean_table_cell_text(content) for _, content in HTML_CELL_RE.findall(row_html)]
        if cells:
            rows.append(cells)
    return rows



def _rows_from_table(raw_table: str, raw_format: str) -> list[list[str]]:
    if raw_format == "html":
        return _parse_html_table_rows(raw_table)
    return _parse_markdown_table_rows(raw_table)



def extract_page_heading_context(page_markdown: str, *, limit: int = 3) -> list[str]:
    """提取前几个标题，用于补强表格或图片标题。"""
    headings: list[str] = []
    seen: set[str] = set()
    for line in str(page_markdown or "").splitlines():
        match = HEADING_MARKDOWN_RE.match(line.strip())
        if not match:
            continue
        heading = re.sub(r"\s+", " ", match.group(1)).strip()
        if not heading or heading.casefold() in seen:
            continue
        headings.append(heading)
        seen.add(heading.casefold())
        if len(headings) >= limit:
            break
    return headings



def build_normalized_table_text(
    raw_table: str,
    raw_format: str,
    caption: str,
    headers: list[str],
    page_label: str,
    *,
    semantic_summary: str = "",
    section_title: str = "",
) -> str:
    """把表格压平成更适合检索的纯文本表示。"""
    rows = _rows_from_table(raw_table, raw_format)
    normalized_headers = headers or (rows[0] if rows else [])
    data_rows = rows[1:] if len(rows) > 1 else []
    parts: list[str] = []

    display_title = build_display_title(section_title, caption)
    if display_title:
        parts.append(f"标题：{display_title}")
    elif caption:
        parts.append(caption)
    if semantic_summary and semantic_summary not in parts:
        parts.append(semantic_summary)
    parts.append(f"页码：{page_label}")
    if normalized_headers:
        parts.append(f"列={'|'.join(normalized_headers)}")
    for index, row in enumerate(data_rows[:50], start=1):
        parts.append(f"行{index}={'|'.join(row)}")

    return "；".join(part for part in parts if part)



def _dedupe_preserve_order(items: list[Any]) -> list[Any]:
    seen: set[Any] = set()
    result: list[Any] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result



def _format_logical_page_label(page_labels: list[str]) -> str:
    normalized = [str(label).strip() for label in page_labels if str(label).strip()]
    if not normalized:
        return ""
    if len(normalized) == 1:
        return normalized[0]
    if all(label.isdigit() for label in normalized):
        page_numbers = [int(label) for label in normalized]
        if page_numbers == list(range(page_numbers[0], page_numbers[-1] + 1)):
            return f"{page_numbers[0]}-{page_numbers[-1]}"
    return ",".join(normalized)



def _merge_table_rows_for_logical_table(
    fragments: list[dict[str, object]],
    headers: list[str],
) -> list[list[str]]:
    merged_rows: list[list[str]] = []
    for fragment in fragments:
        raw_table = str(fragment.get("raw_table", "") or "")
        raw_format = str(fragment.get("raw_format", "markdown") or "markdown")
        rows = _rows_from_table(raw_table, raw_format)
        if not rows:
            continue

        fragment_headers = [
            str(header).strip()
            for header in (fragment.get("headers", []) or [])
            if str(header).strip()
        ]
        first_row = [str(cell).strip() for cell in rows[0]]
        if fragment_headers and first_row == fragment_headers:
            data_rows = rows[1:]
        elif headers and first_row == headers:
            data_rows = rows[1:]
        else:
            data_rows = rows

        merged_rows.extend(data_rows)
    return merged_rows



def _build_logical_table_block(fragments: list[dict[str, object]]) -> dict[str, object]:
    first = fragments[0]
    caption = next((str(item.get("caption", "")).strip() for item in fragments if str(item.get("caption", "")).strip()), "")
    semantic_summary = next(
        (str(item.get("semantic_summary", "")).strip() for item in fragments if str(item.get("semantic_summary", "")).strip()),
        "",
    )
    headers = next(
        (
            [str(header).strip() for header in (item.get("headers", []) or []) if str(header).strip()]
            for item in fragments
            if item.get("headers")
        ),
        [],
    )
    page_numbers = _dedupe_preserve_order([
        int(item.get("page_no"))
        for item in fragments
        if item.get("page_no") is not None
    ])
    page_labels = _dedupe_preserve_order([
        str(item.get("page_label", "")).strip()
        for item in fragments
        if str(item.get("page_label", "")).strip()
    ])
    section_titles = _dedupe_preserve_order([
        str(item.get("section_title", "")).strip()
        for item in fragments
        if str(item.get("section_title", "")).strip()
    ])
    section_title = section_titles[0] if section_titles else ""
    logical_page_label = _format_logical_page_label(page_labels)
    logical_table_id = str(first.get("table_id", "")).strip() or str(first.get("block_id", "")).strip()
    fragment_table_ids = _dedupe_preserve_order([
        str(item.get("table_id", "")).strip()
        for item in fragments
        if str(item.get("table_id", "")).strip()
    ])
    merged_rows = _merge_table_rows_for_logical_table(fragments, headers)
    display_title = build_display_title(section_title, caption or semantic_summary)
    normalized_parts: list[str] = []
    if display_title:
        normalized_parts.append(f"标题：{display_title}")
    elif caption:
        normalized_parts.append(caption)
    if semantic_summary and semantic_summary not in normalized_parts:
        normalized_parts.append(semantic_summary)
    normalized_parts.append(f"页码：{logical_page_label or first.get('page_label', '') or '-'}")
    if headers:
        normalized_parts.append(f"列={'|'.join(headers)}")
    for index, row in enumerate(merged_rows[:120], start=1):
        normalized_parts.append(f"行{index}={'|'.join(str(cell).strip() for cell in row)}")

    summary = semantic_summary or caption or display_title or f"第 {logical_page_label or first.get('page_label', '-') or '-'} 页表格"
    raw_table_sections = []
    logical_table_sections = []
    for item in fragments:
        section_page_label = str(item.get("page_label", "")).strip() or str(item.get("page_no", "")).strip() or "-"
        raw_table = str(item.get("raw_table", "") or "").strip()
        if not raw_table:
            continue
        raw_table_sections.append(raw_table)
        rows = _rows_from_table(raw_table, str(item.get("raw_format", "markdown") or "markdown"))
        logical_table_sections.append(
            {
                "table_id": str(item.get("table_id", "") or ""),
                "page_label": section_page_label,
                "page_no": item.get("page_no"),
                "row_count": len(rows),
            }
        )

    return {
        "block_id": f"{first.get('doc_id', '')}_{logical_table_id}",
        "table_id": logical_table_id,
        "logical_table_id": logical_table_id,
        "fragment_table_ids": fragment_table_ids,
        "fragment_count": len(fragment_table_ids) or len(fragments),
        "logical_page_numbers": page_numbers,
        "logical_page_labels": page_labels,
        "logical_table_sections": logical_table_sections,
        "page_no": page_numbers[0] if page_numbers else first.get("page_no"),
        "page_label": logical_page_label or str(first.get("page_label", "")).strip(),
        "section_title": section_title,
        "display_title": display_title,
        "block_type": "table",
        "summary": summary,
        "caption": caption,
        "semantic_summary": semantic_summary,
        "raw_table": "\n\n".join(raw_table_sections),
        "raw_format": str(first.get("raw_format", "markdown") or "markdown"),
        "normalized_table_text": "；".join(part for part in normalized_parts if part),
        "headers": headers,
        "table_type": str(first.get("table_type", "simple") or "simple"),
        "continued_from_prev": False,
        "continues_to_next": False,
        "bbox_normalized": first.get("bbox_normalized", []),
        "origin": str(first.get("origin", "pdf_ocr") or "pdf_ocr"),
        "doc_id": str(first.get("doc_id", "") or ""),
        "source_path": str(first.get("source_path", "") or ""),
    }



def merge_logical_table_blocks(table_blocks: list[dict[str, object]]) -> list[dict[str, object]]:
    """把跨页表格碎片合并成一个逻辑检索块。"""
    if not table_blocks:
        return []

    logical_blocks: list[dict[str, object]] = []
    current_fragments: list[dict[str, object]] = []
    current_open = False

    for block in sorted(table_blocks, key=lambda item: (int(item.get("page_no") or 10**9), str(item.get("table_id", "") or ""))):
        continued_from_prev = bool(block.get("continued_from_prev", False))
        if continued_from_prev and current_fragments and current_open:
            current_fragments.append(block)
        else:
            if current_fragments:
                logical_blocks.append(_build_logical_table_block(current_fragments))
            current_fragments = [block]
        current_open = bool(block.get("continues_to_next", False))
        if not current_open:
            logical_blocks.append(_build_logical_table_block(current_fragments))
            current_fragments = []

    if current_fragments:
        logical_blocks.append(_build_logical_table_block(current_fragments))

    return logical_blocks



def build_pdf_ocr_table_blocks(
    ocr_doc_dir: str,
    doc_id: str,
    source_path: str,
    page_labels: dict[int, str] | None = None,
) -> list[dict[str, object]]:
    """根据 OCR 页面 markdown 与原始元数据构建标准化表格块。"""
    ocr_dir = Path(ocr_doc_dir)
    page_labels = page_labels or {}
    page_table_blocks: list[dict[str, object]] = []

    for raw_page_path in sorted(ocr_dir.glob("page_*_raw.md")):
        page_no = parse_page_number(raw_page_path.name)
        if page_no is None:
            continue

        page_md_path = ocr_dir / f"page_{page_no:04d}.md"
        if not page_md_path.exists():
            continue

        page_markdown = strip_trailing_region_metadata(page_md_path.read_text(encoding="utf-8"))
        raw_markdown = raw_page_path.read_text(encoding="utf-8")
        page_headings = extract_page_heading_context(page_markdown)
        section_title = " / ".join(page_headings[:2])
        page_label = page_labels.get(page_no, str(page_no))

        table_defs = extract_tables_from_raw_markdown(raw_markdown, page_no)
        remaining_defs = list(table_defs)
        parsed_tables = extract_table_blocks_from_markdown(page_markdown)

        for index, parsed in enumerate(parsed_tables, start=1):
            parsed_table_id = normalize_page_scoped_id(parsed.get("table_id", ""), page_no, "table")
            matched_def = None
            if parsed_table_id:
                for def_index, table_def in enumerate(remaining_defs):
                    if table_def["id"] == parsed_table_id:
                        matched_def = remaining_defs.pop(def_index)
                        break
            if matched_def is None and remaining_defs:
                matched_def = remaining_defs.pop(0)

            final_table_id = parsed_table_id or (matched_def["id"] if matched_def else f"table_p{page_no:02d}_{index:03d}")
            caption = matched_def["caption"] if matched_def else ""
            semantic_summary = matched_def["semantic_summary"] if matched_def else ""
            headers = matched_def["headers"] if matched_def else []
            table_type = matched_def["type"] if matched_def else "simple"
            raw_table = parsed["raw_table"]
            raw_format = parsed["raw_format"]
            normalized_table_text = build_normalized_table_text(
                raw_table=raw_table,
                raw_format=raw_format,
                caption=caption,
                headers=headers,
                page_label=page_label,
                semantic_summary=semantic_summary,
                section_title=section_title,
            )
            display_title = build_display_title(section_title, caption or semantic_summary)
            summary = semantic_summary or caption or display_title or f"第 {page_label} 页表格"

            page_table_blocks.append(
                {
                    "block_id": f"{doc_id}_{final_table_id}",
                    "table_id": final_table_id,
                    "block_type": "table",
                    "summary": summary,
                    "caption": caption,
                    "semantic_summary": semantic_summary,
                    "raw_table": raw_table,
                    "raw_format": raw_format,
                    "normalized_table_text": normalized_table_text,
                    "headers": headers,
                    "table_type": table_type,
                    "continued_from_prev": matched_def["continued_from_prev"] if matched_def else False,
                    "continues_to_next": matched_def["continues_to_next"] if matched_def else False,
                    "bbox_normalized": matched_def["bbox_normalized"] if matched_def else [],
                    "origin": "pdf_ocr",
                    "doc_id": doc_id,
                    "page_no": page_no,
                    "page_label": page_label,
                    "section_title": section_title,
                    "display_title": display_title,
                    "source_path": source_path,
                }
            )

    return merge_logical_table_blocks(page_table_blocks)


__all__ = [
    "build_normalized_table_text",
    "build_pdf_ocr_table_blocks",
    "extract_page_heading_context",
    "extract_table_blocks_from_markdown",
    "merge_logical_table_blocks",
]
