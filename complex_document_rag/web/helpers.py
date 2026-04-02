"""网页层用于序列化与产物 URL 处理的标准辅助函数。

这些函数负责把内部检索节点转换成前端可消费的载荷，并把产物引用改写成
稳定 URL。把真实实现放回 `web/` 包后，模块边界也就和职责一致了。
"""

from __future__ import annotations

import os
import re
from html import escape
from typing import Any
from urllib.parse import quote

from complex_document_rag.core.paths import project_root_from_file
from complex_document_rag.ingestion.ocr_layout import materialize_missing_pdf_region_images


PROJECT_ROOT = project_root_from_file(__file__)
ARTIFACTS_ROOT = os.path.join(PROJECT_ROOT, "complex_document_rag", "ingestion_output")
LEGACY_ARTIFACT_PREFIXES = (
    "complex_document_rag/ingestion_output/",
    "p0_basic_rag/ingestion_output/",
)


def build_artifact_url(path: str, artifacts_root: str = ARTIFACTS_ROOT) -> str:
    """把摄入产物路径映射成前端使用的 `/artifacts/...` URL。"""
    if not path:
        return ""

    normalized_path = str(path).replace("\\", "/")
    abs_path = os.path.abspath(path)
    abs_root = os.path.abspath(artifacts_root)

    try:
        rel_path = os.path.relpath(abs_path, abs_root)
    except ValueError:
        return ""

    if rel_path.startswith(".."):
        for prefix in LEGACY_ARTIFACT_PREFIXES:
            if prefix not in normalized_path:
                continue
            rel_path = normalized_path.split(prefix, 1)[1].lstrip("/")
            if rel_path:
                return f"/artifacts/{quote(rel_path)}"
        return ""

    return f"/artifacts/{quote(rel_path.replace(os.sep, '/'))}"


def _build_image_url_if_exists(image_path: str, artifacts_root: str) -> str:
    """仅在解析后的产物文件真实存在时，才返回图片 URL。"""
    if not image_path:
        return ""
    # image_path 可能是绝对路径、旧项目路径，或容器里的 ingestion_output 路径。
    if os.path.isfile(image_path):
        return build_artifact_url(image_path, artifacts_root)

    normalized_path = str(image_path).replace("\\", "/")
    candidate_paths = [os.path.join(artifacts_root, image_path.lstrip("/"))]
    for prefix in LEGACY_ARTIFACT_PREFIXES:
        if prefix not in normalized_path:
            continue
        rel_path = normalized_path.split(prefix, 1)[1].lstrip("/")
        if rel_path:
            candidate_paths.append(os.path.join(artifacts_root, rel_path))

    seen_candidates: set[str] = set()
    for candidate in candidate_paths:
        abs_candidate = os.path.abspath(candidate)
        if abs_candidate in seen_candidates:
            continue
        seen_candidates.add(abs_candidate)
        if os.path.isfile(abs_candidate):
            return build_artifact_url(abs_candidate, artifacts_root)
    return ""


def _node_kind(metadata: dict[str, Any]) -> str:
    """把不同来源的节点元数据归一成 text/image/table 三种类型。"""
    node_type = metadata.get("type", "")
    block_type = metadata.get("block_type", "")

    if node_type == "image_description" or block_type == "image":
        return "image"
    if node_type == "table_block" or block_type == "table":
        return "table"
    return "text"


BBOX_SRC_RE = re.compile(r"^bbox://", re.IGNORECASE)
HTML_IMG_TAG_RE = re.compile(r"<img\b[^>]*>", re.IGNORECASE)
MARKDOWN_TABLE_ALIGN_RE = re.compile(r"^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*$")


def _find_ocr_doc_dir(doc_root: str) -> str:
    """如果已落盘，则定位单个文档对应的 OCR 目录。"""
    raw_ocr_root = os.path.join(doc_root, "raw_pdf_ocr")
    if not os.path.isdir(raw_ocr_root):
        return ""

    for name in sorted(os.listdir(raw_ocr_root)):
        candidate = os.path.join(raw_ocr_root, name)
        if os.path.isdir(candidate):
            return candidate
    return ""


def _resolve_table_image_url(
    src: str,
    alt: str,
    doc_id: str,
    page_no: int | None,
    artifacts_root: str,
) -> str:
    """把 markdown/html 表格中的图片引用解析成可访问的产物 URL。"""
    normalized_src = src.strip()
    if not normalized_src or not doc_id:
        return ""

    if normalized_src.startswith("images/"):
        return f"/artifacts/{quote(doc_id)}/{normalized_src}"

    if not BBOX_SRC_RE.match(normalized_src) or not alt:
        return normalized_src

    doc_root = os.path.join(artifacts_root, doc_id)
    target_image_path = os.path.join(doc_root, "images", f"{alt}.png")
    if not os.path.exists(target_image_path):
        ocr_doc_dir = _find_ocr_doc_dir(doc_root)
        if ocr_doc_dir:
            page_numbers = {int(page_no)} if page_no is not None else None
            materialize_missing_pdf_region_images(
                ocr_doc_dir=ocr_doc_dir,
                target_images_dir=os.path.join(doc_root, "images"),
                page_numbers=page_numbers,
            )

    if os.path.exists(target_image_path):
        return build_artifact_url(target_image_path, artifacts_root=artifacts_root)
    return ""


def normalize_table_asset_paths(
    raw_table: str,
    raw_format: str,
    doc_id: str,
    page_no: int | None = None,
    artifacts_root: str = ARTIFACTS_ROOT,
) -> str:
    """重写表格内的图片引用，确保浏览器可以正确加载。"""
    if not raw_table or not doc_id:
        return raw_table

    def replace_markdown(match: re.Match[str]) -> str:
        alt = match.group(1)
        src = match.group(2)
        normalized_src = _resolve_table_image_url(
            src=src,
            alt=alt,
            doc_id=doc_id,
            page_no=page_no,
            artifacts_root=artifacts_root,
        )
        if not normalized_src:
            return alt or ""
        return f"![{alt}]({normalized_src})"

    def replace_html(match: re.Match[str]) -> str:
        tag = match.group(0)
        src_match = re.search(r'''src=(["'])([^"']+)\1''', tag, flags=re.IGNORECASE)
        alt_match = re.search(r'''alt=(["'])([^"']*)\1''', tag, flags=re.IGNORECASE)
        if not src_match:
            return tag

        alt = alt_match.group(2) if alt_match else ""
        normalized_src = _resolve_table_image_url(
            src=src_match.group(2),
            alt=alt,
            doc_id=doc_id,
            page_no=page_no,
            artifacts_root=artifacts_root,
        )
        if not normalized_src:
            return alt or ""
        return re.sub(
            r'''src=(["'])([^"']+)\1''',
            f'src="{normalized_src}"',
            tag,
            count=1,
            flags=re.IGNORECASE,
        )

    normalized = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", replace_markdown, raw_table)
    if (raw_format or "").lower() == "html":
        normalized = HTML_IMG_TAG_RE.sub(replace_html, normalized)
    return normalized


def _render_inline_markdown(text: str) -> str:
    """渲染回答片段里用到的最小 inline markdown 子集。"""
    escaped = escape(text or "")
    escaped = escaped.replace("&lt;br&gt;", "<br>").replace("&lt;br/&gt;", "<br>").replace("&lt;br /&gt;", "<br>")
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    return escaped


def _is_markdown_table_start(lines: list[str], index: int) -> bool:
    """通过表头分隔行判断当前位置是否为 markdown 表格开头。"""
    if index + 1 >= len(lines):
        return False
    return "|" in lines[index] and bool(MARKDOWN_TABLE_ALIGN_RE.match(lines[index + 1].strip()))


def _split_markdown_row(line: str) -> list[str]:
    """把一行 markdown 表格拆成去空白后的单元格值。"""
    stripped = line.strip().strip("|")
    return [cell.strip() for cell in stripped.split("|")]


def _render_markdown_table_block(lines: list[str]) -> str:
    """把一个 markdown 表格块渲染成紧凑 HTML。"""
    rows = [_split_markdown_row(line) for line in lines if line.strip()]
    if len(rows) < 2:
        return f"<p>{_render_inline_markdown(' '.join(lines))}</p>"

    header = rows[0]
    body = rows[2:]
    head_html = "".join(f"<th>{_render_inline_markdown(cell)}</th>" for cell in header)
    body_html = "".join(
        "<tr>" + "".join(f"<td>{_render_inline_markdown(cell)}</td>" for cell in row) + "</tr>"
        for row in body
    )
    return f"<table><thead><tr>{head_html}</tr></thead><tbody>{body_html}</tbody></table>"


def render_answer_markdown_html(markdown: str) -> str:
    """把当前支持的回答 markdown 子集渲染成轻量 HTML。"""
    text = (markdown or "").strip()
    if not text:
        return "<p>本次未生成回答，请展开下方证据面板查看召回结果。</p>"

    lines = text.splitlines()
    parts: list[str] = []
    index = 0

    while index < len(lines):
        line = lines[index].rstrip()
        stripped = line.strip()
        if not stripped:
            index += 1
            continue

        if _is_markdown_table_start(lines, index):
            table_lines = [lines[index], lines[index + 1]]
            index += 2
            while index < len(lines) and "|" in lines[index]:
                table_lines.append(lines[index])
                index += 1
            parts.append(_render_markdown_table_block(table_lines))
            continue

        if re.match(r"^#{1,3}\s+", stripped):
            heading = re.sub(r"^#{1,3}\s+", "", stripped)
            parts.append(f"<h3>{_render_inline_markdown(heading)}</h3>")
            index += 1
            continue

        if re.match(r"^[-*]\s+", stripped):
            items = []
            while index < len(lines) and re.match(r"^[-*]\s+", lines[index].strip()):
                items.append(re.sub(r"^[-*]\s+", "", lines[index].strip()))
                index += 1
            parts.append("<ul>" + "".join(f"<li>{_render_inline_markdown(item)}</li>" for item in items) + "</ul>")
            continue

        if re.match(r"^\d+\.\s+", stripped):
            items = []
            while index < len(lines) and re.match(r"^\d+\.\s+", lines[index].strip()):
                items.append(re.sub(r"^\d+\.\s+", "", lines[index].strip()))
                index += 1
            parts.append("<ol>" + "".join(f"<li>{_render_inline_markdown(item)}</li>" for item in items) + "</ol>")
            continue

        if stripped.startswith("```"):
            fence_lang = stripped[3:].strip().lower()
            index += 1
            code_lines: list[str] = []
            while index < len(lines) and not lines[index].strip().startswith("```"):
                code_lines.append(lines[index].rstrip())
                index += 1
            if index < len(lines) and lines[index].strip().startswith("```"):
                index += 1
            code_content = escape("\n".join(code_lines))
            if fence_lang == "mermaid":
                parts.append(f'<pre class="mermaid">{code_content}</pre>')
            else:
                parts.append(f"<pre><code>{code_content}</code></pre>")
            continue

        paragraph_lines = [stripped]
        index += 1
        while index < len(lines):
            candidate = lines[index].strip()
            if not candidate:
                index += 1
                break
            if _is_markdown_table_start(lines, index):
                break
            if re.match(r"^(#{1,3}\s+|[-*]\s+|\d+\.\s+)", candidate):
                break
            paragraph_lines.append(candidate)
            index += 1
        parts.append(f"<p>{_render_inline_markdown(' '.join(paragraph_lines))}</p>")

    return "".join(parts)


def serialize_scored_node(node: Any, artifacts_root: str = ARTIFACTS_ROOT) -> dict[str, Any]:
    """把单个检索节点序列化成前端期望的 JSON 结构。"""
    metadata = getattr(node, "metadata", {}) or {}
    text = getattr(node, "text", "") or ""
    score = getattr(node, "score", 0.0) or 0.0
    kind = _node_kind(metadata)

    base_payload = {
        "kind": kind,
        "score": round(float(score), 4),
        "doc_id": metadata.get("doc_id", ""),
        "page_no": metadata.get("page_no"),
        "page_label": metadata.get("page_label", ""),
        "source_path": metadata.get("source_path", ""),
        "summary": metadata.get("summary", ""),
    }

    if kind == "image":
        image_path = metadata.get("image_path", "") or metadata.get("source_image_path", "")
        return {
            **base_payload,
            "image_id": metadata.get("image_id", ""),
            "image_path": image_path,
            "image_url": _build_image_url_if_exists(image_path, artifacts_root),
            "summary": metadata.get("summary", "") or text[:140],
        }

    if kind == "table":
        raw_format = metadata.get("raw_format", "markdown")
        raw_table = normalize_table_asset_paths(
            metadata.get("raw_table", ""),
            raw_format,
            metadata.get("doc_id", ""),
            page_no=metadata.get("page_no"),
            artifacts_root=artifacts_root,
        )
        return {
            **base_payload,
            "table_id": metadata.get("table_id", ""),
            "caption": metadata.get("caption", ""),
            "semantic_summary": metadata.get("semantic_summary", "") or metadata.get("summary", ""),
            "headers": metadata.get("headers", []),
            "raw_table": raw_table,
            "raw_format": raw_format,
            "normalized_table_text": metadata.get("normalized_table_text", ""),
            "summary": metadata.get("semantic_summary", "") or metadata.get("summary", "") or metadata.get("caption", ""),
        }

    return {
        **base_payload,
        "text": text,
        "snippet": text[:240].replace("\n", " "),
        "block_id": metadata.get("block_id", ""),
    }


def serialize_retrieval_bundle(
    retrieval: dict[str, list[Any]],
    artifacts_root: str = ARTIFACTS_ROOT,
) -> dict[str, list[dict[str, Any]]]:
    """把完整检索结果包序列化成查询接口响应结构。"""
    return {
        "text_results": [
            serialize_scored_node(node, artifacts_root=artifacts_root)
            for node in retrieval.get("text_results", [])
        ],
        "image_results": [
            serialize_scored_node(node, artifacts_root=artifacts_root)
            for node in retrieval.get("image_results", [])
        ],
        "table_results": [
            serialize_scored_node(node, artifacts_root=artifacts_root)
            for node in retrieval.get("table_results", [])
        ],
    }


def serialize_answer_sources(
    source_nodes: list[Any],
    artifacts_root: str = ARTIFACTS_ROOT,
) -> list[dict[str, Any]]:
    """按与检索结果一致的结构序列化回答证据节点。"""
    return [
        serialize_scored_node(node, artifacts_root=artifacts_root)
        for node in source_nodes
    ]


__all__ = [
    "ARTIFACTS_ROOT",
    "LEGACY_ARTIFACT_PREFIXES",
    "PROJECT_ROOT",
    "build_artifact_url",
    "normalize_table_asset_paths",
    "render_answer_markdown_html",
    "serialize_answer_sources",
    "serialize_retrieval_bundle",
    "serialize_scored_node",
]
