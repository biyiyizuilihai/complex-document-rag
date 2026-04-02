"""摄入流程中用于处理 DOCX 的专用辅助函数。"""

from __future__ import annotations

from typing import Iterator

try:
    from docx import Document as DocxDocumentFactory
    from docx.document import Document as DocxDocument
    from docx.oxml.table import CT_Tbl
    from docx.oxml.text.paragraph import CT_P
    from docx.table import Table
    from docx.text.paragraph import Paragraph
except ImportError:  # pragma: no cover - surfaced to the caller when used
    DocxDocumentFactory = None
    DocxDocument = object
    CT_Tbl = object
    CT_P = object
    Table = object
    Paragraph = object



def escape_markdown_cell(text: str) -> str:
    """转义会破坏 markdown 表格结构的字符。"""
    return text.replace("|", "\\|").replace("\n", "<br>").strip()



def markdown_table_from_rows(rows: list[list[str]]) -> str:
    """把表格行数据序列化成简单 markdown 表格。"""
    if not rows:
        return ""

    width = max(len(row) for row in rows)
    normalized_rows = [row + [""] * (width - len(row)) for row in rows]
    header = [escape_markdown_cell(cell) for cell in normalized_rows[0]]
    body = [
        [escape_markdown_cell(cell) for cell in row]
        for row in normalized_rows[1:]
    ]

    lines = [
        f"| {' | '.join(header)} |",
        f"| {' | '.join(['---'] * width)} |",
    ]
    for row in body:
        lines.append(f"| {' | '.join(row)} |")
    return "\n".join(lines)



def iter_block_items(document: DocxDocument) -> Iterator[Paragraph | Table]:
    """按文档中的原始顺序依次产出段落和表格。"""
    body = document.element.body
    for child in body.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, document)
        elif isinstance(child, CT_Tbl):
            yield Table(child, document)



def _flush_text_buffer(
    text_buffer: list[str],
    blocks: list[dict[str, object]],
    index: int,
) -> int:
    """把累积的段落文本提交成一个逻辑文本块。"""
    content = "\n\n".join(part for part in text_buffer if part.strip()).strip()
    if content:
        blocks.append(
            {
                "block_id": f"text_{index:04d}",
                "block_type": "text",
                "content": content,
                "origin": "native_docx",
                "page_no": None,
            }
        )
        index += 1
    text_buffer.clear()
    return index



def extract_docx_native_blocks(docx_path: str) -> list[dict[str, object]]:
    """在不经过 OCR 的情况下，从 DOCX 中提取段落和表格。"""
    if DocxDocumentFactory is None:
        raise ImportError("请先安装 python-docx")

    document = DocxDocumentFactory(docx_path)
    blocks: list[dict[str, object]] = []
    text_buffer: list[str] = []
    block_index = 1

    for item in iter_block_items(document):
        if isinstance(item, Paragraph):
            text = item.text.strip()
            if text:
                text_buffer.append(text)
            continue

        if isinstance(item, Table):
            block_index = _flush_text_buffer(text_buffer, blocks, block_index)
            rows = [[cell.text.strip() for cell in row.cells] for row in item.rows]
            table_markdown = markdown_table_from_rows(rows)
            if table_markdown:
                blocks.append(
                    {
                        "block_id": f"table_{block_index:04d}",
                        "block_type": "table",
                        "content": table_markdown,
                        "origin": "native_docx",
                        "page_no": None,
                    }
                )
                block_index += 1

    _flush_text_buffer(text_buffer, blocks, block_index)
    return blocks



def get_docx_inline_shape_count(docx_path: str) -> int:
    """统计内嵌图形数量，供流水线判断是否需要 OCR。"""
    if DocxDocumentFactory is None:
        raise ImportError("请先安装 python-docx")
    document = DocxDocumentFactory(docx_path)
    return len(document.inline_shapes)



def build_document_markdown(text_blocks: list[dict[str, object]]) -> str:
    """把提取出的块重新拼成一个 markdown 文档。"""
    return "\n\n".join(
        str(block["content"]).strip()
        for block in text_blocks
        if str(block.get("content", "")).strip()
    ).strip()


__all__ = [
    "build_document_markdown",
    "escape_markdown_cell",
    "extract_docx_native_blocks",
    "get_docx_inline_shape_count",
    "iter_block_items",
    "markdown_table_from_rows",
]
