"""摄入流程中负责产物收集与文件系统读写的辅助函数。"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import fitz
from PIL import Image

from complex_document_rag.ingestion.common import (
    build_page_aware_markdown,
    parse_page_number,
    strip_table_markers,
    strip_trailing_region_metadata,
)
from complex_document_rag.ingestion.tables import build_pdf_ocr_table_blocks
from complex_document_rag.ingestion.files import list_image_filenames



def collect_pdf_ocr_output(
    ocr_doc_dir: str,
    doc_id: str,
    source_path: str,
    page_labels: dict[int, str] | None = None,
) -> tuple[str, list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    """从 OCR 输出目录中收集标准化后的文本、图片和表格产物。"""
    ocr_dir = Path(ocr_doc_dir)
    page_labels = page_labels or {}

    text_blocks: list[dict[str, object]] = []
    page_files = sorted(ocr_dir.glob("page_[0-9][0-9][0-9][0-9].md"))
    if not page_files:
        raise FileNotFoundError(f"未找到 OCR 生成的逐页 Markdown 文件: {ocr_doc_dir}")

    for page_file in page_files:
        page_no = parse_page_number(page_file.name)
        page_label = page_labels.get(page_no or 0, str(page_no or ""))
        page_content = strip_table_markers(strip_trailing_region_metadata(page_file.read_text(encoding="utf-8")))
        text_blocks.append(
            {
                "block_id": f"{doc_id}_text_p{page_no:04d}" if page_no is not None else f"{doc_id}_text",
                "block_type": "text",
                "content": page_content,
                "origin": "pdf_ocr",
                "page_no": page_no,
                "page_label": page_label,
                "source_path": source_path,
            }
        )

    image_blocks: list[dict[str, object]] = []
    images_dir = ocr_dir / "images"
    if images_dir.exists():
        for image_path in sorted(images_dir.glob("*")):
            if not image_path.is_file():
                continue
            page_no = parse_page_number(image_path.name)
            image_blocks.append(
                {
                    "block_id": f"{doc_id}_{image_path.stem}",
                    "block_type": "image",
                    "image_path": str(image_path.resolve()),
                    "origin": "pdf_ocr",
                    "page_no": page_no,
                    "page_label": page_labels.get(page_no or 0, str(page_no or "")),
                    "source_path": source_path,
                }
            )

    table_blocks = build_pdf_ocr_table_blocks(
        ocr_doc_dir=ocr_doc_dir,
        doc_id=doc_id,
        source_path=source_path,
        page_labels=page_labels,
    )

    return build_page_aware_markdown(text_blocks), text_blocks, image_blocks, table_blocks



def collect_folder_ocr_output(
    ocr_output_dir: str,
    doc_id: str,
    source_path: str,
) -> tuple[str, list[dict[str, object]], list[dict[str, object]]]:
    """从 visual-parser 目录结构中收集文本和图片产物。"""
    root = Path(ocr_output_dir)
    text_blocks: list[dict[str, object]] = []
    image_blocks: list[dict[str, object]] = []
    merged_parts: list[str] = []

    page_dirs = sorted(
        (path for path in root.iterdir() if path.is_dir() and parse_page_number(path.name) is not None),
        key=lambda path: parse_page_number(path.name) or 0,
    )

    for page_dir in page_dirs:
        page_no = parse_page_number(page_dir.name)
        md_path = page_dir / f"{page_dir.name}.md"
        if md_path.exists():
            content = md_path.read_text(encoding="utf-8").strip()
            if content:
                merged_parts.append(content)
                text_blocks.append(
                    {
                        "block_id": f"{doc_id}_visual_p{page_no:04d}",
                        "block_type": "text",
                        "content": content,
                        "origin": "visual_page",
                        "page_no": page_no,
                        "source_path": source_path,
                    }
                )

        images_dir = page_dir / "images"
        if images_dir.exists():
            for image_name in list_image_filenames(str(images_dir)):
                image_blocks.append(
                    {
                        "block_id": f"{doc_id}_{page_dir.name}_{Path(image_name).stem}",
                        "block_type": "image",
                        "image_path": str((images_dir / image_name).resolve()),
                        "origin": "visual_page",
                        "page_no": page_no,
                        "source_path": source_path,
                    }
                )

    return "\n\n---\n\n".join(merged_parts), text_blocks, image_blocks



def should_run_visual_parse(
    inline_image_count: int,
    visual_component_count: int,
    diagram_score: float,
    image_threshold: int = 5,
    component_threshold: int = 12,
    diagram_threshold: float = 0.18,
) -> bool:
    """判断页面是否需要额外视觉解析的启发式门槛。"""
    return (
        inline_image_count > image_threshold
        or visual_component_count >= component_threshold
        or diagram_score >= diagram_threshold
    )



def estimate_page_visual_metrics(image_path: str) -> dict[str, float]:
    """估计一张页面渲染图中图形元素的密集程度。"""
    with Image.open(image_path) as image:
        grayscale = image.convert("L")
        grayscale.thumbnail((240, 240))
        width, height = grayscale.size
        pixels = list(grayscale.getdata())

    binary = [1 if value < 220 else 0 for value in pixels]
    component_count = 0
    visited = set()
    active_pixels = sum(binary)

    for index, value in enumerate(binary):
        if value == 0 or index in visited:
            continue

        stack = [index]
        visited.add(index)
        size = 0

        while stack:
            current = stack.pop()
            size += 1
            x = current % width
            y = current // width
            neighbors = []
            if x > 0:
                neighbors.append(current - 1)
            if x + 1 < width:
                neighbors.append(current + 1)
            if y > 0:
                neighbors.append(current - width)
            if y + 1 < height:
                neighbors.append(current + width)

            for neighbor in neighbors:
                if binary[neighbor] == 1 and neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)

        if size >= 6:
            component_count += 1

    dark_ratio = active_pixels / max(len(binary), 1)
    diagram_score = min(1.0, component_count / 40.0 + dark_ratio)
    return {
        "component_count": float(component_count),
        "diagram_score": float(diagram_score),
        "dark_ratio": float(dark_ratio),
    }



def render_docx_to_pdf(docx_path: str, output_dir: str) -> str:
    """通过 LibreOffice 把 DOCX 渲染成 PDF，供后续视觉处理使用。"""
    os.makedirs(output_dir, exist_ok=True)
    command = [
        "soffice",
        "--headless",
        "--convert-to",
        "pdf",
        "--outdir",
        output_dir,
        docx_path,
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("未找到 soffice，请先安装 LibreOffice。") from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or exc.stdout or "").strip()
        raise RuntimeError(f"DOCX 转 PDF 失败: {stderr}") from exc

    pdf_path = os.path.join(output_dir, f"{Path(docx_path).stem}.pdf")
    if not os.path.exists(pdf_path):
        raise RuntimeError(f"DOCX 转 PDF 失败，未生成文件: {pdf_path}")
    return pdf_path



def render_pdf_to_images(pdf_path: str, output_dir: str, dpi: int = 200) -> list[str]:
    """把 PDF 渲染成逐页 PNG 图片。"""
    os.makedirs(output_dir, exist_ok=True)
    document = fitz.open(pdf_path)
    image_paths: list[str] = []
    matrix = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    try:
        for index, page in enumerate(document, start=1):
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            image_path = os.path.join(output_dir, f"page_{index:04d}.png")
            pixmap.save(image_path)
            image_paths.append(image_path)
    finally:
        document.close()

    return image_paths



def copy_images_to_standard_dir(image_blocks: list[dict[str, object]], target_dir: str) -> list[dict[str, object]]:
    """把图片产物复制到标准化摄入输出目录。"""
    os.makedirs(target_dir, exist_ok=True)
    normalized_blocks: list[dict[str, object]] = []
    seen_names: dict[str, int] = {}

    for block in image_blocks:
        source_image_path = str(block["image_path"])
        filename = os.path.basename(source_image_path)
        count = seen_names.get(filename, 0)
        seen_names[filename] = count + 1
        if count:
            stem = Path(filename).stem
            suffix = Path(filename).suffix
            filename = f"{stem}_{count}{suffix}"
        destination = os.path.join(target_dir, filename)
        shutil.copy2(source_image_path, destination)

        normalized = dict(block)
        normalized["image_path"] = os.path.abspath(destination)
        normalized_blocks.append(normalized)

    return normalized_blocks



def write_manifest(
    manifest_path: str,
    doc_id: str,
    source_path: str,
    document_markdown_path: str,
    text_blocks: list[dict[str, object]],
    image_blocks: list[dict[str, object]],
    table_blocks: list[dict[str, object]] | None = None,
) -> dict[str, Any]:
    """写出后续索引步骤消费的标准化 manifest。"""
    manifest = {
        "doc_id": doc_id,
        "source_path": os.path.abspath(source_path),
        "document_markdown_path": os.path.abspath(document_markdown_path),
        "text_blocks": text_blocks,
        "image_blocks": image_blocks,
        "table_blocks": table_blocks or [],
    }
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return manifest


__all__ = [
    "collect_folder_ocr_output",
    "collect_pdf_ocr_output",
    "copy_images_to_standard_dir",
    "estimate_page_visual_metrics",
    "render_docx_to_pdf",
    "render_pdf_to_images",
    "should_run_visual_parse",
    "write_manifest",
]
