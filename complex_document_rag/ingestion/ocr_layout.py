"""用于处理 OCR 原始 markdown 载荷及 region/table 元数据的辅助函数。"""

from __future__ import annotations

import json
import re
from pathlib import Path

from PIL import Image

from complex_document_rag.ingestion.common import parse_page_number



def extract_ocr_metadata_payload(raw_markdown: str) -> dict[str, object]:
    """解析 OCR 步骤追加在末尾的 JSON 载荷。"""
    matches = re.findall(r"```json\s*(\{[\s\S]*?\})\s*(?:```|$)", raw_markdown)
    if not matches:
        return {}

    for candidate in reversed(matches):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and ("regions" in payload or "tables" in payload):
            return payload
    return {}



def normalize_page_scoped_id(raw_id: str, page_no: int, prefix: str) -> str:
    """把 OCR 局部 id 规范化成带页范围的 id，例如 `img_p01_001`。"""
    normalized = raw_id.strip()
    if not normalized:
        return normalized
    if re.match(rf"^{prefix}_p\d+_", normalized):
        return normalized
    if normalized.startswith(f"{prefix}_"):
        suffix = normalized[len(prefix) + 1 :]
        return f"{prefix}_p{page_no:02d}_{suffix}"
    return normalized



def extract_regions_from_raw_markdown(raw_markdown: str, page_no: int) -> list[dict[str, object]]:
    """从 OCR 原始 markdown 中读取图片或示意图区域。"""
    payload = extract_ocr_metadata_payload(raw_markdown)
    if not payload:
        return []

    regions = []
    for region in payload.get("regions", []):
        raw_region_id = str(region.get("id", "")).strip()
        if not raw_region_id:
            continue
        normalized_id = normalize_page_scoped_id(raw_region_id, page_no, "img")
        regions.append(
            {
                "id": normalized_id,
                "caption": str(region.get("caption", "")).strip(),
                "type": str(region.get("type", "other")).strip() or "other",
                "bbox_normalized": region.get("bbox_normalized", []),
            }
        )
    return regions



def extract_tables_from_raw_markdown(raw_markdown: str, page_no: int) -> list[dict[str, object]]:
    """从 OCR 原始 markdown 中读取表格元数据。"""
    payload = extract_ocr_metadata_payload(raw_markdown)
    if not payload:
        return []

    tables = []
    for index, table in enumerate(payload.get("tables", []), start=1):
        raw_table_id = str(table.get("id", "")).strip() or f"table_{index:03d}"
        normalized_id = normalize_page_scoped_id(raw_table_id, page_no, "table")
        raw_headers = table.get("headers", [])
        headers = [str(header).strip() for header in raw_headers if str(header).strip()]
        tables.append(
            {
                "id": normalized_id,
                "caption": str(table.get("caption", "")).strip(),
                "semantic_summary": str(table.get("semantic_summary", "")).strip(),
                "type": str(table.get("type", "simple")).strip() or "simple",
                "headers": headers,
                "continued_from_prev": bool(table.get("continued_from_prev", False)),
                "continues_to_next": bool(table.get("continues_to_next", False)),
                "bbox_normalized": table.get("bbox_normalized", []),
            }
        )
    return tables



def _crop_bbox_image(
    page_image_path: Path,
    output_path: Path,
    bbox_normalized: list[float],
    padding: float = 0.01,
) -> bool:
    """根据归一化 bbox 从页面渲染图中裁出区域图像。"""
    if len(bbox_normalized) != 4:
        return False

    nx1, ny1, nx2, ny2 = [float(value) for value in bbox_normalized]
    nx1, ny1 = max(0.0, nx1 - padding), max(0.0, ny1 - padding)
    nx2, ny2 = min(1.0, nx2 + padding), min(1.0, ny2 + padding)

    with Image.open(page_image_path) as image:
        width, height = image.size
        x1, y1 = int(nx1 * width), int(ny1 * height)
        x2, y2 = int(nx2 * width), int(ny2 * height)
        if x2 <= x1 or y2 <= y1:
            return False

        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.crop((x1, y1, x2, y2)).save(output_path)
    return True



def materialize_missing_pdf_region_images(
    ocr_doc_dir: str,
    target_images_dir: str | None = None,
    padding: float = 0.01,
    page_numbers: set[int] | None = None,
) -> list[str]:
    """当区域图缺失时，依据 OCR bbox 元数据补裁图片。"""
    ocr_dir = Path(ocr_doc_dir)
    page_images_dir = ocr_dir / "page_images"
    if not ocr_dir.exists() or not page_images_dir.exists():
        return []

    target_dir = Path(target_images_dir) if target_images_dir else (ocr_dir / "images")
    target_dir.mkdir(parents=True, exist_ok=True)

    created: list[str] = []
    for raw_page_path in sorted(ocr_dir.glob("page_*_raw.md")):
        page_no = parse_page_number(raw_page_path.name)
        if page_no is None:
            continue
        if page_numbers and page_no not in page_numbers:
            continue

        page_image_path = page_images_dir / f"page_{page_no:04d}.png"
        if not page_image_path.exists():
            continue

        raw_markdown = raw_page_path.read_text(encoding="utf-8")
        for region in extract_regions_from_raw_markdown(raw_markdown, page_no):
            image_id = str(region.get("id", "")).strip()
            bbox_normalized = region.get("bbox_normalized", [])
            if not image_id or not isinstance(bbox_normalized, list):
                continue

            output_path = target_dir / f"{image_id}.png"
            if output_path.exists():
                continue

            if _crop_bbox_image(
                page_image_path=page_image_path,
                output_path=output_path,
                bbox_normalized=bbox_normalized,
                padding=padding,
            ):
                created.append(str(output_path))

    return created


__all__ = [
    "extract_ocr_metadata_payload",
    "extract_regions_from_raw_markdown",
    "extract_tables_from_raw_markdown",
    "materialize_missing_pdf_region_images",
    "normalize_page_scoped_id",
]
