"""用于序列化和解析图片描述记录的辅助函数。

图片描述流水线会产出 JSON 记录，后续会被索引层和 Web 层消费。这里把
这套记录格式收敛在一个地方统一维护。
"""

from __future__ import annotations

import os
from typing import Any

from complex_document_rag.core.models import ImageDescription
from complex_document_rag.ingestion.files import IMAGE_EXTENSIONS



def make_image_id(index: int) -> str:
    """根据顺序编号生成稳定的图片标识。"""
    return f"img_{index:05d}"



def build_description_payload(
    desc: ImageDescription, image_path: str, project_root: str
) -> dict[str, Any]:
    """把单条图片描述及其源图像元数据序列化成记录。"""
    abs_image_path = os.path.abspath(image_path)
    rel_image_path = os.path.relpath(abs_image_path, project_root)

    return {
        "summary": desc.summary,
        "detailed_description": desc.detailed_description,
        "nodes": desc.nodes,
        "external_references": [
            {"target": ref.target, "context": ref.context}
            for ref in desc.external_references
        ],
        "tags": desc.tags,
        "source_image_path": rel_image_path,
        "source_image_filename": os.path.basename(abs_image_path),
    }



def resolve_source_image_path(
    img_id: str, desc_data: dict[str, Any], project_root: str
) -> str:
    """从新旧两套元数据格式中解析原始图片路径。

    新格式会持久化 `source_image_path` 和 `source_image_filename`。旧记录
    可能只有推导出来的 image id。这里同时兼容两者，这样索引器就能在不
    先重写历史 JSON 文件的情况下完成重建。
    """
    candidate_paths: list[str] = []

    stored_path = desc_data.get("source_image_path") or desc_data.get("image_path")
    if stored_path:
        if os.path.isabs(stored_path):
            candidate_paths.append(stored_path)
        else:
            candidate_paths.append(os.path.join(project_root, stored_path))

    stored_filename = desc_data.get("source_image_filename")
    if stored_filename:
        candidate_paths.append(os.path.join(project_root, "images", stored_filename))

    for extension in IMAGE_EXTENSIONS:
        candidate_paths.append(os.path.join(project_root, "images", f"{img_id}{extension}"))

    for candidate_path in candidate_paths:
        if candidate_path and os.path.exists(candidate_path):
            return os.path.abspath(candidate_path)

    raise FileNotFoundError(f"找不到图片原文件: {img_id}，已尝试 {candidate_paths}")


__all__ = [
    "build_description_payload",
    "make_image_id",
    "resolve_source_image_path",
]
