"""摄入与索引流程共用的文件系统列举辅助函数。"""

from __future__ import annotations

import os


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def list_image_filenames(image_dir: str) -> list[str]:
    """返回稳定排序的图片文件名，同时忽略占位文件和子目录。"""
    return sorted(
        filename
        for filename in os.listdir(image_dir)
        if filename.lower().endswith(IMAGE_EXTENSIONS)
        and os.path.isfile(os.path.join(image_dir, filename))
    )



def list_document_files(docs_dir: str) -> list[str]:
    """返回真实文档文件，并跳过 `.gitkeep` 这类隐藏占位文件。"""
    return sorted(
        filename
        for filename in os.listdir(docs_dir)
        if not filename.startswith(".")
        and os.path.isfile(os.path.join(docs_dir, filename))
    )


__all__ = ["IMAGE_EXTENSIONS", "list_document_files", "list_image_filenames"]
