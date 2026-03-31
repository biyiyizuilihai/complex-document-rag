"""
============================================================
Step 0: 单文档摄入编排
============================================================
功能：
    支持 PDF 单文档输入，自动抽取正文、表格、整块流程图，
    并直接进入当前 RAG 的文本索引与图片索引。

Phase 1 约束：
    - PDF 直接复用 ../scripts/batch_ocr.py
    - 当前默认不处理 DOCX / Word 文件
============================================================
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import OPENAI_API_KEY, OPENAI_BASE_URL
from llama_index.core import Document
from complex_document_rag.document_ingestion import (
    build_pdf_ocr_image_descriptions,
    collect_pdf_ocr_output,
    copy_images_to_standard_dir,
    extract_pdf_page_labels,
    materialize_missing_pdf_region_images,
    sanitize_doc_id,
    write_manifest,
)
from complex_document_rag.pipeline_utils import project_root_from_file
from complex_document_rag.step2_vector_indexing import build_text_index_from_documents
from complex_document_rag.step3_image_indexing import batch_index_images
from complex_document_rag.step3_table_indexing import batch_index_tables
from complex_document_rag.table_summary import summarize_table_blocks


PROJECT_ROOT = project_root_from_file(__file__)
BATCH_OCR_SCRIPT = os.path.abspath(os.path.join(PROJECT_ROOT, "..", "scripts", "batch_ocr.py"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="单 PDF 文档摄入并接入当前 RAG")
    parser.add_argument("--input", required=True, help="输入文件路径，仅支持 .pdf")
    parser.add_argument(
        "--output-dir",
        default=os.path.join(PROJECT_ROOT, "complex_document_rag", "ingestion_output"),
        help="标准化中间产物输出目录",
    )
    parser.add_argument("--ocr-model", default="qwen3.5-plus", help="复用外部 OCR 脚本时使用的模型名")
    parser.add_argument("--workers", type=int, default=4, help="OCR 并发数")
    parser.add_argument("--dpi", type=int, default=220, help="PDF OCR 的 DPI")
    parser.add_argument(
        "--skip-ocr",
        action="store_true",
        help="跳过 OCR 阶段，复用已有的 raw_pdf_ocr/ 产物，直接重新生成中间产物并重新索引",
    )
    return parser


def relpath_from_project(path: str) -> str:
    abs_path = os.path.abspath(path)
    try:
        return os.path.relpath(abs_path, PROJECT_ROOT)
    except ValueError:
        return abs_path


def ensure_script_exists() -> None:
    if not os.path.exists(BATCH_OCR_SCRIPT):
        raise FileNotFoundError(f"找不到外部 OCR 脚本: {BATCH_OCR_SCRIPT}")


def run_batch_ocr(input_path: str, output_dir: str, model: str, workers: int, dpi: int) -> None:
    ensure_script_exists()
    os.makedirs(output_dir, exist_ok=True)

    command = [
        sys.executable,
        BATCH_OCR_SCRIPT,
        "--input",
        input_path,
        "--output-dir",
        output_dir,
        "--model",
        model,
        "--workers",
        str(workers),
        "--dpi",
        str(dpi),
    ]
    env = os.environ.copy()
    if OPENAI_API_KEY and not env.get("DASHSCOPE_API_KEY"):
        env["DASHSCOPE_API_KEY"] = OPENAI_API_KEY
    subprocess.run(command, check=True, env=env)


def attach_text_block_metadata(
    blocks: list[dict[str, object]],
    doc_id: str,
    source_path: str,
) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    for index, block in enumerate(blocks, start=1):
        merged = dict(block)
        merged["doc_id"] = doc_id
        merged["source_doc_id"] = doc_id
        merged["source_path"] = os.path.abspath(source_path)
        merged.setdefault("block_id", f"{doc_id}_text_{index:04d}")
        normalized.append(merged)
    return normalized


def attach_image_block_metadata(
    blocks: list[dict[str, object]],
    doc_id: str,
    source_path: str,
) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    for index, block in enumerate(blocks, start=1):
        merged = dict(block)
        merged["doc_id"] = doc_id
        merged["source_doc_id"] = doc_id
        merged["source_path"] = os.path.abspath(source_path)
        merged.setdefault("block_id", f"{doc_id}_image_{index:04d}")
        normalized.append(merged)
    return normalized


def write_document_markdown(document_path: str, content: str) -> None:
    with open(document_path, "w", encoding="utf-8") as f:
        f.write(content.strip())


def build_llama_documents(text_blocks: list[dict[str, object]]) -> list[Document]:
    documents: list[Document] = []
    for block in text_blocks:
        content = str(block.get("content", "")).strip()
        if not content:
            continue
        metadata = {
            "doc_id": block.get("doc_id", ""),
            "source_doc_id": block.get("source_doc_id", ""),
            "source_path": block.get("source_path", ""),
            "page_no": block.get("page_no"),
            "page_label": block.get("page_label", ""),
            "block_type": block.get("block_type", "text"),
            "origin": block.get("origin", ""),
            "block_id": block.get("block_id", ""),
        }
        documents.append(Document(text=content, metadata=metadata))
    return documents


def write_image_descriptions(
    descriptions: dict[str, dict],
    descriptions_path: str,
) -> bool:
    if not descriptions:
        print("未生成图片描述记录。")
        return False
    with open(descriptions_path, "w", encoding="utf-8") as f:
        json.dump(descriptions, f, ensure_ascii=False, indent=2)
    return True


def write_table_blocks(
    table_blocks: list[dict[str, object]],
    table_blocks_path: str,
) -> bool:
    if not table_blocks:
        print("未生成表格记录。")
        return False
    with open(table_blocks_path, "w", encoding="utf-8") as f:
        json.dump(table_blocks, f, ensure_ascii=False, indent=2)
    return True


def index_image_descriptions(descriptions_path: str) -> None:
    if not os.path.exists(descriptions_path):
        print("未找到图片描述文件，跳过图片索引。")
        return

    batch_index_images(descriptions_path)


def index_table_blocks(table_blocks_path: str) -> None:
    if not os.path.exists(table_blocks_path):
        print("未找到表格块文件，跳过表格索引。")
        return

    batch_index_tables(table_blocks_path)


def prepare_pdf_input(args: argparse.Namespace, input_path: str, work_dir: str, doc_id: str, skip_ocr: bool = False):
    raw_ocr_dir = os.path.join(work_dir, "raw_pdf_ocr")
    if skip_ocr and os.path.isdir(raw_ocr_dir):
        print(f"--skip-ocr: 复用已有 OCR 产物 {raw_ocr_dir}")
    else:
        run_batch_ocr(input_path, raw_ocr_dir, args.ocr_model, args.workers, args.dpi)

    ocr_doc_dir = os.path.join(raw_ocr_dir, Path(input_path).stem)
    materialize_missing_pdf_region_images(ocr_doc_dir=ocr_doc_dir)
    page_labels = extract_pdf_page_labels(input_path)
    merged_text, text_blocks, image_blocks, table_blocks = collect_pdf_ocr_output(
        ocr_doc_dir=ocr_doc_dir,
        doc_id=doc_id,
        source_path=input_path,
        page_labels=page_labels,
    )
    if table_blocks:
        table_blocks = summarize_table_blocks(table_blocks)

    text_blocks = attach_text_block_metadata(text_blocks, doc_id, input_path)
    image_blocks = attach_image_block_metadata(image_blocks, doc_id, input_path)
    standard_images_dir = os.path.join(work_dir, "images")
    normalized_images = copy_images_to_standard_dir(image_blocks, standard_images_dir)
    image_descriptions = build_pdf_ocr_image_descriptions(
        ocr_doc_dir=ocr_doc_dir,
        project_root=PROJECT_ROOT,
        doc_id=doc_id,
        source_path=input_path,
        page_labels=page_labels,
        images_dir=standard_images_dir,
    )

    document_md_path = os.path.join(work_dir, "document.md")
    write_document_markdown(document_md_path, merged_text)

    manifest = write_manifest(
        manifest_path=os.path.join(work_dir, "manifest.json"),
        doc_id=doc_id,
        source_path=input_path,
        document_markdown_path=document_md_path,
        text_blocks=text_blocks,
        image_blocks=normalized_images,
        table_blocks=table_blocks,
    )
    return manifest, text_blocks, normalized_images, table_blocks, document_md_path, image_descriptions


def ingest_document(args: argparse.Namespace) -> None:
    input_path = os.path.abspath(args.input)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    suffix = Path(input_path).suffix.lower()
    if suffix != ".pdf":
        raise ValueError(f"当前仅支持 PDF 输入，收到: {suffix}")

    if args.ocr_model.lower().startswith("qwen"):
        has_dashscope_key = bool(os.environ.get("DASHSCOPE_API_KEY"))
        uses_dashscope_base = "dashscope" in (OPENAI_BASE_URL or "").lower()
        if not has_dashscope_key and not uses_dashscope_base:
            raise RuntimeError(
                "当前 OCR 复用的是 DashScope 脚本，请提供 DASHSCOPE_API_KEY，"
                "或将 OPENAI_BASE_URL 配置为 DashScope 兼容地址。"
            )

    doc_id = sanitize_doc_id(input_path)
    work_dir = os.path.join(os.path.abspath(args.output_dir), doc_id)
    skip_ocr = getattr(args, "skip_ocr", False)
    raw_ocr_dir = os.path.join(work_dir, "raw_pdf_ocr")

    if skip_ocr and os.path.isdir(raw_ocr_dir):
        os.makedirs(work_dir, exist_ok=True)
    else:
        if os.path.isdir(work_dir):
            shutil.rmtree(work_dir)
        os.makedirs(work_dir, exist_ok=True)

    manifest, text_blocks, image_blocks, table_blocks, document_md_path, image_descriptions = prepare_pdf_input(
        args, input_path, work_dir, doc_id, skip_ocr=skip_ocr
    )
    image_descriptions_path = os.path.join(work_dir, "image_descriptions.json")
    has_image_descriptions = write_image_descriptions(
        descriptions=image_descriptions,
        descriptions_path=image_descriptions_path,
    )
    table_blocks_path = os.path.join(work_dir, "table_blocks.json")
    has_table_blocks = write_table_blocks(
        table_blocks=table_blocks,
        table_blocks_path=table_blocks_path,
    )

    documents = build_llama_documents(text_blocks)
    if documents:
        build_text_index_from_documents(documents)
    else:
        print("未生成可入库的文本块，跳过文本索引。")

    if has_image_descriptions:
        index_image_descriptions(image_descriptions_path)
    else:
        print("无图片描述可入库，跳过图片索引。")

    if has_table_blocks:
        index_table_blocks(table_blocks_path)
    else:
        print("无表格块可入库，跳过表格索引。")

    print("\n摄入完成")
    print(f"  doc_id: {doc_id}")
    print(f"  输出目录: {work_dir}")
    print(f"  文档 Markdown: {document_md_path}")
    print(f"  Manifest: {os.path.join(work_dir, 'manifest.json')}")
    print(f"  文本块: {len(manifest['text_blocks'])}")
    print(f"  图片块: {len(manifest['image_blocks'])}")
    print(f"  表格块: {len(manifest['table_blocks'])}")


if __name__ == "__main__":
    parser = build_parser()
    cli_args = parser.parse_args()

    if not OPENAI_API_KEY:
        print("错误: 请先设置 OPENAI_API_KEY 或 DASHSCOPE_API_KEY")
        sys.exit(1)

    try:
        ingest_document(cli_args)
    except Exception as exc:
        print(f"摄入失败: {exc}")
        sys.exit(1)
