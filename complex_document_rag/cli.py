from __future__ import annotations

import argparse
import sys

import uvicorn

from complex_document_rag.core.cleanup import cleanup_upload_jobs
from complex_document_rag.indexing.image_index import batch_index_images
from complex_document_rag.indexing.qdrant import (
    create_qdrant_client,
    delete_doc_vectors,
    drop_managed_collections,
    managed_collection_names,
)
from complex_document_rag.indexing.table_index import batch_index_tables
from complex_document_rag.indexing.text_index import build_text_index
from complex_document_rag.ingestion.pipeline import ingest_document
from complex_document_rag.retrieval.query_console import (
    create_basic_query_engine,
    get_default_test_queries,
    load_indexes,
    test_full_query,
    test_retrieval_only,
)


def _add_ingest_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=True, help="输入文件路径，仅支持 .pdf")
    parser.add_argument(
        "--output-dir",
        default="complex_document_rag/ingestion_output",
        help="标准化中间产物输出目录",
    )
    parser.add_argument("--ocr-model", default="qwen3.5-plus", help="复用外部 OCR 脚本时使用的模型名")
    parser.add_argument("--workers", type=int, default=4, help="OCR 并发数")
    parser.add_argument("--dpi", type=int, default=220, help="PDF OCR 的 DPI")
    parser.add_argument(
        "--skip-ocr",
        action="store_true",
        help="跳过 OCR 阶段，复用已有 raw_pdf_ocr 产物并重新生成中间产物与索引",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Complex Document RAG unified CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve_parser = subparsers.add_parser("serve", help="启动 Web 服务")
    serve_parser.add_argument("--host", default="127.0.0.1", help="绑定地址")
    serve_parser.add_argument("--port", type=int, default=8000, help="监听端口")
    serve_parser.add_argument("--reload", action="store_true", help="启用自动重载")

    ingest_parser = subparsers.add_parser("ingest", help="执行单文档摄入")
    _add_ingest_arguments(ingest_parser)

    text_parser = subparsers.add_parser("index-text", help="构建文本索引")
    text_parser.add_argument("docs_dir", help="文档目录")

    image_parser = subparsers.add_parser("index-images", help="构建图片描述索引")
    image_parser.add_argument(
        "descriptions_file",
        nargs="?",
        default="image_descriptions.json",
        help="图片描述 JSON 文件",
    )

    table_parser = subparsers.add_parser("index-tables", help="构建表格索引")
    table_parser.add_argument(
        "table_blocks_file",
        nargs="?",
        default="table_blocks.json",
        help="表格块 JSON 文件",
    )

    query_parser = subparsers.add_parser("query", help="执行检索或问答")
    query_parser.add_argument("--query", help="查询内容；不传则进入交互模式")
    query_parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="只看检索结果，不生成回答",
    )

    qdrant_parser = subparsers.add_parser("qdrant", help="管理 Qdrant collections")
    qdrant_subparsers = qdrant_parser.add_subparsers(dest="qdrant_command", required=True)
    qdrant_subparsers.add_parser("list", help="列出受管理 collections")
    qdrant_subparsers.add_parser("drop-all", help="删除所有受管理 collections")
    delete_doc_parser = qdrant_subparsers.add_parser("delete-doc", help="按 doc_id 删除向量")
    delete_doc_parser.add_argument("--doc-id", required=True, help="文档 ID")

    cleanup_parser = subparsers.add_parser("cleanup", help="清理运行时目录")
    cleanup_subparsers = cleanup_parser.add_subparsers(dest="cleanup_command", required=True)
    cleanup_upload_parser = cleanup_subparsers.add_parser("upload-jobs", help="清理历史上传任务目录")
    cleanup_upload_parser.add_argument(
        "--keep",
        type=int,
        default=0,
        help="保留最新 N 个任务目录；默认 0 表示全部删除",
    )
    cleanup_upload_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只输出将删除的目录，不执行删除",
    )
    cleanup_upload_parser.add_argument("--root", default=None, help=argparse.SUPPRESS)

    return parser


def _handle_query(args: argparse.Namespace) -> int:
    text_index, image_index, table_index = load_indexes()
    if args.query:
        if args.retrieval_only:
            test_retrieval_only(text_index, image_index, table_index, args.query)
        else:
            query_engine = create_basic_query_engine(text_index, image_index, table_index)
            test_full_query(query_engine, args.query)
        return 0

    if args.retrieval_only:
        for query in get_default_test_queries():
            test_retrieval_only(text_index, image_index, table_index, query)
        return 0

    query_engine = create_basic_query_engine(text_index, image_index, table_index)
    print("\n进入交互式查询模式，输入 q / quit / exit 退出。")
    while True:
        user_query = input("\n请输入问题: ").strip()
        if user_query.lower() in {"q", "quit", "exit"}:
            return 0
        if user_query:
            test_full_query(query_engine, user_query)


def _handle_cleanup(args: argparse.Namespace) -> int:
    if args.cleanup_command == "upload-jobs":
        removed_paths = cleanup_upload_jobs(
            root=args.root,
            keep=args.keep,
            dry_run=args.dry_run,
        )
        if args.dry_run:
            print(f"dry-run: {len(removed_paths)} 个上传任务目录将被删除")
        else:
            print(f"已删除 {len(removed_paths)} 个上传任务目录")
        for path in removed_paths:
            print(path)
        return 0

    raise ValueError(f"unknown cleanup command: {args.cleanup_command}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "serve":
        uvicorn.run("complex_document_rag.web.routes:app", host=args.host, port=args.port, reload=args.reload)
        return 0

    if args.command == "ingest":
        ingest_document(args)
        return 0

    if args.command == "index-text":
        build_text_index(args.docs_dir)
        return 0

    if args.command == "index-images":
        batch_index_images(args.descriptions_file)
        return 0

    if args.command == "index-tables":
        batch_index_tables(args.table_blocks_file)
        return 0

    if args.command == "query":
        return _handle_query(args)

    if args.command == "cleanup":
        return _handle_cleanup(args)

    if args.command == "qdrant":
        if args.qdrant_command == "list":
            for collection_name in managed_collection_names():
                print(collection_name)
            return 0

        client = create_qdrant_client()
        if args.qdrant_command == "drop-all":
            drop_managed_collections(client)
            print("已删除所有受管理的 collections。")
            return 0

        if args.qdrant_command == "delete-doc":
            delete_doc_vectors(client, args.doc_id)
            print(f"已删除 doc_id={args.doc_id} 的向量。")
            return 0

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
