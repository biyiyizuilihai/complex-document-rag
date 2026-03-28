import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from complex_document_rag.qdrant_management import (
    create_qdrant_client,
    delete_doc_vectors,
    drop_managed_collections,
    managed_collection_names,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="管理复杂文档 RAG 相关的 Qdrant collections")
    parser.add_argument("--list", action="store_true", help="列出当前受管理的 collection 名称")
    parser.add_argument("--drop-all", action="store_true", help="删除所有受管理的 collection")
    parser.add_argument("--delete-doc-id", help="按 doc_id 删除单个文档的向量")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    selected = sum(bool(flag) for flag in [args.list, args.drop_all, args.delete_doc_id])
    if selected != 1:
        parser.error("请且仅请传入一个操作：--list / --drop-all / --delete-doc-id")

    if args.list:
        for collection_name in managed_collection_names():
            print(collection_name)
        return 0

    client = create_qdrant_client()
    if args.drop_all:
        drop_managed_collections(client)
        print("已删除所有受管理的 collections。")
        return 0

    delete_doc_vectors(client, args.delete_doc_id)
    print(f"已删除 doc_id={args.delete_doc_id} 的向量。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
