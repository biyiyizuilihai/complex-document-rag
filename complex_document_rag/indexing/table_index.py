"""
============================================================
第 3T 步：表格块入库
============================================================
功能：
    将摄入阶段生成的表格块作为文本节点写入 Qdrant 的 `table_blocks` 集合。
"""

import json
import os
import sys

from complex_document_rag.core.config import (
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_TABLE_COLLECTION,
    EMBEDDING_MODEL,
)
from complex_document_rag.providers.embeddings import create_embedding_model

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient


qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

table_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name=QDRANT_TABLE_COLLECTION,
)

embed_model = create_embedding_model(
    model_name=EMBEDDING_MODEL,
    api_key=OPENAI_API_KEY,
    api_base=OPENAI_BASE_URL,
)

MAX_TABLE_EMBED_TEXT_LENGTH = 8192
TRUNCATION_MARKER = "\n[内容已截断]\n"


def _clip_utf8_head(text: str, byte_limit: int) -> str:
    encoded = text.encode("utf-8")
    if len(encoded) <= byte_limit:
        return text
    clipped = encoded[:byte_limit]
    while clipped:
        try:
            return clipped.decode("utf-8")
        except UnicodeDecodeError:
            clipped = clipped[:-1]
    return ""


def _clip_utf8_tail(text: str, byte_limit: int) -> str:
    encoded = text.encode("utf-8")
    if len(encoded) <= byte_limit:
        return text
    clipped = encoded[-byte_limit:]
    while clipped:
        try:
            return clipped.decode("utf-8")
        except UnicodeDecodeError:
            clipped = clipped[1:]
    return ""


def _truncate_table_embedding_text(text: str, limit: int = MAX_TABLE_EMBED_TEXT_LENGTH) -> str:
    normalized = (text or "").strip()
    encoded = normalized.encode("utf-8")
    if len(encoded) <= limit:
        return normalized

    marker_budget = limit - len(TRUNCATION_MARKER.encode("utf-8"))
    if marker_budget <= 0:
        return _clip_utf8_head(normalized, limit)

    head_budget = int(marker_budget * 0.7)
    tail_budget = marker_budget - head_budget
    head = _clip_utf8_head(normalized, head_budget).rstrip()
    tail = _clip_utf8_tail(normalized, tail_budget).lstrip() if tail_budget > 0 else ""
    return f"{head}{TRUNCATION_MARKER}{tail}"


def index_single_table(table_block: dict[str, object]) -> TextNode:
    normalized_text = str(table_block.get("normalized_table_text", "")).strip()
    raw_table = str(table_block.get("raw_table", "")).strip()
    table_id = str(table_block.get("table_id", "")).strip()
    semantic_summary = str(
        table_block.get("semantic_summary", "") or table_block.get("summary", "")
    ).strip()
    display_title = str(
        table_block.get("display_title", "")
        or table_block.get("caption", "")
        or semantic_summary
        or ""
    ).strip()
    section_title = str(table_block.get("section_title", "") or "").strip()

    text_parts = []
    if semantic_summary:
        text_parts.append(semantic_summary)
    if display_title and display_title not in text_parts:
        text_parts.append(display_title)
    if section_title and section_title not in text_parts:
        text_parts.append(section_title)
    if normalized_text:
        text_parts.append(normalized_text)
    elif raw_table:
        text_parts.append(raw_table)

    text = _truncate_table_embedding_text("\n".join(text_parts))
    if not text:
        raise ValueError(f"表格块缺少可索引文本: {table_id or table_block}")

    metadata = {
        "table_id": table_id,
        "logical_table_id": table_block.get("logical_table_id", table_id),
        "logical_page_numbers": table_block.get("logical_page_numbers", []),
        "logical_page_labels": table_block.get("logical_page_labels", []),
        "logical_table_sections": table_block.get("logical_table_sections", []),
        "fragment_table_ids": table_block.get("fragment_table_ids", []),
        "fragment_count": table_block.get("fragment_count", 1),
        "doc_id": table_block.get("doc_id", ""),
        "source_doc_id": table_block.get("doc_id", ""),
        "source_path": table_block.get("source_path", ""),
        "page_no": table_block.get("page_no"),
        "page_label": table_block.get("page_label", ""),
        "origin": table_block.get("origin", "pdf_ocr"),
        "block_type": table_block.get("block_type", "table"),
        "semantic_summary": semantic_summary,
        "caption": table_block.get("caption", ""),
        "display_title": display_title,
        "section_title": section_title,
        "headers": table_block.get("headers", []),
        "table_type": table_block.get("table_type", "simple"),
        "raw_format": table_block.get("raw_format", "markdown"),
        "raw_table": raw_table,
        "normalized_table_text": normalized_text,
        "continued_from_prev": table_block.get("continued_from_prev", False),
        "continues_to_next": table_block.get("continues_to_next", False),
        "bbox_normalized": table_block.get("bbox_normalized", []),
        "type": "table_block",
    }

    return TextNode(
        text=text,
        metadata=metadata,
        excluded_embed_metadata_keys=list(metadata.keys()),
    )


def batch_index_tables(table_blocks_file: str = "table_blocks.json"):
    if not os.path.exists(table_blocks_file):
        print(f"错误: 找不到表格块文件: {table_blocks_file}")
        sys.exit(1)

    with open(table_blocks_file, "r", encoding="utf-8") as f:
        table_blocks = json.load(f)

    print(f"读取了 {len(table_blocks)} 条表格块")
    nodes = []
    for table_block in table_blocks:
        node = index_single_table(table_block)
        nodes.append(node)
        print(f"  ✓ {table_block.get('table_id', '')}: {str(table_block.get('summary', ''))[:40]}...")

    if not nodes:
        print("错误: 没有可入库的表格块。")
        sys.exit(1)

    storage_context = StorageContext.from_defaults(vector_store=table_store)
    print("\n正在生成表格 embedding 并存入 Qdrant...")
    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )

    print("\n✓ 表格索引构建完成！")
    print(f"  集合名称: {QDRANT_TABLE_COLLECTION}")
    print(f"  共 {len(nodes)} 条记录")
    print(f"  Qdrant Dashboard: http://{QDRANT_HOST}:{QDRANT_PORT}/dashboard")
    return index


def load_table_index() -> VectorStoreIndex:
    index = VectorStoreIndex.from_vector_store(
        vector_store=table_store,
        embed_model=embed_model,
    )
    print(f"✓ 已加载表格索引: {QDRANT_TABLE_COLLECTION}")
    return index


def main() -> int:
    """表格块入库命令行入口。"""
    if not OPENAI_API_KEY:
        print("错误: 请设置 OPENAI_API_KEY 环境变量")
        return 1

    batch_index_tables()
    return 0


if __name__ == "__main__":
    sys.exit(main())
