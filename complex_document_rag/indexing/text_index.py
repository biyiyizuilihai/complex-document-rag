"""
============================================================
Step 2: 向量索引构建（文档文本）
============================================================
功能：
    将文档文本分块（chunk），生成 embedding 向量，存入 Qdrant 向量数据库。
    这是 RAG 系统的核心组件之一。

原理：
    RAG 的"R"（Retrieval）依赖向量相似度搜索：
    1. 文档被拆分为小块（chunk），每块约 512-1024 个 token
    2. 每个 chunk 通过 embedding 模型转化为一个高维向量
    3. 向量存入 Qdrant，建立索引
    4. 查询时，用户的问题也被转化为向量，在库中找到最相似的 chunk

    为什么要分块？
    - LLM 的上下文窗口有限，不能把整篇文档都塞进去
    - 小块文本的 embedding 比整篇文档的 embedding 更精确
    - 检索时可以精确定位到相关段落，而不是返回整篇文档

使用方式：
    python step2_vector_indexing.py

前置条件：
    1. 设置 OPENAI_API_KEY 环境变量
    2. 启动 Qdrant: docker run -p 6333:6333 qdrant/qdrant
    3. 在 ../docs/ 目录放入测试文档（.txt, .pdf, .md 等）
============================================================
"""

import os
import sys

from complex_document_rag.core.config import (
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_TEXT_COLLECTION,
    EMBEDDING_MODEL,
)
from complex_document_rag.core.paths import project_root_from_file
from complex_document_rag.ingestion.files import list_document_files
from complex_document_rag.providers.embeddings import create_embedding_model

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core import SimpleDirectoryReader
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient


PROJECT_ROOT = project_root_from_file(__file__)


# ============================================================
# 第一步：初始化 Qdrant 客户端
# ============================================================
# Qdrant 是高性能向量数据库，支持：
#   - 向量相似度搜索（余弦相似度、欧氏距离等）
#   - 元数据过滤（按 tags、type 等字段过滤）
#   - 持久化存储（重启后数据不丢失）
#
# 本地启动方式：
#   运行命令：docker run -p 6333:6333 qdrant/qdrant
#
# Dashboard 地址：
#   http://localhost:6333/dashboard
# ============================================================

qdrant_client = QdrantClient(
    host=QDRANT_HOST,   # 默认 localhost
    port=QDRANT_PORT,   # 默认 6333
)

# 创建向量存储实例 — 对应 Qdrant 中的一个 collection
# collection_name 可以理解成数据库里的表名
text_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name=QDRANT_TEXT_COLLECTION,  # 默认 "text_chunks"
)


# ============================================================
# 第二步：初始化 Embedding 模型
# ============================================================
# `text-embedding-3-small` 是 OpenAI 的文本向量化模型：
#   - 输出维度：1536
#   - 成本很低：$0.02 / 1M tokens
#   - 质量足够好，适合大多数场景
#
# 如果需要更高质量，可以换用 text-embedding-3-large（3072维）
# ============================================================

embed_model = create_embedding_model(
    model_name=EMBEDDING_MODEL,
    api_key=OPENAI_API_KEY,
    api_base=OPENAI_BASE_URL,
)


def build_text_index(docs_dir: str) -> VectorStoreIndex:
    """
    从文档目录构建向量索引
    --------------------------------------------------------
    流程：
        1. SimpleDirectoryReader 自动识别和加载各种格式的文档
           （支持 .txt, .pdf, .md, .docx 等）
        2. LlamaIndex 自动将文档拆分为 chunks
           （默认使用 SentenceSplitter，chunk_size=1024）
        3. 每个 chunk 通过 embedding 模型转化为向量
        4. 向量和元数据一起存入 Qdrant

    参数:
        docs_dir: 文档目录路径

    返回:
        VectorStoreIndex 对象，可直接用于检索

    注意:
        - 首次运行会创建 Qdrant collection
        - 重复运行会追加数据（不会自动去重）
        - 如需重建索引，先删除 Qdrant 中的 collection
    """
    print(f"正在加载文档目录: {docs_dir}")

    # `SimpleDirectoryReader` 会递归扫描目录下的所有文件，
    # 并内置 PDF、Markdown、纯文本等格式的解析能力。
    documents = SimpleDirectoryReader(docs_dir).load_data()
    print(f"加载了 {len(documents)} 个文档")

    # `StorageContext` 用来明确告诉 LlamaIndex：
    # 当前索引要落到 Qdrant，而不是默认的内存存储。
    storage_context = StorageContext.from_defaults(
        vector_store=text_store
    )

    # `VectorStoreIndex.from_documents` 是一站式入口，
    # 会自动完成“分块 → embedding → 存储”的整条流程。
    print("正在构建向量索引（分块 → embedding → 存入 Qdrant）...")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,  # 显示进度条
    )

    print(f"✓ 向量索引构建完成！")
    print(f"  集合名称: {QDRANT_TEXT_COLLECTION}")
    print(f"  Qdrant Dashboard: http://{QDRANT_HOST}:{QDRANT_PORT}/dashboard")
    return index


def build_text_index_from_documents(documents: list) -> VectorStoreIndex:
    """
    从已整理好的 LlamaIndex Document 列表构建文本向量索引。
    --------------------------------------------------------
    该接口供 ingestion pipeline 与统一 CLI 直接复用，避免必须先落盘到 docs/ 目录。
    """
    storage_context = StorageContext.from_defaults(vector_store=text_store)
    print("正在构建向量索引（分块 → embedding → 存入 Qdrant）...")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )
    print("✓ 文本索引构建完成！")
    print(f"  集合名称: {QDRANT_TEXT_COLLECTION}")
    print(f"  Qdrant Dashboard: http://{QDRANT_HOST}:{QDRANT_PORT}/dashboard")
    return index


def load_existing_index() -> VectorStoreIndex:
    """
    加载已有的向量索引（不重新构建）
    --------------------------------------------------------
    如果之前已经构建过索引，可以直接加载使用，无需重新处理文档。
    这在 step4_basic_query.py 中会用到。
    """
    index = VectorStoreIndex.from_vector_store(
        vector_store=text_store,
        embed_model=embed_model,
    )
    print(f"✓ 已加载现有索引: {QDRANT_TEXT_COLLECTION}")
    return index


def main() -> int:
    """文本向量索引构建命令行入口。"""
    if not OPENAI_API_KEY:
        print("错误: 请设置 OPENAI_API_KEY 环境变量")
        return 1

    docs_path = os.path.join(PROJECT_ROOT, "docs")
    if not os.path.exists(docs_path):
        print(f"错误: 文档目录不存在: {docs_path}")
        print("请创建目录并放入测试文档（.txt, .pdf, .md 等）")
        return 1

    doc_files = list_document_files(docs_path)
    if not doc_files:
        print(f"警告: 文档目录为空，跳过文本向量索引构建: {docs_path}")
        print("当前可以继续执行 step3 和 step4，仅验证图片描述检索链路。")
        return 0

    # 构建索引
    index = build_text_index(docs_path)

    # 简单测试：执行一次查询验证索引可用
    print("\n--- 简单测试 ---")
    retriever = index.as_retriever(similarity_top_k=3)
    test_query = "请用一个简单的查询测试检索效果"
    results = retriever.retrieve(test_query)
    print(f"查询: '{test_query}'")
    print(f"返回 {len(results)} 条结果")
    for i, node in enumerate(results):
        score = node.score or 0
        text_preview = node.text[:100].replace("\n", " ")
        print(f"  [{i+1}] 相似度={score:.4f} | {text_preview}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
