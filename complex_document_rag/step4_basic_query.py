"""
============================================================
Step 4: 基础查询测试
============================================================
功能：
    使用前三步构建的向量索引，执行查询测试，验证检索效果。
    这是当前仓库的命令行验证步骤。

原理：
    查询流程：
        1. 用户输入自然语言问题
        2. 问题被转化为 embedding 向量
        3. 在 Qdrant 中搜索最相似的文本块和图片描述
        4. 将检索到的上下文传给 LLM
        5. LLM 基于上下文生成回答

    这是最基础的 RAG 流程，P1 阶段会加入知识图谱增强。

使用方式：
    python step4_basic_query.py

前置条件：
    1. 完成 step2（文本索引已构建）
    2. 完成 step3（图片描述索引已构建）
    3. Qdrant 已启动且有数据
============================================================
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_TEXT_COLLECTION,
    QDRANT_IMAGE_COLLECTION,
    QDRANT_TABLE_COLLECTION,
    EMBEDDING_MODEL,
    TEXT_LLM_MODEL,
    TEXT_SIMILARITY_TOP_K,
    IMAGE_SIMILARITY_TOP_K,
    TABLE_SIMILARITY_TOP_K,
    RERANK_API_BASE,
    RERANK_API_KEY,
    RERANK_ENABLED,
    RERANK_MODEL,
    RERANK_TIMEOUT_SECONDS,
)
from model_provider_utils import create_embedding_model, create_text_llm
from complex_document_rag.reranker import (
    SiliconFlowRerankPostprocessor,
    SiliconFlowReranker,
    rerank_retrieval_bundle,
)
from complex_document_rag.query_utils import build_query_fusion_retriever

from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient


# ============================================================
# 初始化组件
# ============================================================

qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
embed_model = create_embedding_model(
    model_name=EMBEDDING_MODEL,
    api_key=OPENAI_API_KEY,
    api_base=OPENAI_BASE_URL,
)


def collection_exists(collection_name: str) -> bool:
    """检查指定的 Qdrant collection 是否已经存在。"""
    collections = qdrant_client.get_collections().collections
    return any(getattr(collection, "name", None) == collection_name for collection in collections)


def load_indexes():
    """
    加载文本索引和图片描述索引
    --------------------------------------------------------
    从 Qdrant 中加载之前构建的两个索引，
    无需重新处理文档和图片。
    """
    text_index = None
    image_index = None
    table_index = None

    if collection_exists(QDRANT_TEXT_COLLECTION):
        text_store = QdrantVectorStore(
            client=qdrant_client,
            collection_name=QDRANT_TEXT_COLLECTION,
        )
        text_index = VectorStoreIndex.from_vector_store(
            vector_store=text_store,
            embed_model=embed_model,
        )
        print(f"✓ 已加载文本索引: {QDRANT_TEXT_COLLECTION}")
    else:
        print(f"⚠ 未找到文本索引，跳过: {QDRANT_TEXT_COLLECTION}")

    if collection_exists(QDRANT_IMAGE_COLLECTION):
        image_store = QdrantVectorStore(
            client=qdrant_client,
            collection_name=QDRANT_IMAGE_COLLECTION,
        )
        image_index = VectorStoreIndex.from_vector_store(
            vector_store=image_store,
            embed_model=embed_model,
        )
        print(f"✓ 已加载图片描述索引: {QDRANT_IMAGE_COLLECTION}")
    else:
        print(f"⚠ 未找到图片描述索引: {QDRANT_IMAGE_COLLECTION}")

    if collection_exists(QDRANT_TABLE_COLLECTION):
        table_store = QdrantVectorStore(
            client=qdrant_client,
            collection_name=QDRANT_TABLE_COLLECTION,
        )
        table_index = VectorStoreIndex.from_vector_store(
            vector_store=table_store,
            embed_model=embed_model,
        )
        print(f"✓ 已加载表格索引: {QDRANT_TABLE_COLLECTION}")
    else:
        print(f"⚠ 未找到表格索引: {QDRANT_TABLE_COLLECTION}")

    if text_index is None and image_index is None and table_index is None:
        raise RuntimeError("Qdrant 中没有可用索引，请先运行 step0 / step2 / step3。")

    return text_index, image_index, table_index


def build_reranker() -> SiliconFlowReranker | None:
    if not RERANK_ENABLED or not RERANK_API_KEY:
        return None
    return SiliconFlowReranker(
        api_key=RERANK_API_KEY,
        api_base=RERANK_API_BASE,
        model_name=RERANK_MODEL,
        top_n=TEXT_SIMILARITY_TOP_K + IMAGE_SIMILARITY_TOP_K + TABLE_SIMILARITY_TOP_K,
        timeout=RERANK_TIMEOUT_SECONDS,
    )


def create_basic_query_engine(text_index, image_index, table_index):
    """
    创建基础查询引擎
    --------------------------------------------------------
    同时检索文本和图片描述，合并结果后交给 LLM 生成回答。

    这是当前仓库的基础实现：
        - 文本和图片描述分别检索
        - 结果可选经过 SiliconFlow Rerank
        - 由 LLM 综合回答

    P1 阶段会加入图遍历，P3 阶段会加入更完整的优化策略。

    参数:
        text_index:  文本向量索引
        image_index: 图片描述向量索引

    返回:
        RetrieverQueryEngine 对象
    """
    retrievers = []
    if text_index is not None:
        retrievers.append(
            text_index.as_retriever(
                similarity_top_k=TEXT_SIMILARITY_TOP_K,
            )
        )
    if image_index is not None:
        retrievers.append(
            image_index.as_retriever(
                similarity_top_k=IMAGE_SIMILARITY_TOP_K,
            )
        )
    if table_index is not None:
        retrievers.append(
            table_index.as_retriever(
                similarity_top_k=TABLE_SIMILARITY_TOP_K,
            )
        )

    if not retrievers:
        raise RuntimeError("没有可用的检索器，请先构建至少一个索引。")

    if len(retrievers) == 1:
        retriever = retrievers[0]
    else:
        retriever = build_query_fusion_retriever(
            retrievers=retrievers,
            similarity_top_k=TEXT_SIMILARITY_TOP_K + IMAGE_SIMILARITY_TOP_K + TABLE_SIMILARITY_TOP_K,
        )

    # 创建响应合成器
    # compact 模式会将所有上下文压缩到一个 prompt 中
    # 适合上下文不太长的场景
    response_synthesizer = get_response_synthesizer(
        response_mode="compact",
        llm=create_text_llm(
            model_name=TEXT_LLM_MODEL,
            api_key=OPENAI_API_KEY,
            api_base=OPENAI_BASE_URL,
        ),
    )

    node_postprocessors = []
    reranker = build_reranker()
    if reranker is not None:
        node_postprocessors.append(
            SiliconFlowRerankPostprocessor(
                reranker=reranker,
                top_n=TEXT_SIMILARITY_TOP_K + IMAGE_SIMILARITY_TOP_K + TABLE_SIMILARITY_TOP_K,
            )
        )

    # 组装查询引擎
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=node_postprocessors,
    )

    return query_engine


def test_retrieval_only(text_index, image_index, table_index, query: str):
    """
    仅测试检索效果（不生成回答）
    --------------------------------------------------------
    在调优阶段，有时只需要看检索结果是否相关，
    不需要等 LLM 生成回答（更快、更省钱）。
    """
    print(f"\n{'='*60}")
    print(f"查询: {query}")
    print(f"{'='*60}")

    text_results = []
    if text_index is not None:
        text_retriever = text_index.as_retriever(
            similarity_top_k=TEXT_SIMILARITY_TOP_K
        )
        text_results = text_retriever.retrieve(query)

    image_results = []
    if image_index is not None:
        image_retriever = image_index.as_retriever(
            similarity_top_k=IMAGE_SIMILARITY_TOP_K
        )
        image_results = image_retriever.retrieve(query)

    table_results = []
    if table_index is not None:
        table_retriever = table_index.as_retriever(
            similarity_top_k=TABLE_SIMILARITY_TOP_K
        )
        table_results = table_retriever.retrieve(query)

    reranker = build_reranker()
    if reranker is not None:
        reranked_bundle = rerank_retrieval_bundle(
            query=query,
            retrieval={
                "text_results": text_results,
                "image_results": image_results,
                "table_results": table_results,
            },
            reranker=reranker,
            top_n_map={
                "text_results": TEXT_SIMILARITY_TOP_K,
                "image_results": IMAGE_SIMILARITY_TOP_K,
                "table_results": TABLE_SIMILARITY_TOP_K,
            },
        )
        text_results = reranked_bundle["text_results"]
        image_results = reranked_bundle["image_results"]
        table_results = reranked_bundle["table_results"]

    if text_index is not None:
        print(f"\n--- 文本检索结果（top {TEXT_SIMILARITY_TOP_K}）---")
        for i, node in enumerate(text_results):
            score = node.score or 0
            text_preview = node.text[:120].replace("\n", " ")
            doc_id = node.metadata.get("doc_id", "")
            page_no = node.metadata.get("page_no")
            page_label = node.metadata.get("page_label", "")
            source_path = node.metadata.get("source_path", "")
            print(f"  [{i+1}] 相似度={score:.4f}")
            print(f"      {text_preview}...")
            if doc_id or source_path or page_no is not None:
                print(
                    f"      文档={doc_id or '-'} | 页码={page_no if page_no is not None else '-'} | 页标签={page_label or '-'}"
                )
                if source_path:
                    print(f"      来源: {source_path}")
    else:
        print("\n--- 文本检索结果 ---")
        print("  文本索引未构建，已跳过。")

    if image_index is not None:
        print(f"\n--- 图片描述检索结果（top {IMAGE_SIMILARITY_TOP_K}）---")
        for i, node in enumerate(image_results):
            score = node.score or 0
            img_id = node.metadata.get("image_id", "unknown")
            image_path = node.metadata.get("image_path", "")
            doc_id = node.metadata.get("doc_id", "")
            page_no = node.metadata.get("page_no")
            page_label = node.metadata.get("page_label", "")
            summary = node.metadata.get("summary", "")[:80]
            print(f"  [{i+1}] 相似度={score:.4f} | 图片ID={img_id}")
            print(f"      摘要: {summary}")
            if doc_id or page_no is not None:
                print(
                    f"      文档={doc_id or '-'} | 页码={page_no if page_no is not None else '-'} | 页标签={page_label or '-'}"
                )
            if image_path:
                print(f"      原图: {image_path}")
    else:
        print("\n--- 图片描述检索结果 ---")
        print("  图片描述索引未构建，已跳过。")

    if table_index is not None:
        print(f"\n--- 表格检索结果（top {TABLE_SIMILARITY_TOP_K}）---")
        for i, node in enumerate(table_results):
            score = node.score or 0
            table_id = node.metadata.get("table_id", "unknown")
            doc_id = node.metadata.get("doc_id", "")
            page_no = node.metadata.get("page_no")
            page_label = node.metadata.get("page_label", "")
            caption = node.metadata.get("caption", "")[:80]
            print(f"  [{i+1}] 相似度={score:.4f} | 表格ID={table_id}")
            print(f"      标题: {caption or node.metadata.get('summary', '')[:80]}")
            if doc_id or page_no is not None:
                print(
                    f"      文档={doc_id or '-'} | 页码={page_no if page_no is not None else '-'} | 页标签={page_label or '-'}"
                )
    else:
        print("\n--- 表格检索结果 ---")
        print("  表格索引未构建，已跳过。")

    return text_results, image_results, table_results


def test_full_query(query_engine, query: str):
    """
    测试完整查询流程（检索 + LLM 生成回答）
    --------------------------------------------------------
    """
    print(f"\n{'='*60}")
    print(f"查询: {query}")
    print(f"{'='*60}")

    response = query_engine.query(query)

    print(f"\n--- 回答 ---")
    print(response.response)

    # 显示引用的源节点
    if response.source_nodes:
        print(f"\n--- 引用来源（{len(response.source_nodes)} 条）---")
        for i, node in enumerate(response.source_nodes):
            node_type = node.metadata.get("type", "text")
            score = node.score or 0
            if node_type == "image_description":
                img_id = node.metadata.get("image_id", "unknown")
                image_path = node.metadata.get("image_path", "")
                print(f"  [{i+1}] [图片] {img_id} | 相似度={score:.4f}")
                if image_path:
                    print(f"      原图: {image_path}")
            elif node_type == "table_block":
                table_id = node.metadata.get("table_id", "unknown")
                caption = node.metadata.get("caption", "")
                doc_id = node.metadata.get("doc_id", "")
                page_no = node.metadata.get("page_no")
                page_label = node.metadata.get("page_label", "")
                print(f"  [{i+1}] [表格] {table_id} | 相似度={score:.4f}")
                if caption:
                    print(f"      标题: {caption}")
                if doc_id or page_no is not None:
                    print(
                        f"      文档={doc_id or '-'} | 页码={page_no if page_no is not None else '-'} | 页标签={page_label or '-'}"
                    )
            else:
                text_preview = node.text[:80].replace("\n", " ")
                doc_id = node.metadata.get("doc_id", "")
                page_no = node.metadata.get("page_no")
                page_label = node.metadata.get("page_label", "")
                source_path = node.metadata.get("source_path", "")
                print(f"  [{i+1}] [文本] 相似度={score:.4f} | {text_preview}...")
                if doc_id or source_path or page_no is not None:
                    print(
                        f"      文档={doc_id or '-'} | 页码={page_no if page_no is not None else '-'} | 页标签={page_label or '-'}"
                    )
                    if source_path:
                        print(f"      来源: {source_path}")

    return response


def get_default_test_queries():
    """返回与当前职业卫生 PDF 语料更匹配的默认测试问题。"""
    return [
        "使用有毒物品作业场所应设置哪些警示标识？",
        "紧急出口标识有哪些方向类型？",
        "苯的职业病危害告知卡包含哪些信息？",
    ]


# ============================================================
# 主函数
# ============================================================
if __name__ == "__main__":
    if not OPENAI_API_KEY:
        print("错误: 请设置 OPENAI_API_KEY 环境变量")
        sys.exit(1)

    # 加载索引
    text_index, image_index, table_index = load_indexes()

    # 创建查询引擎
    query_engine = create_basic_query_engine(text_index, image_index, table_index)

    # ========================================================
    # 测试查询
    # ========================================================
    # 建议准备 5-10 个典型问题，覆盖不同场景：
    #   - 直接匹配文档内容的问题
    #   - 需要理解流程图的问题
    #   - 跨文档/图片的综合问题
    # ========================================================

    test_queries = get_default_test_queries()

    print("\n" + "=" * 60)
    print(" 复杂文档 RAG — 查询测试")
    print("=" * 60)

    # 先测试纯检索效果
    print("\n\n【模式1: 仅检索，不生成回答】")
    for query in test_queries[:1]:  # 先用第一个问题测试检索
        test_retrieval_only(text_index, image_index, table_index, query)

    # 再测试完整查询
    print("\n\n【模式2: 检索 + 生成回答】")
    for query in test_queries:
        test_full_query(query_engine, query)

    # 交互式查询
    print("\n\n" + "=" * 60)
    print(" 交互式查询（输入 q 退出）")
    print("=" * 60)
    while True:
        user_query = input("\n请输入问题: ").strip()
        if user_query.lower() in ("q", "quit", "exit"):
            print("退出查询。")
            break
        if user_query:
            test_full_query(query_engine, user_query)
