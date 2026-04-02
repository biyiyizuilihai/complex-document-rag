"""供检索与查询入口复用的 Fusion Retriever 辅助函数。"""

from llama_index.core.llms.mock import MockLLM
from llama_index.core.retrievers import QueryFusionRetriever


def build_query_fusion_retriever(retrievers, similarity_top_k: int):
    """构造一个不启用异步扇出的确定性 fusion retriever。"""
    return QueryFusionRetriever(
        retrievers=retrievers,
        llm=MockLLM(),
        mode="simple",
        num_queries=1,
        similarity_top_k=similarity_top_k,
        use_async=False,
    )
