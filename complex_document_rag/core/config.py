"""
============================================================
统一配置模块
============================================================
本文件集中管理所有外部服务的连接配置。
整个项目都从这里导入配置，避免硬编码。

使用方式：
    from complex_document_rag.core.config import (
        OPENAI_API_KEY, QDRANT_HOST, NEO4J_URL, MINIO_ENDPOINT
    )

配置优先级：
    1. 环境变量（最高优先级，适合生产环境）
    2. .env 文件（开发环境推荐）
    3. 本文件中的默认值（仅用于本地开发）

首次使用前：
    1. 复制 .env.example 为 .env
    2. 填入你的实际配置
    或者直接设置环境变量：
        export OPENAI_API_KEY="sk-xxx"
============================================================
"""

import os
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量（如果存在）
# 这样开发时不需要每次手动 export，只需维护 .env 文件即可
load_dotenv()


# ============================================================
# OpenAI 接口 / 兼容接口配置（例如 Qwen DashScope）
# ============================================================
# 用于：图片描述生成、文本向量化、知识图谱实体提取、最终问答生成
# 兼容两种环境变量命名：
# - OPENAI_API_KEY: 当前项目沿用
# - DASHSCOPE_API_KEY: DashScope 官方文档常用
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY", "")

# 兼容 Qwen 等第三方大模型服务时，设置此 Base URL
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

# 多模态 LLM 模型名称 — 用于图片描述生成和最终问答 (如使用Qwen，可改为 "qwen-vl-max")
MULTIMODAL_LLM_MODEL = os.getenv("MULTIMODAL_LLM_MODEL", "gpt-4o")

# 文本 LLM 模型名称 — 用于知识图谱实体提取等轻量任务 (如使用Qwen，可改为 "qwen-plus")
TEXT_LLM_MODEL = os.getenv("TEXT_LLM_MODEL", "gpt-4o-mini")


def _default_web_answer_llm_model() -> str:
    normalized_base = OPENAI_BASE_URL.strip().lower()
    normalized_model = TEXT_LLM_MODEL.strip().lower()
    dashscope_markers = (
        "dashscope.aliyuncs.com",
        "dashscope-intl.aliyuncs.com",
    )
    if any(marker in normalized_base for marker in dashscope_markers) or normalized_model.startswith(
        ("qwen", "qwq")
    ):
        return "qwen3.5-flash"
    return TEXT_LLM_MODEL


# 网页问答模型名称 — 用于前端交互式回答，默认偏向更快的模型以降低首 token 延迟
WEB_ANSWER_LLM_MODEL = os.getenv("WEB_ANSWER_LLM_MODEL", _default_web_answer_llm_model())
WEB_ANSWER_ENABLE_THINKING = os.getenv("WEB_ANSWER_ENABLE_THINKING", "false").lower() in {
    "1",
    "true",
    "yes",
}

# 思考 token 预算上限（仅当 WEB_ANSWER_ENABLE_THINKING=true 时生效）
# 默认 8000 token ≈ 5~10 秒，设为 0 表示不限制（恢复原始无上限行为）
WEB_ANSWER_THINKING_BUDGET_TOKENS = max(0, int(os.getenv("WEB_ANSWER_THINKING_BUDGET_TOKENS", "0")))

# Embedding 模型名称（如使用 Qwen，可改为 "text-embedding-v3"）
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")


# ============================================================
# 重排器配置（可选）
# ============================================================
# 支持使用 SiliconFlow 的 /rerank 接口对检索结果做二次重排
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "true").lower() not in {"0", "false", "no"}
RERANK_API_KEY = os.getenv("RERANK_API_KEY") or os.getenv("SILICONFLOW_API_KEY", "")
RERANK_API_BASE = os.getenv("RERANK_API_BASE", "https://api.siliconflow.cn/v1")
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
RERANK_TIMEOUT_SECONDS = float(os.getenv("RERANK_TIMEOUT_SECONDS", "5"))


# ============================================================
# Qdrant 向量数据库配置
# ============================================================
# Qdrant 负责存储文本和图片描述的 embedding 向量
# 本地启动：docker run -p 6333:6333 qdrant/qdrant
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

# 集合名称 — 文本和图片描述分开存储，便于独立管理和检索
QDRANT_TEXT_COLLECTION = "text_chunks"          # 文档文本分块的向量集合
QDRANT_IMAGE_COLLECTION = "image_descriptions"  # 图片描述的向量集合
QDRANT_TABLE_COLLECTION = "table_blocks"        # 表格块的向量集合


# ============================================================
# Neo4j 知识图谱配置
# ============================================================
# Neo4j 负责存储图片与文档之间的引用关系，支持图遍历查询
# 本地启动：docker run -p 7687:7687 -e NEO4J_AUTH=neo4j/your_password neo4j
NEO4J_URL = os.getenv("NEO4J_URL", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "your_password")


# ============================================================
# MinIO / S3 对象存储配置
# ============================================================
# MinIO 负责存储图片原文件，通过 ID 映射与向量库中的描述关联
# 本地启动：docker run -p 9000:9000 -p 9001:9001 minio/minio server /data --console-address ":9001"
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "rag-images")
MINIO_SECURE = os.getenv("MINIO_SECURE", "false").lower() == "true"


# ============================================================
# Celery 任务队列配置（P2 阶段使用）
# ============================================================
# Celery 用于大规模批处理（数万张图片的描述生成）
# Redis 作为消息代理：docker run -p 6379:6379 redis
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")


# ============================================================
# 检索参数配置
# ============================================================
# 这些参数影响检索质量，建议在 P3 阶段根据实际效果调优

# 向量检索返回的文本结果数量
TEXT_SIMILARITY_TOP_K = int(os.getenv("TEXT_SIMILARITY_TOP_K", "5"))

# 向量检索返回的图片结果数量
IMAGE_SIMILARITY_TOP_K = int(os.getenv("IMAGE_SIMILARITY_TOP_K", "3"))

# 向量检索返回的表格结果数量
TABLE_SIMILARITY_TOP_K = int(os.getenv("TABLE_SIMILARITY_TOP_K", "3"))

# 检索结果过滤阈值（主要针对 Rerank 后分数）
# 分支结果必须满足最低分数线，才会进入最终回答上下文。
TEXT_RETRIEVAL_SCORE_THRESHOLD = float(os.getenv("TEXT_RETRIEVAL_SCORE_THRESHOLD", "0.55"))
IMAGE_RETRIEVAL_SCORE_THRESHOLD = float(os.getenv("IMAGE_RETRIEVAL_SCORE_THRESHOLD", "0.70"))
TABLE_RETRIEVAL_SCORE_THRESHOLD = float(os.getenv("TABLE_RETRIEVAL_SCORE_THRESHOLD", "0.70"))

# 与同分支最佳结果的最大允许分差。
# 例如 top1=0.92、margin=0.10，则低于 0.82 的结果会被丢弃。
TEXT_RETRIEVAL_SCORE_MARGIN = float(os.getenv("TEXT_RETRIEVAL_SCORE_MARGIN", "0.12"))
IMAGE_RETRIEVAL_SCORE_MARGIN = float(os.getenv("IMAGE_RETRIEVAL_SCORE_MARGIN", "0.08"))
TABLE_RETRIEVAL_SCORE_MARGIN = float(os.getenv("TABLE_RETRIEVAL_SCORE_MARGIN", "0.08"))

# 图遍历深度（1 表示只看直接关联，2 表示看两跳内的关联）
GRAPH_TRAVERSAL_DEPTH = int(os.getenv("GRAPH_TRAVERSAL_DEPTH", "1"))

# 最终返回给 LLM 的上下文数量上限
FINAL_TOP_K = int(os.getenv("FINAL_TOP_K", "5"))


# ============================================================
# 文件路径配置
# ============================================================
# 示例数据目录（本地调试时可选）
DOCS_DIR = os.getenv("DOCS_DIR", "./docs/")
IMAGES_DIR = os.getenv("IMAGES_DIR", "./images/")
