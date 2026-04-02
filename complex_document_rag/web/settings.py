"""网页层专用的运行时配置与进程级共享状态。

把这些值集中到这里，可以让路由处理函数专注于传输层行为，而不是被
配置噪音和共享状态管理干扰。
"""

from __future__ import annotations

import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from complex_document_rag.web.helpers import PROJECT_ROOT


WEB_STATIC_DIR = os.path.join(PROJECT_ROOT, "complex_document_rag", "web_static")
INGEST_STATIC_PATH = os.path.join(WEB_STATIC_DIR, "ingest.html")
INGEST_UPLOAD_ROOT = os.path.join(PROJECT_ROOT, "complex_document_rag", "upload_jobs")
INGEST_OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "complex_document_rag", "ingestion_output")
INGEST_DEFAULT_OCR_MODELS = ("qwen3.5-plus", "qwen3.5-flash")
INGEST_OCR_MODEL_OPTIONS = tuple(
    value.strip()
    for value in os.getenv("INGEST_OCR_MODEL_OPTIONS", ",".join(INGEST_DEFAULT_OCR_MODELS)).split(",")
    if value.strip()
) or INGEST_DEFAULT_OCR_MODELS
INGEST_DEFAULT_OCR_MODEL = os.getenv("INGEST_DEFAULT_OCR_MODEL", INGEST_OCR_MODEL_OPTIONS[0])
INGEST_DEFAULT_WORKERS = max(1, int(os.getenv("INGEST_DEFAULT_WORKERS", "10")))
INGEST_MAX_WORKERS = max(INGEST_DEFAULT_WORKERS, int(os.getenv("INGEST_MAX_WORKERS", "20")))
INGEST_DPI = max(72, int(os.getenv("INGEST_DPI", "220")))
INGEST_EXECUTOR = ThreadPoolExecutor(max_workers=2)
INGEST_JOBS: dict[str, dict[str, Any]] = {}
INGEST_JOBS_LOCK = threading.Lock()
RETRIEVAL_RETRIES = 2
RETRIEVAL_RETRY_DELAY_SECONDS = 0.6
ANSWER_ASSET_TOP_K = 5
ANSWER_ASSET_SCORE_THRESHOLD = 0.6
ASSET_JUDGE_ENABLED = os.getenv("ASSET_JUDGE_ENABLED", "true").lower() not in {"0", "false", "no"}
ASSET_JUDGE_MODEL = os.getenv("ASSET_JUDGE_MODEL", "qwen3.5-flash")
ASSET_JUDGE_MAX_ASSETS = max(1, int(os.getenv("ASSET_JUDGE_MAX_ASSETS", "3")))
ASSET_JUDGE_MAX_TOKENS = max(128, int(os.getenv("ASSET_JUDGE_MAX_TOKENS", "400")))
TEXT_CONTEXT_TOP_K = 3
IMAGE_CONTEXT_TOP_K = 2
TABLE_CONTEXT_TOP_K = 2
TEXT_CONTEXT_CHAR_LIMIT = 900
IMAGE_SUMMARY_CHAR_LIMIT = 200
TABLE_SUMMARY_CHAR_LIMIT = 220
TABLE_PREVIEW_ROW_LIMIT = 3
TABLE_PREVIEW_ROW_CHAR_LIMIT = 220
ANSWER_IMAGE_INPUT_TOP_K = max(1, int(os.getenv("ANSWER_IMAGE_INPUT_TOP_K", "1")))
HISTORY_TURN_LIMIT = max(1, int(os.getenv("HISTORY_TURN_LIMIT", "6")))
HISTORY_QUERY_CHAR_LIMIT = max(80, int(os.getenv("HISTORY_QUERY_CHAR_LIMIT", "240")))
HISTORY_ANSWER_CHAR_LIMIT = max(120, int(os.getenv("HISTORY_ANSWER_CHAR_LIMIT", "420")))
RETRIEVAL_HISTORY_ANSWER_CHAR_LIMIT = max(80, int(os.getenv("RETRIEVAL_HISTORY_ANSWER_CHAR_LIMIT", "220")))
RERANK_CONFIDENCE_FLOOR = 0.2
QUERY_EXPANSION_SCORE_PENALTY = 0.02
RERANKED_BRANCH_KEYS = ("text_results",)
_EMBEDDING_CACHE_MAX = 128
QUERY_REWRITE_MODEL = os.getenv("QUERY_REWRITE_MODEL", "qwen3.5-flash")
QUERY_REWRITE_MAX_TOKENS = max(64, int(os.getenv("QUERY_REWRITE_MAX_TOKENS", "200")))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger("rag.timing")
