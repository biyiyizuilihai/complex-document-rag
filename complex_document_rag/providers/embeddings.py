"""向量模型提供方选择与重试策略。"""

from __future__ import annotations

import os
import time
from typing import Any

from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import SSLError as RequestsSSLError
from requests.exceptions import Timeout as RequestsTimeout

from complex_document_rag.providers.common import (
    BaseEmbedding,
    DASHSCOPE_BASE_URL_MARKERS,
    PrivateAttr,
    normalized,
)


RETRYABLE_EMBEDDING_ERROR_MARKERS = (
    "unexpected_eof_while_reading",
    "max retries exceeded",
    "connection reset by peer",
    "temporarily unavailable",
    "connection aborted",
    "read timed out",
)



def _is_retryable_embedding_error(exc: Exception) -> bool:
    """识别值得额外重试一两次的瞬时网络或 TLS 错误。"""
    if isinstance(
        exc,
        (
            RequestsSSLError,
            RequestsConnectionError,
            RequestsTimeout,
            ConnectionResetError,
            TimeoutError,
        ),
    ):
        return True

    message = str(exc).strip().lower()
    return any(marker in message for marker in RETRYABLE_EMBEDDING_ERROR_MARKERS)


class RetryingEmbedding(BaseEmbedding):
    """为 embedding 模型包一层轻量重试逻辑，吸收远端调用抖动。"""

    model_name: str = ""
    embed_batch_size: int = 10
    max_retries: int = 2
    retry_delay_seconds: float = 0.6
    _delegate: Any = PrivateAttr()

    def __init__(
        self,
        delegate: Any,
        *,
        max_retries: int = 2,
        retry_delay_seconds: float = 0.6,
    ) -> None:
        base_kwargs = {
            "model_name": getattr(delegate, "model_name", ""),
            "embed_batch_size": int(getattr(delegate, "embed_batch_size", 10) or 10),
            "callback_manager": getattr(delegate, "callback_manager", None),
            "num_workers": getattr(delegate, "num_workers", None),
            "embeddings_cache": getattr(delegate, "embeddings_cache", None),
            "rate_limiter": getattr(delegate, "rate_limiter", None),
            "max_retries": max_retries,
            "retry_delay_seconds": retry_delay_seconds,
        }
        try:
            super().__init__(**base_kwargs)
        except TypeError:
            # 在精简测试环境里，BaseEmbedding 可能退化成 `object`。
            # 这里仍然要保证包装器实例可以被正常构造。
            super().__init__()

        self.model_name = str(base_kwargs["model_name"])
        self.embed_batch_size = int(base_kwargs["embed_batch_size"])
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds
        self._delegate = delegate

    @classmethod
    def class_name(cls) -> str:
        return "RetryingEmbedding"

    def _run_with_retry(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        """只有在错误看起来像瞬时网络问题时才执行重试。"""
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt >= self.max_retries or not _is_retryable_embedding_error(exc):
                    raise
                time.sleep(self.retry_delay_seconds * (attempt + 1))

        raise last_error or RuntimeError("embedding request failed")

    def _get_query_embedding(self, query: str) -> Any:
        return self._run_with_retry(self._delegate._get_query_embedding, query)

    def _get_text_embedding(self, text: str) -> Any:
        return self._run_with_retry(self._delegate._get_text_embedding, text)

    def _get_text_embeddings(self, texts: list[str]) -> Any:
        return self._run_with_retry(self._delegate._get_text_embeddings, texts)

    async def _aget_query_embedding(self, query: str) -> Any:
        if hasattr(self._delegate, "_aget_query_embedding"):
            return await self._delegate._aget_query_embedding(query)
        return self._run_with_retry(self._delegate._get_query_embedding, query)



def should_use_dashscope_embedding(model_name: str, api_base: str | None = None) -> bool:
    """当模型名或 base URL 指向 DashScope 时，走 DashScope 的 embedding 实现。"""
    normalized_model = normalized(model_name)
    normalized_base = normalized(api_base)
    return (
        any(marker in normalized_base for marker in DASHSCOPE_BASE_URL_MARKERS)
        or normalized_model.startswith("text-embedding-v")
        or normalized_model.startswith("tongyi-embedding-")
        or normalized_model.startswith("qwen")
    )



def _embedding_retry_settings() -> tuple[int, float]:
    """从环境变量读取重试配置，并做防御式解析与兜底。"""
    retry_count_raw = os.getenv("EMBEDDING_RETRY_COUNT", "2")
    retry_delay_raw = os.getenv("EMBEDDING_RETRY_DELAY_SECONDS", "0.6")
    try:
        retry_count = max(0, int(retry_count_raw))
    except ValueError:
        retry_count = 2
    try:
        retry_delay_seconds = max(0.0, float(retry_delay_raw))
    except ValueError:
        retry_delay_seconds = 0.6
    return retry_count, retry_delay_seconds



def create_embedding_model(model_name: str, api_key: str, api_base: str | None = None):
    """创建索引流程统一使用的 embedding 模型包装器。"""
    retry_count, retry_delay_seconds = _embedding_retry_settings()

    if should_use_dashscope_embedding(model_name=model_name, api_base=api_base):
        try:
            from llama_index.embeddings.dashscope import DashScopeEmbedding
        except ImportError as exc:
            raise ImportError(
                "当前配置检测到 DashScope embedding，请先安装依赖："
                "`pip install llama-index-embeddings-dashscope dashscope`"
            ) from exc

        raw_batch_size = os.getenv("DASHSCOPE_EMBED_BATCH_SIZE", "10")
        try:
            embed_batch_size = max(1, min(int(raw_batch_size), 10))
        except ValueError:
            embed_batch_size = 10

        embedding = DashScopeEmbedding(
            model_name=model_name,
            api_key=api_key,
            embed_batch_size=embed_batch_size,
        )
        return RetryingEmbedding(
            delegate=embedding,
            max_retries=retry_count,
            retry_delay_seconds=retry_delay_seconds,
        )

    from llama_index.embeddings.openai import OpenAIEmbedding

    embedding = OpenAIEmbedding(
        model=model_name,
        api_key=api_key,
        api_base=api_base,
    )
    return RetryingEmbedding(
        delegate=embedding,
        max_retries=retry_count,
        retry_delay_seconds=retry_delay_seconds,
    )


__all__ = [
    "RetryingEmbedding",
    "create_embedding_model",
    "should_use_dashscope_embedding",
]
