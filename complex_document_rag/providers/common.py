"""供 provider 模块共用的兼容层辅助函数。

当前 provider 层需要同时支持 OpenAI 和通过 OpenAI 兼容接口接入的
DashScope。这个模块把可选依赖导入和 provider 判定规则集中起来，
让 embedding 与 LLM 工厂保持小而稳定。
"""

from __future__ import annotations

from typing import Any


DASHSCOPE_BASE_URL_MARKERS = (
    "dashscope.aliyuncs.com",
    "dashscope-intl.aliyuncs.com",
)



def normalized(value: str | None) -> str:
    """在做 provider 路由判断前，先规范化配置值。"""
    return (value or "").strip().lower()


try:
    from openai import OpenAI as OpenAIClient
    from llama_index.core.base.embeddings.base import BaseEmbedding
    from llama_index.core.base.llms.types import (
        CompletionResponse,
        CompletionResponseGen,
        LLMMetadata,
    )
    from llama_index.core.bridge.pydantic import PrivateAttr
    from llama_index.core.callbacks import CallbackManager
    from llama_index.core.llms.custom import CustomLLM
    from llama_index.core.types import PydanticProgramMode
except ImportError:
    OpenAIClient = None
    BaseEmbedding = object
    CompletionResponse = None
    CompletionResponseGen = None
    LLMMetadata = None
    CallbackManager = None
    CustomLLM = object
    PydanticProgramMode = None

    def PrivateAttr(default: Any = None) -> Any:  # type: ignore[misc]
        """测试环境缺少可选的 LlamaIndex 组件时使用的回退实现。"""
        return default


__all__ = [
    "BaseEmbedding",
    "CallbackManager",
    "CompletionResponse",
    "CompletionResponseGen",
    "CustomLLM",
    "DASHSCOPE_BASE_URL_MARKERS",
    "LLMMetadata",
    "OpenAIClient",
    "PrivateAttr",
    "PydanticProgramMode",
    "normalized",
]
