"""文本与多模态 LLM 的 provider 选择逻辑。"""

from __future__ import annotations

from typing import Any

from complex_document_rag.providers.common import DASHSCOPE_BASE_URL_MARKERS, normalized
from complex_document_rag.providers.openai_compatible import (
    OpenAICompatibleLLM,
    OpenAICompatibleMultimodalLLM,
)



def should_use_dashscope_llm(model_name: str, api_base: str | None = None) -> bool:
    """当底层 provider 是 DashScope 时，将 LLM 调用路由到 OpenAI 兼容包装器。"""
    normalized_model = normalized(model_name)
    normalized_base = normalized(api_base)
    return any(marker in normalized_base for marker in DASHSCOPE_BASE_URL_MARKERS) or (
        normalized_model.startswith("qwen") or normalized_model.startswith("qwq")
    )



def _thinking_request_kwargs(
    *, disable_thinking: bool, thinking_budget_tokens: int
) -> dict[str, Any]:
    """把 thinking 开关翻译成 provider 所需的 `extra_body` 请求参数。"""
    if disable_thinking:
        return {"extra_body": {"enable_thinking": False}}
    if thinking_budget_tokens > 0:
        return {
            "extra_body": {
                "enable_thinking": True,
                "thinking_budget_tokens": thinking_budget_tokens,
            }
        }
    return {}



def create_text_llm(
    model_name: str,
    api_key: str,
    api_base: str | None = None,
    *,
    disable_thinking: bool = False,
    thinking_budget_tokens: int = 0,
):
    """创建检索侧和回答侧默认使用的纯文本 LLM。"""
    if should_use_dashscope_llm(model_name=model_name, api_base=api_base):
        return OpenAICompatibleLLM(
            model_name=model_name,
            api_key=api_key,
            api_base=api_base,
            additional_request_kwargs=_thinking_request_kwargs(
                disable_thinking=disable_thinking,
                thinking_budget_tokens=thinking_budget_tokens,
            ),
        )

    from llama_index.llms.openai import OpenAI

    return OpenAI(
        model=model_name,
        api_key=api_key,
        api_base=api_base,
    )



def create_multimodal_llm(
    model_name: str,
    api_key: str,
    api_base: str | None = None,
    *,
    disable_thinking: bool = False,
    thinking_budget_tokens: int = 0,
):
    """创建回答阶段使用的多模态模型。

    多模态请求统一走 OpenAI 兼容包装器，因为它能在不同 provider 间一致地
    处理远程图片 URL 和本地图像 data URL。
    """
    additional_request_kwargs: dict[str, Any] = {}
    if should_use_dashscope_llm(model_name=model_name, api_base=api_base):
        additional_request_kwargs = _thinking_request_kwargs(
            disable_thinking=disable_thinking,
            thinking_budget_tokens=thinking_budget_tokens,
        )

    return OpenAICompatibleMultimodalLLM(
        model_name=model_name,
        api_key=api_key,
        api_base=api_base,
        additional_request_kwargs=additional_request_kwargs,
    )


__all__ = [
    "OpenAICompatibleLLM",
    "OpenAICompatibleMultimodalLLM",
    "create_multimodal_llm",
    "create_text_llm",
    "should_use_dashscope_llm",
]
