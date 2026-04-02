"""供 DashScope 与多模态请求使用的 OpenAI 兼容 LLM 包装器。

当 OpenAI 兼容 HTTP API 比 provider 专用的 LlamaIndex 集成更稳定时，
这里会统一接管请求格式、流式行为和本地图像处理逻辑。
"""

from __future__ import annotations

import base64
import mimetypes
import os
from typing import Any, Optional

from complex_document_rag.providers.common import (
    CallbackManager,
    CompletionResponse,
    CompletionResponseGen,
    CustomLLM,
    LLMMetadata,
    OpenAIClient,
    PrivateAttr,
    PydanticProgramMode,
)



def _local_image_path_to_data_url(path: str) -> str:
    """把本地图像编码成 data URL，供 OpenAI 兼容多模态接口使用。"""
    normalized_path = os.path.abspath(path)
    mime_type, _ = mimetypes.guess_type(normalized_path)
    if not mime_type:
        mime_type = "image/png"
    with open(normalized_path, "rb") as handle:
        encoded = base64.b64encode(handle.read()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"



def _build_multimodal_user_content(
    prompt: str,
    *,
    image_paths: Optional[list[str]] = None,
    image_urls: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    """构造同时包含文本与本地/远程图片的 chat 消息内容。"""
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for url in image_urls or []:
        normalized_url = str(url or "").strip()
        if not normalized_url:
            continue
        content.append({"type": "image_url", "image_url": {"url": normalized_url}})
    for path in image_paths or []:
        normalized_path = str(path or "").strip()
        if not normalized_path:
            continue
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": _local_image_path_to_data_url(normalized_path)},
            }
        )
    return content


if OpenAIClient is not None and CompletionResponse is not None and LLMMetadata is not None:

    class OpenAICompatibleLLM(CustomLLM):
        """基于 OpenAI 兼容接口的轻量纯文本 LLM 包装器。"""

        model_name: str = ""
        api_key: str = ""
        api_base: Optional[str] = None
        max_tokens: Optional[int] = 1024
        temperature: float = 0.1
        additional_request_kwargs: dict[str, Any] = {}
        _client: Any = PrivateAttr()

        def __init__(
            self,
            model_name: str,
            api_key: str,
            api_base: Optional[str] = None,
            max_tokens: Optional[int] = 1024,
            temperature: float = 0.1,
            additional_request_kwargs: Optional[dict[str, Any]] = None,
            callback_manager: Optional[CallbackManager] = None,
            system_prompt: Optional[str] = None,
        ) -> None:
            super().__init__(
                model_name=model_name,
                api_key=api_key,
                api_base=api_base,
                max_tokens=max_tokens,
                temperature=temperature,
                additional_request_kwargs=dict(additional_request_kwargs or {}),
                callback_manager=callback_manager or CallbackManager([]),
                system_prompt=system_prompt,
                pydantic_program_mode=PydanticProgramMode.DEFAULT,
            )
            self.model_name = model_name
            self.api_key = api_key
            self.api_base = api_base
            self.max_tokens = max_tokens
            self.temperature = temperature
            self.additional_request_kwargs = dict(additional_request_kwargs or {})
            self._client = OpenAIClient(api_key=api_key, base_url=api_base)

        @classmethod
        def class_name(cls) -> str:
            return "OpenAICompatibleLLM"

        @property
        def metadata(self) -> LLMMetadata:
            return LLMMetadata(
                context_window=131072,
                num_output=self.max_tokens or -1,
                is_chat_model=False,
                model_name=self.model_name,
            )

        def complete(
            self, prompt: str, formatted: bool = False, **kwargs: Any
        ) -> CompletionResponse:
            messages = []
            if getattr(self, "system_prompt", None):
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append({"role": "user", "content": prompt})

            request_kwargs = {
                "model": self.model_name,
                "messages": messages,
                "temperature": kwargs.get("temperature", self.temperature),
            }
            max_tokens = kwargs.get("max_tokens", self.max_tokens)
            if max_tokens is not None:
                request_kwargs["max_tokens"] = max_tokens
            request_kwargs.update(self.additional_request_kwargs)

            response = self._client.chat.completions.create(**request_kwargs)
            text = response.choices[0].message.content or ""
            return CompletionResponse(text=text, raw=response)

        def stream_complete(
            self, prompt: str, formatted: bool = False, **kwargs: Any
        ) -> CompletionResponseGen:
            messages = []
            if getattr(self, "system_prompt", None):
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append({"role": "user", "content": prompt})

            request_kwargs = {
                "model": self.model_name,
                "messages": messages,
                "temperature": kwargs.get("temperature", self.temperature),
                "stream": True,
            }
            max_tokens = kwargs.get("max_tokens", self.max_tokens)
            if max_tokens is not None:
                request_kwargs["max_tokens"] = max_tokens
            request_kwargs.update(self.additional_request_kwargs)

            stream = self._client.chat.completions.create(**request_kwargs)

            def gen() -> CompletionResponseGen:
                for chunk in stream:
                    delta = chunk.choices[0].delta
                    delta_text = getattr(delta, "content", None) or ""
                    reasoning_text = getattr(delta, "reasoning_content", None) or ""
                    if not delta_text and not reasoning_text:
                        continue
                    additional_kwargs: dict[str, Any] = {}
                    if reasoning_text:
                        additional_kwargs["reasoning_delta"] = reasoning_text
                    yield CompletionResponse(
                        text=delta_text or "",
                        delta=delta_text or None,
                        raw=chunk,
                        additional_kwargs=additional_kwargs,
                    )

            return gen()


    class OpenAICompatibleMultimodalLLM(CustomLLM):
        """可同时接收文本与 `image_url` 内容的 OpenAI 兼容包装器。"""

        model_name: str = ""
        api_key: str = ""
        api_base: Optional[str] = None
        max_tokens: Optional[int] = 1024
        temperature: float = 0.1
        additional_request_kwargs: dict[str, Any] = {}
        _client: Any = PrivateAttr()

        def __init__(
            self,
            model_name: str,
            api_key: str,
            api_base: Optional[str] = None,
            max_tokens: Optional[int] = 1024,
            temperature: float = 0.1,
            additional_request_kwargs: Optional[dict[str, Any]] = None,
            callback_manager: Optional[CallbackManager] = None,
            system_prompt: Optional[str] = None,
        ) -> None:
            super().__init__(
                model_name=model_name,
                api_key=api_key,
                api_base=api_base,
                max_tokens=max_tokens,
                temperature=temperature,
                additional_request_kwargs=dict(additional_request_kwargs or {}),
                callback_manager=callback_manager or CallbackManager([]),
                system_prompt=system_prompt,
                pydantic_program_mode=PydanticProgramMode.DEFAULT,
            )
            self.model_name = model_name
            self.api_key = api_key
            self.api_base = api_base
            self.max_tokens = max_tokens
            self.temperature = temperature
            self.additional_request_kwargs = dict(additional_request_kwargs or {})
            self._client = OpenAIClient(api_key=api_key, base_url=api_base)

        @classmethod
        def class_name(cls) -> str:
            return "OpenAICompatibleMultimodalLLM"

        @property
        def metadata(self) -> LLMMetadata:
            return LLMMetadata(
                context_window=131072,
                num_output=self.max_tokens or -1,
                is_chat_model=False,
                model_name=self.model_name,
            )

        def _build_messages(
            self,
            prompt: str,
            *,
            image_paths: Optional[list[str]] = None,
            image_urls: Optional[list[str]] = None,
        ) -> list[dict[str, Any]]:
            messages = []
            if getattr(self, "system_prompt", None):
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append(
                {
                    "role": "user",
                    "content": _build_multimodal_user_content(
                        prompt,
                        image_paths=image_paths,
                        image_urls=image_urls,
                    ),
                }
            )
            return messages

        def complete(
            self,
            prompt: str,
            formatted: bool = False,
            *,
            image_paths: Optional[list[str]] = None,
            image_urls: Optional[list[str]] = None,
            **kwargs: Any,
        ) -> CompletionResponse:
            request_kwargs = {
                "model": self.model_name,
                "messages": self._build_messages(
                    prompt,
                    image_paths=image_paths,
                    image_urls=image_urls,
                ),
                "temperature": kwargs.get("temperature", self.temperature),
            }
            max_tokens = kwargs.get("max_tokens", self.max_tokens)
            if max_tokens is not None:
                request_kwargs["max_tokens"] = max_tokens
            request_kwargs.update(self.additional_request_kwargs)

            response = self._client.chat.completions.create(**request_kwargs)
            text = response.choices[0].message.content or ""
            return CompletionResponse(text=text, raw=response)

        def stream_complete(
            self,
            prompt: str,
            formatted: bool = False,
            *,
            image_paths: Optional[list[str]] = None,
            image_urls: Optional[list[str]] = None,
            **kwargs: Any,
        ) -> CompletionResponseGen:
            request_kwargs = {
                "model": self.model_name,
                "messages": self._build_messages(
                    prompt,
                    image_paths=image_paths,
                    image_urls=image_urls,
                ),
                "temperature": kwargs.get("temperature", self.temperature),
                "stream": True,
            }
            max_tokens = kwargs.get("max_tokens", self.max_tokens)
            if max_tokens is not None:
                request_kwargs["max_tokens"] = max_tokens
            request_kwargs.update(self.additional_request_kwargs)

            stream = self._client.chat.completions.create(**request_kwargs)

            def gen() -> CompletionResponseGen:
                for chunk in stream:
                    delta = chunk.choices[0].delta
                    delta_text = getattr(delta, "content", None) or ""
                    reasoning_text = getattr(delta, "reasoning_content", None) or ""
                    if not delta_text and not reasoning_text:
                        continue
                    additional_kwargs: dict[str, Any] = {}
                    if reasoning_text:
                        additional_kwargs["reasoning_delta"] = reasoning_text
                    yield CompletionResponse(
                        text=delta_text or "",
                        delta=delta_text or None,
                        raw=chunk,
                        additional_kwargs=additional_kwargs,
                    )

            return gen()

else:

    class OpenAICompatibleLLM:  # type: ignore[no-redef]
        """缺少可选运行时依赖时供测试使用的占位实现。"""

        def __init__(
            self,
            model_name: str,
            api_key: str,
            api_base: Optional[str] = None,
            max_tokens: Optional[int] = 1024,
            temperature: float = 0.1,
            additional_request_kwargs: Optional[dict[str, Any]] = None,
            callback_manager: Any = None,
            system_prompt: Optional[str] = None,
        ) -> None:
            self.model_name = model_name
            self.api_key = api_key
            self.api_base = api_base
            self.max_tokens = max_tokens
            self.temperature = temperature
            self.additional_request_kwargs = dict(additional_request_kwargs or {})
            self.callback_manager = callback_manager
            self.system_prompt = system_prompt


    class OpenAICompatibleMultimodalLLM(OpenAICompatibleLLM):  # type: ignore[no-redef]
        """在轻量测试环境中保留构造参数形状的占位实现。"""


__all__ = [
    "OpenAIClient",
    "OpenAICompatibleLLM",
    "OpenAICompatibleMultimodalLLM",
]
