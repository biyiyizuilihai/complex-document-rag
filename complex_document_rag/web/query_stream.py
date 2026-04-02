"""流式查询路由与 SSE 事件组装逻辑。

这个模块负责查询 API 的事件流侧逻辑。它刻意把流式事件组装与普通
HTTP JSON 处理分开，让流控、时延日志与部分失败行为都集中在一个地方。
"""

from __future__ import annotations

import time
from typing import Iterable

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

import complex_document_rag.web.backend as web_backend_module
from complex_document_rag.web.helpers import (
    render_answer_markdown_html,
    serialize_answer_sources,
    serialize_retrieval_bundle,
)
from complex_document_rag.web.schemas import QueryRequest
from complex_document_rag.web.settings import LOGGER


router = APIRouter()


def _stream_query_events(payload: QueryRequest) -> Iterable[str]:
    """为单次查询请求产出 SSE 事件流。

    事件顺序是刻意设计的：
    1. 先发送检索元数据，让前端能尽早渲染证据
    2. 再在 LLM 流式输出过程中持续发送 reasoning/chunk 事件
    3. 无论是否发生部分错误，最后都一定发送 `done` 事件关闭流
    """
    backend = web_backend_module.get_query_backend()
    history = [turn.model_dump() for turn in payload.history]
    t_request = time.perf_counter()
    retrieval_error = ""
    retrieval = web_backend_module._empty_retrieval_bundle()
    try:
        retrieval = backend.retrieve(payload.query, history=history)
    except Exception as exc:
        retrieval_error = str(exc)

    if retrieval_error:
        yield web_backend_module._sse_event(
            "retrieval",
            {
                "retrieval": serialize_retrieval_bundle(retrieval),
                "retrieval_error": retrieval_error,
                "answer_assets": [],
                "answer_sources": [],
            },
        )
        yield web_backend_module._sse_event("error", {"message": retrieval_error})
        yield web_backend_module._sse_event(
            "done",
            {
                "answer": "",
                "answer_html": render_answer_markdown_html(""),
                "answer_assets": [],
                "answer_sources": [],
            },
        )
        return

    if not payload.generate_answer:
        yield web_backend_module._sse_event(
            "retrieval",
            {
                "retrieval": serialize_retrieval_bundle(retrieval),
                "retrieval_error": "",
                "answer_assets": [],
                "answer_sources": [],
            },
        )
        yield web_backend_module._sse_event(
            "done",
            {
                "answer": "",
                "answer_html": render_answer_markdown_html(""),
                "answer_assets": [],
                "answer_sources": [],
            },
        )
        return

    try:
        stream_result = backend.stream_answer(payload.query, retrieval=retrieval, history=history)
    except Exception as exc:
        yield web_backend_module._sse_event(
            "retrieval",
            {
                "retrieval": serialize_retrieval_bundle(retrieval),
                "retrieval_error": "",
                "answer_assets": [],
                "answer_sources": [],
            },
        )
        yield web_backend_module._sse_event("error", {"message": str(exc)})
        yield web_backend_module._sse_event(
            "done",
            {
                "answer": "",
                "answer_html": render_answer_markdown_html(""),
                "answer_assets": [],
                "answer_sources": [],
            },
        )
        return

    if isinstance(stream_result, dict):
        stream_iter = stream_result.get("stream", [])
        raw_answer_sources = stream_result.get("answer_sources", [])
        raw_answer_assets = stream_result.get("answer_assets", [])
    else:
        stream_iter = stream_result
        raw_answer_assets = (
            backend.select_answer_assets(retrieval, query=payload.query)
            if hasattr(backend, "select_answer_assets")
            else []
        )
        raw_answer_sources = (
            backend.select_answer_sources(retrieval, raw_answer_assets)
            if hasattr(backend, "select_answer_sources")
            else []
        )

    answer_sources = serialize_answer_sources(raw_answer_sources)
    answer_assets = serialize_answer_sources(web_backend_module._sort_answer_assets(raw_answer_assets))
    yield web_backend_module._sse_event(
        "retrieval",
        {
            "retrieval": serialize_retrieval_bundle(retrieval),
            "retrieval_error": "",
            "answer_assets": answer_assets,
            "answer_sources": answer_sources,
        },
    )

    answer_text = ""
    t_llm = time.perf_counter()
    first_stream_event = True
    first_visible_token = True
    try:
        for chunk in stream_iter:
            additional_kwargs = getattr(chunk, "additional_kwargs", {}) or {}
            reasoning_delta = str(additional_kwargs.get("reasoning_delta", "") or "")
            delta = getattr(chunk, "delta", "") or getattr(chunk, "text", "")
            if not delta and isinstance(chunk, str):
                delta = chunk
            if not reasoning_delta and not delta:
                continue
            if first_stream_event:
                LOGGER.info("[llm-any ] %5.0fms  first stream event", (time.perf_counter() - t_request) * 1000)
                first_stream_event = False
            if reasoning_delta:
                yield web_backend_module._sse_event("reasoning", {"delta": reasoning_delta})
            if not delta:
                continue
            if first_visible_token:
                LOGGER.info(
                    "[llm-ttft] %5.0fms  (retrieval→first visible token)",
                    (time.perf_counter() - t_request) * 1000,
                )
                LOGGER.info(
                    "[llm-gen ] %5.0fms  waiting for first visible token",
                    (time.perf_counter() - t_llm) * 1000,
                )
                first_visible_token = False
            answer_text += delta
            yield web_backend_module._sse_event("chunk", {"delta": delta})
    except Exception as exc:
        yield web_backend_module._sse_event("error", {"message": str(exc)})
        yield web_backend_module._sse_event(
            "done",
            {
                "answer": answer_text,
                "answer_html": render_answer_markdown_html(answer_text),
                "answer_assets": answer_assets,
                "answer_sources": answer_sources,
            },
        )
        return

    LOGGER.info(
        "[llm-done] %5.0fms  total request  answer=%d chars",
        (time.perf_counter() - t_request) * 1000,
        len(answer_text),
    )
    yield web_backend_module._sse_event(
        "done",
        {
            "answer": answer_text,
            "answer_html": render_answer_markdown_html(answer_text),
            "answer_assets": answer_assets,
            "answer_sources": answer_sources,
        },
    )


@router.post("/api/query/stream")
def stream_query(payload: QueryRequest) -> StreamingResponse:
    """通过 SSE 执行一次流式查询请求。"""
    return StreamingResponse(_stream_query_events(payload), media_type="text/event-stream")


__all__ = ["LOGGER", "router"]
