"""面向同步查询的 HTTP 路由。

这个模块负责聊天页上的普通 HTTP 端点：

- 返回主聊天页面
- 暴露健康检查
- 返回默认示例问题
- 执行一次非流式查询

把这些路由与 SSE 流式接口拆开后，传输层行为更容易推理：这个文件只返回
普通 JSON 载荷，而 `query_stream.py` 专注于长连接事件流。
"""

from __future__ import annotations

import os
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

import complex_document_rag.web.backend as web_backend_module
from complex_document_rag.web.helpers import (
    render_answer_markdown_html,
    serialize_retrieval_bundle,
)
from complex_document_rag.web.schemas import QueryRequest
from complex_document_rag.web.settings import WEB_STATIC_DIR


router = APIRouter()


def _build_query_response(payload: QueryRequest) -> dict[str, Any]:
    """执行一次同步查询，并把结果整理成前端需要的载荷。

    前端即使在回答生成失败时，也需要拿到检索元数据。因此这里会把
    检索与生成视为两个独立阶段，分别记录它们的失败信息。
    """
    backend = web_backend_module.get_query_backend()
    history = [turn.model_dump() for turn in payload.history]
    retrieval_error = ""
    retrieval = web_backend_module._empty_retrieval_bundle()
    try:
        retrieval = backend.retrieve(payload.query, history=history)
    except Exception as exc:
        retrieval_error = str(exc)

    answer = ""
    answer_error = ""
    answer_sources: list[dict[str, Any]] = []
    answer_assets: list[dict[str, Any]] = []

    # 只有在检索成功且调用方明确要求生成回答时，才进入回答生成阶段。
    # 这样可以让接口的传输层行为保持可预期。
    if payload.generate_answer and not retrieval_error:
        try:
            answer_result = backend.answer(payload.query, retrieval=retrieval, history=history)
        except Exception as exc:  # pragma: no cover - exercised via tests with fake backend
            answer_error = str(exc)
        else:
            answer, answer_sources, answer_assets = web_backend_module._serialize_answer_result(answer_result)

    return {
        "query": payload.query,
        "answer": answer,
        "answer_html": render_answer_markdown_html(answer),
        "answer_error": answer_error,
        "retrieval_error": retrieval_error,
        "retrieval": serialize_retrieval_bundle(retrieval),
        "answer_sources": answer_sources,
        "answer_assets": answer_assets,
    }


@router.get("/")
def index() -> FileResponse:
    """返回聊天前端页面。"""
    index_path = os.path.join(WEB_STATIC_DIR, "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="前端页面未生成。")
    return FileResponse(
        index_path,
        headers={"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache"},
    )


@router.get("/api/health")
def health() -> dict[str, str]:
    """暴露一个最小化健康检查接口，供本地编排或探活使用。"""
    return {"status": "ok"}


@router.get("/api/default-queries")
def default_queries() -> dict[str, list[str]]:
    """返回聊天界面展示的示例问题。"""
    return {"queries": web_backend_module.load_default_queries()}


@router.post("/api/query")
def query(payload: QueryRequest) -> dict[str, Any]:
    """执行一次同步查询请求。"""
    return _build_query_response(payload)


__all__ = ["router"]
