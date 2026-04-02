"""用于组装 FastAPI 应用并挂接生命周期的逻辑。

当前 web 包已经按传输层边界拆分职责：

- `query_http.py` 负责普通 JSON/HTML 查询路由
- `query_stream.py` 负责 SSE 流式查询路由
- `ingest_routes.py` 负责上传与任务跟踪路由

这个模块刻意不承载业务逻辑，只负责组装 FastAPI 应用、挂载静态目录，
并在进程启动时预热一次查询后端。
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import complex_document_rag.web.backend as web_backend_module
from complex_document_rag.web.helpers import ARTIFACTS_ROOT
from complex_document_rag.web.ingest_routes import router as ingest_router
from complex_document_rag.web.query_http import router as query_http_router
from complex_document_rag.web.query_stream import router as query_stream_router
from complex_document_rag.web.settings import (
    INGEST_OUTPUT_ROOT,
    INGEST_UPLOAD_ROOT,
    LOGGER,
    WEB_STATIC_DIR,
)


@asynccontextmanager
async def _lifespan(_: FastAPI):
    """预热一次查询后端，避免首个请求承受冷启动开销。"""
    t_start = time.perf_counter()
    try:
        web_backend_module.get_query_backend()
    except Exception:
        LOGGER.exception("[warmup  ] failed to initialize query backend")
    else:
        LOGGER.info("[warmup  ] %5.0fms  query backend ready", (time.perf_counter() - t_start) * 1000)
    yield


def create_app() -> FastAPI:
    """创建 FastAPI 应用、挂载静态资源并注册各组路由。"""
    app = FastAPI(
        title="复杂文档 RAG 解析",
        description="用于复杂文档 OCR、检索和多模态问答的一体化本地前端",
        version="0.2.0",
        lifespan=_lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    os.makedirs(WEB_STATIC_DIR, exist_ok=True)
    os.makedirs(ARTIFACTS_ROOT, exist_ok=True)
    os.makedirs(INGEST_UPLOAD_ROOT, exist_ok=True)
    os.makedirs(INGEST_OUTPUT_ROOT, exist_ok=True)

    app.mount("/artifacts", StaticFiles(directory=ARTIFACTS_ROOT), name="artifacts")
    app.mount("/static", StaticFiles(directory=WEB_STATIC_DIR), name="static")

    app.include_router(query_http_router)
    app.include_router(query_stream_router)
    app.include_router(ingest_router)
    return app


app = create_app()


__all__ = ["app", "create_app"]
