"""面向摄入流程的 HTTP 路由。

这个模块负责上传校验、摄入任务创建和任务状态查询。把它与查询路由拆开，
可以让写入型较强的摄入流程与只读查询 API 保持边界清晰。
"""

from __future__ import annotations

import os
import uuid
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

import complex_document_rag.web.jobs as web_jobs_module
from complex_document_rag.web.settings import (
    INGEST_DEFAULT_OCR_MODEL,
    INGEST_DEFAULT_WORKERS,
    INGEST_MAX_WORKERS,
    INGEST_OCR_MODEL_OPTIONS,
    INGEST_STATIC_PATH,
    INGEST_UPLOAD_ROOT,
)


router = APIRouter()


@router.get("/ingest")
def ingest_page() -> FileResponse:
    """返回摄入/上传页面。"""
    if not os.path.exists(INGEST_STATIC_PATH):
        raise HTTPException(status_code=404, detail="摄入页面未生成。")
    return FileResponse(
        INGEST_STATIC_PATH,
        headers={"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache"},
    )


@router.get("/api/ingest/options")
def ingest_options() -> dict[str, Any]:
    """返回摄入页面默认配置与允许的 OCR 选项。"""
    return {
        "ocr_models": list(INGEST_OCR_MODEL_OPTIONS),
        "default_ocr_model": INGEST_DEFAULT_OCR_MODEL,
        "default_workers": INGEST_DEFAULT_WORKERS,
        "min_workers": 1,
        "max_workers": INGEST_MAX_WORKERS,
    }


@router.post("/api/ingest/jobs", status_code=202)
async def create_ingest_job(
    file: UploadFile = File(...),
    ocr_model: str = Form(INGEST_DEFAULT_OCR_MODEL),
    workers: int = Form(INGEST_DEFAULT_WORKERS),
) -> dict[str, Any]:
    """接收一个 PDF 上传请求，并放入后台摄入队列。"""
    filename = os.path.basename(file.filename or "").strip()
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="仅支持 PDF 文件上传。")

    normalized_model = str(ocr_model or "").strip() or INGEST_DEFAULT_OCR_MODEL
    if workers < 1 or workers > INGEST_MAX_WORKERS:
        raise HTTPException(status_code=400, detail=f"并发数必须在 1 到 {INGEST_MAX_WORKERS} 之间。")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="上传文件为空。")

    job_id = f"job_{uuid.uuid4().hex[:12]}"
    job_dir = os.path.join(INGEST_UPLOAD_ROOT, job_id)
    os.makedirs(job_dir, exist_ok=True)
    upload_path = os.path.join(job_dir, filename)
    with open(upload_path, "wb") as handle:
        handle.write(content)

    job = web_jobs_module.build_queued_ingest_job(
        job_id=job_id,
        filename=filename,
        ocr_model=normalized_model,
        workers=workers,
        upload_path=upload_path,
    )
    web_jobs_module._store_ingest_job(job)
    web_jobs_module._append_ingest_job_log(job_id, f"文件已上传：{filename}")
    web_jobs_module._submit_ingest_job(job_id)
    return web_jobs_module._public_ingest_job_payload(job)


@router.get("/api/ingest/jobs/{job_id}")
def ingest_job_status(job_id: str) -> dict[str, Any]:
    """返回单个摄入任务最新的对外可见状态。"""
    return web_jobs_module._public_ingest_job_payload(web_jobs_module._get_ingest_job_or_404(job_id))


__all__ = ["router"]
