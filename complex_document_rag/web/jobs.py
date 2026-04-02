"""网页层的摄入任务状态与执行辅助逻辑。

HTTP 路由只负责校验请求并把结果转成 JSON。排队任务状态、日志捕获、
旧向量清理和摄入后校验都集中在这里，保证路由模块只关注传输逻辑。
"""

from __future__ import annotations

import argparse
import contextlib
import copy
from datetime import datetime, timezone
import json
import os
import time
from typing import Any

from fastapi import HTTPException

import complex_document_rag.web.backend as web_backend_module
from complex_document_rag.core.config import (
    QDRANT_IMAGE_COLLECTION,
    QDRANT_TABLE_COLLECTION,
    QDRANT_TEXT_COLLECTION,
)
from complex_document_rag.ingestion.common import sanitize_doc_id
from complex_document_rag.indexing.qdrant import (
    count_doc_vectors,
    create_qdrant_client,
    delete_doc_vectors,
)
from complex_document_rag.ingestion.pipeline import ingest_document
from complex_document_rag.web.settings import (
    INGEST_DEFAULT_OCR_MODEL,
    INGEST_DEFAULT_WORKERS,
    INGEST_DPI,
    INGEST_EXECUTOR,
    INGEST_JOBS,
    INGEST_JOBS_LOCK,
    INGEST_OUTPUT_ROOT,
)



def _utc_now_iso() -> str:
    """返回稳定的 UTC 时间戳字符串，用于任务快照与日志。"""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")



def _public_ingest_job_payload(job: dict[str, Any]) -> dict[str, Any]:
    """只返回前端安全且有用的字段。"""
    visible_keys = (
        "job_id",
        "status",
        "filename",
        "ocr_model",
        "workers",
        "created_at",
        "updated_at",
        "started_at",
        "finished_at",
        "logs",
        "error",
        "doc_id",
        "output_dir",
        "index_status",
        "index_message",
        "artifact_warnings",
        "artifact_counts",
        "index_counts",
    )
    return {key: copy.deepcopy(job.get(key)) for key in visible_keys}



def build_queued_ingest_job(
    *,
    job_id: str,
    filename: str,
    ocr_model: str,
    workers: int,
    upload_path: str,
) -> dict[str, Any]:
    """为新上传的摄入任务创建标准内存记录。"""
    timestamp = _utc_now_iso()
    return {
        "job_id": job_id,
        "status": "queued",
        "filename": filename,
        "ocr_model": ocr_model,
        "workers": workers,
        "created_at": timestamp,
        "updated_at": timestamp,
        "started_at": "",
        "finished_at": "",
        "logs": [],
        "error": "",
        "doc_id": "",
        "output_dir": "",
        "index_status": "pending",
        "index_message": "",
        "artifact_warnings": [],
        "artifact_counts": {},
        "index_counts": {},
        "upload_path": upload_path,
    }



def _store_ingest_job(job: dict[str, Any]) -> None:
    """把任务记录保存到进程内注册表。"""
    with INGEST_JOBS_LOCK:
        INGEST_JOBS[job["job_id"]] = job



def _get_ingest_job_or_404(job_id: str) -> dict[str, Any]:
    """返回任务记录副本；若不存在则抛出前端友好的 404。"""
    with INGEST_JOBS_LOCK:
        job = INGEST_JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="摄入任务不存在。")
        return copy.deepcopy(job)



def _update_ingest_job(job_id: str, **fields: Any) -> None:
    """原子地更新任务记录，并刷新修改时间戳。"""
    with INGEST_JOBS_LOCK:
        job = INGEST_JOBS[job_id]
        job.update(fields)
        job["updated_at"] = _utc_now_iso()



def _append_ingest_job_log(job_id: str, message: str) -> None:
    """把捕获到的 stdout/stderr 规范化成按行记录的任务日志。"""
    stripped = str(message or "").rstrip()
    if not stripped:
        return
    with INGEST_JOBS_LOCK:
        job = INGEST_JOBS[job_id]
        logs = job.setdefault("logs", [])
        for line in stripped.splitlines():
            normalized = line.strip()
            if not normalized:
                continue
            logs.append(f"{time.strftime('%H:%M:%S')}  {normalized}")
        job["updated_at"] = _utc_now_iso()


class _IngestJobLogWriter:
    """在重定向 stdout/stderr 时保留未换行的部分内容。"""

    def __init__(self, job_id: str):
        self.job_id = job_id
        self._buffer = ""

    def write(self, text: str) -> int:
        self._buffer += str(text or "")
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            _append_ingest_job_log(self.job_id, line)
        return len(text or "")

    def flush(self) -> None:
        if self._buffer.strip():
            _append_ingest_job_log(self.job_id, self._buffer)
        self._buffer = ""



def _submit_ingest_job(job_id: str) -> None:
    """把长耗时摄入任务提交到后台执行器。"""
    INGEST_EXECUTOR.submit(_run_ingest_job, job_id)



def _refresh_query_backend_after_ingest(job_id: str) -> None:
    """清空缓存后的后端实例，确保新入库向量能立刻被检索到。"""
    cache_clear = getattr(web_backend_module.get_query_backend, "cache_clear", None)
    if callable(cache_clear):
        cache_clear()
    web_backend_module.get_query_backend()
    _append_ingest_job_log(job_id, "问答后端已刷新，可直接去智能问答页提问。")



def _load_json_record_count(path: str) -> int:
    """统计摄入流程产出的 list/dict JSON 文件中的条目数。"""
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        return len(payload)
    if isinstance(payload, list):
        return len(payload)
    return 0



def _load_ingest_artifact_counts(output_dir: str) -> dict[str, int]:
    """汇总磁盘上的产物数量，便于向量校验时做同口径比较。"""
    manifest_path = os.path.join(output_dir, "manifest.json")
    manifest: dict[str, Any] = {}
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle) or {}
    return {
        "text_blocks": len(manifest.get("text_blocks") or []),
        "image_files": len(manifest.get("image_blocks") or []),
        "image_descriptions": _load_json_record_count(os.path.join(output_dir, "image_descriptions.json")),
        "table_blocks": _load_json_record_count(os.path.join(output_dir, "table_blocks.json")),
    }



def _compute_ingest_index_status(
    doc_id: str,
    output_dir: str,
    source_path: str,
    *,
    client: Any | None = None,
) -> dict[str, Any]:
    """对单个文档比较落地产物数量与 Qdrant 向量数量。"""
    artifact_counts = _load_ingest_artifact_counts(output_dir)
    qdrant_client = client or create_qdrant_client()
    index_counts = {
        "text_chunks": count_doc_vectors(
            qdrant_client,
            QDRANT_TEXT_COLLECTION,
            doc_id=doc_id,
            source_path=source_path,
        ),
        "image_descriptions": count_doc_vectors(
            qdrant_client,
            QDRANT_IMAGE_COLLECTION,
            doc_id=doc_id,
            source_path=source_path,
        ),
        "table_blocks": count_doc_vectors(
            qdrant_client,
            QDRANT_TABLE_COLLECTION,
            doc_id=doc_id,
            source_path=source_path,
        ),
    }

    text_expected = artifact_counts["text_blocks"]
    image_expected = artifact_counts["image_descriptions"]
    table_expected = artifact_counts["table_blocks"]

    branch_ok = {
        "text": (text_expected == 0 and index_counts["text_chunks"] == 0)
        or (text_expected > 0 and index_counts["text_chunks"] > 0),
        "image": index_counts["image_descriptions"] == image_expected,
        "table": index_counts["table_blocks"] == table_expected,
    }

    if all(branch_ok.values()):
        index_status = "verified"
    elif any(index_counts.values()):
        index_status = "partial"
    else:
        index_status = "missing"

    artifact_warnings: list[str] = []
    missing_image_descriptions = artifact_counts["image_files"] - artifact_counts["image_descriptions"]
    if missing_image_descriptions > 0:
        artifact_warnings.append(
            f"检测到 {missing_image_descriptions} 张图片未生成图片描述，这部分图片不会参与图片检索。"
        )

    branch_fragments = [
        f"文本块 {text_expected} -> 文本索引 {index_counts['text_chunks']}",
        (
            f"图片描述 {image_expected}"
            + (
                f" / 图片文件 {artifact_counts['image_files']}"
                if artifact_counts["image_files"] != image_expected
                else ""
            )
            + f" -> 图片索引 {index_counts['image_descriptions']}"
        ),
        f"表格块 {table_expected} -> 表格索引 {index_counts['table_blocks']}",
    ]
    status_prefix = {
        "verified": "索引校验通过",
        "partial": "索引校验部分异常",
        "missing": "索引校验失败",
    }[index_status]
    index_message = f"{status_prefix}：{'；'.join(branch_fragments)}"

    return {
        "index_status": index_status,
        "index_message": index_message,
        "artifact_warnings": artifact_warnings,
        "artifact_counts": artifact_counts,
        "index_counts": index_counts,
    }



def _run_ingest_job(job_id: str) -> None:
    """为一个排队任务执行上传文件后的重新入库、校验和后端刷新。"""
    job = _get_ingest_job_or_404(job_id)
    upload_path = str(job.get("upload_path") or "")
    ocr_model = str(job.get("ocr_model") or INGEST_DEFAULT_OCR_MODEL)
    workers = int(job.get("workers") or INGEST_DEFAULT_WORKERS)
    doc_id = sanitize_doc_id(upload_path)
    output_dir = os.path.join(INGEST_OUTPUT_ROOT, doc_id)

    _update_ingest_job(
        job_id,
        status="running",
        started_at=_utc_now_iso(),
        doc_id=doc_id,
        output_dir=output_dir,
        error="",
    )
    _append_ingest_job_log(job_id, f"开始摄入 {os.path.basename(upload_path)}")
    _append_ingest_job_log(job_id, f"OCR 模型: {ocr_model} / 并发数: {workers}")

    log_writer = _IngestJobLogWriter(job_id)
    try:
        try:
            # 重新摄入前先清掉同一文档的旧向量，避免新旧结果混在一起。
            client = create_qdrant_client()
            delete_doc_vectors(client, doc_id, source_path=upload_path)
            _append_ingest_job_log(job_id, f"已清理 doc_id={doc_id} 的旧向量。")
        except Exception as exc:
            _append_ingest_job_log(job_id, f"旧向量清理跳过: {exc}")

        args = argparse.Namespace(
            input=upload_path,
            output_dir=INGEST_OUTPUT_ROOT,
            ocr_model=ocr_model,
            workers=workers,
            dpi=INGEST_DPI,
            skip_ocr=False,
        )

        with contextlib.redirect_stdout(log_writer), contextlib.redirect_stderr(log_writer):
            ingest_document(args)
        log_writer.flush()

        verification = _compute_ingest_index_status(doc_id, output_dir, upload_path)
        _update_ingest_job(job_id, **verification)
        _append_ingest_job_log(job_id, verification["index_message"])
        for warning in verification.get("artifact_warnings", []):
            _append_ingest_job_log(job_id, f"注意: {warning}")

        try:
            _refresh_query_backend_after_ingest(job_id)
        except Exception as exc:
            _append_ingest_job_log(job_id, f"问答后端刷新失败: {exc}")

        _update_ingest_job(
            job_id,
            status="succeeded",
            finished_at=_utc_now_iso(),
            error="",
        )
    except Exception as exc:
        log_writer.flush()
        _append_ingest_job_log(job_id, f"摄入失败: {exc}")
        _update_ingest_job(
            job_id,
            status="failed",
            finished_at=_utc_now_iso(),
            error=str(exc),
        )


__all__ = [
    "INGEST_JOBS",
    "_append_ingest_job_log",
    "_compute_ingest_index_status",
    "_get_ingest_job_or_404",
    "_public_ingest_job_payload",
    "_run_ingest_job",
    "_store_ingest_job",
    "_submit_ingest_job",
    "build_queued_ingest_job",
]
