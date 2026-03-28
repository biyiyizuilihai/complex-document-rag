from __future__ import annotations

import argparse
import contextlib
from concurrent.futures import ThreadPoolExecutor
import copy
from datetime import datetime, timezone
import json
import logging
import os
import re
import sys
import threading
import time
import uuid
from functools import lru_cache
from typing import Any, Iterable

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from llama_index.core.schema import QueryBundle

from config import (
    IMAGE_SIMILARITY_TOP_K,
    IMAGE_RETRIEVAL_SCORE_MARGIN,
    IMAGE_RETRIEVAL_SCORE_THRESHOLD,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    RERANK_API_BASE,
    RERANK_API_KEY,
    RERANK_ENABLED,
    RERANK_MODEL,
    RERANK_TIMEOUT_SECONDS,
    TABLE_SIMILARITY_TOP_K,
    TABLE_RETRIEVAL_SCORE_MARGIN,
    TABLE_RETRIEVAL_SCORE_THRESHOLD,
    TEXT_SIMILARITY_TOP_K,
    TEXT_RETRIEVAL_SCORE_MARGIN,
    TEXT_RETRIEVAL_SCORE_THRESHOLD,
    WEB_ANSWER_ENABLE_THINKING,
    WEB_ANSWER_LLM_MODEL,
)
from model_provider_utils import create_text_llm
from complex_document_rag.document_ingestion import sanitize_doc_id
from complex_document_rag.qdrant_management import create_qdrant_client, delete_doc_vectors
from complex_document_rag.reranker import SiliconFlowReranker, rerank_retrieval_bundle
from complex_document_rag.step0_document_ingestion import ingest_document
from complex_document_rag.web_helpers import (
    ARTIFACTS_ROOT,
    PROJECT_ROOT,
    render_answer_markdown_html,
    serialize_answer_sources,
    serialize_retrieval_bundle,
)


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
INGEST_DEFAULT_WORKERS = max(1, int(os.getenv("INGEST_DEFAULT_WORKERS", "4")))
INGEST_MAX_WORKERS = max(INGEST_DEFAULT_WORKERS, int(os.getenv("INGEST_MAX_WORKERS", "12")))
INGEST_DPI = max(72, int(os.getenv("INGEST_DPI", "220")))
INGEST_EXECUTOR = ThreadPoolExecutor(max_workers=2)
INGEST_JOBS: dict[str, dict[str, Any]] = {}
INGEST_JOBS_LOCK = threading.Lock()
RETRIEVAL_RETRIES = 2
RETRIEVAL_RETRY_DELAY_SECONDS = 0.6
ANSWER_ASSET_TOP_K = 5
ANSWER_ASSET_SCORE_THRESHOLD = 0.6
TEXT_CONTEXT_TOP_K = 3
IMAGE_CONTEXT_TOP_K = 2
TABLE_CONTEXT_TOP_K = 2
TEXT_CONTEXT_CHAR_LIMIT = 900
IMAGE_SUMMARY_CHAR_LIMIT = 200
TABLE_SUMMARY_CHAR_LIMIT = 220
TABLE_PREVIEW_ROW_LIMIT = 3
TABLE_PREVIEW_ROW_CHAR_LIMIT = 220
RERANK_CONFIDENCE_FLOOR = 0.2
QUERY_EXPANSION_SCORE_PENALTY = 0.02
RERANKED_BRANCH_KEYS = ("text_results",)
_EMBEDDING_CACHE_MAX = 128

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger("rag.timing")


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="用户问题")
    generate_answer: bool = Field(default=True, description="是否生成最终回答")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _public_ingest_job_payload(job: dict[str, Any]) -> dict[str, Any]:
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
    )
    return {key: copy.deepcopy(job.get(key)) for key in visible_keys}


def _store_ingest_job(job: dict[str, Any]) -> None:
    with INGEST_JOBS_LOCK:
        INGEST_JOBS[job["job_id"]] = job


def _get_ingest_job_or_404(job_id: str) -> dict[str, Any]:
    with INGEST_JOBS_LOCK:
        job = INGEST_JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="摄入任务不存在。")
        return copy.deepcopy(job)


def _update_ingest_job(job_id: str, **fields: Any) -> None:
    with INGEST_JOBS_LOCK:
        job = INGEST_JOBS[job_id]
        job.update(fields)
        job["updated_at"] = _utc_now_iso()


def _append_ingest_job_log(job_id: str, message: str) -> None:
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
    INGEST_EXECUTOR.submit(_run_ingest_job, job_id)


def _refresh_query_backend_after_ingest(job_id: str) -> None:
    cache_clear = getattr(get_query_backend, "cache_clear", None)
    if callable(cache_clear):
        cache_clear()
    get_query_backend()
    _append_ingest_job_log(job_id, "问答后端已刷新，可直接去智能问答页提问。")


def _run_ingest_job(job_id: str) -> None:
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
            client = create_qdrant_client()
            delete_doc_vectors(client, doc_id)
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


def _safe_score(node: Any) -> float:
    return float(getattr(node, "score", 0.0) or 0.0)


IMAGE_REF_PATTERN = re.compile(r"(img(?:_p\d{2,4})?_\d+)", re.IGNORECASE)
NUMBER_PATTERN = re.compile(r"\d+")
ASCII_TERM_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9._/-]*")
CJK_PATTERN = re.compile(r"[\u3400-\u9fff]")

QUERY_TERM_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "MRB": ("Material Review Board",),
    "QSAT": ("Quality Sensitivity Alert Tag",),
    "8D": ("8D report", "8 disciplines"),
    "CAR": ("Corrective Action Request",),
}

QUERY_PHRASE_EXPANSIONS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("触发时机", "触发条件"), ("trigger criteria", "when to initiate")),
    (("流程图", "flow图"), ("flowchart", "process flow")),
    (("流程",), ("flow", "process")),
    (("关键不良",), ("critical defect", "key defect", "non-conformance")),
    (("不良处理",), ("defect handling", "non-conformance handling")),
    (("客户投诉",), ("customer complaint",)),
    (("质量警报",), ("quality alert", "quality sensitivity alert")),
)


def _dedupe_nodes(nodes: Iterable[Any]) -> list[Any]:
    deduped: list[Any] = []
    seen: set[tuple[str, str, str, Any]] = set()
    for node in nodes:
        metadata = getattr(node, "metadata", {}) or {}
        key = (
            metadata.get("type", ""),
            metadata.get("image_id", ""),
            metadata.get("table_id", ""),
            metadata.get("block_id", ""),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(node)
    return deduped


def _contains_cjk(text: str) -> bool:
    return bool(CJK_PATTERN.search(text or ""))


def _append_unique_terms(terms: list[str], values: Iterable[str]) -> None:
    seen = {term.casefold() for term in terms}
    for value in values:
        normalized = str(value).strip()
        if not normalized:
            continue
        folded = normalized.casefold()
        if folded in seen:
            continue
        terms.append(normalized)
        seen.add(folded)


def _extract_image_ref_id(value: str) -> str:
    match = IMAGE_REF_PATTERN.search(value or "")
    return match.group(1) if match else ""


def _truncate_for_prompt(text: str, limit: int) -> str:
    normalized = (text or "").strip()
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[: max(0, limit - 1)].rstrip()}…"


def _extract_table_embedded_image_ids(node: Any) -> set[str]:
    metadata = getattr(node, "metadata", {}) or {}
    raw_table = str(metadata.get("raw_table", "") or "")
    refs = {_extract_image_ref_id(match) for match in IMAGE_REF_PATTERN.findall(raw_table)}
    return {ref for ref in refs if ref}


def _node_embedded_image_id(node: Any) -> str:
    metadata = getattr(node, "metadata", {}) or {}
    candidates = [
        str(metadata.get("image_id", "") or ""),
        str(metadata.get("image_path", "") or ""),
        str(metadata.get("source_image_path", "") or ""),
        str(getattr(node, "text", "") or ""),
    ]
    for candidate in candidates:
        ref = _extract_image_ref_id(candidate)
        if ref:
            return ref
    return ""


def _safe_page_no(node: Any) -> int:
    metadata = getattr(node, "metadata", {}) or {}
    value = metadata.get("page_no")
    try:
        return int(value)
    except (TypeError, ValueError):
        return 10**9


def _clone_query_bundle(query_bundle: QueryBundle) -> QueryBundle:
    embedding = query_bundle.embedding
    return QueryBundle(
        query_str=query_bundle.query_str,
        embedding=list(embedding) if embedding is not None else None,
    )


def _snapshot_retrieval_bundle(retrieval: dict[str, list[Any]]) -> dict[str, list[Any]]:
    return {
        key: [copy.deepcopy(node) for node in nodes]
        for key, nodes in retrieval.items()
    }


def _node_identity(node: Any) -> tuple[str, str]:
    metadata = getattr(node, "metadata", {}) or {}
    node_type = str(metadata.get("type", "") or metadata.get("block_type", "") or "text")
    for field in ("table_id", "image_id", "block_id", "node_id"):
        value = str(metadata.get(field, "") or "")
        if value:
            return (node_type, value)
    fallback = "|".join(
        [
            str(metadata.get("doc_id", "") or ""),
            str(metadata.get("page_no", "") or ""),
            str(getattr(node, "text", "") or "")[:120],
        ]
    )
    return (node_type, fallback)


def _merge_retrieval_bundles(
    retrievals: list[dict[str, list[Any]]],
    *,
    variant_penalties: list[float],
) -> dict[str, list[Any]]:
    merged: dict[str, list[Any]] = {
        "text_results": [],
        "image_results": [],
        "table_results": [],
    }

    for branch_name in merged:
        selected: dict[tuple[str, str], Any] = {}
        for retrieval, penalty in zip(retrievals, variant_penalties):
            for node in retrieval.get(branch_name, []):
                identity = _node_identity(node)
                adjusted_score = max(0.0, _safe_score(node) - penalty)
                current = selected.get(identity)
                if current is not None and _safe_score(current) >= adjusted_score:
                    continue
                snapshot = copy.deepcopy(node)
                snapshot.score = adjusted_score
                selected[identity] = snapshot

        merged[branch_name] = sorted(selected.values(), key=_safe_score, reverse=True)

    return merged


def _build_table_prompt_preview(node: Any) -> str:
    metadata = getattr(node, "metadata", {}) or {}
    semantic_summary = _truncate_for_prompt(
        str(metadata.get("semantic_summary", "") or metadata.get("summary", "") or ""),
        TABLE_SUMMARY_CHAR_LIMIT,
    )
    headers = metadata.get("headers", []) or []
    normalized_text = str(metadata.get("normalized_table_text", "") or "").strip()

    preview_lines: list[str] = []
    if semantic_summary:
        preview_lines.append(f"摘要={semantic_summary}")

    if headers:
        preview_lines.append(
            "列名=" + "|".join(str(header).strip() for header in headers if str(header).strip())
        )

    row_matches = re.findall(r"行\d+=([^；\n]+)", normalized_text)
    for index, row in enumerate(row_matches[:TABLE_PREVIEW_ROW_LIMIT], start=1):
        preview_lines.append(f"示例行{index}={_truncate_for_prompt(row, TABLE_PREVIEW_ROW_CHAR_LIMIT)}")

    if not preview_lines and normalized_text:
        preview_lines.append(_truncate_for_prompt(normalized_text, TABLE_PREVIEW_ROW_CHAR_LIMIT))

    return "\n".join(preview_lines) or "无表格摘要"


def _filter_branch_nodes(
    nodes: Iterable[Any],
    *,
    min_score: float,
    relative_margin: float,
) -> list[Any]:
    ranked = sorted(list(nodes), key=_safe_score, reverse=True)
    if not ranked:
        return []

    above_threshold = [node for node in ranked if _safe_score(node) >= min_score]
    if not above_threshold:
        return []

    top_score = _safe_score(above_threshold[0])
    min_relative_score = top_score - relative_margin
    return [
        node
        for node in above_threshold
        if _safe_score(node) >= min_relative_score
    ]


def _should_fallback_to_raw_branch(nodes: Iterable[Any], *, confidence_floor: float) -> bool:
    ranked = sorted(list(nodes), key=_safe_score, reverse=True)
    if not ranked:
        return True
    return _safe_score(ranked[0]) < confidence_floor


def _node_reference_label(node: Any) -> str:
    metadata = getattr(node, "metadata", {}) or {}
    node_type = metadata.get("type", "")

    if node_type == "table_block":
        return str(
            metadata.get("caption")
            or metadata.get("semantic_summary")
            or metadata.get("summary")
            or metadata.get("table_id")
            or "相关表格"
        )

    if node_type == "image_description":
        return str(
            metadata.get("summary")
            or metadata.get("caption")
            or metadata.get("image_id")
            or "相关图片"
        )

    return str(metadata.get("block_id") or "相关内容")


def _node_sort_reference(node: Any) -> str:
    metadata = getattr(node, "metadata", {}) or {}
    node_type = metadata.get("type", "")
    if node_type == "table_block":
        return str(metadata.get("caption") or metadata.get("table_id") or "")
    if node_type == "image_description":
        return str(metadata.get("summary") or metadata.get("image_id") or "")
    return str(metadata.get("block_id") or "")


def _node_display_order_key(node: Any) -> tuple[int, tuple[int, ...], str]:
    reference = _node_sort_reference(node)
    numbers = tuple(int(value) for value in NUMBER_PATTERN.findall(reference))
    return (_safe_page_no(node), numbers, reference)


def _sort_nodes_for_display(nodes: Iterable[Any]) -> list[Any]:
    node_list = list(nodes)
    if any(_safe_page_no(node) < 10**9 for node in node_list):
        return sorted(node_list, key=_node_display_order_key)
    return node_list


def _answer_asset_order_key(node: Any) -> tuple[int, int, tuple[int, ...], str]:
    metadata = getattr(node, "metadata", {}) or {}
    node_type = metadata.get("type", "")
    if node_type == "table_block":
        type_priority = 0
    elif node_type == "image_description":
        type_priority = 1
    else:
        type_priority = 2

    page_no, numbers, reference = _node_display_order_key(node)
    return (type_priority, page_no, numbers, reference)


def _sort_answer_assets(nodes: Iterable[Any]) -> list[Any]:
    return sorted(list(nodes), key=_answer_asset_order_key)


class QueryBackend:
    def __init__(self) -> None:
        from complex_document_rag.step4_basic_query import load_indexes

        self.text_index, self.image_index, self.table_index = load_indexes()
        self.embed_model = next(
            (
                getattr(index, "_embed_model", None)
                for index in (self.text_index, self.image_index, self.table_index)
                if index is not None and getattr(index, "_embed_model", None) is not None
            ),
            None,
        )
        self.llm = create_text_llm(
            model_name=WEB_ANSWER_LLM_MODEL,
            api_key=OPENAI_API_KEY,
            api_base=OPENAI_BASE_URL,
            disable_thinking=not WEB_ANSWER_ENABLE_THINKING,
        )
        self.reranker = self._build_reranker()
        self._embedding_cache: dict[str, list[float]] = {}

    def _build_reranker(self) -> SiliconFlowReranker | None:
        if not RERANK_ENABLED or not RERANK_API_KEY:
            return None
        return SiliconFlowReranker(
            api_key=RERANK_API_KEY,
            api_base=RERANK_API_BASE,
            model_name=RERANK_MODEL,
            top_n=max(TEXT_SIMILARITY_TOP_K, IMAGE_SIMILARITY_TOP_K, TABLE_SIMILARITY_TOP_K),
            timeout=RERANK_TIMEOUT_SECONDS,
        )

    def _ensure_embedding_cache(self) -> dict[str, list[float]]:
        cache = getattr(self, "_embedding_cache", None)
        if cache is None:
            cache = {}
            self._embedding_cache = cache
        return cache

    def _get_cached_embedding(self, query: str) -> list[float] | None:
        if self.embed_model is None:
            return None
        cache = self._ensure_embedding_cache()
        if query not in cache:
            if len(cache) >= _EMBEDDING_CACHE_MAX:
                cache.pop(next(iter(cache)))
            cache[query] = self.embed_model.get_query_embedding(query)
        return cache[query]

    def _prefetch_embeddings(self, queries: list[str]) -> None:
        """多个 query variant 的 embedding 并行预取，结果写入缓存。"""
        if self.embed_model is None:
            return
        cache = self._ensure_embedding_cache()
        missing = [q for q in queries if q not in cache]
        if len(missing) <= 1:
            for q in missing:
                self._get_cached_embedding(q)
            return
        with ThreadPoolExecutor(max_workers=len(missing)) as executor:
            futures = [(q, executor.submit(self.embed_model.get_query_embedding, q)) for q in missing]
        for q, fut in futures:
            if len(cache) >= _EMBEDDING_CACHE_MAX:
                cache.pop(next(iter(cache)))
            cache[q] = fut.result()

    def _build_query_bundle(self, query: str) -> QueryBundle:
        return QueryBundle(query_str=query, embedding=self._get_cached_embedding(query))

    def build_query_variants(self, query: str) -> list[str]:
        normalized_query = (query or "").strip()
        if not normalized_query or not _contains_cjk(normalized_query):
            return [normalized_query]

        english_terms: list[str] = []
        ascii_terms = ASCII_TERM_PATTERN.findall(normalized_query)
        uppercase_terms = {term.upper() for term in ascii_terms}

        for acronym in sorted(uppercase_terms):
            if acronym in QUERY_TERM_EXPANSIONS:
                _append_unique_terms(english_terms, QUERY_TERM_EXPANSIONS[acronym])

        matched_phrase = False
        for aliases, expansions in QUERY_PHRASE_EXPANSIONS:
            if any(alias in normalized_query for alias in aliases):
                matched_phrase = True
                _append_unique_terms(english_terms, expansions)

        should_expand = bool(english_terms) and (matched_phrase or bool(uppercase_terms))
        if not should_expand:
            return [normalized_query]

        mixed_query = " ".join([normalized_query, *english_terms]).strip()
        if mixed_query == normalized_query:
            return [normalized_query]
        return [normalized_query, mixed_query]

    def _retrieve_branch(
        self,
        index: Any,
        *,
        similarity_top_k: int,
        query_bundle: QueryBundle,
    ) -> list[Any]:
        if index is None:
            return []
        retriever = index.as_retriever(similarity_top_k=similarity_top_k)
        return retriever.retrieve(_clone_query_bundle(query_bundle))

    def _retrieve_once(self, query: str) -> dict[str, list[Any]]:
        t_total = time.perf_counter()

        query_variants = self.build_query_variants(query)
        LOGGER.info("[query   ] variants=%d  %s", len(query_variants), query_variants)

        branch_specs = [
            ("text_results", self.text_index, TEXT_SIMILARITY_TOP_K),
            ("image_results", self.image_index, IMAGE_SIMILARITY_TOP_K),
            ("table_results", self.table_index, TABLE_SIMILARITY_TOP_K),
        ]
        active_branches = [spec for spec in branch_specs if spec[1] is not None]

        t_embed = time.perf_counter()
        self._prefetch_embeddings(query_variants)
        LOGGER.info("[embed   ] %5.0fms  queries=%d", (time.perf_counter() - t_embed) * 1000, len(query_variants))

        variant_retrievals: list[dict[str, list[Any]]] = []
        variant_penalties: list[float] = []

        for variant_index, variant_query in enumerate(query_variants):
            query_bundle = self._build_query_bundle(variant_query)
            variant_result: dict[str, list[Any]] = {
                "text_results": [],
                "image_results": [],
                "table_results": [],
            }

            t_qdrant = time.perf_counter()
            if len(active_branches) <= 1:
                for key, index, top_k in active_branches:
                    variant_result[key] = self._retrieve_branch(
                        index,
                        similarity_top_k=top_k,
                        query_bundle=query_bundle,
                    )
            else:
                with ThreadPoolExecutor(max_workers=len(active_branches)) as executor:
                    future_map = {
                        executor.submit(
                            self._retrieve_branch,
                            index,
                            similarity_top_k=top_k,
                            query_bundle=query_bundle,
                        ): key
                        for key, index, top_k in active_branches
                    }
                    for future, key in future_map.items():
                        variant_result[key] = future.result()
            LOGGER.info(
                "[qdrant  ] %5.0fms  variant=%d  text=%d img=%d tbl=%d",
                (time.perf_counter() - t_qdrant) * 1000,
                variant_index,
                len(variant_result["text_results"]),
                len(variant_result["image_results"]),
                len(variant_result["table_results"]),
            )

            variant_retrievals.append(variant_result)
            variant_penalties.append(variant_index * QUERY_EXPANSION_SCORE_PENALTY)

        retrieval = _merge_retrieval_bundles(
            variant_retrievals,
            variant_penalties=variant_penalties,
        )
        LOGGER.info(
            "[merge   ]  merged  text=%d img=%d tbl=%d",
            len(retrieval.get("text_results", [])),
            len(retrieval.get("image_results", [])),
            len(retrieval.get("table_results", [])),
        )

        raw_snapshot = _snapshot_retrieval_bundle(retrieval)
        reranked_text_only = {
            "text_results": retrieval.get("text_results", []),
            "image_results": [],
            "table_results": [],
        }
        n_before_rerank = len(reranked_text_only["text_results"])
        t_rerank = time.perf_counter()
        reranked_text_only = rerank_retrieval_bundle(
            query=query,
            retrieval=reranked_text_only,
            reranker=self.reranker,
            top_n_map={
                "text_results": TEXT_SIMILARITY_TOP_K,
            },
        )
        LOGGER.info(
            "[rerank  ] %5.0fms  text %d→%d",
            (time.perf_counter() - t_rerank) * 1000,
            n_before_rerank,
            len(reranked_text_only.get("text_results", [])),
        )

        merged_after_rerank = {
            "text_results": reranked_text_only.get("text_results", []),
            "image_results": raw_snapshot.get("image_results", []),
            "table_results": raw_snapshot.get("table_results", []),
        }
        result = self.filter_retrieval(
            merged_after_rerank,
            raw_retrieval=raw_snapshot,
            reranked_branches=(
                set(RERANKED_BRANCH_KEYS)
                if self.reranker is not None and getattr(self.reranker, "enabled", True)
                else set()
            ),
        )
        LOGGER.info(
            "[filter  ]  after filter  text=%d img=%d tbl=%d",
            len(result.get("text_results", [])),
            len(result.get("image_results", [])),
            len(result.get("table_results", [])),
        )
        LOGGER.info("[retrieve] %5.0fms  total", (time.perf_counter() - t_total) * 1000)
        return result

    def filter_retrieval(
        self,
        retrieval: dict[str, list[Any]],
        *,
        raw_retrieval: dict[str, list[Any]] | None = None,
        reranked_branches: set[str] | None = None,
    ) -> dict[str, list[Any]]:
        reranked_branches = (
            set(reranked_branches)
            if reranked_branches is not None
            else (
                {"text_results", "image_results", "table_results"}
                if self.reranker is not None and getattr(self.reranker, "enabled", True)
                else set()
            )
        )
        branch_rules = {
            "text_results": (TEXT_RETRIEVAL_SCORE_THRESHOLD, TEXT_RETRIEVAL_SCORE_MARGIN),
            "image_results": (IMAGE_RETRIEVAL_SCORE_THRESHOLD, IMAGE_RETRIEVAL_SCORE_MARGIN),
            "table_results": (TABLE_RETRIEVAL_SCORE_THRESHOLD, TABLE_RETRIEVAL_SCORE_MARGIN),
        }
        filtered: dict[str, list[Any]] = {}
        for key, (threshold, margin) in branch_rules.items():
            reranked_nodes = retrieval.get(key, [])
            branch_nodes = reranked_nodes
            use_absolute_threshold = key in reranked_branches
            branch_min_score = threshold if use_absolute_threshold else 0.0

            if use_absolute_threshold and raw_retrieval is not None and _should_fallback_to_raw_branch(
                reranked_nodes,
                confidence_floor=RERANK_CONFIDENCE_FLOOR,
            ):
                branch_nodes = raw_retrieval.get(key, [])
                branch_min_score = 0.0

            filtered[key] = _filter_branch_nodes(
                branch_nodes,
                min_score=branch_min_score,
                relative_margin=margin,
            )

        return filtered

    def retrieve(self, query: str) -> dict[str, list[Any]]:
        last_error: Exception | None = None
        for attempt in range(RETRIEVAL_RETRIES + 1):
            try:
                return self._retrieve_once(query)
            except Exception as exc:
                last_error = exc
                if attempt >= RETRIEVAL_RETRIES:
                    break
                time.sleep(RETRIEVAL_RETRY_DELAY_SECONDS * (attempt + 1))

        raise RuntimeError(str(last_error) if last_error else "检索失败")

    def select_answer_assets(self, retrieval: dict[str, list[Any]]) -> list[Any]:
        candidates = sorted(
            list(retrieval.get("image_results", [])) + list(retrieval.get("table_results", [])),
            key=_safe_score,
            reverse=True,
        )
        filtered = [node for node in candidates if _safe_score(node) >= ANSWER_ASSET_SCORE_THRESHOLD]

        selected: list[Any] = []
        selected_table_image_refs: set[str] = set()

        for node in filtered:
            metadata = getattr(node, "metadata", {}) or {}
            node_type = metadata.get("type", "")

            if node_type == "image_description":
                image_ref = _node_embedded_image_id(node)
                if image_ref and image_ref in selected_table_image_refs:
                    continue
                selected.append(node)
            elif node_type == "table_block":
                selected.append(node)
                selected_table_image_refs.update(_extract_table_embedded_image_ids(node))
                if selected_table_image_refs:
                    selected = [
                        item
                        for item in selected
                        if (
                            (getattr(item, "metadata", {}) or {}).get("type", "") != "image_description"
                            or _node_embedded_image_id(item) not in selected_table_image_refs
                        )
                    ]
                    if node not in selected:
                        selected.append(node)
            else:
                selected.append(node)

            selected = _dedupe_nodes(selected)
            if len(selected) > ANSWER_ASSET_TOP_K:
                selected = selected[:ANSWER_ASSET_TOP_K]

        return _sort_answer_assets(selected[:ANSWER_ASSET_TOP_K])

    def select_answer_sources(self, retrieval: dict[str, list[Any]], answer_assets: list[Any]) -> list[Any]:
        text_results = sorted(retrieval.get("text_results", []), key=_safe_score, reverse=True)
        candidate_sources = text_results[:2] + answer_assets
        return _sort_nodes_for_display(_dedupe_nodes(candidate_sources))

    def build_answer_prompt(
        self,
        query: str,
        retrieval: dict[str, list[Any]],
        answer_assets: list[Any],
    ) -> str:
        text_context = []
        for index, node in enumerate(retrieval.get("text_results", [])[:TEXT_CONTEXT_TOP_K], start=1):
            metadata = getattr(node, "metadata", {}) or {}
            text_context.append(
                f"[文本{index}] 页码={metadata.get('page_label', metadata.get('page_no', '-'))}\n"
                f"{_truncate_for_prompt(str(getattr(node, 'text', '') or ''), TEXT_CONTEXT_CHAR_LIMIT)}"
            )

        image_context = []
        for index, node in enumerate(
            _sort_nodes_for_display(retrieval.get("image_results", [])[:IMAGE_CONTEXT_TOP_K]),
            start=1,
        ):
            metadata = getattr(node, "metadata", {}) or {}
            image_context.append(
                f"[图片{index}] 名称={_node_reference_label(node)} 页码={metadata.get('page_label', metadata.get('page_no', '-'))} "
                f"摘要={_truncate_for_prompt(str(metadata.get('summary', '') or ''), IMAGE_SUMMARY_CHAR_LIMIT)}"
            )

        table_context = []
        for index, node in enumerate(
            _sort_nodes_for_display(retrieval.get("table_results", [])[:TABLE_CONTEXT_TOP_K]),
            start=1,
        ):
            metadata = getattr(node, "metadata", {}) or {}
            table_context.append(
                f"[表格{index}] 标题={_node_reference_label(node)} 页码={metadata.get('page_label', metadata.get('page_no', '-'))}\n"
                f"{_build_table_prompt_preview(node)}"
            )

        answer_asset_ids = []
        for node in _sort_nodes_for_display(answer_assets):
            metadata = getattr(node, "metadata", {}) or {}
            if metadata.get("image_id"):
                answer_asset_ids.append(f"图片:{_node_reference_label(node)}")
            if metadata.get("table_id"):
                answer_asset_ids.append(f"表格:{_node_reference_label(node)}")

        return (
            "你是一个严谨的 RAG 问答助手。请严格基于给定上下文回答，不要编造。\n"
            "如果证据不足，请明确说证据不足。\n"
            "如果输出思考过程，也请全程使用中文，不要输出英文小标题或英文分析。\n"
            "如果下面列出的图片或表格与问题直接相关，请在回答中自然提及“见相关图片/见相关表格”。\n"
            "引用图表时只使用中文标题或名称，不要输出任何内部 ID、文件名或技术标识。\n"
            f"推荐随答案一起返回的素材：{', '.join(answer_asset_ids) if answer_asset_ids else '无'}\n\n"
            f"问题：{query}\n\n"
            "文本证据：\n"
            f"{chr(10).join(text_context) or '无'}\n\n"
            "图片证据：\n"
            f"{chr(10).join(image_context) or '无'}\n\n"
            "表格证据：\n"
            f"{chr(10).join(table_context) or '无'}\n\n"
            "请输出简洁、直接的中文回答。"
        )

    def answer(self, query: str, retrieval: dict[str, list[Any]] | None = None) -> dict[str, Any]:
        retrieval = retrieval or self.retrieve(query)
        answer_assets = self.select_answer_assets(retrieval)
        answer_sources = self.select_answer_sources(retrieval, answer_assets)
        prompt = self.build_answer_prompt(query, retrieval, answer_assets)
        response = self.llm.complete(prompt)
        return {
            "answer": response.text or "",
            "answer_sources": answer_sources,
            "answer_assets": answer_assets,
        }

    def stream_answer(self, query: str, retrieval: dict[str, list[Any]] | None = None):
        retrieval = retrieval or self.retrieve(query)
        answer_assets = self.select_answer_assets(retrieval)
        answer_sources = self.select_answer_sources(retrieval, answer_assets)
        prompt = self.build_answer_prompt(query, retrieval, answer_assets)
        return {
            "stream": self.llm.stream_complete(prompt),
            "answer_sources": answer_sources,
            "answer_assets": answer_assets,
        }


@lru_cache(maxsize=1)
def get_query_backend() -> QueryBackend:
    return QueryBackend()


def load_default_queries() -> list[str]:
    from complex_document_rag.step4_basic_query import get_default_test_queries

    return get_default_test_queries()


def _empty_retrieval_bundle() -> dict[str, list[Any]]:
    return {
        "text_results": [],
        "image_results": [],
        "table_results": [],
    }


def _serialize_answer_result(answer_result: Any) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    if isinstance(answer_result, dict):
        answer = answer_result.get("answer", "") or ""
        answer_sources = serialize_answer_sources(answer_result.get("answer_sources", []))
        answer_assets = serialize_answer_sources(_sort_answer_assets(answer_result.get("answer_assets", [])))
        return answer, answer_sources, answer_assets

    answer = getattr(answer_result, "response", "") or ""
    raw_sources = getattr(answer_result, "source_nodes", []) or []
    answer_sources = serialize_answer_sources(raw_sources)
    return answer, answer_sources, answer_sources


def _sse_event(name: str, payload: dict[str, Any]) -> str:
    return f"event: {name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def create_app() -> FastAPI:
    app = FastAPI(
        title="复杂文档 RAG 解析",
        description="用于本地验证文本、图片、表格召回效果的轻量前端",
        version="0.2.0",
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

    @app.on_event("startup")
    def warm_query_backend() -> None:
        t_start = time.perf_counter()
        try:
            get_query_backend()
        except Exception:
            LOGGER.exception("[warmup  ] failed to initialize query backend")
            return
        LOGGER.info("[warmup  ] %5.0fms  query backend ready", (time.perf_counter() - t_start) * 1000)

    @app.get("/")
    def index() -> FileResponse:
        index_path = os.path.join(WEB_STATIC_DIR, "index.html")
        if not os.path.exists(index_path):
            raise HTTPException(status_code=404, detail="前端页面未生成。")
        return FileResponse(
            index_path,
            headers={"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache"},
        )

    @app.get("/ingest")
    def ingest_page() -> FileResponse:
        if not os.path.exists(INGEST_STATIC_PATH):
            raise HTTPException(status_code=404, detail="摄入页面未生成。")
        return FileResponse(
            INGEST_STATIC_PATH,
            headers={"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache"},
        )

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/default-queries")
    def default_queries() -> dict[str, list[str]]:
        return {"queries": load_default_queries()}

    @app.get("/api/ingest/options")
    def ingest_options() -> dict[str, Any]:
        return {
            "ocr_models": list(INGEST_OCR_MODEL_OPTIONS),
            "default_ocr_model": INGEST_DEFAULT_OCR_MODEL,
            "default_workers": INGEST_DEFAULT_WORKERS,
            "min_workers": 1,
            "max_workers": INGEST_MAX_WORKERS,
        }

    @app.post("/api/ingest/jobs", status_code=202)
    async def create_ingest_job(
        file: UploadFile = File(...),
        ocr_model: str = Form(INGEST_DEFAULT_OCR_MODEL),
        workers: int = Form(INGEST_DEFAULT_WORKERS),
    ) -> dict[str, Any]:
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

        job = {
            "job_id": job_id,
            "status": "queued",
            "filename": filename,
            "ocr_model": normalized_model,
            "workers": workers,
            "created_at": _utc_now_iso(),
            "updated_at": _utc_now_iso(),
            "started_at": "",
            "finished_at": "",
            "logs": [],
            "error": "",
            "doc_id": "",
            "output_dir": "",
            "upload_path": upload_path,
        }
        _store_ingest_job(job)
        _append_ingest_job_log(job_id, f"文件已上传：{filename}")
        _submit_ingest_job(job_id)
        return _public_ingest_job_payload(job)

    @app.get("/api/ingest/jobs/{job_id}")
    def ingest_job_status(job_id: str) -> dict[str, Any]:
        return _public_ingest_job_payload(_get_ingest_job_or_404(job_id))

    @app.post("/api/query")
    def query(payload: QueryRequest) -> dict[str, Any]:
        backend = get_query_backend()
        retrieval_error = ""
        retrieval = _empty_retrieval_bundle()
        try:
            retrieval = backend.retrieve(payload.query)
        except Exception as exc:
            retrieval_error = str(exc)

        answer = ""
        answer_error = ""
        answer_sources: list[dict[str, Any]] = []
        answer_assets: list[dict[str, Any]] = []

        if payload.generate_answer and not retrieval_error:
            try:
                answer_result = backend.answer(payload.query, retrieval=retrieval)
            except Exception as exc:  # pragma: no cover - exercised via tests with fake backend
                answer_error = str(exc)
            else:
                answer, answer_sources, answer_assets = _serialize_answer_result(answer_result)

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

    @app.post("/api/query/stream")
    def stream_query(payload: QueryRequest) -> StreamingResponse:
        backend = get_query_backend()

        def event_stream():
            t_request = time.perf_counter()
            retrieval_error = ""
            retrieval = _empty_retrieval_bundle()
            try:
                retrieval = backend.retrieve(payload.query)
            except Exception as exc:
                retrieval_error = str(exc)

            if retrieval_error:
                yield _sse_event(
                    "retrieval",
                    {
                        "retrieval": serialize_retrieval_bundle(retrieval),
                        "retrieval_error": retrieval_error,
                        "answer_assets": [],
                        "answer_sources": [],
                    },
                )
                yield _sse_event("error", {"message": retrieval_error})
                yield _sse_event(
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
                yield _sse_event(
                    "retrieval",
                    {
                        "retrieval": serialize_retrieval_bundle(retrieval),
                        "retrieval_error": "",
                        "answer_assets": [],
                        "answer_sources": [],
                    },
                )
                yield _sse_event(
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
                stream_result = backend.stream_answer(payload.query, retrieval=retrieval)
            except Exception as exc:
                yield _sse_event(
                    "retrieval",
                    {
                        "retrieval": serialize_retrieval_bundle(retrieval),
                        "retrieval_error": "",
                        "answer_assets": [],
                        "answer_sources": [],
                    },
                )
                yield _sse_event("error", {"message": str(exc)})
                yield _sse_event(
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
                    backend.select_answer_assets(retrieval)
                    if hasattr(backend, "select_answer_assets")
                    else []
                )
                raw_answer_sources = (
                    backend.select_answer_sources(retrieval, raw_answer_assets)
                    if hasattr(backend, "select_answer_sources")
                    else []
                )

            answer_sources = serialize_answer_sources(raw_answer_sources)
            answer_assets = serialize_answer_sources(_sort_answer_assets(raw_answer_assets))
            yield _sse_event(
                "retrieval",
                {
                    "retrieval": serialize_retrieval_bundle(retrieval),
                    "retrieval_error": "",
                    "answer_assets": answer_assets,
                    "answer_sources": answer_sources,
                },
            )

            answer_text = ""
            reasoning_text = ""
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
                        reasoning_text += reasoning_delta
                        yield _sse_event("reasoning", {"delta": reasoning_delta})
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
                    yield _sse_event("chunk", {"delta": delta})
            except Exception as exc:
                yield _sse_event("error", {"message": str(exc)})
                yield _sse_event(
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
            yield _sse_event(
                "done",
                {
                    "answer": answer_text,
                    "answer_html": render_answer_markdown_html(answer_text),
                    "answer_assets": answer_assets,
                    "answer_sources": answer_sources,
                },
            )

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "complex_document_rag.web_app:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )
