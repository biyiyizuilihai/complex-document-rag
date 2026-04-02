# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Setup
```bash
# Python 3.10+ required
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # edit with API keys
docker run -p 6333:6333 qdrant/qdrant
```

### Running the Application
```bash
python -m complex_document_rag serve
# Q&A: http://127.0.0.1:8000/   Upload UI: http://127.0.0.1:8000/ingest
```

### Document Ingestion (CLI)
```bash
python -m complex_document_rag ingest --input "/path/to/file.pdf"
# Options: --ocr-model qwen3.5-plus --workers 4 --dpi 220 --skip-ocr
```

### Other CLI Commands
```bash
python -m complex_document_rag index-text /path/to/docs
python -m complex_document_rag index-images /path/to/image_descriptions.json
python -m complex_document_rag index-tables /path/to/table_blocks.json
python -m complex_document_rag query --query "..."
python -m complex_document_rag qdrant list
python -m complex_document_rag qdrant delete-doc --doc-id doc_foo
python -m complex_document_rag cleanup upload-jobs --keep 0
```

### Testing
```bash
python -m unittest discover -s tests -v
python -m unittest tests.test_web_backend -v
python -m unittest tests.test_web_backend.QueryBackendTests.test_build_query_variants_adds_bilingual_hints_for_cross_lingual_query
python -m unittest tests.test_web_query_http tests.test_web_query_stream -v
```

## Architecture

### High-Level Flow

**Ingestion:**
```
PDF → scripts/batch_ocr.py → ingestion/pipeline.py → ingestion/* (parse) → indexing/* → Qdrant
```

**Query:**
```
User query → web/backend.py:QueryBackend → parallel retrieval (3 collections) → optional reranking → LLM answer
```

### Three Qdrant Collections

- `text_chunks` — text blocks (`indexing/text_index.py`)
- `image_descriptions` — LLM-generated image descriptions (`indexing/image_index.py`)
- `table_blocks` — structured tables (`indexing/table_index.py`)

### Package Map

| Package | Key Modules | Role |
|---------|-------------|------|
| `core/` | `config.py`, `models.py`, `paths.py`, `cleanup.py` | Central config, shared data models, path utilities |
| `ingestion/` | `pipeline.py`, `artifacts.py`, `tables.py`, `images.py`, `ocr_layout.py` | PDF ingestion orchestration and parsing |
| `indexing/` | `qdrant.py`, `text_index.py`, `image_index.py`, `table_index.py` | Vector storage and collection management |
| `retrieval/` | `query_console.py`, `reranking.py`, `fusion.py` | CLI retrieval, SiliconFlow reranking, multi-index fusion |
| `providers/` | `llms.py`, `embeddings.py`, `openai_compatible.py` | LLM/embedding factories; DashScope vs OpenAI auto-detection |
| `web/` | `routes.py`, `backend.py`, `jobs.py`, `query_http.py`, `query_stream.py`, `ingest_routes.py`, `helpers.py`, `schemas.py`, `settings.py` | FastAPI app, retrieval orchestration, SSE streaming, job queue |

### `QueryBackend` (`web/backend.py`)

Central retrieval class. Key methods:
- `_retrieve_once()` — parallel retrieval across 3 branches via `ThreadPoolExecutor`
- `_retrieve_branch()` — per-collection retrieval with LRU embedding cache (max 128)
- `build_query_variants()` — bilingual hints for cross-lingual queries
- `filter_retrieval_results()` — score thresholds + relative score margins
- `select_answer_assets()` — optional LLM judge for relevant images/tables
- `_apply_evidence_fallback()` — top-2 fallback when all evidence slots are empty
- `_build_answer_prompt()` — context assembly with history truncation
- `answer()` / `stream_answer()` — LLM answer generation (sync / SSE)

### Web API

| Endpoint | Purpose |
|----------|---------|
| `GET /` | Q&A interface |
| `GET /ingest` | Upload interface |
| `GET /api/health` | Health check |
| `POST /api/query` | Synchronous query (JSON) |
| `POST /api/query/stream` | Streaming SSE query |
| `POST /api/ingest/jobs` | Submit ingestion job (202 Accepted) |
| `GET /api/ingest/jobs/{job_id}` | Job status |

Streaming SSE sends events in order: `retrieval` → `reasoning` (if thinking enabled) → `answer` chunks → `done`.

### Ingestion Pipeline (`ingestion/pipeline.py`)

`ingest_document(args)` steps:
1. Run `scripts/batch_ocr.py` (or `--skip-ocr`)
2. `collect_pdf_ocr_output()` → text/image/table blocks
3. `materialize_missing_pdf_region_images()` → region PNGs from PDF
4. `build_pdf_ocr_image_descriptions()` + `summarize_table_blocks()` → LLM-generated descriptions
5. Write `manifest.json`, `image_descriptions.json`, `table_blocks.json` to `ingestion_output/{doc_id}/`
6. Index all three collections in Qdrant

### Configuration

All via `.env` (canonical source: `complex_document_rag/core/config.py`; root `config.py` re-exports for backward compat).

**Key variables:**
```bash
OPENAI_API_KEY=...
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
MULTIMODAL_LLM_MODEL=qwen3.5-flash
TEXT_LLM_MODEL=qwen3.5-flash
EMBEDDING_MODEL=text-embedding-v3
QDRANT_HOST=localhost
QDRANT_PORT=6333
WEB_ANSWER_ENABLE_THINKING=false
WEB_ANSWER_THINKING_BUDGET_TOKENS=8000   # 0 = unlimited; only active when thinking enabled
RERANK_ENABLED=true
RERANK_API_KEY=...                        # or SILICONFLOW_API_KEY
RERANK_MODEL=BAAI/bge-reranker-v2-m3
ASSET_JUDGE_ENABLED=true
ASSET_JUDGE_MODEL=qwen3.5-flash
HISTORY_TURN_LIMIT=6
```

Retrieval thresholds (`TEXT/IMAGE/TABLE_RETRIEVAL_SCORE_THRESHOLD`, `*_SCORE_MARGIN`, `*_SIMILARITY_TOP_K`, `FINAL_TOP_K`) are in `core/config.py`.

### Frontend

Two static HTML files in `complex_document_rag/web_static/` (no build step). Multi-turn chat is in-memory only (not persisted). Artifact files served from disk via `/artifacts/` prefix.
