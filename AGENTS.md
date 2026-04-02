# AGENTS.md

This file provides guidance to Codex when working with code in this repository.

## Commands

### Running the Application
```bash
# Start web service (Q&A + ingestion UI)
python -m complex_document_rag serve
# Access: http://127.0.0.1:8000/ (Q&A), http://127.0.0.1:8000/ingest

# Start required Qdrant vector database
docker run -p 6333:6333 qdrant/qdrant
```

### Document Ingestion (CLI)
```bash
python -m complex_document_rag ingest --input "/path/to/file.pdf"
# With options: --ocr-model qwen3.5-plus --workers 4 --dpi 220 --skip-ocr
```

### Other Useful CLI Commands
```bash
# Build indexes directly
python -m complex_document_rag index-text /path/to/docs
python -m complex_document_rag index-images /path/to/image_descriptions.json
python -m complex_document_rag index-tables /path/to/table_blocks.json

# Query from CLI
python -m complex_document_rag query --query "苯相关作业场所应设置哪些警示标识？"
python -m complex_document_rag query --retrieval-only

# Manage Qdrant collections
python -m complex_document_rag qdrant list
python -m complex_document_rag qdrant delete-doc --doc-id <doc_id>
python -m complex_document_rag qdrant drop-all

# Clean historical upload job directories
python -m complex_document_rag cleanup upload-jobs --keep 0
```

### Testing
```bash
# Run all tests in a module
python -m unittest discover -s tests -p 'test_web_backend.py'

# Run a specific test class
python -m unittest tests.test_web_query_http.WebQueryHttpTests

# Run a specific test method
python -m unittest tests.test_web_backend.QueryBackendTests.test_build_query_variants_adds_bilingual_hints_for_cross_lingual_query

# Run all tests with verbose output
python -m unittest discover -s tests -v
```

### Setup
```bash
# Python 3.10+ required; if local python3 is 3.9, use `conda activate test`
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # then edit with API keys
```

## Architecture

### High-Level Flow

**Ingestion Pipeline:**
```
PDF → scripts/batch_ocr.py (OCR) → ingestion/pipeline.py (orchestrate) → ingestion/* (parse/extract) → indexing/* → Qdrant
```

**Query Pipeline:**
```
User query → web/backend.py:QueryBackend → parallel retrieval from 3 Qdrant collections → optional reranking → LLM answer generation
```

### Three Qdrant Collections

All document content is stored in three separate vector collections:
- `text_chunks` — Document text blocks (indexed by `complex_document_rag/indexing/text_index.py`)
- `image_descriptions` — LLM-generated image/diagram descriptions (indexed by `complex_document_rag/indexing/image_index.py`)
- `table_blocks` — Extracted and structured tables (indexed by `complex_document_rag/indexing/table_index.py`)

### Key Modules

| Module | Role |
|--------|------|
| `complex_document_rag/cli.py` | Unified CLI entrypoint for serve/ingest/index/query/qdrant |
| `complex_document_rag/web/routes.py` | FastAPI app composition, router registration, lifespan warmup |
| `complex_document_rag/web/backend.py` | `QueryBackend`, retrieval orchestration, answer generation, SSE helpers |
| `complex_document_rag/web/helpers.py` | Serialization, artifact URL mapping, markdown→HTML |
| `complex_document_rag/web/jobs.py` | Async ingestion job queue and verification helpers |
| `complex_document_rag/ingestion/pipeline.py` | Canonical PDF ingestion pipeline |
| `complex_document_rag/ingestion/artifacts.py` | OCR artifact collection, manifest writing, filesystem helpers |
| `complex_document_rag/ingestion/tables.py` | Table parsing, logical-table merge, normalized table blocks |
| `complex_document_rag/ingestion/images.py` | Image description payload construction |
| `complex_document_rag/indexing/qdrant.py` | Collection management and doc-level deletion/count helpers |
| `complex_document_rag/retrieval/query_console.py` | CLI query/retrieval helpers |
| `complex_document_rag/retrieval/reranking.py` | Optional SiliconFlow reranking |
| `complex_document_rag/core/config.py` | Environment-driven configuration |
| `complex_document_rag/providers/` | OpenAI-compatible model factories, embeddings, text/multimodal LLM adapters |

### `QueryBackend` (core of `web/backend.py`)

The central retrieval class. Key methods:
- `_retrieve_once()` — Orchestrates parallel retrieval across all 3 branches using `ThreadPoolExecutor`
- `_retrieve_branch()` — Per-collection retrieval with embedding caching (LRU, max 128)
- `build_query_variants()` — Adds bilingual hints for cross-lingual queries
- `filter_retrieval_results()` — Applies score thresholds and relative score margins
- `select_answer_assets()` — Optional LLM judge to pick relevant images/tables
- `_build_answer_prompt()` — Assembles context with history truncation
- `_apply_evidence_fallback()` — Uses filtered image/table evidence when answer assets would otherwise be empty

### Web API Endpoints

- `GET /` → Q&A interface
- `GET /ingest` → Document upload interface
- `POST /api/query` → Synchronous query
- `POST /api/query/stream` → Streaming SSE query
- `POST /api/ingest/jobs` → Submit ingestion job
- `GET /api/ingest/jobs/{job_id}` → Job status

### Configuration

All behavior is controlled via `.env`. Key variables:

```bash
# Required
OPENAI_API_KEY=...           # Or DASHSCOPE_API_KEY
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Current model setup (Qwen via DashScope)
MULTIMODAL_LLM_MODEL=qwen3.5-flash  # For image descriptions + multimodal answers
TEXT_LLM_MODEL=qwen3.5-flash        # For lightweight text tasks
EMBEDDING_MODEL=text-embedding-v3

# DashScope/Qwen compatibility
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
WEB_ANSWER_ENABLE_THINKING=false

# Optional reranking via SiliconFlow
RERANK_ENABLED=true
SILICONFLOW_API_KEY=...
RERANK_MODEL=BAAI/bge-reranker-v2-m3
```

Retrieval thresholds (`TEXT_RETRIEVAL_SCORE_THRESHOLD`, `IMAGE_RETRIEVAL_SCORE_THRESHOLD`, etc.) and top-K values are all tunable via env vars — see `complex_document_rag/core/config.py` for the full list.

### Vector Node Metadata Schema

Each node stored in Qdrant carries metadata including: `doc_id`, `source_path`, `page_no`, `block_type` (`text`/`table`/`image`), `block_id`, `section_title`, `logical_table_id` (for cross-page table groups), `continued_from_prev`/`continues_to_next` (pagination flags), and branch-specific fields like `image_id`, `table_id`, `caption`, `display_title`.

### Frontend

Two static HTML files in `complex_document_rag/web_static/` with no build step. The Q&A interface supports multi-turn chat (in-memory, not persisted). Artifact files (images, tables) are served from disk via `/artifacts/` URL prefix.
