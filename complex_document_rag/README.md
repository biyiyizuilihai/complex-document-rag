# 复杂文档 RAG 核心模块

`complex_document_rag` 是这个仓库真正可运行的核心模块，覆盖了一条完整的本地验证链路：

```text
PDF -> OCR -> 结构化中间产物 -> Qdrant -> Web 问答
```

## 适用范围

- 单个 PDF 摄入
- OCR 后抽取文本块、图片描述、表格块
- 三路证据分别建索引
- Web 端查看回答、证据、附图和附表

当前主路径是 PDF-only。代码里还保留了一些 DOCX 辅助函数，但不是推荐入口。

## 关键入口

- `step0_document_ingestion.py`
  - 单 PDF 摄入编排入口
- `step4_basic_query.py`
  - 命令行查询入口
- `web_app.py`
  - FastAPI 服务入口
- `manage_qdrant.py`
  - Qdrant 集合管理和按 `doc_id` 删除

## 数据流

```text
PDF
  -> ../scripts/batch_ocr.py
  -> document_ingestion.py
  -> document.md / manifest.json / image_descriptions.json / table_blocks.json
  -> step2_vector_indexing.py
  -> step3_image_indexing.py
  -> step3_table_indexing.py
  -> web_app.py / step4_basic_query.py
```

## 运行

在仓库根目录执行：

```bash
python complex_document_rag/step0_document_ingestion.py \
  --input "/absolute/path/to/your.pdf"
```

常用参数：

- `--ocr-model`
- `--workers`
- `--dpi`
- `--output-dir`
- `--skip-ocr`

启动 Web：

```bash
python complex_document_rag/web_app.py
```

访问：

- [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- [http://127.0.0.1:8000/ingest](http://127.0.0.1:8000/ingest)

## 输出目录

每次摄入会在 `complex_document_rag/ingestion_output/<doc_id>/` 生成：

- `document.md`
- `manifest.json`
- `image_descriptions.json`
- `table_blocks.json`
- `images/`
- `raw_pdf_ocr/`

## 相关模块

- `document_ingestion.py`
  - OCR 结果归一化
- `query_utils.py`
  - 多路召回与过滤
- `reranker.py`
  - rerank 封装
- `web_helpers.py`
  - Web 渲染辅助
- `web_static/index.html`
  - 问答页前端
- `web_static/ingest.html`
  - 摄入页前端

## 说明

- OCR 脚本依赖位于仓库根目录的 `scripts/`
- `web_app.py` 已包含上传页、问答页与摄入任务接口
- 新文档摄入完成后，问答页可以直接查询同一个 Qdrant 实例中的内容
