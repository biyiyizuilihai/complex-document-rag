# 复杂文档 RAG 解析

一个可直接运行的复杂文档 RAG 项目：支持在前端上传 PDF，自动完成 OCR、结构化、向量入库，并在同一套 Web 界面里做问答验证。

## 功能概览

- PDF-only 摄入
- 前端上传 PDF 并选择 OCR 模型、并发数
- 文本、图片、表格三路索引到 Qdrant
- Web 问答页支持流式回答、思考过程展示、来源证据展示
- 摄入完成后可直接回到问答页验证效果

## 页面入口

- 智能问答页：`/`
- 文档摄入页：`/ingest`

启动服务后默认地址是 [http://127.0.0.1:8000](http://127.0.0.1:8000)。

## 目录结构

```text
complex-document-rag/
├── README.md
├── .env.example
├── requirements.txt
├── config.py
├── data_models.py
├── model_provider_utils.py
├── scripts/
│   ├── batch_ocr.py
│   ├── postprocess.py
│   └── table_normalizer.py
├── complex_document_rag/
│   ├── README.md
│   ├── step0_document_ingestion.py
│   ├── step4_basic_query.py
│   ├── web_app.py
│   └── web_static/
└── tests/
```

## 环境要求

- Python 3.10+
- 一个可访问的 LLM / embedding API
- 本地 Qdrant
- Git
- Docker Desktop 或可用的 Docker 环境

推荐先准备虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 配置

复制环境变量模板：

```bash
cp .env.example .env
```

至少需要配置：

- `OPENAI_API_KEY`
- 如果使用 DashScope / Qwen，再配置：
  - `DASHSCOPE_API_KEY`
  - `OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1`
  - `MULTIMODAL_LLM_MODEL`
  - `TEXT_LLM_MODEL`
  - `WEB_ANSWER_LLM_MODEL`
  - `EMBEDDING_MODEL`

可选配置：

- `WEB_ANSWER_ENABLE_THINKING=true`
- `SILICONFLOW_API_KEY`
- `RERANK_ENABLED=true`

## 新电脑安装

如果对方是一台全新的 Mac，建议按这个顺序装：

1. 安装 Homebrew

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

2. 安装 Git、Python 和 Docker Desktop

```bash
brew install git python@3.11
brew install --cask docker
```

3. 验证基础环境

```bash
git --version
python3 --version
docker --version
```

4. 克隆仓库并安装依赖

```bash
git clone <your-repo-url>
cd complex-document-rag
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

5. 复制环境变量模板并填入 API Key

```bash
cp .env.example .env
```

6. 启动 Docker Desktop 后再运行 Qdrant

```bash
docker run -p 6333:6333 qdrant/qdrant
```

7. 启动服务

```bash
python complex_document_rag/web_app.py
```

更完整的安装说明见 [docs/setup-new-machine.md](/Users/biyiyi/Downloads/ocr-markdown%202/complex-document-rag/docs/setup-new-machine.md)。

## 快速开始

### 1. 启动 Qdrant

```bash
docker run -p 6333:6333 qdrant/qdrant
```

### 2. 启动 Web 服务

```bash
python complex_document_rag/web_app.py
```

### 3. 打开页面

- 问答页：[http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- 摄入页：[http://127.0.0.1:8000/ingest](http://127.0.0.1:8000/ingest)

## 使用方式

### 方式 A：前端上传 PDF

1. 打开 `/ingest`
2. 选择一个 PDF
3. 选择 OCR 模型和并发数
4. 点击“开始摄入”
5. 摄入完成后回到问答页提问

### 方式 B：命令行摄入

```bash
python complex_document_rag/step0_document_ingestion.py \
  --input "/absolute/path/to/your.pdf" \
  --ocr-model qwen3.5-plus \
  --workers 4
```

然后启动问答页：

```bash
python complex_document_rag/web_app.py
```

## 运行产物

每次摄入会在 `complex_document_rag/ingestion_output/<doc_id>/` 下生成：

- `document.md`
- `manifest.json`
- `image_descriptions.json`
- `table_blocks.json`
- `images/`
- `raw_pdf_ocr/`

这些目录已经被 `.gitignore` 忽略，不建议提交到仓库。

## 测试

先跑最有价值的定向用例：

```bash
python -m unittest discover -s tests -p 'test_web_static_frontend.py'
python -m unittest discover -s tests -p 'test_web_static_ingest_frontend.py'
python -m unittest discover -s tests -p 'test_web_app.py'
```

## 当前边界

- 只支持 PDF 上传
- 主要面向本地调试和原型验证
- 默认使用单机 FastAPI + 本地 Qdrant
- 不包含用户系统、权限、任务队列和生产部署方案

## 发布到 GitHub

如果你要把这个目录单独发到 GitHub：

```bash
cd /path/to/complex-document-rag
git init
git add .
git commit -m "Initial commit"
```

然后在 GitHub 创建空仓库，再执行：

```bash
git remote add origin <your-repo-url>
git branch -M main
git push -u origin main
```

## 补充说明

- OCR 脚本已经一并放进 `scripts/`，这个仓库是自包含的
- 如果你打开了 `WEB_ANSWER_ENABLE_THINKING=true`，前端会显示模型思考过程
- 如果上传页返回 `{\"detail\":\"Not Found\"}`，通常是服务没重启，或者访问的不是当前仓库启动的进程
