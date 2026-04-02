# 新电脑安装指南

这份指南按 macOS 新电脑准备，目标是把 `complex-document-rag` 跑起来，并能完成一次 PDF 摄入和问答验证。

## 1. 安装基础工具

### 安装 Homebrew

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 安装 Git 和 Python

```bash
brew install git python@3.11
```

### 安装 Docker Desktop

```bash
brew install --cask docker
```

安装后先手动打开一次 Docker Desktop，等它进入可用状态。

## 2. 获取代码

```bash
git clone <your-repo-url>
cd complex-document-rag
```

## 3. 创建 Python 虚拟环境

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 4. 配置环境变量

复制模板：

```bash
cp .env.example .env
```

至少填写：

- `OPENAI_API_KEY`

如果你使用 DashScope / Qwen，再补：

- `DASHSCOPE_API_KEY`
- `OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1`
- `MULTIMODAL_LLM_MODEL`
- `TEXT_LLM_MODEL`
- `WEB_ANSWER_LLM_MODEL`
- `EMBEDDING_MODEL`

## 5. 启动 Qdrant

```bash
docker run -p 6333:6333 qdrant/qdrant
```

如果端口已占用，先停掉旧容器，或者改映射端口并同步修改 `.env`。

## 6. 启动服务

```bash
python -m complex_document_rag serve
```

打开：

- [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- [http://127.0.0.1:8000/ingest](http://127.0.0.1:8000/ingest)

## 7. 验证是否安装成功

建议按这个顺序验：

1. 打开 `/ingest`
2. 上传一份 PDF
3. 选择 OCR 模型和并发数
4. 等任务完成
5. 回到 `/` 提一个问题

如果能看到检索结果、附图/附表和最终回答，说明环境已经跑通。

## 常见问题

### `{"detail":"Not Found"}`

通常是服务没重启，或者你访问的不是当前仓库启动的进程。

### `ModuleNotFoundError`

大多数情况是没有激活虚拟环境，或者 `pip install -r requirements.txt` 没跑完。

### Qdrant 连不上

先检查 Docker Desktop 是否启动，再检查：

```bash
curl http://127.0.0.1:6333
```

### 首个 token 很慢

默认已经关闭 thinking。若你手动打开了 `WEB_ANSWER_ENABLE_THINKING=true`，模型会先输出思考过程。
