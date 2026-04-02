# RAG 修复与索引优化设计文档

**日期**：2026-04-01  
**状态**：已批准

---

## 背景

通过运行日志与代码审查发现六个问题，分为运行时修复和索引质量优化两类：

**运行时修复（三个）：**
1. 部分文档图片文件不在磁盘但向量库有引用，前端显示破图（404）
2. 双重门槛过滤导致部分查询 evidence 全空，LLM 无上下文作答
3. Thinking 模式无预算上限，LLM 回答延迟 30~47 秒

**索引质量优化（三个）：**
4. `summary` 与 `semantic_summary` 字段完全重复，浪费存储
5. `logical_table_sections` 存储所有碎片原文，跨页大表单节点 metadata 过大
6. 表格 embedding 文本顺序不优，语义密度最高的 `semantic_summary` 排在表格正文之后被稀释

---

## 修复一：图片 404 两层修复

### 根因

`doc_cleaned_qcg000001_ca` 等文档入库时图片文件未正确落盘，但 Qdrant 元数据中的 `image_path` 已写入。查询时 `web/helpers.py` 直接用 `image_path` 构建 `image_url`，不检查文件是否存在，前端收到坏链后显示破图。

### 方案

**治标（服务端过滤）— `web/helpers.py`**

在 `serialize_scored_node` 中，构建 `image_url` 之前先做 `os.path.exists()` 检查。文件不存在则 `image_url` 返回空字符串，前端不渲染图片卡片。

```python
# web/helpers.py: serialize_scored_node, kind == "image" 分支
actual_path = _resolve_artifact_abs_path(image_path, artifacts_root)
image_url = build_artifact_url(image_path, artifacts_root) if actual_path and os.path.exists(actual_path) else ""
```

新增辅助函数 `_resolve_artifact_abs_path(rel_path, artifacts_root) -> str`，将相对路径解析为绝对路径（复用 `build_artifact_url` 内的逻辑）。

**治本（重新入库）**

文档 `doc_cleaned_qcg000001_ca` 需用户在 `/ingest` 页面重新提交入库，重新生成图片文件。此步骤为手动操作，不在代码修改范围内，在日志中补充说明。

---

## 修复二：召回空上下文 Fallback 机制

### 根因

查询流水线有两道独立过滤：
1. `filter_retrieval_results()`：按分支阈值过滤（TEXT=0.55、IMAGE/TABLE=0.70）
2. `select_answer_assets()`：再次用 `ANSWER_ASSET_SCORE_THRESHOLD=0.60` 过滤，之后还有 LLM judge

两道叠加后部分查询（如"对PFAS的管控要求"）出现 `ALL evidence slots are empty`，LLM 无上下文作答。

### 方案

在 `web/backend.py` 查询主流程（同步与流式两个入口）中，`build_answer_prompt` 调用前加 fallback 检查：

```
if text_results == [] and answer_assets == []:
    # 第一优先：从 filtered_retrieval 取最优 table/image（跳过 judge）
    fallback_assets = sorted(
        filtered["table_results"] + filtered["image_results"],
        key=score, reverse=True
    )[:2]

    # 第二优先：若 table/image 也空，从 raw_retrieval 取 text（score > FALLBACK_TEXT_SCORE_FLOOR）
    if not fallback_assets:
        fallback_text = [n for n in raw["text_results"] if score(n) > FALLBACK_TEXT_SCORE_FLOOR][:2]
```

**关键设计决策：**
- 不改任何现有阈值，正常路径完全不变
- fallback 触发时写 `WARNING` 日志 `[fallback] evidence empty, using N low-confidence nodes`
- 新增配置 `FALLBACK_TEXT_SCORE_FLOOR = 0.40`（可 env 覆盖）
- fallback 节点不进入 `answer_assets`（不触发 judge），直接作为文本上下文送入 prompt

---

## 修复三：Thinking 预算上限

### 根因

`.env` 中 `WEB_ANSWER_ENABLE_THINKING=true`，DashScope 调用未设 thinking token 上限，模型无限思考，`[llm-ttft]` 延迟达 30~47 秒。

### 方案

**`config.py`**：新增
```python
WEB_ANSWER_THINKING_BUDGET_TOKENS = int(os.getenv("WEB_ANSWER_THINKING_BUDGET_TOKENS", "8000"))
```

**`model_provider_utils.py`**：在 DashScope 流式/非流式调用路径中，当 `disable_thinking=False` 时同时传入 budget：
```python
extra_body = {"enable_thinking": True, "thinking_budget": WEB_ANSWER_THINKING_BUDGET_TOKENS}
```

`disable_thinking=True` 时维持现有 `{"enable_thinking": False}`，不传 budget。

**`.env.example`**：补充 `WEB_ANSWER_THINKING_BUDGET_TOKENS=8000` 注释说明。

预期效果：thinking 耗时从 30~47 秒降至约 5~10 秒。

---

## 优化四：去除 `summary` 冗余字段

### 根因

`table_summary.py` 中同时设置了两个相同值：
```python
enriched["semantic_summary"] = semantic_summary
enriched["summary"] = semantic_summary   # 完全重复
```
`indexing/table_index.py` metadata 中也同时存了两份。`summary` 已无独立语义。

### 方案

- **`table_summary.py`**：删除 `enriched["summary"] = semantic_summary` 这一行
- **`indexing/table_index.py`**：metadata 中删除 `"summary": semantic_summary`，只保留 `"semantic_summary"`
- **`web/helpers.py`** (`serialize_scored_node` table 分支)：读取时改为：
  ```python
  "summary": metadata.get("semantic_summary") or metadata.get("caption", ""),
  ```
  保证旧节点（仍有 `summary` 字段）也能兼容
- **`web/backend.py`** (`_node_reference_label` 等读 `summary` 的地方)：同样 fallback 到 `semantic_summary`

---

## 优化五：精简 `logical_table_sections` 存储

### 根因

`logical_table_sections` 存储每页碎片的完整 raw_table，5 页表格 → 5 份 raw_table 全写入单个 Qdrant 节点 metadata，大型表格单节点可达数十 KB。

### 方案

**`ingestion/tables.py`** 中 `_build_logical_table_block()`：`logical_table_sections` 只保留每页的轻量摘要，不存完整 raw_table：

```python
"logical_table_sections": [
    {
        "table_id": frag["table_id"],
        "page_label": frag.get("page_label", ""),
        "page_no": frag.get("page_no"),
        "row_count": len(frag.get("rows", [])),
        # 不存 raw_table
    }
    for frag in fragments
]
```

完整表格内容通过合并后的顶层 `raw_table` 字段获取（已在 `_build_logical_table_block` 中合并），无需各页重复存储。

---

## 优化六：表格 Embedding 文本顺序优化

### 根因

当前顺序：`display_title → section_title → semantic_summary → normalized_table_text`

`semantic_summary` 是语义密度最高的字段，但被表格正文（可能很长）排在后面，embedding 对靠前内容权重更高，导致语义特征被稀释。

### 方案

**`indexing/table_index.py`** `index_single_table()` 中调整 `text_parts` 顺序：

```python
text_parts = []
if semantic_summary:
    text_parts.append(semantic_summary)      # 1. 语义摘要（最先）
if display_title and display_title not in text_parts:
    text_parts.append(display_title)         # 2. 标题
if section_title and section_title not in text_parts:
    text_parts.append(section_title)         # 3. 章节
if normalized_text:
    text_parts.append(normalized_text)       # 4. 表格正文（压后）
elif raw_table:
    text_parts.append(raw_table)
```

---

## 改动范围汇总

| 文件 | 改动 |
|------|------|
| `complex_document_rag/web/helpers.py` | 图片 URL 存在性检查；`summary` fallback 读取 |
| `complex_document_rag/web/backend.py` | fallback 机制；`summary` 读取兼容 |
| `complex_document_rag/ingestion/tables.py` | `logical_table_sections` 精简 |
| `complex_document_rag/indexing/table_index.py` | 删 `summary` 字段；调整 embedding 文本顺序 |
| `complex_document_rag/table_summary.py` | 删 `enriched["summary"]` 赋值 |
| `config.py` | 新增 `FALLBACK_TEXT_SCORE_FLOOR`、`WEB_ANSWER_THINKING_BUDGET_TOKENS` |
| `model_provider_utils.py` | 传入 thinking budget |
| `.env.example` | 新增变量注释 |

**不改动**：`ingest.html`、`indexing/image_index.py`、`indexing/text_index.py`

---

## 验证标准

1. 含 404 图片的查询结果：图片卡片静默消失，无破图图标
2. 查询"对PFAS的管控要求"：日志出现 `[fallback]` 标记，LLM 收到上下文
3. 开启 thinking 时：`[llm-ttft]` ≤ 15 秒
4. 新入库表格节点：Qdrant metadata 中无 `summary` 字段，`logical_table_sections` 无 `raw_table` 子字段
5. 表格检索质量：embedding 顺序调整后，语义相关查询命中率不下降
