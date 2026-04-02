# 进度交接文件
_最后更新: 2026-04-01_

## 已完成 (Tasks 1–7 代码全部提交)

| # | 内容 | Commit |
|---|------|--------|
| 1 | `config.py` + `.env.example` 新增 `WEB_ANSWER_THINKING_BUDGET_TOKENS` | a6f28a5 |
| 2 | `model_provider_utils.py` + `web/backend.py` 注入 thinking budget（8000 token 上限）| e94b2e7 |
| 3 | `web/helpers.py` 图片 URL 文件存在性检查，文件不存在返回空串 | fbbed14 |
| 4 | 删除 `table_summary.py` + `indexing/table_index.py` 中冗余 `summary` 字段 | db64df7 |
| 5 | `indexing/table_index.py` embedding 文本调序：`semantic_summary` 排第一 | 4a8542f |
| 6 | `ingestion/tables.py` `logical_table_sections` 精简，去掉 `raw_table` | a4f56dd |
| 7 | `web/backend.py` 新增 `_apply_evidence_fallback`，空 evidence 兜底 | c84f673 |

## 尚未完成

### Task 7 code quality review 已完成

#### Task 7 是干什么的

- 目标：修复“检索分支有结果，但最终 answer evidence 为空”时，回答侧拿不到任何图表证据的问题。
- 触发条件：`text_results=[]` 且 `answer_assets=[]`。
- 具体行为：[`complex_document_rag/web/backend.py`](/Users/biyiyi/Downloads/ocr-markdown%202/complex-document-rag/complex_document_rag/web/backend.py#L1228) 中新增 `_apply_evidence_fallback`，从已过滤的 `table_results + image_results` 里按分数降序取前 2 个节点，直接作为 `answer_assets`，并跳过 judge。
- 调用位置：[`answer()`](/Users/biyiyi/Downloads/ocr-markdown%202/complex-document-rag/complex_document_rag/web/backend.py#L1407) 和 [`stream_answer()`](/Users/biyiyi/Downloads/ocr-markdown%202/complex-document-rag/complex_document_rag/web/backend.py#L1426)。
- 直接收益：避免 prompt 中“文本证据为空、图表证据也为空”，让模型至少看到低置信度但仍可用的表格/图片证据；日志会打印 `[fallback]` 便于线上排查。
- 测试覆盖：[`EvidenceFallbackTests`](/Users/biyiyi/Downloads/ocr-markdown%202/complex-document-rag/tests/test_web_backend.py) 已覆盖 4 个场景：不触发 fallback、已有 assets 不触发、空 evidence 时取 top 节点、retrieval 也为空时安全返回空列表。

- **Spec compliance review** ✅ 已通过（逻辑正确，两处 call site 均确认）
- **Code quality review** ✅ 已人工完成（基于 `9226b6a..c84f673` diff 静态审查）
- **Review 结论**：未发现 Critical / Important 级别问题，Task 7 可关闭
- **Residual risk（非阻塞）**：
  - 当前 fallback 会有意绕过 judge，直接从 filtered `table_results + image_results` 中取 top 节点；这是本任务设计，不视为缺陷
  - 当前测试覆盖 helper 本身，但未覆盖 `answer()` / `stream_answer()` 的集成路径

#### 下次恢复时要做什么

- 无需再发起 Task 7 review；如要补强，只需补一条 `answer()` / `stream_answer()` 集成测试

#### Code quality review checklist

- [x] helper 的命名、docstring、触发条件与真实行为一致
- [x] fallback 只读取 filtered retrieval，不会把 raw 节点带回回答链路
- [x] top-2 选择、排序规则、日志文案足够直接，便于线上排查
- [x] `answer()` 和 `stream_answer()` 两条路径保持一致
- [x] fallback 仅改写 `answer_assets`，`answer_sources` 仍由统一逻辑生成
- [x] helper 级单测已覆盖“retrieval 全空 / text 非空 / assets 非空 / 混合 image+table 排序”
- [ ] 可选增强：补 `image-only` / `table-only` 单测，以及 `answer()` / `stream_answer()` 集成测试

#### 完成标准

- [x] 无 Critical / Important 级别问题
- [x] Minor / residual risk 已记录，不阻塞关闭
- [x] Task 7 已完成

### 验收测试（环境问题，非阻塞）
- `conda activate test` 环境可运行测试（Python 3.12.11）
- 已验证：`tests/test_web_backend.py` 中 `EvidenceFallbackTests` 4 个测试全部通过
- 已修复 `web/helpers.py` 图片 URL 路径兼容问题：
  - 支持从 `p0_basic_rag/ingestion_output/...` 恢复为当前 `/artifacts/...` URL
  - 支持从 `/workspace/complex_document_rag/ingestion_output/...` 恢复为当前 `/artifacts/...` URL
- 已验证：`python -m unittest discover -s tests -p 'test_web_*.py' -v` 已全部通过
- 因此 Task 7 本身不再受环境阻塞，web 相关测试模块也已回绿
- 上线后仍建议手工验证三点：
  1. 含 404 图片文档查询 → 图片卡片消失
  2. 查询"PFAS管控要求" → 日志出现 `[fallback]`
  3. 开启 thinking 查询 → TTFT ≤ 15 秒

## 下次开始时的操作

```bash
# 1. 进入可用环境
conda activate test

# 2. 运行验收
python -m unittest discover -s tests -p 'test_web_*.py' -v

# 3. 如需全量验收，再补跑
python -m unittest discover -s tests -v

# 4. 如验收通过，更新 CLAUDE.md（旧版，已过期）
```
