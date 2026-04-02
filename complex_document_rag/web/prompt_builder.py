"""LLM 提示词构造与 JSON 响应解析工具函数。

从 backend.py 中提取的无状态辅助函数，专门负责：
- 构造各类 prompt（查询改写、素材筛查 judge、历史对话压缩）
- 格式化表格预览（供 judge prompt 使用）
- 解析 LLM 返回的 JSON 响应（兼容 markdown fence 包裹）

所有函数均为纯函数，不依赖任何实例状态，可在测试中直接调用。
"""

from __future__ import annotations

import json
import re
from typing import Any

# prompt 长度限制常量
from complex_document_rag.web.settings import (
    ASSET_JUDGE_MAX_ASSETS,        # judge 最多保留的素材数（在 prompt 中告知 LLM）
    HISTORY_ANSWER_CHAR_LIMIT,     # 历史回答截断字符数（防止历史过长占满 prompt）
    HISTORY_QUERY_CHAR_LIMIT,      # 历史问题截断字符数
    HISTORY_TURN_LIMIT,            # 保留的最近对话轮数
    TABLE_PREVIEW_ROW_CHAR_LIMIT,  # 表格预览中单行数据截断字符数
    TABLE_PREVIEW_ROW_LIMIT,       # 表格预览中展示的最大行数
    TABLE_SUMMARY_CHAR_LIMIT,      # 表格摘要截断字符数
)

# 节点读取工具（避免在此模块重复实现）
from complex_document_rag.web.retrieval_utils import (
    _node_asset_id,          # 读取节点唯一素材 ID（image_id / table_id）
    _node_reference_label,   # 返回适合展示的节点标题
    _truncate_for_prompt,    # 按字符数截断字符串
)


# ===========================================================================
# JSON 响应解析
# ===========================================================================

def _clean_json_response_text(text: str) -> str:
    """清除 LLM 响应中的 markdown 代码块标记，提取纯 JSON 文本。

    LLM 有时会把 JSON 包裹在 ```json ... ``` 或 ``` ... ``` 中返回，
    此函数剥离这些标记，方便后续 json.loads 解析。
    """
    cleaned = str(text or "").strip()
    # 去掉开头的 markdown fence
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    # 去掉结尾的 markdown fence
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


def _parse_json_response(text: str) -> dict[str, Any]:
    """解析 LLM 返回的 JSON 字符串，兼容多种格式。

    解析策略：
    1. 先清除 markdown fence 标记
    2. 直接 json.loads 解析
    3. 若失败，尝试从文本中提取第一个 {...} 块再解析
       （兼容 LLM 在 JSON 前后添加解释文字的情况）

    Raises:
        ValueError: 响应为空或解析结果不是 dict
        json.JSONDecodeError: JSON 格式无法解析（正则提取也失败时）
    """
    cleaned = _clean_json_response_text(text)
    if not cleaned:
        raise ValueError("empty judge response")

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        # 尝试从文本中提取第一个 JSON 对象
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        if not match:
            raise  # 无法提取，原样抛出 JSONDecodeError
        parsed = json.loads(match.group(0))

    if not isinstance(parsed, dict):
        raise ValueError("judge response is not a JSON object")
    return parsed


# ===========================================================================
# 历史对话压缩
# ===========================================================================

def _build_history_prompt(history: Any) -> str:
    """把最近几轮对话压缩成长度受控的 prompt 片段。

    只保留最近 HISTORY_TURN_LIMIT 轮，每轮的问题和回答分别截断到
    HISTORY_QUERY_CHAR_LIMIT / HISTORY_ANSWER_CHAR_LIMIT 字符，
    防止历史对话过长占满 LLM 的上下文窗口。

    返回格式：
        对话历史：
        上一轮用户：...
        上一轮助手：...

    无历史或所有轮次都为空时返回空串。
    """
    if not history:
        return ""

    lines: list[str] = []
    # 只取最近 N 轮
    recent_turns = list(history)[-HISTORY_TURN_LIMIT:]
    for turn in recent_turns:
        query = _truncate_for_prompt(str((turn or {}).get("query", "") or ""), HISTORY_QUERY_CHAR_LIMIT)
        answer = _truncate_for_prompt(str((turn or {}).get("answer", "") or ""), HISTORY_ANSWER_CHAR_LIMIT)
        if query:
            lines.append(f"上一轮用户：{query}")
        if answer:
            lines.append(f"上一轮助手：{answer}")

    if not lines:
        return ""
    return "对话历史：\n" + "\n".join(lines) + "\n\n"


# ===========================================================================
# 表格预览格式化
# ===========================================================================

def _build_table_prompt_preview(node: Any) -> str:
    """把表格节点格式化成 LLM 可读的简洁预览。

    预览内容（按优先级）：
    1. semantic_summary（语义摘要）— 最重要，排第一
    2. headers（列名）— 帮助 LLM 理解表格结构
    3. 示例行（从 normalized_table_text 中提取）— 展示具体数据

    若以上字段均为空但 normalized_table_text 存在，则直接截断 normalized_table_text。
    全部为空时返回 "无表格摘要"。
    """
    metadata = getattr(node, "metadata", {}) or {}

    # 优先使用 semantic_summary，回退到 summary
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
        # 过滤掉空列名后拼接
        preview_lines.append(
            "列名=" + "|".join(str(header).strip() for header in headers if str(header).strip())
        )

    # 从 normalized_table_text 中提取示例行（格式：行N=内容；）
    row_matches = re.findall(r"行\d+=([^；\n]+)", normalized_text)
    for index, row in enumerate(row_matches[:TABLE_PREVIEW_ROW_LIMIT], start=1):
        preview_lines.append(f"示例行{index}={_truncate_for_prompt(row, TABLE_PREVIEW_ROW_CHAR_LIMIT)}")

    # 所有字段均为空但 normalized_text 存在时的兜底
    if not preview_lines and normalized_text:
        preview_lines.append(_truncate_for_prompt(normalized_text, TABLE_PREVIEW_ROW_CHAR_LIMIT))

    return "\n".join(preview_lines) or "无表格摘要"


# ===========================================================================
# 查询改写 Prompt
# ===========================================================================

def _build_query_rewrite_prompt(current_query: str, last_query: str) -> str:
    """构造用于把追问改写成独立检索查询的 prompt。

    场景：用户在多轮对话中使用代词或指代（"它"、"这个"、"刚才说的"），
    向量检索无法理解这些指代，需要把追问展开为完整独立的查询语句。

    LLM 应仅返回改写后的查询文本，不含任何解释或引号。

    Args:
        current_query: 用户最新的问题（可能包含代词指代）
        last_query: 上一轮用户问题（提供指代的上下文）
    """
    return (
        "你是一个检索查询改写助手。\n\n"
        "根据下方对话历史和用户最新问题，将最新问题改写为一个完整独立的检索查询：\n"
        "- 如果问题本身完整独立（不依赖上文、无代词指代），原样返回\n"
        "- 如果问题引用了上文（\u201c它\u201d、\u201c这个\u201d、\u201c刚才\u201d、\u201c上面\u201d等），将引用展开为完整表述\n"
        "- 只返回改写后的查询，不要任何解释、不要引号\n\n"
        f"上一问：{last_query}\n\n"
        f"最新问题：{current_query}"
    )


# ===========================================================================
# 素材筛查 Judge Prompt
# ===========================================================================

def _build_asset_judge_prompt(query: str, candidates: list[Any]) -> str:
    """构造用于给图片/表格候选素材排序和筛选的结构化 prompt。

    为每个候选素材生成一个结构化文本块，包含：
    - 图片节点：id、kind、doc、page、name、section、summary、detail
    - 表格节点：id、kind、doc、page、title、section、preview

    Judge 规则：
    1. 流程图问题 → 优先保留图片
    2. 数据/清单问题 → 优先保留表格
    3. 不因关键词沾边就判定相关（避免假阳性）
    4. 多文档场景 → 优先同一文档内相互印证的一组素材
    5. 最多选 ASSET_JUDGE_MAX_ASSETS 个，按相关性降序输出

    LLM 只返回 JSON，格式：{"selected_ids": ["id1", "id2"]}

    Args:
        query: 用户查询文本
        candidates: 候选图片/表格节点列表
    """
    candidate_blocks: list[str] = []
    for node in candidates:
        metadata = getattr(node, "metadata", {}) or {}
        node_type = metadata.get("type", "")
        asset_id = _node_asset_id(node)
        page_label = metadata.get("page_label", metadata.get("page_no", "-"))

        if node_type == "image_description":
            # 优先使用 detailed_description，回退到 summary
            detailed_description = _truncate_for_prompt(
                str(metadata.get("detailed_description", "") or metadata.get("summary", "") or ""),
                260,
            )
            candidate_blocks.append("\n".join([
                f"- id={asset_id}",
                "  kind=image",
                f"  doc={metadata.get('doc_id', '-')}",
                f"  page={page_label}",
                f"  name={_node_reference_label(node)}",
                f"  section={_truncate_for_prompt(str(metadata.get('section_title', '') or '无'), 120)}",
                f"  summary={_truncate_for_prompt(str(metadata.get('summary', '') or ''), 180)}",
                f"  detail={detailed_description or '无'}",
            ]))
            continue

        if node_type == "table_block":
            candidate_blocks.append("\n".join([
                f"- id={asset_id}",
                "  kind=table",
                f"  doc={metadata.get('doc_id', '-')}",
                f"  page={page_label}",
                f"  title={_node_reference_label(node)}",
                f"  section={_truncate_for_prompt(str(metadata.get('section_title', '') or '无'), 120)}",
                f"  preview={_build_table_prompt_preview(node)}",
            ]))
            continue

        # 未知类型节点跳过（不加入候选块）

    return (
        "你是一个 RAG 素材筛查器。请只保留能直接支持回答用户问题的图片或表格，过滤掉只是关键词沾边但并不直接回答问题的素材。\n"
        "判断规则：\n"
        "1. 如果用户问题明显在找流程图、示意图、图片内容，优先保留图片；表格只有在直接解释同一问题时才保留。\n"
        "2. 如果用户问题明显在找表格、矩阵、清单，优先保留表格；图片只有在直接补充该表格时才保留。\n"
        "3. 不要因为素材里出现了相同关键词（如 critical defect、flow、MRB）就判定为相关。\n"
        "4. 如果候选素材来自不同文档，优先保留来自同一份文档、且能相互印证的一组素材；不要混入另一份文档里仅仅语义相近的干扰项。\n"
        f"5. 最多选择 {ASSET_JUDGE_MAX_ASSETS} 个素材，按\u201c最值得展示给用户\u201d的顺序输出。\n"
        "6. 只返回 JSON，不要解释。\n\n"
        f"用户问题：{query}\n\n"
        "候选素材：\n"
        f"{chr(10).join(candidate_blocks) or '无'}\n\n"
        '输出格式：{"selected_ids":["素材id1","素材id2"]}'
    )
