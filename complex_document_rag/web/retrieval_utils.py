"""节点检查、检索结果处理和排序工具函数。

从 backend.py 中提取的无状态辅助函数，不依赖 QueryBackend 实例。
所有函数均为纯函数或只读操作，可在测试中直接调用。

模块结构：
- 正则与常量     — 查询扩展规则、过滤参数
- 基础工具       — 安全读取分数、字符串截断
- 节点元数据读取 — doc_id、asset_id、标题、页码等
- 词法匹配       — CJK 检测、查询词提取、词面加分
- 焦点文档选择   — 多文档场景下找出主导文档
- 图片/表格引用  — 提取嵌入图片 ID、解析本地路径
- 检索结果处理   — 克隆/快照/合并/过滤 QueryBundle
- 节点去重与排序 — 去重、展示排序、素材排序
- 流程图检测     — 判断是否需要生成 Mermaid 图
- 跨页表格合并   — 把同一逻辑表格的碎片节点合并为一个
"""

from __future__ import annotations

import copy
import os
import re
from typing import Any, Iterable

from llama_index.core.schema import QueryBundle

# 表格碎片合并逻辑：把跨页分片拼接成完整的逻辑表格
from complex_document_rag.ingestion.tables import merge_logical_table_blocks
# 项目根目录：用于把相对路径转成绝对路径
from complex_document_rag.web.helpers import PROJECT_ROOT


# ===========================================================================
# 正则与查询扩展常量
# ===========================================================================

# 匹配图片引用 ID，例如 "img_001"、"img_p003_12"
IMAGE_REF_PATTERN = re.compile(r"(img(?:_p\d{2,4})?_\d+)", re.IGNORECASE)

# 从排序标题中提取数字序号（用于自然排序）
NUMBER_PATTERN = re.compile(r"\d+")

# 匹配 ASCII 英文词（包括带点/斜线的型号编号，如 "MRB"、"8D"、"CAR"）
ASCII_TERM_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9._/-]*")

# 匹配 CJK 汉字（用于判断查询语言）
CJK_PATTERN = re.compile(r"[\u3400-\u9fff]")

# 缩写 → 英文全称：中英混合查询时追加英文扩展，提升向量召回率
QUERY_TERM_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "MRB": ("Material Review Board",),
    "QSAT": ("Quality Sensitivity Alert Tag",),
    "8D": ("8D report", "8 disciplines"),
    "CAR": ("Corrective Action Request",),
}

# 中文短语 → 英文扩展：把中文查询对应的英文术语追加到变体查询中
# 格式：((中文别名集合), (英文扩展集合))
QUERY_PHRASE_EXPANSIONS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("触发时机", "触发条件"), ("trigger criteria", "when to initiate")),
    (("流程图", "flow图"), ("flowchart", "process flow")),
    (("流程",), ("flow", "process")),
    (("关键不良",), ("critical defect", "key defect", "non-conformance")),
    (("不良处理",), ("defect handling", "non-conformance handling")),
    (("客户投诉",), ("customer complaint",)),
    (("质量警报",), ("quality alert", "quality sensitivity alert")),
)

# 出现这些关键词时，在 prompt 中追加 Mermaid 流程图生成指令
FLOWCHART_HINT_KEYWORDS = (
    "流程图",
    "flow图",
    "flowchart",
    "process flow",
    "mermaid",
)

# 用于从查询中提取有意义的词元（英文 2+ 字符或 CJK 2+ 汉字）
QUERY_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]+|[\u3400-\u9fff]{2,}")

# 通用词汇黑名单：这些词在词面匹配时不计为有效查询词
GENERIC_QUERY_TOKENS = {
    "这个", "那个", "一下", "详细", "解释", "详细解释", "帮我", "请",
    "吗", "呢", "表格", "图片", "流程图", "flowchart", "flow", "diagram", "图", "表",
}

# 焦点文档选择参数
FOCUS_DOC_BRANCH_LIMIT = 2      # 每个分支只取前 N 名参与文档评分
FOCUS_DOC_SCORE_MARGIN = 0.08   # 第一名与第二名分差超过此值才认定存在主导文档
FOCUS_DOC_MODALITY_BONUS = 0.05 # 文档覆盖多个模态（文本+图片+表格）的加分

# 词面匹配加分上限：防止过多关键词命中导致分数失真
LEXICAL_MATCH_BONUS_CAP = 0.18


# ===========================================================================
# 基础工具
# ===========================================================================

def _safe_score(node: Any) -> float:
    """防御式读取 `node.score`，兼容测试中使用的轻量假对象。

    LlamaIndex NodeWithScore 的 score 可能为 None（未评分节点），
    也可能在单元测试中使用 SimpleNamespace 等简单对象，
    直接访问 node.score 可能引发 AttributeError 或 TypeError。
    """
    return float(getattr(node, "score", 0.0) or 0.0)


def _truncate_for_prompt(text: str, limit: int) -> str:
    """按字符数截断字符串，超出时在末尾追加省略号。

    用于把过长的元数据字段截断到 prompt 安全长度，
    避免单个字段占满 LLM 的上下文窗口。
    """
    normalized = (text or "").strip()
    if len(normalized) <= limit:
        return normalized
    # 末尾保留省略号，rstrip 去掉截断点前的空格
    return f"{normalized[: max(0, limit - 1)].rstrip()}…"


# ===========================================================================
# 节点元数据读取
# ===========================================================================

def _node_doc_id(node: Any) -> str:
    """读取节点所属文档的 ID（优先 doc_id，回退到 source_doc_id）。"""
    metadata = getattr(node, "metadata", {}) or {}
    return str(metadata.get("doc_id", "") or metadata.get("source_doc_id", "") or "").strip()


def _node_asset_id(node: Any) -> str:
    """读取节点的唯一素材 ID，供 judge prompt 和前端引用。

    优先级：image_id > logical_table_id > table_id
    logical_table_id 优先于 table_id 是因为跨页合并后只有 logical_table_id 是稳定的。
    """
    metadata = getattr(node, "metadata", {}) or {}
    return str(
        metadata.get("image_id")
        or metadata.get("logical_table_id")
        or metadata.get("table_id")
        or ""
    ).strip()


def _node_identity(node: Any) -> tuple[str, str]:
    """为节点生成去重/合并用的唯一标识 (type, id)。

    表格节点优先使用 logical_table_id，确保跨页碎片被识别为同一实体。
    其他节点依次尝试 table_id / image_id / block_id / node_id，
    全部缺失时回退到 doc_id + page_no + 文本前缀 的拼接字符串。
    """
    metadata = getattr(node, "metadata", {}) or {}
    node_type = str(metadata.get("type", "") or metadata.get("block_type", "") or "text")

    # 表格节点：优先使用逻辑表格 ID（跨页碎片共享同一 ID）
    if node_type == "table_block":
        logical_table_id = str(metadata.get("logical_table_id", "") or metadata.get("table_id", "") or "")
        if logical_table_id:
            return (node_type, logical_table_id)

    # 通用字段：依次尝试各种 ID 字段
    for field in ("table_id", "image_id", "block_id", "node_id"):
        value = str(metadata.get(field, "") or "")
        if value:
            return (node_type, value)

    # 最后回退：用文档+页码+文本前缀构造唯一键
    fallback = "|".join([
        str(metadata.get("doc_id", "") or ""),
        str(metadata.get("page_no", "") or ""),
        str(getattr(node, "text", "") or "")[:120],
    ])
    return (node_type, fallback)


def _safe_page_no(node: Any) -> int:
    """安全读取节点页码，缺失时返回极大值（用于把无页码节点排到末尾）。"""
    metadata = getattr(node, "metadata", {}) or {}
    value = metadata.get("page_no")
    try:
        return int(value)
    except (TypeError, ValueError):
        return 10**9  # 极大值，确保无页码节点排在已知页码之后


def _node_reference_label(node: Any) -> str:
    """返回适合在 UI 或 prompt 中展示的节点标题。

    表格：优先 display_title（人工设定）> caption > section_title > semantic_summary > table_id
    图片：优先 display_title > summary > section_title > caption > image_id
    其他：使用 block_id
    """
    metadata = getattr(node, "metadata", {}) or {}
    node_type = metadata.get("type", "")

    if node_type == "table_block":
        return str(
            metadata.get("display_title") or metadata.get("caption")
            or metadata.get("section_title") or metadata.get("semantic_summary")
            or metadata.get("summary") or metadata.get("table_id") or "相关表格"
        )
    if node_type == "image_description":
        return str(
            metadata.get("display_title") or metadata.get("summary")
            or metadata.get("section_title") or metadata.get("caption")
            or metadata.get("image_id") or "相关图片"
        )
    return str(metadata.get("block_id") or "相关内容")


def _node_sort_reference(node: Any) -> str:
    """返回用于排序的节点标题字符串（不包含摘要字段，避免长文本干扰数字提取）。"""
    metadata = getattr(node, "metadata", {}) or {}
    node_type = metadata.get("type", "")
    if node_type == "table_block":
        return str(
            metadata.get("display_title") or metadata.get("caption")
            or metadata.get("section_title") or metadata.get("table_id") or ""
        )
    if node_type == "image_description":
        return str(
            metadata.get("display_title") or metadata.get("summary")
            or metadata.get("section_title") or metadata.get("image_id") or ""
        )
    return str(metadata.get("block_id") or "")


def _clear_answer_asset_rank(node: Any) -> None:
    """删除 judge 注入的临时 `_answer_asset_rank` 字段。

    每次调用 select_answer_assets 前都要清除，防止上一次 judge 的排名
    污染本次的候选列表排序。
    """
    metadata = getattr(node, "metadata", None)
    if isinstance(metadata, dict):
        metadata.pop("_answer_asset_rank", None)


# ===========================================================================
# 词法匹配与查询扩展
# ===========================================================================

def _contains_cjk(text: str) -> bool:
    """判断字符串是否包含 CJK 汉字（用于区分中英文查询策略）。"""
    return bool(CJK_PATTERN.search(text or ""))


def _extract_query_terms(query: str) -> list[str]:
    """从查询中提取有效词元，过滤通用词和过短词。

    提取的词元用于计算词面匹配加分（_lexical_match_bonus）。
    全部转小写处理，保证大小写不敏感匹配。
    """
    normalized = str(query or "").casefold()
    seen: set[str] = set()
    terms: list[str] = []
    for raw_term in QUERY_TOKEN_PATTERN.findall(normalized):
        term = raw_term.strip()
        # 过滤：长度不足 2、通用词、已出现
        if len(term) < 2 or term in GENERIC_QUERY_TOKENS or term in seen:
            continue
        seen.add(term)
        terms.append(term)
    return terms


def _node_searchable_text(node: Any) -> str:
    """把节点元数据与正文拼成用于词面匹配的文本面（全部小写）。

    包含 display_title、section_title、caption、semantic_summary、summary、
    detailed_description、normalized_table_text 和节点正文，
    确保词面匹配能覆盖节点的所有可检索字段。
    """
    metadata = getattr(node, "metadata", {}) or {}
    parts = [
        str(metadata.get("display_title", "") or ""),
        str(metadata.get("section_title", "") or ""),
        str(metadata.get("caption", "") or ""),
        str(metadata.get("semantic_summary", "") or ""),
        str(metadata.get("summary", "") or ""),
        str(metadata.get("detailed_description", "") or ""),
        # 表格文本较长，截断后再加入，避免匹配文本过大
        _truncate_for_prompt(str(metadata.get("normalized_table_text", "") or ""), 800),
        _truncate_for_prompt(str(getattr(node, "text", "") or ""), 500),
    ]
    return "\n".join(part for part in parts if part).casefold()


def _lexical_match_bonus(query: str, node: Any) -> float:
    """计算查询与节点的词面匹配加分（语义分的补充）。

    规则：
    1. 完整查询（4-48 字符）出现在节点文本中：+0.08
    2. 长词（≥4 字符）命中：每词 +0.035
    3. 短词（2-3 字符）命中：每词 +0.02
    4. 加分上限：LEXICAL_MATCH_BONUS_CAP（防止过多词命中导致分数失真）
    """
    searchable = _node_searchable_text(node)
    if not searchable:
        return 0.0

    bonus = 0.0
    # 完整短语匹配加分
    normalized_query = re.sub(r"\s+", " ", str(query or "").casefold()).strip()
    if normalized_query and 4 <= len(normalized_query) <= 48 and normalized_query in searchable:
        bonus += 0.08

    # 逐词匹配加分
    for term in _extract_query_terms(query):
        if term in searchable:
            bonus += 0.035 if len(term) >= 4 else 0.02

    return min(LEXICAL_MATCH_BONUS_CAP, bonus)


def _append_unique_terms(terms: list[str], values: Iterable[str]) -> None:
    """向 terms 列表追加去重后的词元（大小写不敏感去重）。

    用于构建查询扩展词列表，确保同一词的不同大小写形式只出现一次。
    """
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
    """从字符串中提取第一个图片引用 ID（如 "img_001"）。

    用于检测表格内嵌图片引用，避免同一张图片以表格碎片和独立图片两种形式
    同时出现在答案素材中。
    """
    match = IMAGE_REF_PATTERN.search(value or "")
    return match.group(1) if match else ""


# ===========================================================================
# 焦点文档选择
# ===========================================================================

def _select_focus_doc_id(query: str, retrieval: dict[str, list[Any]]) -> str:
    """当高分结果明显聚集在同一文档时，选出主导文档。

    评分策略：
    1. 取每个分支的前 FOCUS_DOC_BRANCH_LIMIT 名节点
    2. 每个节点的分数 = 向量分 + 词面匹配加分
    3. 文档覆盖多个模态（文本+图片+表格）额外加分
    4. 第一名与第二名的分差 >= FOCUS_DOC_SCORE_MARGIN 时，确认主导文档

    返回空串表示没有明显主导文档（多文档均衡分布或只有一个文档）。
    """
    doc_scores: dict[str, dict[str, Any]] = {}

    for branch_name in ("text_results", "image_results", "table_results"):
        # 每个分支只取前 N 名，避免低分节点过多影响文档评分
        for node in retrieval.get(branch_name, [])[:FOCUS_DOC_BRANCH_LIMIT]:
            doc_id = _node_doc_id(node)
            if not doc_id:
                continue
            entry = doc_scores.setdefault(doc_id, {"score": 0.0, "modalities": set()})
            entry["score"] += _safe_score(node) + _lexical_match_bonus(query, node)
            entry["modalities"].add(branch_name)

    if not doc_scores:
        return ""

    # 多模态覆盖奖励：鼓励选择在文本、图片、表格三路都有命中的文档
    for entry in doc_scores.values():
        entry["score"] += max(0, len(entry["modalities"]) - 1) * FOCUS_DOC_MODALITY_BONUS

    ranked = sorted(doc_scores.items(), key=lambda item: item[1]["score"], reverse=True)

    # 只有一个文档时直接选择（无需比较）
    if len(ranked) == 1:
        return ranked[0][0]

    # 分差不足时认为无明显主导文档，返回空串（不限制素材范围）
    if ranked[0][1]["score"] < ranked[1][1]["score"] + FOCUS_DOC_SCORE_MARGIN:
        return ""

    return ranked[0][0]


def _restrict_to_focus_doc(nodes: list[Any], focus_doc_id: str) -> list[Any]:
    """把候选节点限制在焦点文档内（无焦点文档时原样返回）。

    安全检查：
    - 若过滤后结果为空，原样返回（避免焦点文档无素材时全部丢弃）
    - 若原列表中没有来自其他文档的节点，也原样返回（单文档场景无需过滤）
    """
    if not focus_doc_id:
        return nodes

    in_doc = [node for node in nodes if _node_doc_id(node) == focus_doc_id]
    if not in_doc:
        # 焦点文档中没有候选节点，回退到全量列表
        return nodes

    other_docs = {_node_doc_id(n) for n in nodes if _node_doc_id(n) and _node_doc_id(n) != focus_doc_id}
    if not other_docs:
        # 原本就只有一个文档，不需要过滤
        return nodes

    return in_doc


# ===========================================================================
# 图片和表格引用提取
# ===========================================================================

def _extract_table_embedded_image_ids(node: Any) -> set[str]:
    """提取表格 raw_table 中内嵌的图片引用 ID 集合。

    表格单元格中可能包含图片引用（如 "见 img_p003_12"），
    提取这些引用后可以去除重复展示的独立图片节点。
    """
    metadata = getattr(node, "metadata", {}) or {}
    raw_table = str(metadata.get("raw_table", "") or "")
    refs = {_extract_image_ref_id(match) for match in IMAGE_REF_PATTERN.findall(raw_table)}
    return {ref for ref in refs if ref}


def _node_embedded_image_id(node: Any) -> str:
    """从图片节点的各字段中提取第一个图片引用 ID。

    依次检查 image_id、image_path、source_image_path、节点文本，
    返回第一个成功匹配的 ID，全部失败时返回空串。
    """
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


def _resolve_local_node_image_path(node: Any) -> str:
    """把节点元数据中的图片路径解析为可访问的本地绝对路径。

    支持两种路径格式：
    - 绝对路径：直接检查文件是否存在
    - 相对路径：相对于 PROJECT_ROOT 拼接后检查

    若路径不存在或字段为空，返回空串（调用方需做空值检查）。
    """
    metadata = getattr(node, "metadata", {}) or {}
    # 优先检查 image_path，回退到 source_image_path（旧版字段）
    for candidate in (metadata.get("image_path", ""), metadata.get("source_image_path", "")):
        normalized = str(candidate or "").strip()
        if not normalized:
            continue
        # 相对路径补全为绝对路径
        absolute_path = normalized if os.path.isabs(normalized) else os.path.join(PROJECT_ROOT, normalized)
        absolute_path = os.path.abspath(absolute_path)
        if os.path.exists(absolute_path):
            return absolute_path
    return ""


# ===========================================================================
# 检索结果合并与过滤
# ===========================================================================

def _clone_query_bundle(query_bundle: QueryBundle) -> QueryBundle:
    """深拷贝 QueryBundle，防止多线程并发检索修改共享 embedding 对象。

    embedding 是 list[float]，list() 浅拷贝已足够（float 不可变）。
    query_str 是不可变字符串，直接复用。
    """
    embedding = query_bundle.embedding
    return QueryBundle(
        query_str=query_bundle.query_str,
        embedding=list(embedding) if embedding is not None else None,
    )


def _snapshot_retrieval_bundle(retrieval: dict[str, list[Any]]) -> dict[str, list[Any]]:
    """对整个检索结果做深拷贝快照。

    用于在重排前保存原始顺序：重排后若置信度不足，可回退到此快照。
    深拷贝确保后续对节点的修改（如写入 metadata["_answer_asset_rank"]）
    不影响快照中的节点。
    """
    return {
        key: [copy.deepcopy(node) for node in nodes]
        for key, nodes in retrieval.items()
    }


def _merge_retrieval_bundles(
    retrievals: list[dict[str, list[Any]]],
    *,
    variant_penalties: list[float],
) -> dict[str, list[Any]]:
    """合并多个查询变体的检索结果，并保留每个节点的最佳分数。

    对于同一个节点（由 _node_identity 判断）：
    - 计算经变体惩罚后的调整分数（adjusted_score = score - penalty）
    - 只保留调整分数最高的那一份拷贝
    最终各分支按调整分数降序排列。

    Args:
        retrievals: 每个查询变体的检索结果列表
        variant_penalties: 与 retrievals 等长的惩罚值列表（第 i 个变体的惩罚）
    """
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
                # 若该节点已有更高调整分，跳过（保留最优版本）
                if current is not None and _safe_score(current) >= adjusted_score:
                    continue
                # 深拷贝节点并写入调整后的分数
                snapshot = copy.deepcopy(node)
                snapshot.score = adjusted_score
                selected[identity] = snapshot
        # 按调整分数降序排列
        merged[branch_name] = sorted(selected.values(), key=_safe_score, reverse=True)
    return merged


def _filter_branch_nodes(
    nodes: Iterable[Any],
    *,
    min_score: float,
    relative_margin: float,
) -> list[Any]:
    """对单个分支应用分数阈值与相对分差过滤。

    双重过滤规则：
    1. 绝对阈值：分数低于 min_score 的节点直接丢弃
    2. 相对分差：分数低于"最高分 - relative_margin"的节点也丢弃
       （防止大量低分节点混入高质量结果中）

    Args:
        nodes: 待过滤的节点列表
        min_score: 绝对分数下限（重排分支用真实阈值，其他分支用 0）
        relative_margin: 相对于最高分的最大分差
    """
    ranked = sorted(list(nodes), key=_safe_score, reverse=True)
    if not ranked:
        return []

    # 先过滤绝对阈值
    above_threshold = [node for node in ranked if _safe_score(node) >= min_score]
    if not above_threshold:
        return []

    # 再过滤相对分差
    top_score = _safe_score(above_threshold[0])
    min_relative_score = top_score - relative_margin
    return [node for node in above_threshold if _safe_score(node) >= min_relative_score]


def _should_fallback_to_raw_branch(nodes: Iterable[Any], *, confidence_floor: float) -> bool:
    """检查重排后最高分是否低于置信下限，低于则建议回退到原始排序。

    重排器有时会给所有节点打低分（网络波动、模型不确定等），
    此时重排结果不可信，应回退到向量相似度排序。
    """
    ranked = sorted(list(nodes), key=_safe_score, reverse=True)
    if not ranked:
        return True  # 空列表视为需要回退
    return _safe_score(ranked[0]) < confidence_floor


# ===========================================================================
# 节点去重与排序
# ===========================================================================

def _dedupe_nodes(nodes: Iterable[Any]) -> list[Any]:
    """移除指向同一底层块的重复素材节点，保留先出现的节点。

    去重 key：(type, image_id, logical_table_id 或 table_id, block_id)
    这种组合确保同一图片或同一表格的不同分片版本只保留第一个。
    """
    deduped: list[Any] = []
    seen: set[tuple[str, str, str, Any]] = set()
    for node in nodes:
        metadata = getattr(node, "metadata", {}) or {}
        key = (
            metadata.get("type", ""),
            metadata.get("image_id", ""),
            metadata.get("logical_table_id", "") or metadata.get("table_id", ""),
            metadata.get("block_id", ""),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(node)
    return deduped


def _node_display_order_key(node: Any) -> tuple[int, tuple[int, ...], str]:
    """生成节点的展示排序键：(页码, 标题中的数字序列, 标题字符串)。

    数字序列提取自标题（如 "图 3-2" → (3, 2)），实现数值自然排序，
    避免字符串排序把 "图10" 排在 "图2" 前面。
    """
    reference = _node_sort_reference(node)
    numbers = tuple(int(value) for value in NUMBER_PATTERN.findall(reference))
    return (_safe_page_no(node), numbers, reference)


def _sort_nodes_for_display(nodes: Iterable[Any]) -> list[Any]:
    """按页码 + 标题自然排序（用于来源列表展示）。

    若所有节点都没有页码信息（page_no 为默认极大值），
    则保持原有顺序（通常是分数降序）。
    """
    node_list = list(nodes)
    if any(_safe_page_no(node) < 10**9 for node in node_list):
        return sorted(node_list, key=_node_display_order_key)
    return node_list


def _answer_asset_order_key(node: Any) -> tuple[int, int, int, tuple[int, ...], str]:
    """生成答案素材的排序键，judge 选中的素材排在前面。

    排序优先级：
    1. judge 选中的节点（_answer_asset_rank 不为 None）按 rank 升序排列
    2. 未被 judge 选中的节点：先按类型（表格 > 图片），再按页码和标题
    """
    metadata = getattr(node, "metadata", {}) or {}
    judge_rank = metadata.get("_answer_asset_rank")

    if judge_rank is not None:
        # judge 选中：第一维为 -1（排在所有未选中节点前）
        page_no, numbers, reference = _node_display_order_key(node)
        try:
            normalized_rank = int(judge_rank)
        except (TypeError, ValueError):
            normalized_rank = 10**9
        return (-1, normalized_rank, page_no, numbers, reference)

    # 未被 judge 选中：按类型优先级排序
    node_type = metadata.get("type", "")
    type_priority = 0 if node_type == "table_block" else (1 if node_type == "image_description" else 2)
    page_no, numbers, reference = _node_display_order_key(node)
    return (0, type_priority, page_no, numbers, reference)


def _sort_answer_assets(nodes: Iterable[Any]) -> list[Any]:
    """对答案素材排序，保证 UI 和测试里的顺序都是确定的。

    judge 选中的素材按 rank 顺序排在最前，其余按类型+页码排序。
    """
    return sorted(list(nodes), key=_answer_asset_order_key)


# ===========================================================================
# 流程图检测
# ===========================================================================

def _should_request_mermaid_diagram(query: str, answer_assets: Iterable[Any]) -> bool:
    """判断是否应在 prompt 中追加 Mermaid 流程图生成指令。

    检测范围：查询文本 + 所有答案素材的 summary / caption / semantic_summary / detailed_description
    只要任意字段包含流程图关键词（如 "流程图"、"flowchart"），就返回 True。
    """
    haystacks = [str(query or "")]
    for node in answer_assets:
        metadata = getattr(node, "metadata", {}) or {}
        haystacks.extend([
            str(metadata.get("summary", "") or ""),
            str(metadata.get("caption", "") or ""),
            str(metadata.get("semantic_summary", "") or ""),
            str(metadata.get("detailed_description", "") or ""),
        ])
    normalized = "\n".join(haystacks).casefold()
    return any(keyword.casefold() in normalized for keyword in FLOWCHART_HINT_KEYWORDS)


# ===========================================================================
# 跨页表格碎片合并
# ===========================================================================

def _coalesce_logical_table_nodes(nodes: Iterable[Any]) -> list[Any]:
    """在选择回答素材和构造 prompt 前，先合并跨页表格碎片。

    处理逻辑：
    1. 非表格节点直接透传（passthrough）
    2. 无 logical_table_id 的表格节点也透传
    3. 只有一个碎片的逻辑表格：补全元数据后直接使用
    4. 多个碎片的逻辑表格：
       a. 按页码排序
       b. 把每个碎片的元数据构造成 fragment dict
       c. 调用 merge_logical_table_blocks 合并
       d. 以分数最高的碎片为基础节点，写入合并后的元数据

    最终结果按分数降序排列（透传节点 + 合并节点混合排序）。
    """
    grouped: dict[str, list[Any]] = {}  # logical_table_id → 碎片节点列表
    passthrough: list[Any] = []         # 非表格或无 logical_table_id 的节点

    for node in nodes:
        metadata = getattr(node, "metadata", {}) or {}
        if metadata.get("type", "") != "table_block":
            passthrough.append(node)
            continue
        logical_table_id = str(metadata.get("logical_table_id") or metadata.get("table_id") or "").strip()
        if not logical_table_id:
            passthrough.append(node)
            continue
        grouped.setdefault(logical_table_id, []).append(node)

    merged_nodes: list[Any] = []
    for logical_table_id, group in grouped.items():
        if len(group) == 1:
            # 单碎片：补全 logical_table_id 等字段后直接使用
            node = copy.deepcopy(group[0])
            metadata = getattr(node, "metadata", None)
            if isinstance(metadata, dict):
                metadata.setdefault("logical_table_id", logical_table_id)
                metadata.setdefault(
                    "logical_page_numbers",
                    [metadata.get("page_no")] if metadata.get("page_no") is not None else [],
                )
                page_label = str(metadata.get("page_label", "")).strip()
                metadata.setdefault("logical_page_labels", [page_label] if page_label else [])
                metadata.setdefault(
                    "fragment_table_ids",
                    [metadata.get("table_id")] if metadata.get("table_id") else [],
                )
                metadata.setdefault("fragment_count", len(metadata.get("fragment_table_ids", [])) or 1)
            merged_nodes.append(node)
            continue

        # 多碎片：按页码排序后构造 fragment dict 列表，交给 merge_logical_table_blocks
        fragments: list[dict[str, object]] = []
        sorted_group = sorted(group, key=_safe_page_no)
        for index, node in enumerate(sorted_group):
            metadata = copy.deepcopy(getattr(node, "metadata", {}) or {})
            fragments.append({
                "block_id": str(metadata.get("block_id") or metadata.get("table_id") or logical_table_id),
                "table_id": str(metadata.get("table_id") or logical_table_id),
                "caption": str(metadata.get("caption", "") or ""),
                "summary": str(metadata.get("summary", "") or ""),
                "semantic_summary": str(metadata.get("semantic_summary", "") or ""),
                "raw_table": str(metadata.get("raw_table", "") or ""),
                "raw_format": str(metadata.get("raw_format", "markdown") or "markdown"),
                "normalized_table_text": str(metadata.get("normalized_table_text", "") or ""),
                "headers": copy.deepcopy(metadata.get("headers", []) or []),
                "table_type": str(metadata.get("table_type", "simple") or "simple"),
                "continued_from_prev": index > 0,           # 非第一片则为续表
                "continues_to_next": index < len(sorted_group) - 1,  # 非最后片则续
                "bbox_normalized": copy.deepcopy(metadata.get("bbox_normalized", []) or []),
                "origin": str(metadata.get("origin", "pdf_ocr") or "pdf_ocr"),
                "doc_id": str(metadata.get("doc_id", "") or ""),
                "source_path": str(metadata.get("source_path", "") or ""),
                "page_no": metadata.get("page_no"),
                "page_label": str(metadata.get("page_label", "") or ""),
                "section_title": str(metadata.get("section_title", "") or ""),
                "display_title": str(metadata.get("display_title", "") or ""),
                "logical_table_id": logical_table_id,
            })

        # 合并碎片，取第一个结果（merge_logical_table_blocks 返回列表）
        merged_block = merge_logical_table_blocks(fragments)[0]

        # 以分数最高的碎片为基础节点（继承其 score 和基本结构）
        base_node = copy.deepcopy(max(group, key=_safe_score))
        base_node.score = max(_safe_score(node) for node in group)
        base_node.text = merged_block.get("normalized_table_text", "") or getattr(base_node, "text", "")

        # 把合并后的元数据写回节点
        metadata = getattr(base_node, "metadata", None)
        if isinstance(metadata, dict):
            metadata.update(merged_block)
            metadata["logical_table_id"] = logical_table_id

        merged_nodes.append(base_node)

    # 透传节点与合并节点混合，按分数降序排列
    return sorted(passthrough + merged_nodes, key=_safe_score, reverse=True)
