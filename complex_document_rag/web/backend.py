"""网页应用里的检索与回答生成后端。

当前 web 包已经把 HTTP 传输层与检索编排层拆开。这个模块负责完整的
检索/回答链路，让路由模块只处理请求接入与响应传输。

无状态的辅助函数已拆分至：
- retrieval_utils.py  — 节点读取、词法匹配、检索结果处理、排序
- prompt_builder.py   — LLM prompt 构造与 JSON 响应解析
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# 标准库
# ---------------------------------------------------------------------------
from concurrent.futures import ThreadPoolExecutor  # 并行检索三个分支
import json  # SSE 帧序列化
import time  # 检索耗时计时
from functools import lru_cache  # 后端单例缓存
from typing import Any

# ---------------------------------------------------------------------------
# 向量检索配置：各分支的 top-K 与分数阈值
# ---------------------------------------------------------------------------
from complex_document_rag.core.config import (
    IMAGE_SIMILARITY_TOP_K,
    IMAGE_RETRIEVAL_SCORE_MARGIN,
    IMAGE_RETRIEVAL_SCORE_THRESHOLD,
    MULTIMODAL_LLM_MODEL,
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
    WEB_ANSWER_THINKING_BUDGET_TOKENS,
)

# ---------------------------------------------------------------------------
# 模型工厂：创建文本 LLM 和多模态 LLM
# ---------------------------------------------------------------------------
from complex_document_rag.providers.llms import create_multimodal_llm, create_text_llm

# ---------------------------------------------------------------------------
# 重排器：可选的 SiliconFlow cross-encoder 重排
# ---------------------------------------------------------------------------
from complex_document_rag.retrieval.reranking import SiliconFlowReranker, rerank_retrieval_bundle

# ---------------------------------------------------------------------------
# Web 层设置：prompt 长度限制、judge 参数、日志器等
# ---------------------------------------------------------------------------
from complex_document_rag.web.settings import (
    ANSWER_ASSET_SCORE_THRESHOLD,   # 进入答案素材候选的最低分
    ANSWER_ASSET_TOP_K,             # 最多带入回答的素材数
    ANSWER_IMAGE_INPUT_TOP_K,       # 多模态调用时传入的最大图片数
    ASSET_JUDGE_ENABLED,            # 是否启用 LLM judge 筛选素材
    ASSET_JUDGE_MAX_ASSETS,         # judge 最多保留的素材数
    ASSET_JUDGE_MAX_TOKENS,         # judge 调用的 max_tokens 上限
    ASSET_JUDGE_MODEL,              # judge 使用的模型名称
    HISTORY_QUERY_CHAR_LIMIT,       # 历史问题截断字符数（用于查询改写）
    IMAGE_CONTEXT_TOP_K,            # prompt 中最多引用的图片数
    IMAGE_SUMMARY_CHAR_LIMIT,       # prompt 中单张图片摘要截断字符数
    LOGGER,                         # 模块级别 Logger
    QUERY_EXPANSION_SCORE_PENALTY,  # 扩展变体分数惩罚值
    QUERY_REWRITE_MAX_TOKENS,       # 查询改写调用的 max_tokens
    QUERY_REWRITE_MODEL,            # 查询改写使用的模型名称
    RERANK_CONFIDENCE_FLOOR,        # 重排分数低于此值时回退到原始排序
    RERANKED_BRANCH_KEYS,           # 哪些分支经过了重排（影响阈值逻辑）
    RETRIEVAL_RETRIES,              # 检索失败时的重试次数
    RETRIEVAL_RETRY_DELAY_SECONDS,  # 重试之间的等待秒数（乘以 attempt）
    TABLE_CONTEXT_TOP_K,            # prompt 中最多引用的表格数
    TEXT_CONTEXT_CHAR_LIMIT,        # prompt 中单个文本块截断字符数
    TEXT_CONTEXT_TOP_K,             # prompt 中最多引用的文本块数
    _EMBEDDING_CACHE_MAX,           # embedding LRU 缓存的最大条目数
)

# ---------------------------------------------------------------------------
# 无状态辅助函数（从本文件拆出，保持类代码整洁）
# ---------------------------------------------------------------------------
from complex_document_rag.web.retrieval_utils import (
    # 查询扩展相关常量
    ASCII_TERM_PATTERN,        # 匹配 ASCII 词的正则，用于提取缩写
    QUERY_TERM_EXPANSIONS,     # 缩写 → 英文全称映射（如 MRB）
    QUERY_PHRASE_EXPANSIONS,   # 中文短语 → 英文扩展映射（如 流程图 → flowchart）
    # 基础工具
    _safe_score,               # 安全读取 node.score，测试用假对象不会报错
    _truncate_for_prompt,      # 按字符数截断字符串，末尾加省略号
    # 节点元数据读取
    _node_asset_id,            # 读取节点的唯一素材 ID（image_id / table_id）
    _node_reference_label,     # 返回适合在 UI 展示的节点标题
    _clear_answer_asset_rank,  # 删除 judge 注入的临时排名字段
    # 词法匹配
    _contains_cjk,             # 判断字符串是否包含 CJK 汉字
    _append_unique_terms,      # 去重追加英文扩展词到列表
    # 焦点文档选择
    _select_focus_doc_id,      # 当高分结果集中于单一文档时，返回该文档 ID
    _restrict_to_focus_doc,    # 把候选节点限制在焦点文档内
    # 图片 / 表格引用提取
    _extract_table_embedded_image_ids,  # 从表格 raw_table 中提取内嵌图片 ID
    _node_embedded_image_id,            # 从图片节点的各字段提取引用 ID
    _resolve_local_node_image_path,     # 把节点元数据中的图片路径解析为本地绝对路径
    # 检索结果处理
    _clone_query_bundle,         # 深拷贝 QueryBundle（防止多线程共享 embedding）
    _snapshot_retrieval_bundle,  # 对整个检索结果做深拷贝快照
    _merge_retrieval_bundles,    # 合并多变体检索结果，保留每节点最高分
    _filter_branch_nodes,        # 按绝对分阈值 + 相对分差过滤单个分支
    _should_fallback_to_raw_branch,  # 重排后最高分不及置信下限时触发回退
    # 排序与去重
    _dedupe_nodes,           # 按 (type, image_id, table_id, block_id) 去重
    _sort_nodes_for_display, # 按页码 + 标题自然排序（用于来源列表）
    _sort_answer_assets,     # 按 judge rank 优先、再按 type + 页码排序素材
    # 跨页表格合并
    _coalesce_logical_table_nodes,  # 把同一逻辑表格的多个分片合并为一个节点
    # 流程图检测
    _should_request_mermaid_diagram,  # 问题或素材涉及流程图时返回 True
)

# ---------------------------------------------------------------------------
# prompt 构造与 JSON 解析（从本文件拆出）
# ---------------------------------------------------------------------------
from complex_document_rag.web.prompt_builder import (
    _build_history_prompt,       # 把最近对话压缩成 prompt 片段
    _build_table_prompt_preview, # 把表格节点格式化成 LLM 可读预览
    _build_query_rewrite_prompt, # 构造追问 → 独立查询的改写 prompt
    _build_asset_judge_prompt,   # 构造素材筛查 judge 的 prompt
    _parse_json_response,        # 解析 LLM 返回的 JSON（兼容 markdown fence）
)

# ---------------------------------------------------------------------------
# 序列化工具：把节点列表转成前端期望的 dict 列表
# ---------------------------------------------------------------------------
from complex_document_rag.web.helpers import serialize_answer_sources


# ===========================================================================
# 核心检索/回答类
# ===========================================================================

class QueryBackend:
    """负责 Web 请求从检索到回答的完整流水线。

    生命周期：
    1. __init__       — 加载三个 Qdrant 索引、创建各 LLM 实例和可选重排器
    2. retrieve()     — 公开入口，查询改写 → 多变体检索 → 重排 → 过滤
    3. select_answer_assets() — 从过滤结果中选图片/表格素材
    4. build_answer_prompt()  — 组装最终发给 LLM 的 prompt
    5. answer() / stream_answer() — 生成回答并返回结构化载荷
    """

    # -----------------------------------------------------------------------
    # 初始化
    # -----------------------------------------------------------------------

    def __init__(self) -> None:
        """加载索引，并初始化 Web 应用使用的辅助模型。

        使用延迟导入 load_indexes，避免在非 Web 场景下触发 Qdrant 连接。
        """
        from complex_document_rag.retrieval.query_console import load_indexes

        # 三个向量集合的索引对象（缺少集合时对应值为 None）
        self.text_index, self.image_index, self.table_index = load_indexes()

        # 优先复用索引内置的 embed_model，避免重复实例化
        self.embed_model = next(
            (
                getattr(index, "_embed_model", None)
                for index in (self.text_index, self.image_index, self.table_index)
                if index is not None and getattr(index, "_embed_model", None) is not None
            ),
            None,
        )

        # 回答生成主模型（支持 thinking budget）
        self.llm = create_text_llm(
            model_name=WEB_ANSWER_LLM_MODEL,
            api_key=OPENAI_API_KEY,
            api_base=OPENAI_BASE_URL,
            disable_thinking=not WEB_ANSWER_ENABLE_THINKING,
            thinking_budget_tokens=WEB_ANSWER_THINKING_BUDGET_TOKENS if WEB_ANSWER_ENABLE_THINKING else 0,
        )

        # 多模态回答模型：当素材里有图片时优先使用
        self.multimodal_llm = create_multimodal_llm(
            model_name=MULTIMODAL_LLM_MODEL,
            api_key=OPENAI_API_KEY,
            api_base=OPENAI_BASE_URL,
            disable_thinking=not WEB_ANSWER_ENABLE_THINKING,
            thinking_budget_tokens=WEB_ANSWER_THINKING_BUDGET_TOKENS if WEB_ANSWER_ENABLE_THINKING else 0,
        )

        # 素材筛查 judge（可选，关闭时退化为启发式排序）
        self.asset_judge_llm = (
            create_text_llm(
                model_name=ASSET_JUDGE_MODEL,
                api_key=OPENAI_API_KEY,
                api_base=OPENAI_BASE_URL,
                disable_thinking=True,  # judge 不需要思考过程
            )
            if ASSET_JUDGE_ENABLED
            else None
        )

        # 查询改写模型（追问 → 独立检索查询）
        self.query_rewrite_llm = create_text_llm(
            model_name=QUERY_REWRITE_MODEL,
            api_key=OPENAI_API_KEY,
            api_base=OPENAI_BASE_URL,
            disable_thinking=True,  # 改写不需要思考过程
        )

        self.reranker = self._build_reranker()
        # embedding LRU 缓存：多个查询变体共用同一 embedding 时避免重复调用
        self._embedding_cache: dict[str, list[float]] = {}

    def _ensure_runtime_state(self) -> None:
        """为绕过 `__init__` 的测试补齐懒加载属性。

        部分单元测试通过 object.__new__ 直接创建实例，跳过 __init__，
        此方法保证被调用的属性始终存在，防止 AttributeError。
        """
        if not hasattr(self, "_embedding_cache") or self._embedding_cache is None:
            self._embedding_cache = {}
        if not hasattr(self, "asset_judge_llm"):
            self.asset_judge_llm = None
        if not hasattr(self, "multimodal_llm"):
            self.multimodal_llm = None

    def _build_reranker(self) -> SiliconFlowReranker | None:
        """仅在功能开启且配置了 API Key 时，创建可选的重排器实例。"""
        if not RERANK_ENABLED or not RERANK_API_KEY:
            return None
        return SiliconFlowReranker(
            api_key=RERANK_API_KEY,
            api_base=RERANK_API_BASE,
            model_name=RERANK_MODEL,
            # top_n 取三个分支里最大的，保证重排不截断原始候选
            top_n=max(TEXT_SIMILARITY_TOP_K, IMAGE_SIMILARITY_TOP_K, TABLE_SIMILARITY_TOP_K),
            timeout=RERANK_TIMEOUT_SECONDS,
        )

    # -----------------------------------------------------------------------
    # Embedding 缓存
    # -----------------------------------------------------------------------

    def _get_cached_embedding(self, query: str) -> list[float] | None:
        """缓存 embedding，因为查询变体经常会复用同一段文本。

        缓存容量由 `_EMBEDDING_CACHE_MAX` 控制，超出后淘汰最早插入的条目
        （近似 FIFO，不是严格 LRU，但对少量变体已足够）。
        """
        self._ensure_runtime_state()
        if self.embed_model is None:
            return None
        if query not in self._embedding_cache:
            # 超出容量时移除最早的 key
            if len(self._embedding_cache) >= _EMBEDDING_CACHE_MAX:
                self._embedding_cache.pop(next(iter(self._embedding_cache)))
            self._embedding_cache[query] = self.embed_model.get_query_embedding(query)
        return self._embedding_cache[query]

    def _prefetch_embeddings(self, queries: list[str]) -> None:
        """在分支检索开始前，先并行预热所有查询变体的 embedding。

        单个变体时退化为串行调用，避免线程池开销。
        多个变体时并发请求，减少端到端延迟。
        """
        self._ensure_runtime_state()
        if self.embed_model is None:
            return
        missing = [q for q in queries if q not in self._embedding_cache]
        if len(missing) <= 1:
            # 仅一个缺失时直接串行
            for q in missing:
                self._get_cached_embedding(q)
            return
        # 并行请求多个 embedding
        with ThreadPoolExecutor(max_workers=len(missing)) as executor:
            futures = [(q, executor.submit(self.embed_model.get_query_embedding, q)) for q in missing]
        for q, fut in futures:
            if len(self._embedding_cache) >= _EMBEDDING_CACHE_MAX:
                self._embedding_cache.pop(next(iter(self._embedding_cache)))
            self._embedding_cache[q] = fut.result()

    def _build_query_bundle(self, query: str) -> Any:
        """把原始查询文本包装成 LlamaIndex retriever 期望的 QueryBundle。

        将 embedding 一并注入，避免 retriever 内部再次调用 embed_model。
        """
        from llama_index.core.schema import QueryBundle
        return QueryBundle(query_str=query, embedding=self._get_cached_embedding(query))

    # -----------------------------------------------------------------------
    # 查询变体生成
    # -----------------------------------------------------------------------

    def build_query_variants(self, query: str) -> list[str]:
        """对缩写较多的中英混合查询做扩展，提升召回率。

        策略：
        1. 纯英文查询不扩展（Qdrant 语义搜索已能覆盖）
        2. 含 CJK 的查询提取大写缩写和中文短语，映射到英文扩展词
        3. 返回 [原始查询, 原始+英文扩展] 两个变体，或仅 [原始查询]

        扩展变体在合并时会被施加 `QUERY_EXPANSION_SCORE_PENALTY` 分数惩罚，
        防止低质扩展命中的节点排在主变体命中结果前面。
        """
        normalized_query = (query or "").strip()
        if not normalized_query or not _contains_cjk(normalized_query):
            # 纯英文或空查询，不做扩展
            return [normalized_query]

        english_terms: list[str] = []
        # 提取查询中的 ASCII 词，转为全大写后查映射表
        ascii_terms = ASCII_TERM_PATTERN.findall(normalized_query)
        uppercase_terms = {term.upper() for term in ascii_terms}

        for acronym in sorted(uppercase_terms):
            if acronym in QUERY_TERM_EXPANSIONS:
                _append_unique_terms(english_terms, QUERY_TERM_EXPANSIONS[acronym])

        # 检查中文短语并追加对应英文扩展
        matched_phrase = False
        for aliases, expansions in QUERY_PHRASE_EXPANSIONS:
            if any(alias in normalized_query for alias in aliases):
                matched_phrase = True
                _append_unique_terms(english_terms, expansions)

        # 只有实际找到扩展词才生成第二个变体，避免无意义的重复查询
        should_expand = bool(english_terms) and (matched_phrase or bool(uppercase_terms))
        if not should_expand:
            return [normalized_query]

        mixed_query = " ".join([normalized_query, *english_terms]).strip()
        if mixed_query == normalized_query:
            return [normalized_query]
        return [normalized_query, mixed_query]

    # -----------------------------------------------------------------------
    # 检索查询构建（含多轮改写）
    # -----------------------------------------------------------------------

    def _build_retrieval_query(self, query: str, history: list[dict[str, Any]] | None = None) -> str:
        """把当前问题与最少必要的历史对话融合，再按需做查询改写。

        改写逻辑：
        - 无历史或无上一轮问题 → 直接返回当前问题
        - 调用 query_rewrite_llm 把代词/指代展开为完整独立查询
        - LLM 调用失败或返回空 → 退回当前问题
        """
        normalized_query = (query or "").strip()
        if not normalized_query or not history:
            return normalized_query

        # 从最近的历史对话里找出最后一个非空问题
        last_turn = None
        for turn in reversed(history):
            if str((turn or {}).get("query", "") or "").strip():
                last_turn = turn
                break
        if not last_turn:
            return normalized_query

        # 截断历史问题，防止过长的 prompt
        last_query = _truncate_for_prompt(
            str((last_turn or {}).get("query", "") or ""),
            HISTORY_QUERY_CHAR_LIMIT,
        )

        llm = getattr(self, "query_rewrite_llm", None)
        if llm is None:
            return normalized_query

        prompt = _build_query_rewrite_prompt(normalized_query, last_query)
        try:
            response = llm.complete(prompt, temperature=0, max_tokens=QUERY_REWRITE_MAX_TOKENS)
            rewritten = (getattr(response, "text", "") or "").strip()
            if rewritten:
                if rewritten != normalized_query:
                    LOGGER.info("[rewrite ] %r → %r", normalized_query, rewritten)
                return rewritten
        except Exception as exc:
            LOGGER.warning("[rewrite ] LLM error, falling back to original query: %s", exc)

        return normalized_query

    # -----------------------------------------------------------------------
    # 分支检索与多变体合并
    # -----------------------------------------------------------------------

    def _retrieve_branch(
        self,
        index: Any,
        *,
        similarity_top_k: int,
        query_bundle: Any,
    ) -> list[Any]:
        """使用已准备好的 QueryBundle 执行单个检索分支（text / image / table）。

        index 为 None 时安全返回空列表（集合未初始化时的防御）。
        QueryBundle 先经过 _clone_query_bundle 深拷贝，防止多线程修改共享状态。
        """
        if index is None:
            return []
        retriever = index.as_retriever(similarity_top_k=similarity_top_k)
        return retriever.retrieve(_clone_query_bundle(query_bundle))

    def _retrieve_once(self, query: str) -> dict[str, list[Any]]:
        """为单个用户问题检索文本、图片和表格候选项。

        完整流程：
        1. 生成查询变体（原始 + 扩展）
        2. 并行预热所有变体的 embedding
        3. 对每个变体并行检索三个分支
        4. 合并所有变体结果，保留每节点最高分
        5. 对文本分支做可选重排
        6. 过滤低分节点
        """
        t_total = time.perf_counter()

        query_variants = self.build_query_variants(query)
        LOGGER.info("[query   ] variants=%d  %s", len(query_variants), query_variants)

        # 三个分支的配置：(结果 key, 索引对象, top_k)
        branch_specs = [
            ("text_results", self.text_index, TEXT_SIMILARITY_TOP_K),
            ("image_results", self.image_index, IMAGE_SIMILARITY_TOP_K),
            ("table_results", self.table_index, TABLE_SIMILARITY_TOP_K),
        ]
        # 过滤掉索引为 None 的分支（仅在对应集合存在时参与检索）
        active_branches = [spec for spec in branch_specs if spec[1] is not None]

        # 并行预热 embedding，减少首次检索延迟
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
                # 单分支时串行执行（避免线程池初始化开销）
                for key, index, top_k in active_branches:
                    variant_result[key] = self._retrieve_branch(
                        index,
                        similarity_top_k=top_k,
                        query_bundle=query_bundle,
                    )
            else:
                # 多分支时并发检索，三路 IO 并行
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
            # 扩展变体的分数按 penalty 递增，越靠后的变体惩罚越重
            variant_penalties.append(variant_index * QUERY_EXPANSION_SCORE_PENALTY)

        # 跨变体去重合并，每个节点保留经惩罚后的最高分
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

        # 保留重排前的原始快照：图片/表格分支不走重排，需要从此快照恢复
        raw_snapshot = _snapshot_retrieval_bundle(retrieval)

        # 仅对文本分支做重排（图片和表格语义分数已足够）
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

        # 将重排后的文本分支与原始快照里的图片/表格分支合并
        merged_after_rerank = {
            "text_results": reranked_text_only.get("text_results", []),
            "image_results": raw_snapshot.get("image_results", []),
            "table_results": raw_snapshot.get("table_results", []),
        }

        # 应用分数阈值过滤
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
        """对原始检索结果应用分数门槛，并在必要时启用重排感知的回退逻辑。

        分支处理规则：
        - 经过重排的分支：使用绝对分阈值，同时检查是否需要回退到原始排序
        - 未经重排的分支：仅使用相对分差过滤（min_score=0.0）
        - 所有分支：过滤后只保留 top 分数与最高分相差在 margin 以内的节点
        - 表格结果：额外做跨页碎片合并

        `reranked_branches` 为 None 时自动根据 reranker 状态判断。
        """
        reranked_branches = (
            set(reranked_branches)
            if reranked_branches is not None
            else (
                {"text_results", "image_results", "table_results"}
                if self.reranker is not None and getattr(self.reranker, "enabled", True)
                else set()
            )
        )

        # 各分支的（绝对分阈值, 相对分差）配置
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

            # 重排后最高分低于置信下限时，回退到重排前的原始顺序
            if use_absolute_threshold and raw_retrieval is not None and _should_fallback_to_raw_branch(
                reranked_nodes,
                confidence_floor=RERANK_CONFIDENCE_FLOOR,
            ):
                branch_nodes = raw_retrieval.get(key, [])
                branch_min_score = 0.0  # 回退后不再应用绝对阈值

            filtered[key] = _filter_branch_nodes(
                branch_nodes,
                min_score=branch_min_score,
                relative_margin=margin,
            )

        # 合并跨页表格碎片，使同一逻辑表格在下游只出现一次
        filtered["table_results"] = _coalesce_logical_table_nodes(filtered.get("table_results", []))

        return filtered

    def retrieve(self, query: str, history: list[dict[str, Any]] | None = None) -> dict[str, list[Any]]:
        """同步路由与流式路由共用的公开检索入口。

        在检索失败时按指数退避重试，超过 `RETRIEVAL_RETRIES` 次后抛出异常。
        """
        # 按需把追问改写为完整独立查询
        retrieval_query = self._build_retrieval_query(query, history=history)
        last_error: Exception | None = None
        for attempt in range(RETRIEVAL_RETRIES + 1):
            try:
                return self._retrieve_once(retrieval_query)
            except Exception as exc:
                last_error = exc
                if attempt >= RETRIEVAL_RETRIES:
                    break
                # 每次重试等待时间线性递增
                time.sleep(RETRIEVAL_RETRY_DELAY_SECONDS * (attempt + 1))

        raise RuntimeError(str(last_error) if last_error else "检索失败")

    # -----------------------------------------------------------------------
    # 素材选择
    # -----------------------------------------------------------------------

    def _judge_answer_assets(self, query: str, candidates: list[Any]) -> list[Any]:
        """使用轻量 judge 模型筛掉不值得带入回答的素材。

        judge 开启时：
        1. 构造包含所有候选素材摘要的 prompt 发给 judge LLM
        2. 解析返回的 {"selected_ids": [...]}
        3. 按 judge 顺序排列素材并注入 _answer_asset_rank 字段

        judge 关闭或 LLM 调用失败时：回退到 _sort_answer_assets 启发式排序。
        """
        self._ensure_runtime_state()
        llm = getattr(self, "asset_judge_llm", None)
        if llm is None or not candidates:
            # judge 未启用，直接按启发式规则排序并截断
            return _sort_answer_assets(candidates[:ANSWER_ASSET_TOP_K])

        prompt = _build_asset_judge_prompt(query, candidates)
        try:
            response = llm.complete(prompt, temperature=0, max_tokens=ASSET_JUDGE_MAX_TOKENS)
            payload = _parse_json_response(getattr(response, "text", "") or "")
            # 兼容两种可能的 key 名称
            selected_ids = payload.get("selected_ids") or payload.get("selected_asset_ids") or []
            if not isinstance(selected_ids, list):
                raise ValueError("selected_ids must be a list")
        except Exception as exc:
            LOGGER.warning("[judge   ] fallback to heuristic asset ordering: %s", exc)
            return _sort_answer_assets(candidates[:ANSWER_ASSET_TOP_K])

        # 构建 id → rank 映射，用于后续排序
        selected_id_order = [str(value).strip() for value in selected_ids if str(value).strip()]
        rank_map = {asset_id: index for index, asset_id in enumerate(selected_id_order)}
        judged = [node for node in candidates if _node_asset_id(node) in rank_map]
        judged.sort(key=lambda node: rank_map.get(_node_asset_id(node), 10**9))
        judged = _dedupe_nodes(judged)[:ASSET_JUDGE_MAX_ASSETS]

        # 把 judge 排名写入节点 metadata，供 _sort_answer_assets 使用
        for index, node in enumerate(judged):
            metadata = getattr(node, "metadata", None)
            if isinstance(metadata, dict):
                metadata["_answer_asset_rank"] = index

        return _sort_answer_assets(judged)

    def select_answer_assets(self, retrieval: dict[str, list[Any]], query: str = "") -> list[Any]:
        """选择需要随最终回答一起展示的图片或表格。

        处理流程：
        1. 合并图片/表格候选，按分数排序
        2. 清除上轮 judge 留下的排名字段
        3. 过滤低于 ANSWER_ASSET_SCORE_THRESHOLD 的候选
        4. 去重并限制总数，同时剔除被表格已内嵌的图片
        5. 合并跨页表格碎片
        6. 焦点文档过滤（多文档场景下避免混入无关文档的素材）
        7. 调用 _judge_answer_assets 做最终精选
        """
        self._ensure_runtime_state()
        # 合并两类素材并按分数降序排列
        candidates = sorted(
            list(retrieval.get("image_results", [])) + list(retrieval.get("table_results", [])),
            key=_safe_score,
            reverse=True,
        )
        # 清除旧的 judge rank，防止上一次调用的排名影响本次
        for node in candidates:
            _clear_answer_asset_rank(node)

        # 只保留分数足够高的候选
        filtered = [node for node in candidates if _safe_score(node) >= ANSWER_ASSET_SCORE_THRESHOLD]

        selected: list[Any] = []
        selected_table_image_refs: set[str] = set()  # 记录已被表格内嵌的图片 ID

        for node in filtered:
            metadata = getattr(node, "metadata", {}) or {}
            node_type = metadata.get("type", "")

            if node_type == "image_description":
                # 若该图片已被某个表格内嵌，则跳过（避免重复展示）
                image_ref = _node_embedded_image_id(node)
                if image_ref and image_ref in selected_table_image_refs:
                    continue
                selected.append(node)

            elif node_type == "table_block":
                selected.append(node)
                # 记录该表格内嵌的图片引用
                selected_table_image_refs.update(_extract_table_embedded_image_ids(node))
                # 若有新的内嵌图片引用，从已选列表中剔除对应的独立图片节点
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

            # 每次迭代后去重并限制总数
            selected = _dedupe_nodes(selected)
            if len(selected) > ANSWER_ASSET_TOP_K:
                selected = selected[:ANSWER_ASSET_TOP_K]

        selected = selected[:ANSWER_ASSET_TOP_K]
        if not selected:
            LOGGER.info("[assets  ] no candidates passed ANSWER_ASSET_SCORE_THRESHOLD=%.2f", ANSWER_ASSET_SCORE_THRESHOLD)
            return []

        LOGGER.info(
            "[assets  ] before focus_doc: %d candidates  scores=%s",
            len(selected),
            [round(_safe_score(n), 3) for n in selected],
        )

        # 合并同一逻辑表格的碎片节点
        selected = _coalesce_logical_table_nodes(selected)

        # 当高分命中集中于单一文档时，把素材限制在该文档内
        focus_doc_id = _select_focus_doc_id(query, retrieval) if query else ""
        LOGGER.info("[assets  ] focus_doc_id=%r", focus_doc_id)
        selected = _restrict_to_focus_doc(selected, focus_doc_id)

        LOGGER.info(
            "[assets  ] after focus_doc restrict: %d candidates  scores=%s",
            len(selected),
            [round(_safe_score(n), 3) for n in selected],
        )

        # 调用 judge 做最终精选
        judged = self._judge_answer_assets(query, selected)
        LOGGER.info(
            "[assets  ] after judge: %d assets  titles=%s",
            len(judged),
            [_node_reference_label(n) for n in judged],
        )
        return judged

    def _apply_evidence_fallback(
        self,
        retrieval: dict[str, list[Any]],
        answer_assets: list[Any],
    ) -> list[Any]:
        """当文本检索为空时，回退到分数最高的非文本证据。

        触发条件：text_results 为空 且 answer_assets 也为空。
        行为：从过滤后的 table_results + image_results 中取分数最高的 2 个节点，
        直接作为 answer_assets，跳过 judge 流程。
        日志：打印 [fallback] 便于线上排查。
        """
        # 有文本结果或已有素材时，不触发 fallback
        if retrieval.get("text_results") or answer_assets:
            return answer_assets

        # 按分数降序取 top-2 非文本节点
        candidates = sorted(
            list(retrieval.get("table_results", [])) + list(retrieval.get("image_results", [])),
            key=_safe_score,
            reverse=True,
        )[:2]

        if candidates:
            LOGGER.warning(
                "[fallback] evidence empty — using %d low-confidence node(s) from filtered retrieval",
                len(candidates),
            )
        return candidates

    def select_answer_sources(self, retrieval: dict[str, list[Any]], answer_assets: list[Any]) -> list[Any]:
        """构造前端回答下方展示的来源列表。

        来源 = 分数最高的 2 个文本块 + 所有答案素材，去重后按页码排序。
        """
        text_results = sorted(retrieval.get("text_results", []), key=_safe_score, reverse=True)
        candidate_sources = text_results[:2] + answer_assets
        return _sort_nodes_for_display(_dedupe_nodes(candidate_sources))

    # -----------------------------------------------------------------------
    # Prompt 构造
    # -----------------------------------------------------------------------

    def build_answer_prompt(
        self,
        query: str,
        retrieval: dict[str, list[Any]],
        answer_assets: list[Any],
        history: list[dict[str, Any]] | None = None,
    ) -> str:
        """用历史对话、文本、表格和图片组装最终回答 prompt。

        Prompt 结构：
        - 系统指令（严格基于证据、中文回答）
        - 可选 Mermaid 流程图指令（当查询或素材涉及流程图时）
        - 推荐素材清单（供 LLM 在回答中提及）
        - 对话历史（最近几轮，长度受控）
        - 当前问题
        - 文本证据（最多 TEXT_CONTEXT_TOP_K 块）
        - 图片证据（最多 IMAGE_CONTEXT_TOP_K 张）
        - 表格证据（最多 TABLE_CONTEXT_TOP_K 张）
        """
        # 文本证据：截断至 TEXT_CONTEXT_CHAR_LIMIT 字符
        text_context = []
        for index, node in enumerate(retrieval.get("text_results", [])[:TEXT_CONTEXT_TOP_K], start=1):
            metadata = getattr(node, "metadata", {}) or {}
            text_context.append(
                f"[文本{index}] 页码={metadata.get('page_label', metadata.get('page_no', '-'))}\n"
                f"{_truncate_for_prompt(str(getattr(node, 'text', '') or ''), TEXT_CONTEXT_CHAR_LIMIT)}"
            )

        # 图片证据：从 answer_assets 中提取图片节点
        selected_image_nodes = [
            node
            for node in answer_assets
            if (getattr(node, "metadata", {}) or {}).get("type", "") == "image_description"
        ][:IMAGE_CONTEXT_TOP_K]
        image_context = []
        for index, node in enumerate(selected_image_nodes, start=1):
            metadata = getattr(node, "metadata", {}) or {}
            image_context.append(
                f"[图片{index}] 名称={_node_reference_label(node)} 页码={metadata.get('page_label', metadata.get('page_no', '-'))} "
                f"摘要={_truncate_for_prompt(str(metadata.get('summary', '') or ''), IMAGE_SUMMARY_CHAR_LIMIT)}"
            )

        # 表格证据：使用 _build_table_prompt_preview 格式化预览
        selected_table_nodes = [
            node
            for node in answer_assets
            if (getattr(node, "metadata", {}) or {}).get("type", "") == "table_block"
        ][:TABLE_CONTEXT_TOP_K]
        table_context = []
        for index, node in enumerate(selected_table_nodes, start=1):
            metadata = getattr(node, "metadata", {}) or {}
            table_context.append(
                f"[表格{index}] 标题={_node_reference_label(node)} 页码={metadata.get('page_label', metadata.get('page_no', '-'))}\n"
                f"{_build_table_prompt_preview(node)}"
            )

        # 构造推荐素材清单（LLM 在回答中可自然提及）
        answer_asset_ids = []
        for node in _sort_nodes_for_display(answer_assets):
            metadata = getattr(node, "metadata", {}) or {}
            if metadata.get("image_id"):
                answer_asset_ids.append(f"图片:{_node_reference_label(node)}")
            if metadata.get("table_id"):
                answer_asset_ids.append(f"表格:{_node_reference_label(node)}")

        # 若查询或素材涉及流程图，追加 Mermaid 生成指令
        mermaid_instruction = ""
        if _should_request_mermaid_diagram(query, answer_assets):
            mermaid_instruction = (
                "如果问题涉及流程图、流程单、flow图或 flowchart，并且证据足够，请先输出文字说明，再额外输出一个 ```mermaid "
                "代码块来概括主流程。Mermaid 中只保留关键节点与判断分支，节点名称使用中文；若某条连线或条件无法从证据中确认，"
                "就在文字说明中明确写出不确定，不要在 Mermaid 中编造。\n"
            )

        # 压缩历史对话成 prompt 片段
        history_context = _build_history_prompt(history)

        LOGGER.info(
            "[prompt  ] text_chunks=%d  image_chunks=%d  table_chunks=%d  asset_ids=%s",
            len(text_context),
            len(image_context),
            len(table_context),
            answer_asset_ids,
        )
        if not text_context and not table_context and not image_context:
            LOGGER.warning("[prompt  ] ALL evidence slots are empty — LLM will see no context!")

        return (
            "你是一个严谨的 RAG 问答助手。请严格基于给定上下文回答，不要编造。\n"
            "如果证据不足，请明确说证据不足。\n"
            "如果输出思考过程，也请全程使用中文，不要输出英文小标题或英文分析。\n"
            "如果下面列出的图片或表格与问题直接相关，请在回答中自然提及\u201c见相关图片/见相关表格\u201d。\n"
            "引用图表时只使用中文标题或名称，不要输出任何内部 ID、文件名或技术标识。\n"
            f"{mermaid_instruction}"
            f"推荐随答案一起返回的素材：{', '.join(answer_asset_ids) if answer_asset_ids else '无'}\n\n"
            f"{history_context}"
            f"当前问题：{query}\n\n"
            "文本证据：\n"
            f"{chr(10).join(text_context) or '无'}\n\n"
            "图片证据：\n"
            f"{chr(10).join(image_context) or '无'}\n\n"
            "表格证据：\n"
            f"{chr(10).join(table_context) or '无'}\n\n"
            "请输出简洁、直接的中文回答。"
            "请默认使用分点回答：先用一句话给出结论，再用 1. 2. 3. 列出关键点；如果有注意事项、例外条件或补充材料，再单独分点说明。"
        )

    # -----------------------------------------------------------------------
    # 图片路径解析
    # -----------------------------------------------------------------------

    def collect_answer_image_paths(self, answer_assets: list[Any]) -> list[str]:
        """为多模态回答收集本地可访问的图像文件路径。

        仅处理 image_description 类型的节点，跳过路径不可解析或文件不存在的节点。
        结果去重，并限制在 ANSWER_IMAGE_INPUT_TOP_K 张以内（防止 token 超限）。
        """
        image_paths: list[str] = []
        seen: set[str] = set()
        for node in answer_assets:
            metadata = getattr(node, "metadata", {}) or {}
            if metadata.get("type", "") != "image_description":
                continue
            image_path = _resolve_local_node_image_path(node)
            if not image_path or image_path in seen:
                continue
            seen.add(image_path)
            image_paths.append(image_path)
            if len(image_paths) >= ANSWER_IMAGE_INPUT_TOP_K:
                break
        return image_paths

    # -----------------------------------------------------------------------
    # 回答生成（同步 + 流式）
    # -----------------------------------------------------------------------

    def _complete_answer(self, prompt: str, answer_assets: list[Any]):
        """一次性生成完整回答，并在有价值时启用多模态输入。

        优先路径：有图片 + multimodal_llm 可用 → 多模态调用
        降级路径：多模态调用失败 或 无图片 → 纯文本 LLM
        """
        self._ensure_runtime_state()
        image_paths = self.collect_answer_image_paths(answer_assets)
        multimodal_llm = getattr(self, "multimodal_llm", None)
        if image_paths and multimodal_llm is not None:
            try:
                LOGGER.info(
                    "[visual  ] multimodal answer  images=%d  model=%s",
                    len(image_paths),
                    getattr(multimodal_llm, "model_name", MULTIMODAL_LLM_MODEL),
                )
                response = multimodal_llm.complete(prompt, image_paths=image_paths)
                LOGGER.info(
                    "[visual-ok] multimodal answer active  images=%d  chars=%d",
                    len(image_paths),
                    len(getattr(response, "text", "") or ""),
                )
                return response
            except Exception as exc:
                LOGGER.warning("[visual  ] multimodal answer fallback to text llm: %s", exc)
        return self.llm.complete(prompt)

    def _stream_answer(self, prompt: str, answer_assets: list[Any]):
        """返回 token 流，并在有价值时启用多模态输入。

        逻辑与 _complete_answer 相同，但返回生成器而非完整字符串。
        """
        self._ensure_runtime_state()
        image_paths = self.collect_answer_image_paths(answer_assets)
        multimodal_llm = getattr(self, "multimodal_llm", None)
        if image_paths and multimodal_llm is not None:
            try:
                LOGGER.info(
                    "[visual  ] multimodal stream  images=%d  model=%s",
                    len(image_paths),
                    getattr(multimodal_llm, "model_name", MULTIMODAL_LLM_MODEL),
                )
                stream = multimodal_llm.stream_complete(prompt, image_paths=image_paths)
                LOGGER.info(
                    "[visual-ok] multimodal stream active  images=%d",
                    len(image_paths),
                )
                return stream
            except Exception as exc:
                LOGGER.warning("[visual  ] multimodal stream fallback to text llm: %s", exc)
        return self.llm.stream_complete(prompt)

    # -----------------------------------------------------------------------
    # 公开接口
    # -----------------------------------------------------------------------

    def answer(
        self,
        query: str,
        retrieval: dict[str, list[Any]] | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """生成包含回答正文、来源和素材的完整载荷。

        外部传入 retrieval 时复用已有结果，不重复检索（用于同步路由预先检索的场景）。
        """
        retrieval = retrieval or self.retrieve(query)
        answer_assets = self.select_answer_assets(retrieval, query=query)
        # 文本证据为空时触发 fallback，确保 LLM 至少看到非文本证据
        answer_assets = self._apply_evidence_fallback(retrieval, answer_assets)
        answer_sources = self.select_answer_sources(retrieval, answer_assets)
        prompt = self.build_answer_prompt(query, retrieval, answer_assets, history=history)
        response = self._complete_answer(prompt, answer_assets)
        return {
            "answer": response.text or "",
            "answer_sources": answer_sources,
            "answer_assets": answer_assets,
        }

    def stream_answer(
        self,
        query: str,
        retrieval: dict[str, list[Any]] | None = None,
        history: list[dict[str, Any]] | None = None,
    ):
        """在复用相同素材选择逻辑的前提下，准备流式回答状态。

        返回字典中 "stream" 为生成器，路由层负责把 token 推送给前端 SSE。
        """
        retrieval = retrieval or self.retrieve(query)
        answer_assets = self.select_answer_assets(retrieval, query=query)
        answer_assets = self._apply_evidence_fallback(retrieval, answer_assets)
        answer_sources = self.select_answer_sources(retrieval, answer_assets)
        prompt = self.build_answer_prompt(query, retrieval, answer_assets, history=history)
        return {
            "stream": self._stream_answer(prompt, answer_assets),
            "answer_sources": answer_sources,
            "answer_assets": answer_assets,
        }


# ===========================================================================
# 模块级工具函数（供路由层调用）
# ===========================================================================

@lru_cache(maxsize=1)
def get_query_backend() -> QueryBackend:
    """在整个进程生命周期内保留一个预热后的后端实例。

    lru_cache(maxsize=1) 保证全局单例语义：
    第一次调用触发索引加载和 LLM 初始化，后续调用直接返回缓存对象。
    """
    return QueryBackend()


def load_default_queries() -> list[str]:
    """返回前端展示的内置示例问题。"""
    from complex_document_rag.retrieval.query_console import get_default_test_queries

    return get_default_test_queries()


def _empty_retrieval_bundle() -> dict[str, list[Any]]:
    """创建 Web 层统一使用的标准检索结果结构（三个空列表）。"""
    return {
        "text_results": [],
        "image_results": [],
        "table_results": [],
    }


def _serialize_answer_result(answer_result: Any) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    """兼容字典式与对象式后端，统一回答载荷结构。

    字典格式：QueryBackend.answer() 的返回值
    对象格式：LlamaIndex QueryEngine 的返回值（保持向后兼容）
    """
    if isinstance(answer_result, dict):
        answer = answer_result.get("answer", "") or ""
        answer_sources = serialize_answer_sources(answer_result.get("answer_sources", []))
        # answer_assets 需要先排序再序列化，确保前端展示顺序一致
        answer_assets = serialize_answer_sources(_sort_answer_assets(answer_result.get("answer_assets", [])))
        return answer, answer_sources, answer_assets

    # 兼容对象格式（LlamaIndex QueryEngine）
    answer = getattr(answer_result, "response", "") or ""
    raw_sources = getattr(answer_result, "source_nodes", []) or []
    answer_sources = serialize_answer_sources(raw_sources)
    return answer, answer_sources, answer_sources


def _sse_event(name: str, payload: dict[str, Any]) -> str:
    """编码单个 Server-Sent Event 帧。

    格式：`event: <name>\ndata: <json>\n\n`
    ensure_ascii=False 保证中文字符直接输出，减小传输体积。
    """
    return f"event: {name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
