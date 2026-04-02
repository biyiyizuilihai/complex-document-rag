# 文档内引用导航现状与改进报告

日期：2026-04-02

## 1. 结论摘要

当前问题不是 `Appendix` 识别不准，而是系统没有“文档内引用导航层”。

现状上，这个项目已经能把 OCR 结果拆成文本块、表格块、图片描述块，并分别入到 `text_chunks`、`table_blocks`、`image_descriptions` 三个向量集合里。但查询阶段仍然只有一次语义召回，没有能力把“命中的引用线索”继续解析成“目标正文块”。

因此：

- 命中目录表时，系统会直接拿目录表回答
- 命中普通表格里的一句 `see xxx` 时，系统也不会继续跳转
- 问题并不只发生在 `Appendix`
- 只按 `appendix / attachment / notice / reference / section` 枚举类型继续扩展，也仍然是不对的

推荐方案是：

**让大模型在摄入阶段抽取“锚点”和“引用关系”，系统只消费结构，不消费词表。**

对应地，检索阶段改成：

1. 第一跳仍然做语义召回
2. 如果高分节点里存在已解析的引用关系
3. 就沿引用关系做同文档二跳召回
4. 把目标正文块、目标表格块、目标图片块合并回上下文

目录表只是这个机制里的一个高质量引用源，不再是主逻辑。

## 2. 现状

### 2.1 检索链路仍然是单跳

当前 `QueryBackend` 的主流程是：

- 构造 query variants
- 并行检索文本、图片、表格三个分支
- 合并结果
- 对文本分支做可选 rerank
- 过滤低分结果
- 直接进入回答阶段

对应代码：

- `complex_document_rag/web/backend.py`
  - `_retrieve_once()`
  - `retrieve()`
  - `select_answer_assets()`
  - `build_answer_prompt()`

关键问题：

- 没有“命中引用节点后继续找目标节点”的二跳逻辑
- `select_answer_assets()` 只是在已有候选里筛选，不会扩展召回
- `build_answer_prompt()` 只消费第一跳结果

### 2.2 索引里没有“引用导航”元数据

#### 文本块

`attach_text_block_metadata()` 和 `build_llama_documents()` 目前只写基础字段，例如：

- `doc_id`
- `source_doc_id`
- `source_path`
- `page_no`
- `page_label`
- `block_type`
- `origin`
- `block_id`

当前没有：

- 这个块定义了什么对象
- 这个块引用了什么对象
- 这些引用是否已经解析到文档内其他块

对应代码：

- `complex_document_rag/ingestion/pipeline.py`

#### 表格块

表格目前的重点是“可检索文本化”，不是“可导航结构化”。

`build_normalized_table_text()` 会把：

- 标题
- 语义摘要
- 页码
- 表头
- 每行内容

全部压平成 embedding 文本。这样做对语义召回有效，但副作用是：

- 目录表、索引表、修订记录表、普通引用表都会因为关键词密度高而强命中
- 系统却不知道这些表里哪一列其实是“导航目标”

对应代码：

- `complex_document_rag/ingestion/tables.py`
- `complex_document_rag/indexing/table_index.py`

#### 图片描述块

图片描述分支也只是普通语义节点，当前同样没有“引用目标”或“被引用对象”元数据。

对应代码：

- `complex_document_rag/ingestion/images.py`
- `complex_document_rag/indexing/image_index.py`

### 2.3 当前数据其实已经在库里

就本次 `ST 客户特殊要求` 场景来看：

- page 9 的目录表已经被抽出来
- `Appendix 1A ~ 1G` 的正文页也已经存在于后续页

所以当前故障不是“没抽到”，而是：

- 引用线索已经入库
- 目标正文也已经入库
- 但两者之间没有结构化连接

### 2.4 当前代码里已经有“引用”概念，但没接进主链路

`complex_document_rag/core/models.py` 里已经有：

- `ExternalReference`
- `DocReference`
- `references_to`
- `referenced_by`

说明仓库作者已经意识到“引用关系”是重要概念；但这层结构目前还没有接到：

- 摄入 metadata
- Qdrant payload
- 查询二跳扩展

## 3. 根因分析

### 3.1 问题不在 `Appendix`，而在“缺少关系层”

用户真正要解决的问题不是：

- 识别 `Appendix`

而是：

- 识别“某个块在定义一个可跳转对象”
- 识别“某个块在引用另一个对象”
- 让系统沿这个引用关系继续找目标内容

如果只解决 `Appendix`：

- 目录页的问题会缓解
- 普通表里的 `see attachment xxx`
- 正文段落里的 `refer to notice yyy`
- 图片说明里的 `see drawing zzz`

这些场景仍然会失败。

### 3.2 继续枚举对象类型，仍然是半硬编码

把方案从：

- `appendix`

扩成：

- `appendix / attachment / notice / reference / section`

看起来更通用，但本质上仍然是：

- 代码在消费固定词表

这不适合当前项目，因为你这套 OCR 和结构抽取本身就是大模型驱动的。模型在不同文档里可能会识别出：

- `Appendix 1A`
- `Attachment 8C`
- `Notice 12`
- `Customer-Specific Annex`
- `Ref. Doc R-004`

如果代码需要不断把这些词加入白名单，系统会越来越脆。

### 3.3 需要把“识别”交给模型，把“结构”交给系统

正确边界应该是：

- 模型负责从块内容里识别“定义”和“引用”
- 系统负责把结果落成稳定 schema
- 检索负责利用这些 schema 做导航

也就是说：

- 不能让代码决定哪些词是对象类型
- 也不能让模型的自由文本直接决定系统分支

系统应只消费统一结构，例如：

- 这个块有哪些 `anchors`
- 这个块有哪些 `references`
- 这些 `references` 是否已被解析到某些 `anchors`

## 4. 推荐抽象：Anchor / Reference 图层

### 4.1 Anchor

`anchor` 表示“这个块自己定义了一个可被引用的对象”。

它可以出现在：

- 正文标题
- 表格标题
- 图纸标题
- 图片说明
- 目录中的目标项定义

它不要求对象一定叫 `Appendix`。只要模型判断：

- 这里定义了一个可被文档内部其他位置引用的对象

就可以产生一个 `anchor`。

### 4.2 Reference

`reference` 表示“这个块提到了另一个对象，但自己不是那个对象的正文”。

它可以出现在：

- 目录表
- 普通表格单元格
- 修订记录表
- 正文段落
- 图片描述

只要模型判断：

- 这里存在跳转线索

就可以产生一个 `reference`。

### 4.3 代码不应该按对象类型分支

对象的名字、类别、写法可以由模型自由识别，但系统逻辑不能写成：

- 如果是 `appendix` 就二跳
- 如果是 `notice` 就二跳

而应该统一写成：

- 如果高分节点里有已解析的 `references`
- 就沿这些引用去找对应 `anchors`

对象类别如果要保留，也只能用于：

- 展示
- 排查
- 调试
- 辅助排序

不能成为主分支条件。

## 5. 建议数据模型

建议每个索引节点都带两类结构：

- `anchors`
- `references`

### 5.1 `anchors`

建议字段：

```json
[
  {
    "anchor_id": "doc:crqa100m01:anchor:17",
    "label": "Appendix 1A",
    "title": "Customer Special Requirement for ST",
    "aliases": ["1A", "Appendix 1A"],
    "model_type": "appendix",
    "confidence": 0.95
  }
]
```

说明：

- `anchor_id`
  - 系统内稳定 ID，由摄入阶段生成
- `label`
  - 模型抽取到的主标识
- `title`
  - 可选标题
- `aliases`
  - 便于解析引用时做匹配
- `model_type`
  - 可选，仅保留模型认知结果，不参与主逻辑分支
- `confidence`
  - 抽取置信度

### 5.2 `references`

建议字段：

```json
[
  {
    "surface": "See Appendix 1A",
    "target_hint": "Appendix 1A",
    "relation": "refers_to",
    "resolved_anchor_ids": ["doc:crqa100m01:anchor:17"],
    "confidence": 0.91
  }
]
```

说明：

- `surface`
  - 原始引用片段
- `target_hint`
  - 模型认为目标对象的文本标签
- `relation`
  - 关系类型，可先统一成 `refers_to`
- `resolved_anchor_ids`
  - 解析成功后对应到的目标锚点 ID 列表
- `confidence`
  - 抽取或解析置信度

### 5.3 为什么保留 `model_type` 但不让它驱动逻辑

`model_type` 有价值，但只能是辅助字段。

保留它的原因：

- 方便调试模型输出
- 方便前端做解释
- 方便后续人工分析误识别

不让它驱动逻辑的原因：

- 它本质上还是模型自由文本
- 不稳定
- 文档域变化大
- 一旦进入主分支，系统又会回到“枚举类型”的老路上

## 6. 目标链路设计

### 6.1 摄入阶段：模型抽取 anchor/reference

建议新增一层通用引用抽取，不区分文本、表格、图片分支的业务类型。

输入对象：

- 文本块
- 表格块
- 图片描述块

输出对象：

- `anchors`
- `references`

建议新增模块：

- `complex_document_rag/ingestion/references.py`

职责：

- 调用结构化输出模型
- 从块内容里抽取 anchor/reference
- 返回标准化结果

### 6.2 解析阶段：在同文档内做 reference resolution

抽取出来的 `references` 还只是线索，不一定已知目标是谁。

因此需要一个文档内解析步骤：

1. 收集当前文档所有 `anchors`
2. 用 `target_hint`、`aliases`、标题文本、页码邻近性做匹配
3. 将匹配结果写入 `resolved_anchor_ids`
4. 未匹配成功的引用保留为 unresolved，不能强连

这一步仍然是通用的，不应该按 `Appendix` 或 `Notice` 写分支。

### 6.3 索引阶段：把关系写入三个集合的 metadata

建议把以下信息写入：

- 文本集合 `text_chunks`
- 表格集合 `table_blocks`
- 图片集合 `image_descriptions`

至少应带：

- `anchors`
- `references`
- `anchor_ids`
- `resolved_reference_anchor_ids`

其中：

- `anchors` / `references` 便于调试和前端解释
- `anchor_ids` / `resolved_reference_anchor_ids` 便于过滤与快速二跳

### 6.4 查询阶段：从单跳改成“语义召回 + 引用扩展”

建议改造 `QueryBackend` 为两阶段：

#### 第一跳：语义召回

保持现有逻辑：

- 文本、图片、表格并行召回
- 可选 rerank
- 过滤

#### 第二跳：引用扩展

对高分结果做一次轻量扫描：

- 如果某个节点带 `resolved_reference_anchor_ids`
- 则在同一 `doc_id` 内按这些 anchor 定向拉取目标节点

目标节点包括：

- 锚点正文文本
- 与锚点关联的表格
- 与锚点关联的图片描述

#### 排序原则

建议排序从高到低：

1. 直接回答问题的目标正文节点
2. 与目标锚点关联的补充表格/图片
3. 提供跳转线索的引用源节点

也就是说：

- 目录表、修订记录表、普通引用表都应作为“辅助证据”
- 不应长期霸占主上下文

### 6.5 回答阶段：优先消费目标节点

`build_answer_prompt()` 不应默认把第一跳高分表格直接塞进主要上下文，而应优先使用：

- 二跳召回到的目标节点
- 目标节点对应的补充表格/图片

引用源节点只保留为：

- 来源解释
- 溯源证据
- 无正文时的 fallback

## 7. 对当前代码的具体改动建议

### 7.1 摄入层

建议新增或调整：

- `complex_document_rag/ingestion/references.py`
  - 新增通用引用抽取与解析逻辑
- `complex_document_rag/ingestion/pipeline.py`
  - 给文本块附加 `anchors/references`
  - 在写 `Document.metadata` 前加入标准字段
- `complex_document_rag/ingestion/tables.py`
  - 给表格块附加 `anchors/references`
- `complex_document_rag/ingestion/images.py`
  - 给图片描述块附加 `anchors/references`

### 7.2 模型层

建议优先复用：

- `complex_document_rag/core/models.py`
  - `DocReference`
  - `references_to`
  - `referenced_by`

但需要适当扩展，使其能表达：

- anchor 定义
- resolved reference
- unresolved reference

### 7.3 索引层

建议调整：

- `complex_document_rag/indexing/table_index.py`
- `complex_document_rag/indexing/image_index.py`
- 文本索引链路对应 metadata 写入位置

目标：

- 让三类集合都能携带同一套引用导航元数据

### 7.4 检索层

建议调整：

- `complex_document_rag/web/backend.py`

重点方法：

- `_retrieve_once()`
  - 保留首轮召回，但增加二跳扩展
- `select_answer_assets()`
  - 不再把“高分引用表”默认视为最终答案素材
- `build_answer_prompt()`
  - 优先消费目标正文节点

## 8. 为什么这个方案比枚举类别更好

### 8.1 更符合你当前 OCR 体系

你现在的文档解析本来就是大模型驱动的。

那就应该：

- 让模型负责理解“这里是什么引用关系”
- 让系统负责保存和利用关系

而不是：

- 让模型先识别
- 再让代码把识别结果硬塞回固定词表

### 8.2 能同时覆盖目录表、普通表、正文、图片说明

这个方案下：

- 目录表是引用源
- 普通表格也可以是引用源
- 正文段落也可以是引用源
- 图片描述也可以是引用源
- 真正的对象正文页是锚点源

逻辑统一，不需要按来源类型反复加补丁。

### 8.3 对未知文档域更稳

不同客户文档、不同供应链模板、不同语言环境里，可跳转对象的叫法会一直变化。

如果系统只依赖：

- `anchors`
- `references`
- `resolved_anchor_ids`

那么即使新文档里叫法完全不同，只要模型能识别出引用关系，检索链路就仍然成立。

## 9. 风险与注意事项

### 风险 1：模型会抽错引用关系

这是最主要风险。

需要控制方式：

- 只对高置信度引用做自动二跳
- 低置信度引用只保留展示，不参与扩展
- unresolved 引用绝不强行连边

### 风险 2：同名对象可能冲突

文档内可能存在：

- 同名标题
- 相似编号
- 目录页和正文页同名

因此解析时不能只看一个字符串相等，需要综合：

- label
- aliases
- title
- 页码邻近性
- 同文档范围

### 风险 3：摄入成本会上升

给每个块多做一轮结构化抽取，会增加：

- token 成本
- 摄入耗时

可行的降本方式：

- 只对长度足够、包含显著结构线索的块做抽取
- 对明显无引用的纯正文块跳过
- 使用轻量模型做初筛，必要时再升级

### 风险 4：需要重建索引

因为 metadata schema 会变化，现有索引不足以承载新逻辑。

这意味着：

- 需要重新 ingest
- 需要重建至少文本和表格索引
- 图片分支如果也要支持引用导航，图片索引也应一并重建

## 10. 验收建议

### 10.1 单元测试

建议覆盖：

- 文本块能正确抽出 `anchors`
- 表格块能正确抽出 `references`
- 普通表格中的引用线索能解析到正文锚点
- unresolved 引用不会被错误强连
- 同名对象冲突时不会乱连

### 10.2 集成测试

建议至少验证四类问题：

1. `ST客户特殊要求`
   - 首轮可能命中目录表
   - 二跳后应补入 `Appendix 1A ~ 1G` 正文

2. 普通表格中的引用问题
   - 某张非目录表内含 `see xxx`
   - 二跳后应能补入目标正文

3. 正文段落中的引用问题
   - 某段正文写 `refer to ...`
   - 应能补入目标块

4. 直接对象查询
   - 用户直接问对象标题或编号
   - 应直接命中目标锚点正文，而不是只返回引用源

## 11. 最终建议

这次问题应当从“目录页专项修复”升级成“文档内引用导航能力建设”。

推荐落地顺序：

1. 在摄入阶段新增 LLM 抽取的 `anchors/references`
2. 在同文档内做一次 reference resolution
3. 把解析结果写入三类索引 metadata
4. 在查询阶段增加轻量二跳扩展
5. 在回答阶段优先消费目标节点，而不是引用源节点

一句话总结：

**不要让代码去识别 `Appendix`、`Notice`、`Reference` 这些词；让模型识别“谁定义了对象、谁引用了对象”，让系统沿关系去检索。**
