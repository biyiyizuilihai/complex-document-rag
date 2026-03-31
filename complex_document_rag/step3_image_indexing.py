"""
============================================================
Step 3: 图片描述入库
============================================================
功能：
    将 step1 生成的图片结构化描述作为文本节点（TextNode）
    存入 Qdrant 向量数据库的 image_descriptions 集合。

原理：
    图片本身存储在对象存储（MinIO/S3）中，
    但向量数据库中需要存的是它的"文本替身"——即 LLM 生成的描述。

    检索时的流程是：
        用户查询 → embedding → 在 image_descriptions 中搜索
        → 找到匹配的描述 → 通过 metadata 中的 image_id 定位原图
        → 将原图传给多模态 LLM 生成最终答案

    所以这一步的关键是：
        1. 用 detailed_description 作为文本内容（用于生成 embedding）
        2. 在 metadata 中保留所有附加信息（image_id, path, tags 等）

使用方式：
    python step3_image_indexing.py

前置条件：
    1. 完成 step1（生成了 image_descriptions.json）
    2. Qdrant 已启动
============================================================
"""

import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_IMAGE_COLLECTION,
    EMBEDDING_MODEL,
)
from model_provider_utils import create_embedding_model
from complex_document_rag.pipeline_utils import (
    project_root_from_file,
    resolve_source_image_path,
)

from llama_index.core.schema import TextNode
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient


PROJECT_ROOT = project_root_from_file(__file__)


# ============================================================
# 初始化 Qdrant 图片描述集合
# ============================================================
# 注意：图片描述和文本文档使用不同的 collection，
# 这样可以独立管理、独立检索，也方便后续按不同权重混合结果。
# ============================================================

qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

image_desc_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name=QDRANT_IMAGE_COLLECTION,  # 默认 "image_descriptions"
)

embed_model = create_embedding_model(
    model_name=EMBEDDING_MODEL,
    api_key=OPENAI_API_KEY,
    api_base=OPENAI_BASE_URL,
)


def index_single_image(img_id: str, desc_data: dict,
                       img_path: str) -> TextNode:
    """
    将单张图片的描述作为 TextNode 存入向量库
    --------------------------------------------------------
    核心思路：
        - text 字段 = detailed_description（用于 embedding 和检索）
        - metadata 字段 = 所有附加信息（用于过滤和定位原图）

    为什么用 TextNode 而不是 ImageNode？
        因为我们存的是"图片的文本描述"，本质上是文本。
        ImageNode 用于直接存储图片数据，不适合这里的场景。

    参数:
        img_id:    图片唯一标识（如 "img_00001"）
        desc_data: 图片描述字典（step1 的输出）
        img_path:  图片原文件路径

    返回:
        创建的 TextNode 对象
    """
    node = TextNode(
        # text 是核心内容，embedding 就是根据这个字段生成的
        text=desc_data["detailed_description"],

        # metadata 不参与 embedding 计算，但会随搜索结果一起返回
        # 这些信息用于：
        #   - image_id: 关联知识图谱中的节点
        #   - image_path: 定位原始图片文件
        #   - summary: 快速预览
        #   - tags: 支持元数据过滤
        #   - type: 区分文本和图片结果
        metadata={
            "image_id": img_id,
            "source_doc_id": desc_data.get("source_doc_id") or desc_data.get("doc_id", ""),
            "image_path": img_path,
            "image_filename": desc_data.get(
                "source_image_filename", os.path.basename(img_path)
            ),
            "image_rel_path": desc_data.get("source_image_path", ""),
            "doc_id": desc_data.get("doc_id", ""),
            "source_path": desc_data.get("source_path", ""),
            "source_document_path": desc_data.get("source_document_path", ""),
            "page_no": desc_data.get("page_no"),
            "page_label": desc_data.get("page_label", ""),
            "origin": desc_data.get("origin", "image_description"),
            "block_type": desc_data.get("block_type", "image"),
            "summary": desc_data.get("summary", ""),
            "tags": desc_data.get("tags", []),
            "nodes_in_chart": desc_data.get("nodes", []),
            "type": "image_description",  # 标记类型，便于混合检索时区分
        },
    )
    return node


def batch_index_images(descriptions_file: str = "image_descriptions.json"):
    """
    批量将图片描述存入向量库
    --------------------------------------------------------
    从 step1 保存的 JSON 文件读取所有描述，
    批量创建 TextNode 并存入 Qdrant。

    参数:
        descriptions_file: step1 输出的 JSON 文件路径
    """
    # 读取 step1 的输出
    if not os.path.exists(descriptions_file):
        print(f"错误: 找不到描述文件: {descriptions_file}")
        print("请先运行 step1_image_description.py 生成描述")
        sys.exit(1)

    with open(descriptions_file, "r", encoding="utf-8") as f:
        all_descriptions = json.load(f)

    print(f"读取了 {len(all_descriptions)} 条图片描述")

    # 构建所有 TextNode
    nodes = []

    for img_id, desc_data in all_descriptions.items():
        try:
            img_path = resolve_source_image_path(img_id, desc_data, PROJECT_ROOT)
        except FileNotFoundError as exc:
            print(f"  ✗ {img_id}: {exc}")
            continue

        node = index_single_image(img_id, desc_data, img_path)
        nodes.append(node)
        print(f"  ✓ {img_id}: {desc_data.get('summary', '')[:40]}...")

    if not nodes:
        print("错误: 没有可入库的图片描述。请重新运行 step1 生成带源路径的新描述文件。")
        sys.exit(1)

    # 批量存入 Qdrant
    # 使用 VectorStoreIndex 会自动处理 embedding 生成和存储
    storage_context = StorageContext.from_defaults(
        vector_store=image_desc_store
    )

    print(f"\n正在生成 embedding 并存入 Qdrant...")
    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )

    print(f"\n✓ 图片描述索引构建完成！")
    print(f"  集合名称: {QDRANT_IMAGE_COLLECTION}")
    print(f"  共 {len(nodes)} 条记录")
    print(f"  Qdrant Dashboard: http://{QDRANT_HOST}:{QDRANT_PORT}/dashboard")

    return index


def load_image_index() -> VectorStoreIndex:
    """
    加载已有的图片描述索引
    --------------------------------------------------------
    在 step4 查询时使用，无需重新构建。
    """
    index = VectorStoreIndex.from_vector_store(
        vector_store=image_desc_store,
        embed_model=embed_model,
    )
    print(f"✓ 已加载图片描述索引: {QDRANT_IMAGE_COLLECTION}")
    return index


# ============================================================
# 主函数
# ============================================================
if __name__ == "__main__":
    if not OPENAI_API_KEY:
        print("错误: 请设置 OPENAI_API_KEY 环境变量")
        sys.exit(1)

    batch_index_images()
