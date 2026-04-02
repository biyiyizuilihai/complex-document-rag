"""
============================================================
Step 1: 图片描述生成
============================================================
功能：
    将每张 Mermaid 流程图发送给多模态 LLM（GPT-4o），
    生成结构化的自然语言描述。这是整个系统的基础。

原理：
    多模态 LLM 能"看懂"图片内容。我们通过精心设计的 Prompt，
    引导它返回结构化 JSON，包含：
    - summary:              一句话概括流程图作用
    - detailed_description: 详细描述每个步骤（用于生成 embedding）
    - nodes:                节点名称列表
    - external_references:  引用的外部资源
    - tags:                 相关标签

    为什么需要生成描述而不是直接用图片做 embedding？
    因为当前的图片 embedding 模型对流程图这类结构化图片的语义理解
    远不如文本 embedding 精确。通过 LLM 先"翻译"成文本，再做
    文本 embedding，能获得更好的检索效果。

使用方式：
    python step1_image_description.py

前置条件：
    1. 设置 OPENAI_API_KEY 环境变量
    2. 在 ../images/ 目录放入测试图片（.png 或 .jpg）
============================================================
"""

import base64
import json
import os
import sys

from complex_document_rag.core.config import OPENAI_API_KEY, OPENAI_BASE_URL, MULTIMODAL_LLM_MODEL
from complex_document_rag.core.models import ImageDescription
from complex_document_rag.core.paths import project_root_from_file
from complex_document_rag.ingestion.files import list_image_filenames
from complex_document_rag.ingestion.image_records import (
    build_description_payload,
    make_image_id,
)

# ============================================================
# 为什么直接使用 openai 客户端而不是 LlamaIndex 封装？
# ============================================================
# LlamaIndex 提供的 OpenAIMultiModal 和 OpenAI 类会校验模型名称，
# 只允许 OpenAI 官方模型（gpt-4o 等），Qwen 等第三方模型会被拒绝。
#
# DashScope（通义千问）提供了完全兼容 OpenAI 的接口，
# 所以直接使用 openai Python 客户端是最可靠的方案：
#   - 不校验模型名称，任意模型都能用
#   - 完整支持多模态消息格式（文本 + 图片）
#   - openai 包已作为 llama-index 的依赖自动安装
# ============================================================
from openai import OpenAI as OpenAIClient


# ============================================================
# 提示词设计（核心）
# ============================================================
# 这个 Prompt 的质量直接决定了：
#   1. 描述的准确性 → 影响向量检索的精度
#   2. 引用提取的完整性 → 影响知识图谱的覆盖率
#   3. 标签的合理性 → 影响元数据过滤的效果
#
# 优化建议：
#   - 先用 10 张图测试效果，确认质量满意后再批量处理
#   - 如果领域有专业术语，在 Prompt 中提供术语表
#   - 对于复杂图片，可以让 LLM 分区域分析后合并
# ============================================================

DESCRIBE_PROMPT = """
请分析这张流程图图片，返回 JSON 格式。

重要要求：
- summary: 一句话，20字以内
- detailed_description: 精炼概括，100字以内，只写核心流程和关键步骤，不要展开细节
- nodes: 只列出关键节点名称（原文）
- external_references: 只提取图中明确提到的外部文档/系统/接口名称
- tags: 3-5个关键词

{
  "summary": "一句话概括流程图主题",
  "detailed_description": "精炼描述核心流程（100字以内）",
  "nodes": ["关键节点名称"],
  "external_references": [
    {"target": "外部资源名称", "context": "出现位置"}
  ],
  "tags": ["关键词"]
}
只返回 JSON，不要其他内容。
"""


# ============================================================
# 初始化 OpenAI 兼容客户端
# ============================================================
# 通过 base_url 参数指向 DashScope API，实现对 Qwen 模型的调用。
# 同样适用于其他兼容 OpenAI 接口的服务（如 Ollama、vLLM 等）。
# ============================================================

client = OpenAIClient(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
)
PROJECT_ROOT = project_root_from_file(__file__)


def describe_image(image_path: str) -> ImageDescription:
    """
    为单张图片生成结构化描述
    --------------------------------------------------------
    流程：
        1. 读取图片并编码为 base64
        2. 构造 OpenAI 格式的多模态消息（文本 + 图片）
        3. 调用 Chat Completions API
        4. 解析返回的 JSON 并转换为 ImageDescription

    参数:
        image_path: 图片文件的本地路径

    返回:
        ImageDescription 对象，包含结构化描述信息
    """
    # 读取图片并编码为 base64
    with open(image_path, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode("utf-8")

    # 根据后缀确定 MIME 类型
    ext = os.path.splitext(image_path)[1].lower()
    mime_type = "image/png" if ext == ".png" else "image/jpeg"

    # 调用多模态 Chat Completions API
    # 消息格式完全兼容 OpenAI — DashScope 原生支持
    response = client.chat.completions.create(
        model=MULTIMODAL_LLM_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": DESCRIBE_PROMPT},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{base64_image}"
                    },
                },
            ],
        }],
        max_tokens=2000,
    )

    # 提取返回的文本内容
    result_text = response.choices[0].message.content.strip()

    # 清理可能包含的 markdown 代码块标记
    # 有些模型会返回 ```json ... ``` 包裹的 JSON
    if result_text.startswith("```json"):
        result_text = result_text[7:]
    elif result_text.startswith("```"):
        result_text = result_text[3:]
    if result_text.endswith("```"):
        result_text = result_text[:-3]
    result_text = result_text.strip()

    # 解析 JSON 响应并转换为数据对象
    raw_data = json.loads(result_text)
    return ImageDescription.from_dict(raw_data)


def batch_describe_images(image_dir: str) -> dict[str, dict]:
    """
    批量为目录下所有图片生成描述
    --------------------------------------------------------
    遍历指定目录下的所有 .png 和 .jpg 文件，逐个生成描述。
    这是当前仓库的简单串行实现，后续如需大规模并行可接入任务队列。

    参数:
        image_dir: 图片目录路径

    返回:
        dict，key 为图片ID（如 "img_00000"），value 为可直接保存的描述记录

    注意:
        - 当前默认以本地调试为主，串行处理即可
        - 建议抽样检查 5% 的描述结果，确认质量
    """
    results = {}

    # 只对真实图片编号，避免 .gitkeep 等占位文件打乱 ID
    filenames = list_image_filenames(image_dir)

    for idx, filename in enumerate(filenames):
        img_path = os.path.join(image_dir, filename)
        img_id = make_image_id(idx)

        print(f"[{idx + 1}] 正在处理: {filename} -> {img_id}")

        try:
            desc = describe_image(img_path)
            results[img_id] = build_description_payload(desc, img_path, PROJECT_ROOT)
            print(f"    ✓ 摘要: {desc.summary[:50]}...")
        except json.JSONDecodeError as e:
            print(f"    ✗ JSON 解析失败: {e}")
        except Exception as e:
            print(f"    ✗ 处理失败: {e}")

    print(f"\n完成！成功处理 {len(results)}/{len(filenames)} 张图片")
    return results


def save_descriptions(results: dict[str, dict],
                      output_path: str = "image_descriptions.json"):
    """
    将描述结果保存到 JSON 文件
    --------------------------------------------------------
    保存后可以复用，避免重复调用 LLM（节省成本）。
    后续 step3 会从这个文件读取描述并入库。
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"描述结果已保存到: {output_path}")


def main() -> int:
    """多模态 LLM 图片描述生成命令行入口。"""
    # 检查 API Key
    if not OPENAI_API_KEY:
        print("错误: 请设置 OPENAI_API_KEY 环境变量")
        print("方法1: export OPENAI_API_KEY='sk-xxx'")
        print("方法2: 在 ../.env 文件中配置")
        return 1

    # 检查图片目录
    images_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "images"
    )
    if not os.path.exists(images_path):
        print(f"错误: 图片目录不存在: {images_path}")
        print("请创建目录并放入测试图片（.png 或 .jpg）")
        return 1

    image_files = list_image_filenames(images_path)
    if not image_files:
        print(f"错误: 图片目录为空: {images_path}")
        print("请放入 10 张 Mermaid 流程图作为测试数据")
        return 1

    print(f"找到 {len(image_files)} 张图片，开始生成描述...\n")

    # 批量生成描述
    results = batch_describe_images(images_path)

    # 保存结果（避免重复调用 LLM）
    if results:
        save_descriptions(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
