"""
================================================================================
递归字符文本分割器 - RecursiveCharacterTextSplitter 示例
================================================================================
功能说明：展示如何使用 RecursiveCharacterTextSplitter 智能分割文本。
          与固定字符数分割不同，递归分割器会按优先级尝试多种分隔符，
          尽可能保持文本的语义完整性。

适用场景：需要保持语义边界的文本、中英文混合文档、结构化文档

技术栈：LangChain RecursiveCharacterTextSplitter

推荐使用：这是最常用的文本分割器，能够在保持语义的同时控制块大小
================================================================================
"""

# ============================================================================
# 导入必要的库
# ============================================================================

from langchain.text_splitter import RecursiveCharacterTextSplitter
# RecursiveCharacterTextSplitter: 递归字符分割器
# 功能：按分隔符优先级递归地分割文本，尽可能保持语义完整性
#
# 参数说明：
#   - separators: 分隔符列表，按优先级从高到低排列
#       默认值: ["\n\n", "\n", " ", ""]
#       工作原理：
#         1. 首先尝试用第一个分隔符（如 "\n\n"）分割
#         2. 如果分割后的块仍超过 chunk_size，则用下一个分隔符（如 "\n"）
#         3. 依此类推，直到使用空字符串 ""（按字符分割）
#
#   - chunk_size: 每个块的最大字符数（默认 4000）
#   - chunk_overlap: 块之间的重叠字符数（默认 200）
#   - length_function: 计算文本长度的函数（默认 len）
#   - keep_separator: 是否保留分隔符（默认 False）
#
# 与 CharacterTextSplitter 的区别：
#   - CharacterTextSplitter: 只使用一个分隔符，不够灵活
#   - RecursiveCharacterTextSplitter: 尝试多个分隔符，更智能
#
# 示例：
#   separators=["\n\n", "\n", "。", "，", " ", ""]
#   表示：
#     1. 优先在段落之间分割（双换行）
#     2. 如果段落太大，在句子之间分割（单换行）
#     3. 如果句子太大，在中文句号处分割
#     4. 如果还太大，在中文逗号处分割
#     5. 最后才按字符强制分割

from langchain_community.document_loaders import TextLoader
# TextLoader: 纯文本文件加载器


# ============================================================================
# 第一步：文档加载
# ============================================================================

# 创建文本加载器
loader = TextLoader("../../data/C2/txt/蜂医.txt", encoding="utf-8")

# 加载文档
# 返回值：List[Document]
docs = loader.load()


# ============================================================================
# 第二步：初始化递归分割器
# ============================================================================

# 创建递归字符分割器
# 针对中英文混合文本，定义一个全面的分隔符列表
text_splitter = RecursiveCharacterTextSplitter(
    # 分隔符列表，按优先级排序
    separators=[
        "\n\n",   # 1. 段落分隔（双换行）- 优先级最高
        "\n",     # 2. 行分隔（单换行）
        "。",     # 3. 中文句号
        "，",     # 4. 中文逗号
        " ",      # 5. 空格（英文单词分隔）
        ""        # 6. 按字符强制分割 - 最后手段
    ],
    chunk_size=200,      # 每个块最多 200 个字符
    chunk_overlap=10     # 相邻块重叠 10 个字符
)

# 工作流程示例：
# 假设有一个 500 字符的段落：
#
# 原文: [第一段...]\n\n[第二段...]\n\n[第三段...] (500字)
#
# 步骤1: 按 "\n\n" 分割
#   结果: [第一段...] (150字), [第二段...] (200字), [第三段...] (150字)
#
# 步骤2: 检查每段大小
#   - 第一段: 150字 < 200 ✓ 保持完整
#   - 第二段: 200字 = 200 ✓ 保持完整
#   - 第三段: 150字 < 200 ✓ 保持完整
#
# 如果某段超过 200 字，则继续用 "\n" 分割
# 如果某行超过 200 字，则继续用 "。" 分割
# 依此类推...


# ============================================================================
# 第三步：执行分块
# ============================================================================

# 执行文本分割
# split_documents() 会：
#   1. 遍历所有文档
#   2. 对每个文档应用递归分割逻辑
#   3. 保留原始 metadata
#   4. 返回分割后的 Document 列表
chunks = text_splitter.split_documents(docs)


# ============================================================================
# 第四步：结果展示
# ============================================================================

# 打印分割结果统计
print(f"文本被切分为 {len(chunks)} 个块。\n")

# 展示前 5 个块的内容
print("--- 前5个块内容示例 ---")
for i, chunk in enumerate(chunks[:5]):
    print("=" * 60)
    # chunk 是 Document 对象
    # .page_content 包含文本内容
    # .metadata 包含元数据
    print(f'块 {i+1} (长度: {len(chunk.page_content)}): "{chunk.page_content}"')


# ============================================================================
# 伪流程图：递归字符分割流程
# ============================================================================
"""
┌─────────────────────────────────────────────────────────────────────┐
│                    递归字符分割流程                                   │
└─────────────────────────────────────────────────────────────────────┘

  输入: 长文本（包含段落、句子、词汇等结构）
  ┌──────────────────────────────────────────────────────────────┐
  │                                                              │
  │  [1. 文档加载]                                                │
  │      ↓                                                       │
  │  TextLoader → docs (Document 对象)                           │
  │      ↓                                                       │
  │                                                              │
  │  [2. 初始化递归分割器]                                        │
  │      ↓                                                       │
  │  RecursiveCharacterTextSplitter(                             │
  │      separators=["\n\n", "\n", "。", "，", " ", ""]          │
  │      chunk_size=200,                                         │
  │      chunk_overlap=10                                        │
  │  )                                                           │
  │      ↓                                                       │
  │                                                              │
  │  [3. 递归分割过程]                                            │
  │      ↓                                                       │
  │  ┌─────────────────────────────────────────────────────┐    │
  │  │                                                     │    │
  │  │  原文: 段落1\n\n段落2很长很长...                      │    │
  │  │       共500字，其中段落2有300字                       │    │
  │  │                                                     │    │
  │  │  ↓ 尝试分隔符1: "\n\n"                               │    │
  │  │                                                     │    │
  │  │  块A: 段落1 (150字) ✓                               │    │
  │  │  块B: 段落2很长很长... (300字) ✗ 超过200!            │    │
  │  │                                                     │    │
  │  │  ↓ 对块B继续尝试分隔符2: "\n"                        │    │
  │  │                                                     │    │
  │  │  块B1: 段落2的\n第一行 (100字) ✓                     │    │
  │  │  块B2: 第二行很长... (200字) ✓                      │    │
  │  │                                                     │    │
  │  │  最终结果:                                          │    │
  │  │    块1: 段落1 (150字)                               │    │
  │  │    块2: 段落2的\n第一行 (100字)                     │    │
  │  │    块3: 第二行很长... (200字)                        │    │
  │  │                                                     │    │
  │  └─────────────────────────────────────────────────────┘    │
  │      ↓                                                       │
  │  chunks (按语义边界分割的块)                                 │
  │                                                              │
  │  [4. 添加重叠]                                               │
  │      ↓                                                       │
  │  相邻块之间添加 10 个字符的重叠                               │
  │      ↓                                                       │
  │  输出: 保持语义完整性的文本块列表                              │
  │                                                              │
  └──────────────────────────────────────────────────────────────┘

  分隔符优先级示例：
  ┌──────────────────────────────────────────────────────────────┐
  │                                                              │
  │  separators = ["\n\n", "\n", "。", "，", " ", ""]            │
  │                                                              │
  │  文本: "第一段\n\n第二段第一句。第二句，还有内容"              │
  │                                                              │
  │  尝试1: 按 "\n\n" 分割                                       │
  │    → ["第一段", "第二段第一句。第二句，还有内容"]              │
  │                                                              │
  │  尝试2: 如果某块仍太大，按 "\n" 分割                          │
  │    → 继续细分...                                             │
  │                                                              │
  │  尝试3: 如果某行仍太大，按 "。" 分割                          │
  │    → 在句号处切分，保持句子完整                               │
  │                                                              │
  │  尝试4: 如果某句仍太大，按 "，" 分割                          │
  │    → 在逗号处切分，保持从句完整                               │
  │                                                              │
  │  尝试5: 最后按 " " 或 "" 强制分割                             │
  │    → 确保不超过 chunk_size                                   │
  │                                                              │
  └──────────────────────────────────────────────────────────────┘
"""


# ============================================================================
# 进阶用法说明
# ============================================================================
"""
# 1. 针对不同语言优化分隔符

# 英文文本
en_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
    chunk_size=1000,
    chunk_overlap=100
)

# 中文文本
zh_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
    chunk_size=500,
    chunk_overlap=50
)

# 中英文混合
mixed_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", "。", ". ", "！", "! ", "？", "? ", "；", "; ",
                "，", ", ", " ", ""],
    chunk_size=500,
    chunk_overlap=50
)

# 2. 代码文件分割
code_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\nclass ", "\n\ndef ", "\n\n", "\n", " ", ""],
    chunk_size=1000,
    chunk_overlap=100
)

# 3. Markdown 文档分割
md_splitter = RecursiveCharacterTextSplitter(
    separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],
    chunk_size=1000,
    chunk_overlap=100
)

# 4. 监控分割过程
from langchain.schema import Document

def split_with_info(docs):
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "。", " ", ""],
        chunk_size=200,
        chunk_overlap=10
    )
    chunks = splitter.split_documents(docs)

    # 分析分割质量
    for i, chunk in enumerate(chunks[:5]):
        content = chunk.page_content
        # 检查是否在不合适的位置截断
        ends_with_period = content.rstrip().endswith('。')
        ends_with_comma = content.rstrip().endswith('，')
        print(f"块{i+1}: 长度={len(content)}, "
              f"以句号结尾={ends_with_period}, "
              f"以逗号结尾={ends_with_comma}")

    return chunks

# 5. 自定义 length_function
import tiktoken

encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

def token_length(text: str) -> int:
    return len(encoding.encode(text))

token_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", "。", " ", ""],
    chunk_size=500,        # 500 tokens
    chunk_overlap=50,      # 50 tokens
    length_function=token_length
)

# 6. 保留分隔符
splitter_with_sep = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", "。", "，", " ", ""],
    chunk_size=200,
    chunk_overlap=10,
    keep_separator=True  # 在块的开头保留分隔符
)
"""
