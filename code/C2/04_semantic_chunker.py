"""
================================================================================
语义感知文本分割器 - SemanticChunker 示例
================================================================================
功能说明：展示如何使用 SemanticChunker 基于语义相似度分割文本。
          与基于规则的分块方式不同，语义分割器使用嵌入模型计算文本片段
          之间的语义相似度，在语义边界处进行分割。

适用场景：需要保持内容连贯性的文档、主题变化明显的文章、专业文档

技术栈：LangChain SemanticChunker + HuggingFaceEmbeddings

优点：生成的每个块都是语义连贯的，不会在主题中间截断
缺点：需要嵌入模型，计算开销较大，速度较慢
================================================================================
"""

# ============================================================================
# 导入必要的库
# ============================================================================

from langchain_experimental.text_splitter import SemanticChunker
# SemanticChunker: 语义感知分割器
# 功能：基于语义相似度智能分割文本，保持内容的语义连贯性
#
# 工作原理：
#   1. 将文本分割成较小的句子
#   2. 使用嵌入模型计算相邻句子之间的语义相似度
#   3. 在相似度较低的边界处进行分割（即主题变化的地方）
#   4. 合并语义相似的句子形成最终的分块
#
# 参数说明：
#   - embeddings: 嵌入模型实例
#   - breakpoint_threshold_type: 断点阈值类型
#       决定如何计算分割阈值，可选值：
#       - "percentile": 基于百分位数（默认，推荐）
#       - "standard_deviation": 基于标准差
#       - "interquartile": 基于四分位距
#       - "gradient": 基于梯度变化
#   - number_of_chunks: 初始分割的句子数量（影响粒度）
#   - buffer_size: 缓冲区大小（用于计算相似度）
#
# 与其他分割器的对比：
#   - CharacterTextSplitter: 按固定字符数分割，不考虑语义
#   - RecursiveCharacterTextSplitter: 按分隔符优先级分割，保持基本结构
#   - SemanticChunker: 按语义相似度分割，保持内容连贯性（最智能）

from langchain_community.embeddings import HuggingFaceEmbeddings
# HuggingFaceEmbeddings: Hugging Face 嵌入模型
# 用于将文本转换为向量表示，计算语义相似度

from langchain_community.document_loaders import TextLoader
# TextLoader: 纯文本文件加载器


# ============================================================================
# 第一步：初始化嵌入模型
# ============================================================================

# 创建中文嵌入模型
# BAAI/bge-small-zh-v1.5 是一个轻量级的中文嵌入模型
# 参数说明：
#   - model_name: 模型名称
#   - model_kwargs: 模型参数
#       - device: 运行设备
#           - 'cpu': CPU 运行（兼容性好）
#           - 'cuda': GPU 运行（速度快，需要显卡）
#   - encode_kwargs: 编码参数
#       - normalize_embeddings: 归一化向量
#           - True: 使用余弦相似度（推荐）
#           - False: 使用点积相似度
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)


# ============================================================================
# 第二步：初始化语义分割器
# ============================================================================

# 创建语义分割器
# breakpoint_threshold_type 参数说明：
#
# 1. "percentile" (百分位数) - 推荐
#    - 计算所有相邻句子对的相似度
#    - 选择指定百分位数作为阈值
#    - 相似度低于阈值的边界作为分割点
#    - 示例：如果 percentile=50，则在相似度低于中位数的地方分割
#
# 2. "standard_deviation" (标准差)
#    - 计算相似度的均值和标准差
#    - 阈值 = 均值 - 标准差
#    - 适合相似度分布较均匀的情况
#
# 3. "interquartile" (四分位距)
#    - 使用 IQR (四分位距) 方法
#    - 对异常值较鲁棒
#
# 4. "gradient" (梯度)
#    - 基于相似度的梯度变化
#    - 在梯度变化最大的地方分割
text_splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile"  # 使用百分位数阈值
    # 也可以尝试其他值：
    # "standard_deviation"  # 基于标准差
    # "interquartile"       # 基于四分位距
    # "gradient"           # 基于梯度
)


# ============================================================================
# 第三步：文档加载
# ============================================================================

# 创建文本加载器
loader = TextLoader("../../data/C2/txt/蜂医.txt", encoding="utf-8")

# 加载文档
documents = loader.load()


# ============================================================================
# 第四步：执行语义分割
# ============================================================================

# 执行语义分割
# split_documents() 会：
#   1. 将文档分割成句子
#   2. 计算相邻句子之间的语义相似度
#   3. 根据阈值确定分割点
#   4. 合并语义相似的句子形成块
#   5. 返回分割后的 Document 列表
docs = text_splitter.split_documents(documents)


# ============================================================================
# 第五步：结果展示
# ============================================================================

# 打印分割结果统计
print(f"文本被切分为 {len(docs)} 个块。\n")

# 展示前 2 个块的内容
print("--- 前2个块内容示例 ---")
for i, chunk in enumerate(docs[:2]):
    print("=" * 60)
    print(f'块 {i+1} (长度: {len(chunk.page_content)}):')
    print(f'"{chunk.page_content}"')


# ============================================================================
# 伪流程图：语义分割流程
# ============================================================================
"""
┌─────────────────────────────────────────────────────────────────────┐
│                       语义分割流程                                    │
└─────────────────────────────────────────────────────────────────────┘

  输入: 长文本文档
  ┌──────────────────────────────────────────────────────────────┐
  │                                                              │
  │  [1. 初始化嵌入模型]                                          │
  │      ↓                                                       │
  │  HuggingFaceEmbeddings("BAAI/bge-small-zh-v1.5")             │
  │      ↓                                                       │
  │  用于计算文本的语义向量表示                                    │
  │                                                              │
  │  [2. 初始化语义分割器]                                        │
  │      ↓                                                       │
  │  SemanticChunker(                                           │
  │      embeddings=embeddings,                                  │
  │      breakpoint_threshold_type="percentile"                  │
  │  )                                                           │
  │      ↓                                                       │
  │                                                              │
  │  [3. 句子分割]                                                │
  │      ↓                                                       │
  │  原文 → [句子1, 句子2, 句子3, ..., 句子N]                     │
  │      ↓                                                       │
  │                                                              │
  │  [4. 计算语义相似度]                                          │
  │      ↓                                                       │
  │  ┌─────────────────────────────────────────────────────┐    │
  │  │                                                     │    │
  │  │  句子1 ──┐                                          │    │
  │  │         ├── 相似度: 0.85 (高) ── 保持               │    │
  │  │  句子2 ──┘                                          │    │
  │  │                                                     │    │
  │  │  句子2 ──┐                                          │    │
  │  │         ├── 相似度: 0.92 (高) ── 保持               │    │
  │  │  句子3 ──┘                                          │    │
  │  │                                                     │    │
  │  │  句子3 ──┐                                          │    │
  │  │         ├── 相似度: 0.35 (低) ── 分割!  ← 边界       │    │
  │  │  句子4 ──┘                                          │    │
  │  │                                                     │    │
  │  │  句子4 ──┐                                          │    │
  │  │         ├── 相似度: 0.88 (高) ── 保持               │    │
  │  │  句子5 ──┘                                          │    │
  │  │                                                     │    │
  │  └─────────────────────────────────────────────────────┘    │
  │      ↓                                                       │
  │                                                              │
  │  [5. 根据阈值确定分割点]                                      │
  │      ↓                                                       │
  │  使用 "percentile" 方法：                                     │
  │  - 计算所有相似度的百分位数                                   │
  │  - 相似度低于阈值的边界作为分割点                              │
  │      ↓                                                       │
  │                                                              │
  │  [6. 合并语义相似的句子]                                      │
  │      ↓                                                       │
  │  ┌─────────────────────────────────────────────────────┐    │
  │  │                                                     │    │
  │  │  块1: 句子1 + 句子2 + 句子3  (主题: 医疗知识)         │    │
  │  │                                                     │    │
  │  │  块2: 句子4 + 句子5 + 句子6  (主题: 蜂蜜功效)         │    │
  │  │                                                     │    │
  │  │  块3: 句子7 + 句子8         (主题: 使用方法)         │    │
  │  │                                                     │    │
  │  └─────────────────────────────────────────────────────┘    │
  │      ↓                                                       │
  │  docs (语义连贯的文本块)                                      │
  │      ↓                                                       │
  │  输出: 按语义边界分割的文本块                                  │
  │                                                              │
  └──────────────────────────────────────────────────────────────┘

  相似度阈值类型对比：
  ┌──────────────────────────────────────────────────────────────┐
  │                                                              │
  │  百分位数 (percentile) - 推荐                                │
  │  ┌────────────────────────────────────────────────────┐      │
  │  │ 相似度: [0.9, 0.8, 0.7, 0.4, 0.3, 0.2, 0.1]       │      │
  │  │ 阈值 = 50% = 0.4                                    │      │
  │  │ 分割点: 相似度 < 0.4 的地方                          │      │
  │  └────────────────────────────────────────────────────┘      │
  │                                                              │
  │  标准差 (standard_deviation)                                 │
  │  ┌────────────────────────────────────────────────────┐      │
  │  │ 均值 = 0.5, 标准差 = 0.3                             │      │
  │  │ 阈值 = 0.5 - 0.3 = 0.2                              │      │
  │  │ 分割点: 相似度 < 0.2 的地方                          │      │
  │  └────────────────────────────────────────────────────┘      │
  │                                                              │
  │  四分位距 (interquartile)                                    │
  │  ┌────────────────────────────────────────────────────┐      │
  │  │ Q1 = 0.25, Q3 = 0.75                                 │      │
  │  │ 阈值 = Q1 = 0.25                                    │      │
  │  │ 分割点: 相似度 < 0.25 的地方                         │      │
  │  └────────────────────────────────────────────────────┘      │
  │                                                              │
  └──────────────────────────────────────────────────────────────┘
"""


# ============================================================================
# 进阶用法说明
# ============================================================================
"""
# 1. 调整分割粒度
from langchain_experimental.text_splitter import SemanticChunker

# 更细粒度的分割
fine_splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile",
    buffer_size=1,  # 较小的缓冲区，更敏感
    number_of_chunks=20  # 初始分割更多句子
)

# 更粗粒度的分割
coarse_splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile",
    buffer_size=5,  # 较大的缓冲区，更平滑
    number_of_chunks=5  # 初始分割更少句子
)

# 2. 尝试不同的阈值类型
splitter_sd = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="standard_deviation"
)

splitter_iqr = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="interquartile"
)

splitter_grad = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="gradient"
)

# 3. 监控分割过程
from langchain_community.embeddings import HuggingFaceEmbeddings
import numpy as np

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'}
)

def semantic_split_with_analysis(docs):
    splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile"
    )
    chunks = splitter.split_documents(docs)

    # 分析每个块的语义连贯性
    for i, chunk in enumerate(chunks[:3]):
        content = chunk.page_content
        # 计算块内部的语义变化
        sentences = content.split('。')
        if len(sentences) > 1:
            vecs = embeddings.embed_documents(sentences)
            # 计算相邻句子的相似度
            similarities = []
            for j in range(len(vecs) - 1):
                sim = np.dot(vecs[j], vecs[j+1])
                similarities.append(sim)

            avg_sim = np.mean(similarities)
            print(f"块{i+1}: {len(sentences)}句, 平均相似度={avg_sim:.3f}")

    return chunks

# 4. 结合递归分割使用
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 先用递归分割预处理，再用语义分割精化
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

semantic_splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile"
)

# 第一步：递归分割
pre_chunks = recursive_splitter.split_documents(docs)
# 第二步：语义分割
final_chunks = semantic_splitter.split_documents(pre_chunks)

# 5. 使用不同的嵌入模型
from langchain_community.embeddings import OpenAIEmbeddings

# OpenAI 嵌入（英文效果好）
openai_embeddings = OpenAIEmbeddings()
splitter_openai = SemanticChunker(openai_embeddings)

# 多语言模型
from langchain_community.embeddings import HuggingFaceEmbeddings

multilingual_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
splitter_multi = SemanticChunker(multilingual_embeddings)
"""
