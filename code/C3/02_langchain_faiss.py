"""
================================================================================
FAISS 向量存储 - LangChain 集成示例
================================================================================
功能说明：展示如何使用 FAISS（Facebook AI Similarity Search）创建、
          保存和加载向量索引。FAISS 是一个高效的相似性搜索库，
          适合大规模向量检索。

适用场景：大规模向量检索、需要持久化索引的场景

技术栈：LangChain + FAISS + HuggingFaceEmbeddings

优点：快速、高效、支持 GPU 加速、可持久化到磁盘
================================================================================
"""

# ============================================================================
# 导入必要的库
# ============================================================================

from langchain_community.vectorstores import FAISS
# FAISS: Facebook AI Similarity Search
# 功能：高效的向量相似性搜索和密集向量聚类
#
# 主要特点：
#   - 速度快：支持多种索引算法（如 IndexFlatIP、IndexIVFFlat）
#   - 可扩展：支持十亿级别的向量搜索
#   - 灵活：支持 CPU 和 GPU 计算
#   - 持久化：可以保存和加载索引
#
# 常用方法：
#   - from_documents(): 从文档列表创建 FAISS 索引
#   - from_texts(): 从文本列表创建 FAISS 索引
#   - save_local(): 保存索引到本地磁盘
#   - load_local(): 从本地磁盘加载索引
#   - similarity_search(): 相似度搜索
#   - similarity_search_with_score(): 带分数的相似度搜索
#   - max_marginal_relevance_search(): 最大边际相关性搜索（提高多样性）

from langchain_community.embeddings import HuggingFaceEmbeddings
# HuggingFaceEmbeddings: Hugging Face 嵌入模型
# 用于将文本转换为向量表示

from langchain_core.documents import Document
# Document: 文档对象
# 包含 page_content（文本内容）和 metadata（元数据）


# ============================================================================
# 第一步：准备示例数据和嵌入模型
# ============================================================================

# 示例文本列表
# 这些文本将被转换为向量并存储在 FAISS 索引中
texts = [
    "张三是法外狂徒",
    "FAISS是一个用于高效相似性搜索和密集向量聚类的库。",
    "LangChain是一个用于开发由语言模型驱动的应用程序的框架。"
]

# 将文本转换为 Document 对象列表
# 每个 Document 对象包含：
#   - page_content: 文档的文本内容
#   - metadata: 元数据字典（可选，如来源、作者等）
docs = [Document(page_content=t) for t in texts]

# 创建嵌入模型
# 用于将文本转换为向量表示
# BAAI/bge-small-zh-v1.5 是一个轻量级的中文嵌入模型
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'}
)

# 此时，每个文本都会被转换为一个向量（如 768 维的浮点数数组）
# 例如：[0.12, -0.34, 0.56, ..., 0.78]


# ============================================================================
# 第二步：创建 FAISS 向量存储并保存到本地
# ============================================================================

# 从文档列表创建 FAISS 索引
# FAISS.from_documents() 会：
#   1. 使用嵌入模型将每个文档的 page_content 转换为向量
#   2. 将向量存储在 FAISS 索引中
#   3. 保留文档与向量的映射关系
#   4. 返回 FAISS 对象
vectorstore = FAISS.from_documents(docs, embeddings)

# 定义本地保存路径
local_faiss_path = "./faiss_index_store"
# 这会创建以下文件：
#   - faiss_index_store/index.faiss  # FAISS 索引文件
#   - faiss_index_store/index.pkl    # 元数据文件

# 将 FAISS 索引保存到本地磁盘
# save_local() 方法会：
#   1. 将 FAISS 索引写入磁盘
#   2. 保存文档元数据
#   3. 保存索引配置信息
vectorstore.save_local(local_faiss_path)

print(f"FAISS index has been saved to {local_faiss_path}")


# ============================================================================
# 第三步：加载索引并执行查询
# ============================================================================

# 从本地磁盘加载 FAISS 索引
# load_local() 参数说明：
#   - folder_path: 索引所在目录
#   - embeddings: 嵌入模型（必须与创建索引时使用的模型相同）
#   - allow_dangerous_deserialization: 允许反序列化
#       注意：加载 pickle 文件可能存在安全风险，因此需要显式允许
loaded_vectorstore = FAISS.load_local(
    local_faiss_path,
    embeddings,
    allow_dangerous_deserialization=True
)

# 定义查询问题
query = "FAISS是做什么的？"

# 执行相似度搜索
# similarity_search() 参数说明：
#   - query: 查询文本
#   - k: 返回最相似的 k 个文档（默认 4）
# 返回值：List[Document]，按相似度排序
results = loaded_vectorstore.similarity_search(query, k=1)

# 打印查询结果
print(f"\n查询: '{query}'")
print("相似度最高的文档:")
for doc in results:
    print(f"- {doc.page_content}")


# ============================================================================
# 伪流程图：FAISS 索引创建与查询流程
# ============================================================================
"""
┌─────────────────────────────────────────────────────────────────────┐
│                    FAISS 索引创建与查询流程                           │
└─────────────────────────────────────────────────────────────────────┘

  阶段 1: 索引创建（Index Creation）
  ┌──────────────────────────────────────────────────────────────┐
  │                                                              │
  │  [1. 准备文档]                                                │
  │      ↓                                                       │
  │  texts = ["张三是法外狂徒", "FAISS是...", ...]                │
  │      ↓                                                       │
  │  docs = [Document(page_content=t) for t in texts]           │
  │      ↓                                                       │
  │                                                              │
  │  [2. 文档向量化]                                             │
  │      ↓                                                       │
  │  HuggingFaceEmbeddings                                      │
  │      ↓                                                       │
  │  ┌──────────────────────────────────────────────────┐       │
  │  │  文本1 → 向量1: [0.12, -0.34, 0.56, ...]        │       │
  │  │  文本2 → 向量2: [-0.23, 0.45, -0.67, ...]        │       │
  │  │  文本3 → 向量3: [0.78, 0.12, -0.34, ...]        │       │
  │  └──────────────────────────────────────────────────┘       │
  │      ↓                                                       │
  │                                                              │
  │  [3. 构建 FAISS 索引]                                        │
  │      ↓                                                       │
  │  FAISS.from_documents(docs, embeddings)                     │
  │      ↓                                                       │
  │  创建索引数据结构                                             │
  │      ↓                                                       │
  │  vectorstore (FAISS 对象)                                    │
  │                                                              │
  │  [4. 持久化到磁盘]                                           │
  │      ↓                                                       │
  │  vectorstore.save_local("./faiss_index_store")               │
  │      ↓                                                       │
  │  ┌──────────────────────────────────────────────────┐       │
  │  │  faiss_index_store/                              │       │
  │  │    ├── index.faiss  (FAISS 索引文件)            │       │
  │  │    └── index.pkl    (元数据文件)                │       │
  │  └──────────────────────────────────────────────────┘       │
  │                                                              │
  └──────────────────────────────────────────────────────────────┘

  阶段 2: 索引加载与查询（Index Loading & Query）
  ┌──────────────────────────────────────────────────────────────┐
  │                                                              │
  │  [5. 加载索引]                                                │
  │      ↓                                                       │
  │  FAISS.load_local(                                           │
  │      "./faiss_index_store",                                  │
  │      embeddings,                                             │
  │      allow_dangerous_deserialization=True                    │
  │  )                                                           │
  │      ↓                                                       │
  │  loaded_vectorstore (FAISS 对象)                             │
  │                                                              │
  │  [6. 执行查询]                                               │
  │      ↓                                                       │
  │  query = "FAISS是做什么的？"                                 │
  │      ↓                                                       │
  │  [7. 查询向量化]                                             │
  │      ↓                                                       │
  │  query → 向量: [0.34, -0.12, 0.78, ...]                      │
  │      ↓                                                       │
  │                                                              │
  │  [8. 相似度搜索]                                             │
  │      ↓                                                       │
  │  similarity_search(query, k=1)                              │
  │      ↓                                                       │
  │  ┌──────────────────────────────────────────────────┐       │
  │  │                                                  │       │
  │  │  计算: query向量 vs 所有文档向量                   │       │
  │  │                                                  │       │
  │  │  相似度1 = dot(query向量, 向量1) = 0.12           │       │
  │  │  相似度2 = dot(query向量, 向量2) = 0.89 ← 最高    │       │
  │  │  相似度3 = dot(query向量, 向量3) = 0.34           │       │
  │  │                                                  │       │
  │  └──────────────────────────────────────────────────┘       │
  │      ↓                                                       │
  │                                                              │
  │  [9. 返回结果]                                               │
  │      ↓                                                       │
  │  results = [Document("FAISS是...")]                          │
  │      ↓                                                       │
  │  输出: 相似度最高的文档                                       │
  │                                                              │
  └──────────────────────────────────────────────────────────────┘
"""


# ============================================================================
# 进阶用法说明
# ============================================================================
"""
# 1. 带分数的相似度搜索
results_with_scores = loaded_vectorstore.similarity_search_with_score(
    "FAISS是做什么的？", k=2
)
for doc, score in results_with_scores:
    print(f"分数: {score:.4f}, 内容: {doc.page_content}")

# 2. 最大边际相关性搜索（MMR）
# 提高结果的多样性，避免返回过于相似的文档
mmr_results = loaded_vectorstore.max_marginal_relevance_search(
    "FAISS是做什么的？",
    k=2,
    fetch_k=3  # 从前3个结果中选择k个最多样性的
)

# 3. 添加新文档到现有索引
new_docs = [Document(page_content="新添加的文档内容")]
loaded_vectorstore.add_documents(new_docs)
# 保存更新后的索引
loaded_vectorstore.save_local(local_faiss_path)

# 4. 从文本直接创建索引
from langchain_community.vectorstores import FAISS
vectorstore_from_texts = FAISS.from_texts(
    texts=["文本1", "文本2", "文本3"],
    embedding=embeddings
)

# 5. 使用不同的 FAISS 索引类型
# 默认使用 IndexFlatIP（内积索引）
# 可以使用 IVF（倒排文件索引）提高大规模搜索速度
import faiss

# 创建量化器
quantizer = faiss.IndexFlatL2(768)  # 768 是向量维度
index_ivf = faiss.IndexIVFFlat(quantizer, 768, 100)  # 100 个聚类中心

# 6. 过滤搜索
# 基于元数据过滤搜索结果
docs_with_metadata = [
    Document(page_content="Python教程", metadata={"category": "programming"}),
    Document(page_content="Java教程", metadata={"category": "programming"}),
    Document(page_content="烹饪技巧", metadata={"category": "cooking"})
]
vectorstore = FAISS.from_documents(docs_with_metadata, embeddings)

# 只搜索特定类别的文档
# 这需要自定义实现或使用其他支持过滤的向量存储

# 7. 批量搜索
queries = ["FAISS是做什么的？", "LangChain有什么用？"]
all_results = []
for query in queries:
    results = loaded_vectorstore.similarity_search(query, k=1)
    all_results.extend(results)
"""
