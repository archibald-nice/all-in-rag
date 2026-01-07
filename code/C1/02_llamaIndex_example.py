"""
================================================================================
RAG 系统完整实现 - 基于 LlamaIndex 框架
================================================================================
功能说明：这是一个使用 LlamaIndex 框架实现的 RAG 系统示例，展示了如何用
          更简洁的 API 构建从文档加载、索引构建到查询生成的完整流程。

适用场景：快速原型开发、简化 RAG 系统构建、初学者入门

技术栈：LlamaIndex + DeepSeek + BGE 嵌入模型
与 LangChain 对比：LlamaIndex 提供了更高级的抽象，代码更简洁
================================================================================
"""

# ============================================================================
# 第一部分：导入必要的库和模块
# ============================================================================

import os  # Python 标准库，用于环境变量操作

# Hugging Face 镜像设置（可选）
# 如果在国内网络环境下无法下载模型，取消下面这行的注释
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from dotenv import load_dotenv  # 从 .env 文件加载环境变量
# 用途：安全地管理 API 密钥等敏感信息

# LlamaIndex 核心模块
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
# 功能说明：
#   - VectorStoreIndex: 向量存储索引，LlamaIndex 的核心索引类型
#       特点：自动处理文档向量化、存储和检索
#       其他索引类型：
#           - ListIndex: 列表索引，顺序扫描所有文档
#           - TreeIndex: 树形索引，层次化组织文档
#           - KeywordTableIndex: 关键词表索引，基于关键词匹配
#
#   - SimpleDirectoryReader: 简单的目录/文件读取器
#       参数：
#           - input_files: 指定要读取的文件列表
#           - input_dir: 指定要读取的目录（读取目录下所有文件）
#           - required_exts: 指定文件扩展名过滤（如 [".md", ".txt"]）
#           - recursive: 是否递归读取子目录（默认 False）
#       返回：包含多个文档的集合
#
#   - Settings: LlamaIndex 的全局配置对象
#       用于设置默认的 LLM、嵌入模型、分块器等
#       配置项：
#           - Settings.llm: 默认的大语言模型
#           - Settings.embed_model: 默认的嵌入模型
#           - Settings.text_splitter: 默认的文本分割器
#           - Settings.chunk_size: 默认的文档块大小
#           - Settings.chunk_overlap: 默认的块重叠大小

# DeepSeek 大语言模型
from llama_index.llms.deepseek import DeepSeek
# 功能：调用 DeepSeek API 进行文本生成
# 其他 LLM 选项：
#   - llama_index.llms.openai.OpenAI: OpenAI GPT 系列
#   - llama_index.llms.anthropic.Anthropic: Anthropic Claude 系列
#   - llama_index.llms.ollama.Ollama: 本地模型（如 Llama）
# 参数说明：
#   - model: 模型名称
#       - "deepseek-chat": 对话模型（推荐）
#       - "deepseek-coder": 代码生成模型
#   - api_key: API 密钥
#   - temperature: 温度参数（0.0-1.0），默认 0.1
#   - max_tokens: 最大生成 token 数

# Hugging Face 嵌入模型
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# 功能：使用 Hugging Face 上的开源嵌入模型
# 其他嵌入模型：
#   - llama_index.embeddings.openai.OpenAIEmbedding: OpenAI 嵌入
#   - llama_index.embeddings.cohere.CohereEmbedding: Cohere 嵌入
# 常用中文嵌入模型：
#   - "BAAI/bge-small-zh-v1.5": 轻量级中文模型（推荐）
#   - "BAAI/bge-base-zh-v1.5": 基础中文模型
#   - "BAAI/bge-large-zh-v1.5": 大型中文模型（精度更高）
# 参数说明：
#   - model_name: 模型名称或路径
#   - device: 运行设备（"cpu"、"cuda"、"mps"）
#   - embed_batch_size: 批处理大小

# 加载环境变量
load_dotenv()


# ============================================================================
# 第二部分：全局配置
# ============================================================================

# 配置默认的大语言模型
# 这会设置全局的 LLM，后续所有索引和查询引擎都会使用这个模型
# 也可以在创建查询引擎时单独指定，覆盖全局设置
Settings.llm = DeepSeek(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY")
)

# 配置默认的嵌入模型
# 这会设置全局的嵌入模型，用于文档向量化和查询向量化
# HuggingFaceEmbedding 会自动下载模型到本地缓存
Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-small-zh-v1.5")


# ============================================================================
# 第三部分：数据加载与索引构建
# ============================================================================

# 加载文档
# SimpleDirectoryReader 是 LlamaIndex 提供的简单文档加载器
# 参数说明：
#   - input_files: 文件路径列表
# 返回值：Document 对象列表
# 注意：也可以用 input_dir 参数指定整个目录
docs = SimpleDirectoryReader(
    input_files=["../../data/C1/markdown/easy-rl-chapter1.md"]
).load_data()
# load_data() 会返回一个 Document 列表
# 每个 Document 包含：
#   - text: 文档内容
#   - metadata: 元数据（如文件名、路径等）

# 构建向量索引
# VectorStoreIndex.from_documents() 是一步到位的方法，它会自动：
#   1. 使用配置的嵌入模型将文档向量化
#   2. 构建向量索引存储在内存中
#   3. 返回可查询的索引对象
# 参数说明：
#   - documents: Document 对象列表
#   - show_progress: 是否显示进度条（默认 True）
# 返回值：VectorStoreIndex 对象
index = VectorStoreIndex.from_documents(docs)


# ============================================================================
# 第四部分：查询与生成
# ============================================================================

# 创建查询引擎
# as_query_engine() 方法会基于索引创建一个查询引擎
# 该引擎会：
#   1. 接收用户查询
#   2. 将查询向量化
#   3. 在向量索引中检索相关文档
#   4. 使用 LLM 基于检索结果生成回答
# 可选参数：
#   - similarity_top_k: 检索的文档数量（默认 2）
#   - text_qa_template: 自定义 QA 模板
#   - refine_template: 自定义精炼模板
#   - streaming: 是否流式输出（默认 False）
query_engine = index.as_query_engine(
    similarity_top_k=3,  # 检索最相关的 3 个文档块
)

# 打印当前使用的提示词模板
# get_prompts() 方法返回查询引擎使用的所有提示词模板
# 这对于调试和理解 LlamaIndex 的工作原理很有帮助
# 返回一个字典，包含：
#   - "response_synthesizer:prompt": 主 QA 模板
#   - "retrieve:prompt": 检索模板（如果有）
print(query_engine.get_prompts())

# 执行查询
# query() 方法会：
#   1. 将问题向量化
#   2. 在向量索引中检索相关文档
#   3. 将检索结果和问题组合成提示词
#   4. 调用 LLM 生成回答
# 参数说明：
#   - query_str: 查询问题文本
#   - streaming: 是否流式输出
# 返回值：Response 对象，包含：
#   - response: 生成的回答文本
#   - source_nodes: 检索到的文档节点列表
#   - metadata: 其他元数据
response = query_engine.query("文中举了哪些例子?")
print(response)


# ============================================================================
# 伪流程图：LlamaIndex RAG 系统执行流程
# ============================================================================
"""
┌─────────────────────────────────────────────────────────────────────┐
│                    LlamaIndex RAG 系统执行流程                        │
└─────────────────────────────────────────────────────────────────────┘

  阶段 1: 配置与初始化（Configuration & Initialization）
  ┌──────────────────────────────────────────────────────────────┐
  │                                                              │
  │  [1. 全局配置]                                                │
  │      ↓                                                       │
  │  Settings.llm = DeepSeek                                      │
  │  Settings.embed_model = HuggingFaceEmbedding                  │
  │      ↓                                                       │
  │  设置默认的 LLM 和嵌入模型                                    │
  │                                                              │
  └──────────────────────────────────────────────────────────────┘

  阶段 2: 数据加载与索引构建（Data Loading & Indexing）
  ┌──────────────────────────────────────────────────────────────┐
  │                                                              │
  │  [2. 文档加载]                                                │
  │      ↓                                                       │
  │  SimpleDirectoryReader(input_files=[...])                    │
  │      ↓                                                       │
  │  docs (Document 对象列表)                                     │
  │      ↓                                                       │
  │                                                              │
  │  [3. 自动索引构建]                                           │
  │      ↓                                                       │
  │  VectorStoreIndex.from_documents(docs)                       │
  │      ↓                                                       │
  │  自动完成以下步骤：                                           │
  │    - 文本分块（默认 chunk_size=1024）                         │
  │    - 向量化（使用 HuggingFaceEmbedding）                      │
  │    - 索引存储（内存向量存储）                                 │
  │      ↓                                                       │
  │  index (VectorStoreIndex 对象)                                │
  │                                                              │
  └──────────────────────────────────────────────────────────────┘

  阶段 3: 查询与生成（Query & Generation）
  ┌──────────────────────────────────────────────────────────────┐
  │                                                              │
  │  [4. 创建查询引擎]                                           │
  │      ↓                                                       │
  │  index.as_query_engine(similarity_top_k=3)                   │
  │      ↓                                                       │
  │  query_engine (QueryEngine 对象)                             │
  │                                                              │
  │  [5. 执行查询]                                               │
  │      ↓                                                       │
  │  query_engine.query("文中举了哪些例子?")                      │
  │      ↓                                                       │
  │  自动执行以下步骤：                                           │
  │    a) 问题向量化                                             │
  │         ↓                                                   │
  │    b) 向量检索（Top-K=3）                                     │
  │         ↓                                                   │
  │    c) 提示词组装                                             │
  │         ↓                                                   │
  │    d) LLM 生成                                               │
  │         ↓                                                   │
  │  response (Response 对象)                                    │
  │      ↓                                                       │
  │  print(response) → 输出回答                                   │
  │                                                              │
  └──────────────────────────────────────────────────────────────┘

  与 LangChain 对比：
  ┌──────────────────────────────────────────────────────────────┐
  │                                                              │
  │  LangChain:                          LlamaIndex:              │
  │  需要手动创建各个组件                自动化处理更多步骤       │
  │  - 手动选择加载器                    - 自动推断文档类型       │
  │  - 手动创建分割器                    - 自动分块               │
  │  - 手动向量化                        - 自动向量化             │
  │  - 手动创建向量存储                  - 自动创建索引           │
  │  - 手动组装提示词                    - 自动生成提示词         │
  │                                                              │
  │  优点：更灵活，可控性强              优点：更简洁，快速开发   │
  │  缺点：代码较长                     缺点：定制化较难         │
  │                                                              │
  └──────────────────────────────────────────────────────────────┘
"""

# ============================================================================
# 高级用法说明
# ============================================================================
"""
# 自定义文本分割器
from llama_index.core.node_parser import SentenceSplitter

Settings.text_splitter = SentenceSplitter(
    chunk_size=500,
    chunk_overlap=50,
)

# 流式输出
query_engine = index.as_query_engine(streaming=True)
response = query_engine.query("你的问题")
for token in response.response_gen:
    print(token, end="")

# 获取检索到的文档
response = query_engine.query("你的问题")
for node in response.source_nodes:
    print(f"得分: {node.score:.4f}")
    print(f"内容: {node.text}\n")

# 持久化索引
index.storage_context.persist("index_dir")

# 从磁盘加载索引
from llama_index.core import StorageContext, load_index_from_storage
storage_context = StorageContext.from_defaults(persist_dir="index_dir")
index = load_index_from_storage(storage_context)
"""
