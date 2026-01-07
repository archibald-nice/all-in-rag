"""
================================================================================
RAG 系统完整实现 - 基于 LangChain 框架
================================================================================
功能说明：这是一个完整的 RAG（检索增强生成）系统示例，展示了从文档加载、分块、向量化、检索到生成的完整流程。

适用场景：基于本地文档的问答系统、知识库检索、智能客服等

技术栈：LangChain + DeepSeek + BGE 嵌入模型 + 内存向量存储
================================================================================
"""

# ============================================================================
# 第一部分：导入必要的库和模块
# ============================================================================

import os  # Python 标准库，用于环境变量和文件路径操作

# Hugging Face 镜像设置（可选）
# 如果在国内网络环境下无法下载模型，取消下面这行的注释
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# 可选镜像站点：
#   - https://hf-mirror.com (国内常用镜像)
#   - https://huggingface.co (官方源，可能需要科学上网)

from dotenv import load_dotenv  # 从 .env 文件加载环境变量
# 用途：安全地管理 API 密钥等敏感信息，避免硬编码在代码中

# LangChain 社区文档加载器
from langchain_community.document_loaders import UnstructuredMarkdownLoader
# 功能：加载 Markdown 格式的文档
# 其他可选加载器：
#   - PyPDFLoader: 加载 PDF 文件
#   - Docx2txtLoader: 加载 Word 文档
#   - WebBaseLoader: 加载网页内容
#   - TextLoader: 加载纯文本文件

# LangChain 文本分割器
from langchain_text_splitters import RecursiveCharacterTextSplitter
# 功能：将长文档智能分割成适合检索的小块
# 其他可选分割器：
#   - CharacterTextSplitter: 按固定字符数分割
#   - MarkdownHeaderTextSplitter: 按 Markdown 标题分割
#   - RecursiveCharacterTextSplitter: 递归分割（推荐，保持语义完整性）

# Hugging Face 嵌入模型
from langchain_huggingface import HuggingFaceEmbeddings
# 功能：将文本转换为向量表示
# 其他嵌入模型：
#   - OpenAIEmbeddings: OpenAI 的嵌入模型（需付费）
#   - CohereEmbeddings: Cohere 的嵌入模型
#   - 也可使用其他开源模型如 sentence-transformers 系列

# 内存向量存储
from langchain_core.vectorstores import InMemoryVectorStore
# 功能：在内存中存储向量，适合小规模数据
# 其他向量存储：
#   - FAISS: Facebook 的向量检索库（推荐用于生产环境）
#   - Chroma: 本地向量数据库
#   - Milvus: 分布式向量数据库（适合大规模数据）
#   - Pinecone: 云端向量数据库服务

# 提示词模板
from langchain_core.prompts import ChatPromptTemplate
# 功能：结构化地构建 LLM 提示词
# 其他提示词模板：
#   - PromptTemplate: 基础提示词模板
#   - MessagesPlaceholder: 用于多轮对话

# DeepSeek 聊天模型
from langchain_deepseek import ChatDeepSeek
# 功能：调用 DeepSeek 大语言模型进行生成
# 其他 LLM：
#   - ChatOpenAI: OpenAI 的 GPT 系列
#   - ChatAnthropic: Anthropic 的 Claude 系列
#   - Tongyi: 阿里通义千问

# 加载环境变量（从 .env 文件读取）
# 这会读取项目根目录下的 .env 文件中的配置
load_dotenv()


# ============================================================================
# 第二部分：数据准备阶段
# ============================================================================

# 定义 Markdown 文件路径
# 路径说明：相对于当前脚本文件的路径
# ../../ 表示向上两级目录（从 code/C1/ 到项目根目录）
markdown_path = "../../data/C1/markdown/easy-rl-chapter1.md"

# 创建 Markdown 文档加载器
# UnstructuredMarkdownLoader: 使用 unstructured 库解析 Markdown
# 参数说明：
#   - file_path: 文件路径（必需）
#   - mode: 加载模式，可选 "single"（单文档）或 "elements"（元素级）
loader = UnstructuredMarkdownLoader(markdown_path)

# 执行文档加载
# 返回值：List[Document]，每个 Document 包含 page_content 和 metadata
# Document 结构：
#   - page_content: 文档的文本内容
#   - metadata: 元数据字典（如来源、文件路径等）
docs = loader.load()

# 创建文本分割器
# RecursiveCharacterTextSplitter: 递归地按不同分隔符分割文本
# 优点：能够保持文本的语义完整性，不会在句子中间强行切断
# 默认参数（可自定义）：
#   - chunk_size: 每个块的最大字符数，默认 4000
#   - chunk_overlap: 块之间的重叠字符数，默认 200（保持上下文连贯性）
#   - length_function: 计算文本长度的函数，默认 len
#   - separators: 分隔符列表，按优先级尝试 ["\n\n", "\n", " ", ""]
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,        # 每块最大 500 字符
    chunk_overlap=50,      # 块之间重叠 50 字符
    separators=["\n\n", "\n", "。", "；", " ", ""]  # 中文友好的分隔符
)

# 执行文本分割
# 返回值：List[Document]，分割后的文档块列表
chunks = text_splitter.split_documents(docs)
print(f"原始文档数: {len(docs)}, 分割后块数: {len(chunks)}")


# ============================================================================
# 第三部分：索引构建阶段
# ============================================================================

# 创建嵌入模型
# HuggingFaceEmbeddings: 使用 Hugging Face 上的开源嵌入模型
# 参数说明：
#   - model_name: 模型名称（Hugging Face 模型 ID）
#       可选模型：
#       - BAAI/bge-small-zh-v1.5: 中文轻量级模型（推荐）
#       - BAAI/bge-base-zh-v1.5: 中文基础模型
#       - BAAI/bge-large-zh-v1.5: 中文大模型（精度更高）
#       - sentence-transformers/all-MiniLM-L6-v2: 英文模型
#   - model_kwargs: 模型参数
#       - device: 运行设备
#           - 'cpu': CPU 运行（兼容性好，速度慢）
#           - 'cuda': GPU 运行（需要 NVIDIA 显卡，速度快）
#           - 'mps': Apple Silicon GPU（M1/M2 芯片）
#   - encode_kwargs: 编码参数
#       - normalize_embeddings: 是否归一化向量
#           - True: 归一化，使用余弦相似度（推荐）
#           - False: 不归一化，使用点积相似度
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# 创建向量存储
# InMemoryVectorStore: 内存中的向量存储
# 优点：简单快速，无需额外配置
# 缺点：重启后数据丢失，不适合大规模数据
# 适用场景：原型开发、小规模数据（< 10000 个文档块）
vectorstore = InMemoryVectorStore(embeddings)

# 将文档块添加到向量存储
# 这一步会：
# 1. 使用嵌入模型将每个文本块转换为向量
# 2. 将向量和文档内容存储在内存中
# 后续可以通过向量相似度搜索来检索相关文档
vectorstore.add_documents(chunks)
print(f"向量索引构建完成，共索引 {len(chunks)} 个文档块")


# ============================================================================
# 第四部分：检索与生成阶段
# ============================================================================

# 创建提示词模板
# ChatPromptTemplate: 结构化的提示词模板
# 使用模板字符串，其中 {context} 和 {question} 是占位符
# 在实际使用时会被替换为具体内容
prompt = ChatPromptTemplate.from_template(
    """请根据下面提供的上下文信息来回答问题。
请确保你的回答完全基于这些上下文。
如果上下文中没有足够的信息来回答问题，请直接告知："抱歉，我无法根据提供的上下文找到相关信息来回答此问题。"

上下文:
{context}

问题: {question}

回答:"""
)

# 创建大语言模型
# ChatDeepSeek: DeepSeek 的聊天模型接口
# 参数说明：
#   - model: 模型名称
#       可选模型：
#       - deepseek-chat: DeepSeek 对话模型（推荐）
#       - deepseek-coder: DeepSeek 代码模型
#   - temperature: 温度参数（0.0 - 2.0）
#       - 0.0: 最确定性，输出固定（适合事实性问答）
#       - 0.7: 平衡创造性和确定性（推荐）
#       - 1.0+: 更随机，更有创造性（适合创意写作）
#   - max_tokens: 最大生成 token 数
#       - 4096: 约 2000-3000 个中文字符
#       - 根据需求调整，越大成本越高
#   - api_key: API 密钥
#       从环境变量读取，避免硬编码
llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0.7,      # 较低的温度确保回答更准确
    max_tokens=4096,      # 足够生成详细的回答
    api_key=os.getenv("DEEPSEEK_API_KEY")  # 从环境变量获取
)

# 定义用户查询问题
# 这是用户想要问的问题，会被用于向量检索
question = "文中举了哪些例子？"

# ============================================================================
# 伪流程图：RAG 系统执行流程
# ============================================================================
"""
┌─────────────────────────────────────────────────────────────────────┐
│                         RAG 系统执行流程                              │
└─────────────────────────────────────────────────────────────────────┘

  阶段 1: 离线准备（Offline Preparation）
  ┌──────────────────────────────────────────────────────────────┐
  │                                                              │
  │  [1. 文档加载]                                                │
  │      ↓                                                       │
  │  Markdown 文件 → UnstructuredMarkdownLoader                   │
  │      ↓                                                       │
  │  原始文档 (docs)                                              │
  │                                                              │
  │  [2. 文本分割]                                                │
  │      ↓                                                       │
  │  RecursiveCharacterTextSplitter                              │
  │      ↓                                                       │
  │  文档块 (chunks) ← 每个 chunk 约 500 字符，重叠 50 字符       │
  │                                                              │
  │  [3. 向量化]                                                 │
  │      ↓                                                       │
  │  HuggingFaceEmbeddings (BAAI/bge-small-zh-v1.5)              │
  │      ↓                                                       │
  │  向量数组 (每个 chunk → 768 维向量)                           │
  │      ↓                                                       │
  │  [4. 索引构建]                                               │
  │      ↓                                                       │
  │  InMemoryVectorStore                                         │
  │      ↓                                                       │
  │  向量索引 (vectorstore) ← 存储在内存中                        │
  │                                                              │
  └──────────────────────────────────────────────────────────────┘

  阶段 2: 在线查询（Online Query）
  ┌──────────────────────────────────────────────────────────────┐
  │                                                              │
  │  [5. 用户提问]                                                │
  │      ↓                                                       │
  │  question = "文中举了哪些例子？"                              │
  │                                                              │
  │  [6. 问题向量化]                                             │
  │      ↓                                                       │
  │  embeddings.embed_query(question)                            │
  │      ↓                                                       │
  │  问题向量 (768 维)                                            │
  │                                                              │
  │  [7. 向量检索]                                               │
  │      ↓                                                       │
  │  vectorstore.similarity_search(question, k=3)                │
  │      ↓                                                       │
  │  计算: 余弦相似度 (问题向量 vs 所有文档块向量)                │
  │      ↓                                                       │
  │  Top-K 相关文档块 (retrieved_docs) ← 最相关的 3 个块         │
  │                                                              │
  │  [8. 上下文构建]                                             │
  │      ↓                                                       │
  │  提取文档内容并拼接                                           │
  │      ↓                                                       │
  │  docs_content = "\n\n".join([doc1, doc2, doc3])              │
  │                                                              │
  │  [9. 提示词组装]                                             │
  │      ↓                                                       │
  │  prompt.format(question=question, context=docs_content)      │
  │      ↓                                                       │
  │  完整提示词 = 系统指令 + 上下文 + 问题                        │
  │                                                              │
  │  [10. LLM 生成]                                              │
  │      ↓                                                       │
  │  ChatDeepSeek.invoke(完整提示词)                              │
  │      ↓                                                       │
  │  LLM 处理:                                                   │
  │      - 理解上下文                                            │
  │      - 理解问题                                              │
  │      - 生成基于上下文的回答                                   │
  │      ↓                                                       │
  │  answer (最终回答)                                           │
  │                                                              │
  │  [11. 输出结果]                                              │
  │      ↓                                                       │
  │  print(answer) → 展示给用户                                   │
  │                                                              │
  └──────────────────────────────────────────────────────────────┘

  数据流向图:
  ┌──────────┐      ┌──────────┐      ┌──────────┐      ┌──────────┐
  │ 文档加载  │ ───> │ 文本分割  │ ───> │ 向量化   │ ───> │ 索引存储  │
  └──────────┘      └──────────┘      └──────────┘      └──────────┘
                                                                  │
                                                                  ▼
  ┌──────────┐      ┌──────────┐      ┌──────────┐      ┌──────────┐
  │  用户    │ ───> │ 问题向量  │ ───> │ 相似度   │ ───> │ Top-K文档 │
  │  提问    │      │          │      │  检索    │      │          │
  └──────────┘      └──────────┘      └──────────┘      └──────────┘
                                                                  │
                                                                  ▼
  ┌──────────┐      ┌──────────┐      ┌──────────┐
  │ 上下文   │ ───> │ 提示词   │ ───> │ LLM生成  │ ───> │ 最终答案 │
  │  组装    │      │ 组装     │      │          │      │          │
  └──────────┘      └──────────┘      └──────────┘      └──────────┘
"""

# 在向量存储中进行相似度搜索
# similarity_search: 根据问题向量搜索最相似的文档
# 参数说明：
#   - query: 用户的问题文本（会自动转换为向量）
#   - k: 返回最相关的 k 个文档，默认 4
#       k 越大，召回的信息越多，但可能包含不相关的内容
#       k 越小，召回的信息越精准，但可能遗漏重要信息
#       常见取值：3-5
# 返回值：List[Document]，按相似度排序的文档列表
retrieved_docs = vectorstore.similarity_search(question, k=3)

# 提取文档内容并拼接
# 将检索到的多个文档块的内容用双换行符连接
# 这样可以在最终提示词中清晰地区分不同的文档块
docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

# 格式化提示词并调用 LLM
# prompt.format(): 将模板中的占位符替换为实际内容
#   - {question} → 用户的问题
#   - {context} → 检索到的文档内容
# llm.invoke(): 调用 LLM 生成回答
#   返回值：AIMessage 对象，包含生成的文本
answer = llm.invoke(prompt.format(question=question, context=docs_content))

# 输出最终答案
print(answer)
# 如果只想输出文本内容，可以使用：
# print(answer.content)
