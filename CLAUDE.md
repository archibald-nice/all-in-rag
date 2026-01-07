# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在此代码库中工作时提供指导。

## 项目概述

**All-in-RAG** 是 Datawhale 出品的 RAG（检索增强生成）技术全栈教程项目，从理论到实践教授 RAG 技术。这是一个**中文教程**项目，包含完整的文档和从基础到进阶的实用代码实现。

**这是一个教育项目，而非生产级库。** 代码示例优先考虑清晰度和教学概念，而非性能优化。

## 技术栈

- **Python 3.12.7**，使用 pip 管理依赖
- **LangChain 0.3.26+**（主要 RAG 框架）和 **LlamaIndex 0.12.51+**（备选框架）
- **向量数据库**：FAISS（本地）、ChromaDB、Milvus（生产环境）
- **图数据库**：Neo4j 5.0+（用于第九章图 RAG）
- **嵌入模型**：BGE (BAAI/bge-small-zh-v1.5) 中文文本嵌入
- **大语言模型**：DeepSeek-chat、Kimi（Moonshot AI）、OpenAI GPT

## 环境配置

```bash
# 创建 conda 环境
conda create -n rag python=3.12.7
conda activate rag

# 安装所有依赖
cd code
pip install -r requirements.txt

# 或安装特定章节的依赖
cd code/C8  # 或 C9
pip install -r requirements.txt
```

## 基础设施（Docker Compose）

**Milvus 向量数据库**（第三章及以上）：
```bash
cd code
docker-compose up -d  # 启动 etcd、minio、milvus，端口 19530、9091、9000-9001
```

**Neo4j 图数据库**（第九章）：
```bash
cd data/C9
docker-compose up -d  # 启动 Neo4j 并自动导入，端口 7474（Web UI）、7687（bolt）
# 默认凭据：neo4j/all-in-rag
```

## 运行示例

**第一章（基础 RAG）**：
```bash
cd code/C1
python 01_langchain_example.py   # 基于 LangChain 的 RAG
python 02_llamaIndex_example.py  # 基于 LlamaIndex 的 RAG
```

**第八章（完整菜谱 RAG 系统）**：
```bash
cd code/C8
export MOONSHOT_API_KEY="your-key"  # 或在 code/config.py 中设置
python main.py  # 交互式问答模式
```

**第九章（图 RAG 系统）**：
```bash
# 先启动 Neo4j（见上方）
cd code/C9
python main.py
```

**AI 智能体菜谱解析**（第九章）：
```bash
cd code/C9/agent\(代码系ai生成\)
export KIMI_API_KEY="your-key"
python run_ai_agent.py test  # 测试单个菜谱
python run_ai_agent.py /path/to/HowToCook  # 批量处理
```

## 项目架构

### 第八章：完整菜谱 RAG 系统

主入口：`code/C8/main.py`

**模块化架构**：
```python
RecipeRAGSystem
├── DataPreparationModule      # 加载 & 分块 markdown 菜谱
├── IndexConstructionModule    # 构建 FAISS 向量索引
├── RetrievalOptimizationModule # 混合检索（BM25 + 向量）
└── GenerationIntegrationModule # 查询路由 & 答案生成
```

**核心设计模式**：
- **父子分块**：小块用于检索，完整文档用于生成
- **混合检索**：BM25 关键词 + 向量语义，使用 RRF（倒数排名融合）
- **查询路由**：将查询分类为 'list'（列表）、'detail'（详情）或 'general'（通用）类型
- **索引缓存**：保存/加载 FAISS 索引以避免重复构建
- **元数据过滤**：按类别（荤菜/素菜/汤品）和难度过滤

### 第九章：图 RAG 系统

主入口：`code/C9/main.py`

**进阶架构**：
- 在向量检索基础上增加 Neo4j 知识图谱
- 使用 DeepSeek/Kimi LLM 进行 AI 驱动的菜谱解析（`agent(代码系ai生成)/`）
- 向量检索和图检索之间的智能查询路由
- 多模态检索：向量 + 图谱 + 关键词

## 目录结构

```
all-in-rag/
├── docs/              # 教程文档（10 章，中文）
├── code/              # 按章节组织的代码示例（C1-C9）
│   ├── C1/           # 基础 RAG 示例（LangChain、LlamaIndex）
│   ├── C2/           # 数据加载 & 分块
│   ├── C3/           # 嵌入模型 & 向量数据库
│   ├── C4/           # 混合检索 & 查询构建
│   ├── C5/           # 格式化生成
│   ├── C6/           # 评估工具
│   ├── C8/           # 完整菜谱 RAG 系统
│   │   ├── main.py   # 系统主入口
│   │   ├── config.py # 配置管理
│   │   └── rag_modules/  # 模块化 RAG 组件
│   └── C9/           # 图 RAG + Neo4j
│       ├── main.py   # 图 RAG 系统
│       ├── agent(代码系ai生成)/  # AI 驱动的菜谱解析器
│       └── rag_modules/  # 图相关模块
├── data/              # 各章节示例数据
├── models/            # 预训练模型存储
└── Extra-chapter/     # 社区贡献的扩展内容
```

## 重要说明

1. **语言**：所有文档和内容均为中文。使用中文嵌入模型（BGE）和中文大模型（DeepSeek、Kimi）。

2. **无测试框架**：这是一个教育项目，通过交互式脚本进行手动测试。没有单元测试或 CI/CD 流水线。

3. **渐进式学习**：章节内容由简入繁，循序渐进。第一章展示基础四步 RAG；第八章实现生产级系统；第九章增加进阶图 RAG。

4. **需要 API 密钥**：许多示例需要 LLM API 密钥（Moonshot、DeepSeek、OpenAI）。通过环境变量或配置文件设置。

5. **模块化设计是有意的**：每个 RAG 阶段都是独立的模块类，便于学习。易于理解各个组件，未针对速度进行优化。

6. **数据流架构**：
```
用户查询 → 查询路由器 → 查询重写 → 检索（向量/BM25/图）
→ RRF 融合 → 父文档检索 → 生成 → 答案
```

7. **社区贡献**：Extra-chapter 目录包含社区贡献的内容。贡献指南见 `Extra-chapter/README.md`。
