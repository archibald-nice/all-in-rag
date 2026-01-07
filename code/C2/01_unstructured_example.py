"""
================================================================================
Unstructured 库 - 文档解析示例
================================================================================
功能说明：展示如何使用 Unstructured 库自动解析多种格式的文档。
          Unstructured 是一个强大的文档解析库，可以自动识别文档类型
          并提取其中的结构化元素。

适用场景：需要从 PDF、Word、HTML 等多种格式文档中提取文本和结构

技术栈：Unstrained 库
================================================================================
"""

# ============================================================================
# 导入必要的库
# ============================================================================

from unstructured.partition.auto import partition
# partition: Unstructured 的自动分区函数
# 功能：根据文件扩展名自动选择合适的分区器
# 支持的文件类型：
#   - PDF: .pdf
#   - Word: .docx, .doc
#   - HTML: .html, .htm
#   - Markdown: .md
#   - 纯文本: .txt
#   - PowerPoint: .pptx
#   - Excel: .xlsx
#   - 图片: .png, .jpg, .jpeg（使用 OCR）
#
# 参数说明：
#   - filename: 文件路径
#   - content_type: 显式指定内容类型（可选）
#       "application/pdf": PDF 文档
#       "text/plain": 纯文本
#       "text/html": HTML 文档
#       等等...
#   - strategy: 分区策略
#       "fast": 快速分区（默认，适合大多数场景）
#       "hi_res": 高分辨率分区（更精确，但速度较慢）
#   - languages: 语言列表（用于 OCR）
#       ["chi_sim"]: 简体中文
#       ["eng"]: 英语
#   - extract_images_in_pdf: 是否提取 PDF 中的图片（默认 False）
# 返回值：Element 对象列表


# ============================================================================
# 文档解析
# ============================================================================

# 定义 PDF 文件路径
pdf_path = "../../data/C2/pdf/rag.pdf"

# 使用 Unstructured 加载并解析 PDF 文档
# partition 函数会自动：
#   1. 识别文件类型（通过扩展名或 content_type 参数）
#   2. 选择合适的解析器
#   3. 提取文档中的文本和结构
#   4. 返回一组 Element 对象
elements = partition(
    filename=pdf_path,
    content_type="application/pdf"  # 显式指定为 PDF（可选，通过扩展名可自动识别）
)


# ============================================================================
# 结果分析与展示
# ============================================================================

# 打印解析结果统计
# Element 对象结构：
#   - category: 元素类型（如 "Title", "NarrativeText", "ListItem" 等）
#   - text: 元素的文本内容
#   - metadata: 元数据（如页码、文件名等）
print(f"解析完成: {len(elements)} 个元素, {sum(len(str(e)) for e in elements)} 字符")

# 统计元素类型
# Unstructured 会将文档内容分类为不同类型的元素：
#   - Title: 标题
#   - NarrativeText: 叙述性文本（正文段落）
#   - ListItem: 列表项
#   - Table: 表格
#   - Header: 页眉
#   - Footer: 页脚
#   等等...
from collections import Counter
types = Counter(e.category for e in elements)
print(f"元素类型: {dict(types)}")

# 显示所有元素
# 遍历所有解析出的元素，展示其类型和内容
print("\n所有元素:")
for i, element in enumerate(elements, 1):
    print(f"Element {i} ({element.category}):")
    print(element)  # 打印元素的文本内容
    print("=" * 60)  # 分隔线


# ============================================================================
# 伪流程图：Unstructured 文档解析流程
# ============================================================================
"""
┌─────────────────────────────────────────────────────────────────────┐
│                  Unstructured 文档解析流程                           │
└─────────────────────────────────────────────────────────────────────┘

  输入: PDF/Word/HTML 等多格式文档
  ┌──────────────────────────────────────────────────────────────┐
  │                                                              │
  │  [1. 文件类型识别]                                           │
  │      ↓                                                       │
  │  partition(filename=pdf_path)                                │
  │      ↓                                                       │
  │  根据文件扩展名或 content_type 自动选择解析器                │
  │      ↓                                                       │
  │                                                              │
  │  [2. 文档解析]                                               │
  │      ↓                                                       │
  │  提取文本内容 + 识别结构                                      │
  │      ↓                                                       │
  │  ┌────────────┬────────────┬────────────┬────────────┐       │
  │  │  Title     │ Narrative  │  ListItem  │   Table    │       │
  │  │  (标题)    │  Text      │  (列表项)  │   (表格)   │       │
  │  │            │  (正文)    │            │            │       │
  │  └────────────┴────────────┴────────────┴────────────┘       │
  │      ↓                                                       │
  │  elements (Element 对象列表)                                 │
  │                                                              │
  │  [3. 结果处理]                                               │
  │      ↓                                                       │
  │  - 统计元素类型                                               │
  │  - 提取文本内容                                               │
  │  - 获取元数据（页码等）                                        │
  │      ↓                                                       │
  │  输出: 结构化的文档元素                                        │
  │                                                              │
  └──────────────────────────────────────────────────────────────┘

  常见 Element 类型说明:
  ┌──────────────────────────────────────────────────────────────┐
  │                                                              │
  │  Title              - 各级标题（H1, H2, H3...）               │
  │  NarrativeText      - 叙述性文本，通常指正文段落              │
  │  ListItem           - 列表项（有序或无序列表）                 │
  │  Table              - 表格（包含行列数据）                    │
  │  Header             - 页眉                                    │
  │  Footer             - 页脚                                    │
  │  PageBreak          - 分页符                                  │
  │  Image              - 图片                                    │
  │  Formula            - 公式（数学公式或化学式）                │
  │                                                              │
  └──────────────────────────────────────────────────────────────┘
"""


# ============================================================================
# 进阶用法说明
# ============================================================================
"""
# 1. 指定分区策略
elements = partition(
    filename="document.pdf",
    strategy="hi_res"  # 高精度分区，适合复杂布局的文档
)

# 2. 处理中文文档（使用 OCR）
elements = partition(
    filename="scanned.pdf",
    strategy="hi_res",
    languages=["chi_sim", "eng"],  # 支持中文和英文
    extract_images_in_pdf=True     # 提取图片中的文字
)

# 3. 仅提取特定类型的元素
from unstructured.staging.base import dict_to_elements

narrative_texts = [e for e in elements if e.category == "NarrativeText"]
titles = [e for e in elements if e.category == "Title"]

# 4. 获取元素的元数据
for element in elements:
    print(f"类型: {element.category}")
    print(f"内容: {element.text}")
    print(f"页码: {element.metadata.page_number}")
    print(f"文件名: {element.metadata.filename}")

# 5. 与 LangChain 集成
from langchain_community.document_loaders import UnstructuredFileLoader

loader = UnstructuredFileLoader(
    "document.pdf",
    strategy="hi_res",
    mode="elements"  # 返回元素列表，而非整个文档
)
docs = loader.load()
"""
