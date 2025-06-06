# rag-system-llamaindex-langchain
# RAG System - LlamaIndex & LangChain

🚀 一个基于 LlamaIndex 和 LangChain 的综合性 RAG（检索增强生成）系统，专为智能文档处理和问答而设计。

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)

## 📋 项目概述

本项目是一个完整的RAG（Retrieval-Augmented Generation）解决方案，结合了LlamaIndex和LangChain的强大功能，为企业和开发者提供了一个高效、可扩展的智能文档处理和问答系统。

### ✨ 核心特性

- **🔍 智能检索**: 基于向量相似度的高精度文档检索
- **🧠 增强生成**: 结合检索内容生成准确、相关的回答
- **📚 多格式支持**: 支持PDF、TXT、DOCX、MD等多种文档格式
- **⚡ 高性能**: 优化的向量存储和检索策略
- **🔧 易于集成**: 模块化设计，支持快速集成到现有系统
- **📊 完整测试**: 包含完整的测试框架和性能评估工具

## 🏗️ 系统架构

```
RAG System
├── 数据预处理模块
│   ├── 文档解析
│   ├── 文本分块
│   └── 数据清洗
├── 向量化模块
│   ├── 嵌入生成
│   ├── 向量存储
│   └── 索引构建
├── 检索模块
│   ├── 相似性搜索
│   ├── 重排序
│   └── 结果过滤
├── 生成模块
│   ├── 提示构建
│   ├── 模型推理
│   └── 答案生成
└── 评估模块
    ├── 检索评估
    ├── 生成评估
    └── 端到端评估
```

## 🚀 快速开始

### 前置要求

- Python 3.8+
- OpenAI API Key（可选，支持本地模型）
- 至少 8GB RAM
- 支持CUDA的GPU（推荐，用于加速）

### 安装步骤

1. **克隆仓库**
```bash
git clone https://github.com/joytianya/rag-system-llamaindex-langchain.git
cd rag-system-llamaindex-langchain
```

2. **创建虚拟环境**
```bash
python -m venv rag_env
source rag_env/bin/activate  # Windows: rag_env\Scripts\activate
```

3. **安装依赖**
```bash
pip install -r requirements.txt
```

4. **配置环境变量**
```bash
cp .env.example .env
# 编辑 .env 文件，添加必要的API密钥
```

### 基础使用

```python
from rag_system import RAGPipeline

# 初始化RAG系统
rag = RAGPipeline(
    embedding_model="text-embedding-ada-002",
    llm_model="gpt-3.5-turbo",
    vector_store="faiss"
)

# 添加文档
rag.add_documents("./data/documents/")

# 构建索引
rag.build_index()

# 查询
response = rag.query("什么是机器学习？")
print(response.answer)
```

## 📖 详细文档

### 核心组件

#### 1. 文档处理器 (Document Processor)
负责解析和预处理各种格式的文档：

```python
from rag_system.processors import DocumentProcessor

processor = DocumentProcessor()
documents = processor.process_directory("./data/")
```

#### 2. 向量存储 (Vector Store)
支持多种向量存储后端：

- **FAISS**: 高性能本地向量存储
- **Chroma**: 轻量级向量数据库
- **Pinecone**: 云端向量存储服务

```python
from rag_system.storage import VectorStore

store = VectorStore(backend="faiss")
store.add_embeddings(documents, embeddings)
```

#### 3. 检索器 (Retriever)
实现多种检索策略：

```python
from rag_system.retrieval import HybridRetriever

retriever = HybridRetriever(
    dense_weight=0.7,
    sparse_weight=0.3,
    top_k=10
)
```

#### 4. 生成器 (Generator)
支持多种语言模型：

```python
from rag_system.generation import LLMGenerator

generator = LLMGenerator(
    model_name="gpt-3.5-turbo",
    temperature=0.1,
    max_tokens=512
)
```

## 🔧 配置说明

### 配置文件示例 (config.yaml)

```yaml
# 模型配置
models:
  embedding:
    name: "text-embedding-ada-002"
    dimension: 1536
  
  llm:
    name: "gpt-3.5-turbo"
    temperature: 0.1
    max_tokens: 512

# 检索配置
retrieval:
  top_k: 10
  similarity_threshold: 0.7
  rerank: true

# 向量存储配置
vector_store:
  type: "faiss"
  index_type: "IVF"
  nlist: 100

# 数据处理配置
processing:
  chunk_size: 1000
  chunk_overlap: 200
  clean_text: true
```

## 📊 性能评估

系统内置了完整的评估框架：

### 检索评估
- **准确率@K**: 检索结果的准确性
- **召回率@K**: 相关文档的召回能力
- **MRR**: 平均倒数排名

### 生成评估
- **BLEU**: 生成质量评估
- **ROUGE**: 摘要质量评估
- **BERTScore**: 语义相似度评估

### 运行评估

```bash
python evaluate.py --config config.yaml --test-data test_set.json
```

## 🧪 测试

运行完整测试套件：

```bash
# 单元测试
pytest tests/unit/

# 集成测试
pytest tests/integration/

# 性能测试
pytest tests/performance/

# 生成测试报告
pytest --coverage-report html
```

## 🌟 高级功能

### 1. 自定义索引策略

```python
from rag_system.indexing import CustomIndexBuilder

builder = CustomIndexBuilder()
builder.add_metadata_filter("document_type", ["pdf", "txt"])
builder.add_semantic_filter(similarity_threshold=0.8)
index = builder.build()
```

### 2. 多模态支持

```python
from rag_system.multimodal import MultiModalRAG

mm_rag = MultiModalRAG()
mm_rag.add_text_documents("./text_docs/")
mm_rag.add_image_documents("./images/")
```

### 3. 实时更新

```python
# 支持文档的实时添加和删除
rag.add_document_stream(document_stream)
rag.remove_documents(doc_ids)
```

## 🔌 API 接口

系统提供了RESTful API接口：

```bash
# 启动API服务器
uvicorn rag_system.api:app --host 0.0.0.0 --port 8000

# 查询接口
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "什么是深度学习？", "top_k": 5}'
```

## 📁 项目结构

```
rag-system-llamaindex-langchain/
├── rag_system/                # 核心代码
│   ├── __init__.py
│   ├── pipeline.py            # 主要管道
│   ├── processors/            # 文档处理器
│   ├── storage/               # 向量存储
│   ├── retrieval/             # 检索模块
│   ├── generation/            # 生成模块
│   └── evaluation/            # 评估工具
├── tests/                     # 测试代码
│   ├── unit/
│   ├── integration/
│   └── performance/
├── docs/                      # 文档
│   ├── API.md
│   ├── DEPLOYMENT.md
│   └── TROUBLESHOOTING.md
├── examples/                  # 示例代码
├── data/                      # 示例数据
├── requirements.txt           # 依赖列表
├── config.yaml               # 配置文件
├── docker-compose.yml        # Docker配置
└── README.md                 # 本文件
```

## 🚀 部署指南

### Docker 部署

```bash
# 构建镜像
docker build -t rag-system .

# 运行容器
docker run -p 8000:8000 rag-system
```

### Kubernetes 部署

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

## 🤝 贡献指南

我们欢迎所有形式的贡献！请查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解详细信息。

### 贡献步骤

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🆘 问题反馈

如果遇到问题，请通过以下方式联系我们：

- 📧 邮箱: joytianya@example.com
- 🐛 Issue: [GitHub Issues](https://github.com/joytianya/rag-system-llamaindex-langchain/issues)
- 💬 讨论: [GitHub Discussions](https://github.com/joytianya/rag-system-llamaindex-langchain/discussions)

## 🙏 致谢

感谢以下开源项目的支持：

- [LlamaIndex](https://github.com/run-llama/llama_index)
- [LangChain](https://github.com/hwchase17/langchain)
- [FAISS](https://github.com/facebookresearch/faiss)
- [OpenAI](https://openai.com/)

## 📈 更新日志

### v1.0.0 (2024-06-06)
- 🎉 初始发布
- ✨ 完整的RAG系统实现
- 📚 支持多种文档格式
- 🔍 高性能向量检索
- 🧠 智能答案生成
- 📊 完整的评估框架

---

**⭐ 如果这个项目对您有帮助，请不要忘记给我们一个星标！**
