# 📚 RAG 知识库问答系统 - Demo

基于 **Streamlit + Ollama + Milvus Lite** 的本地知识库检索增强生成（RAG）演示应用。

![系统架构](https://img.shields.io/badge/架构-RAG-blue)
![框架](https://img.shields.io/badge/框架-Streamlit-green)
![向量库](https://img.shields.io/badge/向量库-Milvus%20Lite-orange)
![模型](https://img.shields.io/badge/LLM-Ollama-purple)

## 🎯 项目简介

这是一个RAG 系统Demo项目：

1. **什么是 RAG？** 检索增强生成如何结合知识库和大模型
2. **向量数据库** 如何存储和检索语义相似的文本
3. **嵌入（Embedding）** 如何将文字转化为计算机理解的数字
4. **完整流程** 从文档上传到智能问答的全链路

## 📁 项目结构

```
rag_demo/
├── app.py              # 主应用文件（单文件结构，方便教学）
├── requirements.txt    # Python 依赖列表
└── README.md          # 本说明文档
```

## 🛠️ 环境准备

### 1. 安装 Python 依赖

```bash
# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # MacOS/Linux

# 安装依赖
pip install -r requirements.txt
```

**主要依赖包说明：**

| 包名 | 用途 |
|------|------|
| `streamlit` | Web 应用框架 |
| `pymilvus` | Milvus Lite 向量数据库客户端 |
| `ollama` | Ollama 本地大模型接口 |
| `langchain` | LLM 应用开发框架（可选） |
| `numpy` | 数值计算 |
| `pandas` | 数据处理和展示 |

### 2. 安装并启动 Ollama

```bash
# 1. 安装 Ollama（MacOS）
brew install ollama

# 2. 启动 Ollama 服务
ollama serve

# 3. 在另一个终端窗口，下载所需模型
ollama pull qwenembedding0.6b:latest   # 嵌入模型
ollama pull gemma3:1b                   # 聊天模型
```

**模型说明：**

| 模型 | 用途 | 维度 |
|------|------|------|
| `qwenembedding0.6b:latest` | 文本嵌入 | 1024 维 |
| `gemma3:1b` | 对话生成 | - |

## 🚀 启动应用

```bash
# 确保在 rag_demo 目录下
streamlit run app.py
```

应用将自动在浏览器中打开，默认地址：`http://localhost:8501`

## 📖 使用指南

### Tab 1: 知识库问答 💬

1. **上传文档**
   - 在左侧边栏选择 `.md` 或 `.txt` 文件
   - 系统自动执行：解析 → 分段 → 嵌入 → 存储

2. **提问**
   - 在聊天框输入问题
   - 系统会先检索相关知识，再生成回答
   - 可以展开 "查看检索到的背景知识" 查看参考片段

### Tab 2: 数据库透视 🔍

- 查看 Milvus 中存储的所有文本片段
- 支持按关键词搜索过滤
- 查看完整内容和元数据

## 🔬 核心概念解释

### 1. RAG 流程

```
用户问题
    ↓
向量化（Embedding）
    ↓
向量相似度搜索
    ↓
获取 Top-K 相关文本
    ↓
构建 Prompt（问题 + 上下文）
    ↓
大模型生成回答
```

### 2. 文本分段（Chunking）

- **为什么分段？** 大模型有输入长度限制
- **怎么分段？** 滑动窗口，每段 1024 字符，重叠 100 字符
- **重叠的作用？** 保持上下文连贯性

### 3. 向量嵌入（Embedding）

- **是什么？** 将文本转换为固定维度的数字向量
- **为什么？** 计算机只能理解数字，向量可以表示语义
- **相似度？** 语义相似的文本，向量距离更近

### 4. 向量检索

- **原理：** 将查询转为向量，寻找最相似的文档向量
- **度量方式：** 余弦相似度（COSINE）
- **Top-K：** 返回最相似的 K 个结果
