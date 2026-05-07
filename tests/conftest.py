import pytest
from config import RAGConfig


@pytest.fixture
def rag_config():
    return RAGConfig()


@pytest.fixture
def sample_text():
    return """# RAG技术介绍

## 什么是RAG
RAG（Retrieval-Augmented Generation）是一种结合检索和生成的技术。

## 核心组件
1. 检索器：从知识库中找到相关文档
2. 生成器：基于检索结果生成答案

## 优势
- 减少幻觉
- 知识可更新
- 答案可溯源"""


@pytest.fixture
def sample_markdown():
    return """# API文档

## 安装
使用pip安装：`pip install langchain`

## 基本用法
```python
from langchain import LLMChain
chain = LLMChain(llm=llm, prompt=prompt)
```

## 参数说明
| 参数 | 类型 | 说明 |
|------|------|------|
| llm | BaseLLM | 语言模型 |
| prompt | PromptTemplate | 提示模板 |
"""
