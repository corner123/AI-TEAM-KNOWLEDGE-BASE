# AI团队知识库 - RAG全栈系统

为AI技术团队构建的内部知识库智能问答系统，展示RAG技术全栈能力。

## 特性

- **多格式数据支持**: Markdown、PDF、CSV、代码、图片
- **6种检索策略**: 混合检索、Text2SQL、查询重构、元数据过滤、多模态检索、重排序
- **智能路由**: 自动选择最佳检索策略
- **Function Calling**: 集成SQL执行、文档检索等工具
- **RAGAS评估**: 完整的检索和响应评估体系
- **Gradio Web UI**: 交互式问答界面

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 填入API Key

# 下载模型
python scripts/download_models.py

# 构建索引
python scripts/build_index.py

# 启动Web UI
python app.py

# 或使用CLI
python main.py query "什么是RAG?"
```

## 技术栈

- LangChain + Milvus + DeepSeek
- BGE嵌入模型 (稠密/稀疏/多模态)
- RAGAS评估框架
- Gradio Web UI

## 项目结构

详见 `docs/superpowers/specs/2026-05-07-rag-knowledge-base-design.md`
