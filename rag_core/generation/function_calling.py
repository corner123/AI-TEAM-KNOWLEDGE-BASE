import json
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_core.language_models import BaseLLM
from .base import GenerationStrategy
from tools.sql_executor import SQLExecutor
from tools.doc_searcher import DocSearcher
from tools.model_info import ModelInfoTool
from utils.logger import get_logger

logger = get_logger("function_calling")

FC_PROMPT = """你是一个AI技术团队的知识库助手。你可以使用以下工具:

1. execute_sql(sql, database) - 执行SQL查询
2. search_knowledge(query, doc_type?, top_k?) - 检索知识库
3. get_model_benchmark(model_name, task?) - 查询模型性能
4. list_available_data() - 列出可用数据

根据用户问题，决定是否需要调用工具。如果需要，返回工具调用；否则直接回答。

用户问题: {query}

可用上下文:
{context}"""


class FunctionCallingGenerator(GenerationStrategy):
    def __init__(self, llm: BaseLLM, db_path: str = ":memory:", vector_store=None):
        self.llm = llm
        self.tools = {
            "execute_sql": SQLExecutor(db_path=db_path),
            "search_knowledge": DocSearcher(vector_store=vector_store),
            "get_model_benchmark": ModelInfoTool(db_path=db_path),
            "list_available_data": self._list_data,
        }

    def generate(self, query: str, context: List[Document], **kwargs) -> Dict[str, Any]:
        context_text = self._format_context(context)
        prompt = FC_PROMPT.format(query=query, context=context_text)

        try:
            response = self.llm.invoke(prompt)
            answer = response.content

            sources = list({doc.metadata.get("source", "unknown") for doc in context})

            return {
                "answer": answer,
                "sources": sources,
                "confidence": 0.85,
                "strategy_used": "FunctionCalling",
            }
        except Exception as e:
            logger.error(f"Function calling failed: {e}")
            return {"answer": f"生成失败: {e}", "sources": [], "confidence": 0.0, "strategy_used": "FunctionCalling"}

    def _format_context(self, docs: List[Document]) -> str:
        if not docs:
            return "无相关上下文"
        parts = []
        for i, doc in enumerate(docs[:3]):
            parts.append(f"[{i+1}] {doc.page_content[:200]}")
        return "\n".join(parts)

    def _list_data(self) -> Dict[str, Any]:
        return {"tables": ["ai_models", "gpu_usage", "projects"], "doc_types": ["technical_doc", "api_ref", "blog", "code"]}
