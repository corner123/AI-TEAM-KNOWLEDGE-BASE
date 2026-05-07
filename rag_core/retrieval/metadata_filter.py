from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.language_models import BaseLLM
from .base import RetrievalStrategy
from rag_core.index.vector_store import MilvusVectorStore
from utils.logger import get_logger

logger = get_logger("metadata_filter")

EXTRACT_CONDITION_PROMPT = """你是一个元数据过滤条件提取专家。根据用户的查询，提取可用于Milvus向量数据库过滤的条件。

可过滤字段:
- doc_type: str (api_ref, tutorial, faq, code_example, technical_doc)
- has_code: bool (true/false)
- chunk_type: str (text, code, table, image)
- section: str

用户查询: {query}

请提取过滤条件，每行一个条件，格式: 字段名 == "值" 或 字段名 == true/false
如果不需要过滤，回复: NONE"""


class MetadataFilterStrategy(RetrievalStrategy):
    def __init__(self, llm: BaseLLM, vector_store: MilvusVectorStore):
        self.llm = llm
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 5, **kwargs) -> List[Document]:
        conditions = self._extract_conditions(query)
        filter_expr = ""
        if conditions != "NONE":
            lines = [line.strip() for line in conditions.split("\n") if line.strip()]
            filter_expr = " and ".join(lines)
            logger.info(f"Extracted filter conditions: {filter_expr}")

        results = self.vector_store.search_dense(query, top_k=top_k, filter_expr=filter_expr)
        return self._to_documents(results)

    def get_strategy_name(self) -> str:
        return "MetadataFilter"

    def _extract_conditions(self, query: str) -> str:
        try:
            prompt = EXTRACT_CONDITION_PROMPT.format(query=query)
            response = self.llm.invoke(prompt)
            conditions = response.content.strip()
            logger.info(f"LLM returned conditions: {conditions}")
            return conditions
        except Exception as e:
            logger.warning(f"Condition extraction failed: {e}")
            return "NONE"
