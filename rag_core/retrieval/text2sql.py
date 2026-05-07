import re
import sqlite3
from typing import List
from langchain_core.documents import Document
from langchain_core.language_models import BaseLLM
from .base import RetrievalStrategy
from utils.logger import get_logger

logger = get_logger("text2sql")

TEXT2SQL_PROMPT = """你是一个SQL专家。根据用户的自然语言问题，生成对应的SQL查询。

数据库表结构:
{table_schema}

规则:
1. 只生成SELECT语句
2. 不要使用DROP、DELETE、UPDATE、INSERT
3. 如果问题无法用SQL回答，返回 "UNANSWERABLE"

用户问题: {query}

只返回SQL语句，不要其他内容:"""


class Text2SQLStrategy(RetrievalStrategy):
    def __init__(self, llm: BaseLLM, db_path: str = ":memory:", max_retries: int = 3):
        self.llm = llm
        self.db_path = db_path
        self.max_retries = max_retries

    def retrieve(self, query: str, top_k: int = 5, **kwargs) -> List[Document]:
        table_schema = kwargs.get("table_schema", self._get_default_schema())

        for attempt in range(self.max_retries):
            try:
                sql = self._generate_sql(query, table_schema)
                if sql == "UNANSWERABLE":
                    return []

                if not self._is_safe_sql(sql):
                    logger.warning(f"Unsafe SQL rejected: {sql}")
                    continue

                results = self._execute_sql(sql)
                return self._results_to_documents(results, sql)
            except Exception as e:
                logger.warning(f"SQL attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    return []

        return []

    def get_strategy_name(self) -> str:
        return "Text2SQL"

    def _generate_sql(self, query: str, table_schema: str) -> str:
        prompt = TEXT2SQL_PROMPT.format(table_schema=table_schema, query=query)
        response = self.llm.invoke(prompt)
        sql = response.content.strip()
        sql = sql.strip("`").strip()
        if sql.startswith("sql"):
            sql = sql[3:].strip()
        return sql

    def _is_safe_sql(self, sql: str) -> bool:
        dangerous_keywords = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE", "EXEC"]
        sql_upper = sql.upper()
        return not any(keyword in sql_upper for keyword in dangerous_keywords)

    def _execute_sql(self, sql: str) -> List[dict]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute(sql)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def _results_to_documents(self, results: List[dict], sql: str) -> List[Document]:
        if not results:
            return []

        docs = []
        for row in results:
            content = " | ".join([f"{k}: {v}" for k, v in row.items()])
            docs.append(Document(
                page_content=content,
                metadata={"source": "database", "doc_type": "structured", "chunk_type": "table_row", "sql": sql},
            ))
        return docs

    def _get_default_schema(self) -> str:
        return "表: ai_models (name, params, accuracy, speed, task)"
