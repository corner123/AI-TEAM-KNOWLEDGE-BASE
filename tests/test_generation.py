import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document


def test_generation_strategy_interface():
    from rag_core.generation.base import GenerationStrategy
    assert hasattr(GenerationStrategy, "generate")


def test_standard_generation():
    from rag_core.generation.standard import StandardGenerator
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content="这是回答。来源: test.md")
    generator = StandardGenerator(llm=llm)

    docs = [Document(page_content="相关内容", metadata={"source": "test.md", "title": "Test"})]
    result = generator.generate("测试问题", docs)

    assert "answer" in result
    assert "sources" in result


def test_function_calling_tools_registered():
    from rag_core.generation.function_calling import FunctionCallingGenerator
    llm = MagicMock()
    generator = FunctionCallingGenerator(llm=llm)
    assert len(generator.tools) == 4


def test_sql_executor_safe():
    from tools.sql_executor import SQLExecutor
    executor = SQLExecutor(db_path=":memory:")
    assert executor._is_safe_sql("SELECT * FROM test") is True
    assert executor._is_safe_sql("DROP TABLE test") is False
