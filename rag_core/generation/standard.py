from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_core.language_models import BaseLLM
from .base import GenerationStrategy

STANDARD_PROMPT = """你是一个AI技术团队的知识库助手。根据以下检索到的上下文回答用户问题。

规则:
1. 只基于提供的上下文回答，不要编造信息
2. 引用来源文件名
3. 如果上下文不相关，请明确说明无法回答

上下文:
{context}

问题: {query}

请回答(附带来源引用和置信度):"""


class StandardGenerator(GenerationStrategy):
    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def generate(self, query: str, context: List[Document], **kwargs) -> Dict[str, Any]:
        context_text = self._format_context(context)
        prompt = STANDARD_PROMPT.format(context=context_text, query=query)

        response = self.llm.invoke(prompt)
        answer = response.content

        sources = list({
            doc.metadata.get("source", "unknown")
            for doc in context
        })

        return {
            "answer": answer,
            "sources": sources,
            "confidence": self._estimate_confidence(context),
            "strategy_used": "Standard",
        }

    def _format_context(self, docs: List[Document]) -> str:
        parts = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "unknown")
            parts.append(f"[{i+1}] (来源: {source})\n{doc.page_content}")
        return "\n\n".join(parts)

    def _estimate_confidence(self, docs: List[Document]) -> float:
        if not docs:
            return 0.0
        scores = [doc.metadata.get("score", 0) for doc in docs]
        return round(sum(scores) / len(scores), 2) if scores else 0.0
