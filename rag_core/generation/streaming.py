from typing import List, Dict, Any, Generator
from langchain_core.documents import Document
from langchain_core.language_models import BaseLLM
from .standard import StandardGenerator


class StreamingGenerator(StandardGenerator):
    def generate_stream(self, query: str, context: List[Document], **kwargs) -> Generator[str, None, None]:
        context_text = self._format_context(context)
        prompt = f"根据以下上下文回答问题:\n\n{context_text}\n\n问题: {query}\n\n回答:"

        try:
            for chunk in self.llm.stream(prompt):
                if hasattr(chunk, "content"):
                    yield chunk.content
        except Exception:
            result = self.generate(query, context, **kwargs)
            yield result.get("answer", "")
