from abc import ABC, abstractmethod
from typing import List, Dict, Any, Generator, Optional
from langchain_core.documents import Document


class GenerationStrategy(ABC):
    @abstractmethod
    def generate(self, query: str, context: List[Document], **kwargs) -> Dict[str, Any]:
        ...

    def generate_stream(self, query: str, context: List[Document], **kwargs) -> Generator[str, None, None]:
        result = self.generate(query, context, **kwargs)
        yield result.get("answer", "")
