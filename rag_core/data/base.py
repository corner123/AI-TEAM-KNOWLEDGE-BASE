from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document


class DataLoader(ABC):
    def __init__(self, file_path: str):
        self.file_path = file_path

    @abstractmethod
    def load(self) -> List[Document]:
        ...

    def _make_doc(self, content: str, **extra_metadata) -> Document:
        metadata = {
            "source": self.file_path,
            "doc_type": "technical_doc",
            "chunk_type": "text",
        }
        metadata.update(extra_metadata)
        return Document(page_content=content, metadata=metadata)
