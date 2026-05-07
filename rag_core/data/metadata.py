from typing import List, Dict, Any
from langchain_core.documents import Document


def enrich_metadata(docs: List[Document]) -> List[Document]:
    for doc in docs:
        if "has_code" not in doc.metadata:
            doc.metadata["has_code"] = "```" in doc.page_content or "def " in doc.page_content
        if "chunk_type" not in doc.metadata:
            doc.metadata["chunk_type"] = _detect_chunk_type(doc.page_content)
        if "language" not in doc.metadata:
            doc.metadata["language"] = _detect_language(doc.page_content)
    return docs


def _detect_chunk_type(content: str) -> str:
    if content.strip().startswith("|") or "---|" in content:
        return "table"
    if "```" in content or "def " in content or "class " in content:
        return "code"
    if content.strip().startswith("[图片"):
        return "image"
    return "text"


def _detect_language(content: str) -> str:
    chinese_chars = sum(1 for c in content if "一" <= c <= "鿿")
    total_chars = len(content.strip())
    if total_chars == 0:
        return "en"
    return "zh" if chinese_chars / total_chars > 0.1 else "en"


def build_metadata_filter(query_metadata: Dict[str, Any]) -> str:
    conditions = []
    for key, value in query_metadata.items():
        if isinstance(value, bool):
            conditions.append(f"{key} == {str(value).lower()}")
        elif isinstance(value, str):
            conditions.append(f'{key} == "{value}"')
    return " and ".join(conditions) if conditions else ""
