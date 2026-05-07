from typing import List, Dict, Any, Optional


class DocSearcher:
    def __init__(self, vector_store=None):
        self.vector_store = vector_store

    def search(self, query: str, doc_type: Optional[str] = None, top_k: int = 5) -> Dict[str, Any]:
        if self.vector_store is None:
            return {"error": "Vector store not initialized"}

        filter_expr = f'doc_type == "{doc_type}"' if doc_type else ""
        results = self.vector_store.search_dense(query, top_k=top_k, filter_expr=filter_expr)
        return {"results": results, "count": len(results)}
