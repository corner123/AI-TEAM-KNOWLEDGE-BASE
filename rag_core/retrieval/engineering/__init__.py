"""Engineering retrieval primitives.

This package is intentionally independent from the existing RAG engine so it
can be evaluated and integrated without changing the current query pipeline.
"""

from .bm25 import BM25Retriever, engineering_tokenize
from .federated import EngineeringRetriever, FederatedPartition, FederatedRetriever
from .fusion import reciprocal_rank_fusion, rrf_fusion
from .live_code import LiveCodeRetriever
from .live_ast import LiveASTRetriever
from .live_git import GitWorktreeState, LiveGitVerifier
from .models import EngineeringSearchResult
from .rerank import EngineeringCrossEncoderReranker
from .routing import SourceIntent, SourceIntentRouter, SourceRoute

__all__ = [
    "BM25Retriever",
    "EngineeringRetriever",
    "EngineeringSearchResult",
    "EngineeringCrossEncoderReranker",
    "FederatedPartition",
    "FederatedRetriever",
    "LiveCodeRetriever",
    "LiveASTRetriever",
    "LiveGitVerifier",
    "GitWorktreeState",
    "SourceIntent",
    "SourceIntentRouter",
    "SourceRoute",
    "engineering_tokenize",
    "reciprocal_rank_fusion",
    "rrf_fusion",
]
