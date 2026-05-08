import pytest
from unittest.mock import MagicMock


def test_ragas_evaluator_init():
    from rag_core.evaluation.ragas_eval import RAGASEvaluator
    evaluator = RAGASEvaluator()
    assert evaluator is not None


def test_custom_metrics_hit_rate():
    from rag_core.evaluation.custom_metrics import CustomMetrics
    retrieved = ["doc1", "doc2", "doc3"]
    relevant = ["doc1", "doc4"]
    assert CustomMetrics.hit_rate(retrieved, relevant) == 1.0

    retrieved = ["doc2", "doc3", "doc5"]
    assert CustomMetrics.hit_rate(retrieved, relevant) == 0.0


def test_custom_metrics_mrr():
    from rag_core.evaluation.custom_metrics import CustomMetrics
    retrieved = ["doc2", "doc1", "doc3"]
    relevant = ["doc1"]
    assert CustomMetrics.mrr(retrieved, relevant) == 0.5  # rank 2 -> 1/2


def test_custom_metrics_latency():
    from rag_core.evaluation.custom_metrics import CustomMetrics
    latencies = [0.1, 0.2, 0.3, 0.4, 0.5]
    stats = CustomMetrics.latency_stats(latencies)
    assert stats["p50"] == 0.3
    assert stats["p95"] >= 0.4
