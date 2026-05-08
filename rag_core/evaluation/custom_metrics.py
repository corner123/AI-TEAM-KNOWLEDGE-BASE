from typing import List, Dict
import numpy as np


class CustomMetrics:
    @staticmethod
    def hit_rate(retrieved: List[str], relevant: List[str], k: int = None) -> float:
        if k is not None:
            retrieved = retrieved[:k]
        return 1.0 if any(doc in relevant for doc in retrieved) else 0.0

    @staticmethod
    def mrr(retrieved: List[str], relevant: List[str]) -> float:
        for i, doc in enumerate(retrieved):
            if doc in relevant:
                return 1.0 / (i + 1)
        return 0.0

    @staticmethod
    def latency_stats(latencies: List[float]) -> Dict[str, float]:
        arr = np.array(latencies)
        return {
            "mean": float(np.mean(arr)),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    @staticmethod
    def strategy_hit_rate(results: List[Dict]) -> Dict[str, float]:
        strategy_counts = {}
        strategy_hits = {}

        for item in results:
            strategy = item.get("strategy", "unknown")
            is_hit = item.get("is_hit", False)

            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
            strategy_hits[strategy] = strategy_hits.get(strategy, 0) + (1 if is_hit else 0)

        return {
            strategy: strategy_hits.get(strategy, 0) / count
            for strategy, count in strategy_counts.items()
        }
