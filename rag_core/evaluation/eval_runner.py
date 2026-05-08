import json
import time
from typing import List, Dict, Any, Callable, Optional
from pathlib import Path
from langchain_core.documents import Document
from .ragas_eval import RAGASEvaluator
from .custom_metrics import CustomMetrics
from utils.logger import get_logger

logger = get_logger("eval_runner")


class EvalRunner:
    def __init__(self, ragas_evaluator: RAGASEvaluator, retrieve_fn: Callable, generate_fn: Callable):
        self.ragas_eval = ragas_evaluator
        self.retrieve_fn = retrieve_fn
        self.generate_fn = generate_fn

    def load_eval_dataset(self, path: str) -> List[Dict]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def run(self, eval_data: List[Dict], strategies: List[str] = None) -> Dict[str, Any]:
        if strategies is None:
            strategies = ["HybridSearch"]

        all_results = {}

        for strategy_name in strategies:
            logger.info(f"Evaluating strategy: {strategy_name}")
            strategy_results = []

            for item in eval_data:
                question = item["question"]
                ground_truth = item["ground_truth"]

                start_time = time.time()
                docs = self.retrieve_fn(question, strategy=strategy_name)
                latency = time.time() - start_time

                answer_result = self.generate_fn(question, docs)
                answer = answer_result.get("answer", "")

                contexts = [doc.page_content for doc in docs]

                ragas_scores = self.ragas_eval.evaluate_single(
                    question=question,
                    answer=answer,
                    contexts=contexts,
                    ground_truth=ground_truth,
                )

                strategy_results.append({
                    "question": question,
                    "answer": answer,
                    "ground_truth": ground_truth,
                    "latency": latency,
                    "num_docs": len(docs),
                    **ragas_scores,
                })

            avg_scores = {}
            for key in ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "latency"]:
                vals = [r[key] for r in strategy_results if key in r]
                avg_scores[key] = sum(vals) / len(vals) if vals else 0

            all_results[strategy_name] = {
                "avg_scores": avg_scores,
                "details": strategy_results,
            }

        return all_results

    def save_results(self, results: Dict, output_path: str):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"Results saved to {output_path}")
