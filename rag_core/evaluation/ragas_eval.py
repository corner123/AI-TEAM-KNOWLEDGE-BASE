from typing import List, Dict, Any
from langchain_core.language_models import BaseLLM
from utils.logger import get_logger

logger = get_logger("ragas_eval")


class RAGASEvaluator:
    def __init__(self, llm: BaseLLM = None):
        self.llm = llm

    def evaluate_single(self, question: str, answer: str, contexts: List[str], ground_truth: str) -> Dict[str, float]:
        try:
            from ragas import evaluate
            from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
            from datasets import Dataset

            data = {
                "question": [question],
                "answer": [answer],
                "contexts": [contexts],
                "ground_truth": [ground_truth],
            }
            dataset = Dataset.from_dict(data)

            result = evaluate(
                dataset,
                metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            )

            return {
                "faithfulness": result["faithfulness"][0] if "faithfulness" in result else 0.0,
                "answer_relevancy": result["answer_relevancy"][0] if "answer_relevancy" in result else 0.0,
                "context_precision": result["context_precision"][0] if "context_precision" in result else 0.0,
                "context_recall": result["context_recall"][0] if "context_recall" in result else 0.0,
            }
        except ImportError:
            logger.warning("ragas not installed, returning dummy scores")
            return {"faithfulness": 0.0, "answer_relevancy": 0.0, "context_precision": 0.0, "context_recall": 0.0}
        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {e}")
            return {"faithfulness": 0.0, "answer_relevancy": 0.0, "context_precision": 0.0, "context_recall": 0.0}

    def evaluate_batch(self, eval_data: List[Dict]) -> Dict[str, float]:
        all_scores = {"faithfulness": [], "answer_relevancy": [], "context_precision": [], "context_recall": []}

        for item in eval_data:
            scores = self.evaluate_single(
                question=item["question"],
                answer=item["answer"],
                contexts=item["contexts"],
                ground_truth=item["ground_truth"],
            )
            for key in all_scores:
                all_scores[key].append(scores[key])

        return {key: sum(vals) / len(vals) if vals else 0.0 for key, vals in all_scores.items()}
