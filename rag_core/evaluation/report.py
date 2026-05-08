from typing import Dict, Any
from utils.logger import get_logger

logger = get_logger("report")


class ReportGenerator:
    @staticmethod
    def generate_text_report(results: Dict[str, Any]) -> str:
        lines = ["=" * 60, "RAG系统评估报告", "=" * 60, ""]

        for strategy_name, data in results.items():
            lines.append(f"策略: {strategy_name}")
            lines.append("-" * 40)
            avg = data.get("avg_scores", {})
            for metric, value in avg.items():
                lines.append(f"  {metric}: {value:.4f}")
            lines.append(f"  评估样本数: {len(data.get('details', []))}")
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def generate_comparison_table(results: Dict[str, Any]) -> Dict[str, Any]:
        headers = ["策略", "Faithfulness", "Relevancy", "Precision", "Recall", "Latency(s)"]
        rows = []

        for strategy_name, data in results.items():
            avg = data.get("avg_scores", {})
            rows.append([
                strategy_name,
                f"{avg.get('faithfulness', 0):.3f}",
                f"{avg.get('answer_relevancy', 0):.3f}",
                f"{avg.get('context_precision', 0):.3f}",
                f"{avg.get('context_recall', 0):.3f}",
                f"{avg.get('latency', 0):.3f}",
            ])

        return {"headers": headers, "rows": rows}

    @staticmethod
    def export_csv(results: Dict[str, Any], output_path: str):
        import csv
        table = ReportGenerator.generate_comparison_table(results)

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(table["headers"])
            writer.writerows(table["rows"])

        logger.info(f"CSV report exported to {output_path}")
