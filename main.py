import argparse
import sys
from config import RAGConfig
from rag_core.engine import RAGEngine
from utils.logger import get_logger

logger = get_logger("main")


def main():
    parser = argparse.ArgumentParser(description="AI团队知识库 RAG系统")
    subparsers = parser.add_subparsers(dest="command")

    query_parser = subparsers.add_parser("query", help="查询知识库")
    query_parser.add_argument("question", help="查询问题")
    query_parser.add_argument("--strategy", "-s", help="指定检索策略")
    query_parser.add_argument("--top-k", "-k", type=int, default=5, help="返回结果数量")

    build_parser = subparsers.add_parser("build", help="构建索引")
    build_parser.add_argument("--data-dir", "-d", help="数据目录")

    eval_parser = subparsers.add_parser("eval", help="运行评估")
    eval_parser.add_argument("--eval-data", "-e", required=True, help="评估数据集路径")
    eval_parser.add_argument("--strategies", "-s", nargs="+", help="评估策略列表")

    stats_parser = subparsers.add_parser("stats", help="查看系统状态")

    args = parser.parse_args()

    config = RAGConfig()
    engine = RAGEngine(config)

    if args.command == "query":
        engine.initialize()
        result = engine.query(args.question, strategy=args.strategy, top_k=args.top_k)
        print(f"\n回答:\n{result['answer']}")
        print(f"\n来源: {result['sources']}")
        print(f"策略: {result.get('strategy_used', 'N/A')}")
        print(f"置信度: {result.get('confidence', 'N/A')}")
        print(f"耗时: {result.get('latency', 0):.2f}s")

    elif args.command == "build":
        engine.initialize()
        engine.build_index(data_dir=args.data_dir)
        print("索引构建完成")

    elif args.command == "eval":
        engine.initialize()
        results = engine.evaluate(args.eval_data, strategies=args.strategies)
        from rag_core.evaluation import ReportGenerator
        print(ReportGenerator.generate_text_report(results))

    elif args.command == "stats":
        engine.initialize()
        stats = engine.get_stats()
        print(f"系统状态: {stats}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
