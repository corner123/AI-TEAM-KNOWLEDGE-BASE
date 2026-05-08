"""构建索引脚本 - 扫描data/raw目录，构建Milvus索引"""
import sys
sys.path.insert(0, ".")

from config import RAGConfig
from rag_core.engine import RAGEngine
from utils.logger import get_logger

logger = get_logger("build_index")


def main():
    config = RAGConfig()
    engine = RAGEngine(config)
    engine.initialize()

    logger.info("Starting index build...")
    engine.build_index()

    stats = engine.get_stats()
    logger.info(f"Build complete. Stats: {stats}")


if __name__ == "__main__":
    main()
