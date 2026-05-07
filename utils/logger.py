import sys
from loguru import logger
from config import RAGConfig

_configured = False


def get_logger(name: str = "rag"):
    global _configured
    if not _configured:
        config = RAGConfig()
        logger.remove()
        logger.add(sys.stderr, level=config.log_level, format="{time:HH:mm:ss} | {level:<7} | {name}:{function}:{line} - {message}")
        _configured = True
    return logger.bind(name=name)
