from typing import Dict, Any, List, Optional
from config import RAGConfig
from rag_core.index.embeddings import EmbeddingManager
from rag_core.retrieval import (
    HybridSearchStrategy, Text2SQLStrategy, QueryRewriteStrategy,
    MetadataFilterStrategy, MultimodalSearchStrategy, RerankerStrategy,
)
from rag_core.generation import StandardGenerator, FunctionCallingGenerator, StreamingGenerator
from rag_core.query_router import QueryRouter
from rag_core.evaluation import RAGASEvaluator, EvalRunner, ReportGenerator
from utils.logger import get_logger
from utils.helpers import Timer

logger = get_logger("engine")


def _try_connect_milvus(config):
    """Try to connect to Milvus. Returns MilvusVectorStore or None."""
    try:
        from pymilvus import connections
        from rag_core.index.vector_store import MilvusVectorStore
        connections.connect(alias="default", host=config.host, port=config.port)
        logger.info(f"Connected to Milvus at {config.host}:{config.port}")
        return MilvusVectorStore
    except Exception as e:
        logger.warning(f"Milvus not available ({e}), using local FAISS store")
        return None


class RAGEngine:
    def __init__(self, config: RAGConfig = None):
        if config is None:
            config = RAGConfig()
        self.config = config
        self._initialized = False

        self.embedding_manager: Optional[EmbeddingManager] = None
        self.vector_store = None
        self.query_router: Optional[QueryRouter] = None
        self.generator = None
        self.evaluator: Optional[RAGASEvaluator] = None

    def initialize(self):
        if self._initialized:
            return

        logger.info("Initializing RAG Engine...")

        self.embedding_manager = EmbeddingManager(self.config.embedding)

        # Try Milvus first, fall back to local FAISS
        MilvusVectorStore = _try_connect_milvus(self.config.milvus)
        if MilvusVectorStore:
            self.vector_store = MilvusVectorStore(self.config.milvus, self.embedding_manager)
        else:
            from rag_core.index.local_vector_store import LocalVectorStore
            self.vector_store = LocalVectorStore(self.embedding_manager)

        from langchain_deepseek import ChatDeepSeek
        llm = ChatDeepSeek(
            model=self.config.llm.model,
            api_key=self.config.llm.api_key,
            base_url=self.config.llm.base_url,
            temperature=self.config.llm.temperature,
            max_tokens=self.config.llm.max_tokens,
        )

        strategies = {
            "HybridSearch": HybridSearchStrategy(vector_store=self.vector_store),
            "Text2SQL": Text2SQLStrategy(llm=llm),
            "QueryRewrite": QueryRewriteStrategy(llm=llm, vector_store=self.vector_store),
            "MetadataFilter": MetadataFilterStrategy(llm=llm, vector_store=self.vector_store),
            "MultimodalSearch": MultimodalSearchStrategy(),
        }

        self.query_router = QueryRouter(strategies=strategies, llm=llm)
        self.generator = StandardGenerator(llm=llm)
        self.evaluator = RAGASEvaluator(llm=llm)
        self._initialized = True

        logger.info("RAG Engine initialized")

    def query(self, question: str, strategy: str = None, top_k: int = 5) -> Dict[str, Any]:
        if not self._initialized:
            self.initialize()

        with Timer("query") as t:
            if strategy and strategy in self.query_router.strategies:
                selected = self.query_router.strategies[strategy]
                docs = selected.retrieve(question, top_k=top_k)
                strategy_used = selected.get_strategy_name()
            else:
                docs = self.query_router.route_and_retrieve(question, top_k=top_k)
                strategy_used = self.query_router.last_strategy_name

            result = self.generator.generate(question, docs)

        result["latency"] = t.elapsed
        result["strategy_used"] = strategy_used

        return result

    def build_index(self, data_dir: str = None):
        from rag_core.data import LoaderFactory, ChunkerFactory, enrich_metadata
        from pathlib import Path

        if not self._initialized:
            self.initialize()

        data_dir = data_dir or self.config.raw_data_dir
        all_docs = []

        for file_path in Path(data_dir).rglob("*"):
            if file_path.is_file():
                try:
                    docs = LoaderFactory.load_file(str(file_path))
                    for doc in docs:
                        chunker = ChunkerFactory.get_chunker(doc.metadata.get("doc_type", "text"))
                        chunks = chunker.chunk([doc])
                        enriched = enrich_metadata(chunks)
                        all_docs.extend(enriched)
                except Exception as e:
                    logger.warning(f"Failed to process {file_path}: {e}")

        if all_docs:
            self.vector_store.insert_documents(all_docs)
            logger.info(f"Indexed {len(all_docs)} chunks from {data_dir}")

    def evaluate(self, eval_data_path: str, strategies: List[str] = None) -> Dict:
        if not self._initialized:
            self.initialize()

        runner = EvalRunner(
            ragas_evaluator=self.evaluator,
            retrieve_fn=lambda q, strategy: self.query_router.strategies.get(strategy, list(self.query_router.strategies.values())[0]).retrieve(q),
            generate_fn=lambda q, docs: self.generator.generate(q, docs),
        )

        eval_data = runner.load_eval_dataset(eval_data_path)
        results = runner.run(eval_data, strategies=strategies)

        report = ReportGenerator.generate_text_report(results)
        logger.info(f"\n{report}")

        return results

    def get_stats(self) -> Dict[str, Any]:
        if not self._initialized:
            return {"status": "not_initialized"}
        return {
            "status": "initialized",
            "vector_store": self.vector_store.get_stats() if self.vector_store else None,
            "strategies": list(self.query_router.strategies.keys()) if self.query_router else [],
        }
