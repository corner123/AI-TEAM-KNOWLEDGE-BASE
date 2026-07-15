import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

# A process manager or deployment environment must take precedence over the
# developer-only .env file.  In particular, never let a repository file
# replace API tokens or repository roots supplied by the service runtime.
load_dotenv(override=False)


@dataclass
class LLMConfig:
    api_key: str = field(default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", ""))
    base_url: str = field(default_factory=lambda: os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    model: str = field(default_factory=lambda: os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"))
    temperature: float = 0.1
    max_tokens: int = 2048


@dataclass
class MilvusConfig:
    host: str = field(default_factory=lambda: os.getenv("MILVUS_HOST", "localhost"))
    port: int = field(default_factory=lambda: int(os.getenv("MILVUS_PORT", "19530")))
    collection_name: str = field(default_factory=lambda: os.getenv("MILVUS_COLLECTION", "knowledge_base"))
    multimodal_collection: str = "multimodal_index"
    dense_dim: int = 1024
    metric_type: str = "COSINE"
    index_type: str = "HNSW"
    hnsw_m: int = 16
    hnsw_ef_construction: int = 128


@dataclass
class EmbeddingConfig:
    model_path: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL_PATH", "BAAI/bge-large-zh-v1.5"))
    sparse_model_path: str = field(default_factory=lambda: os.getenv("SPARSE_EMBEDDING_MODEL_PATH", "BAAI/bge-m3"))
    multimodal_model_path: str = field(default_factory=lambda: os.getenv("MULTIMODAL_MODEL_PATH", "BAAI/visualized-base"))
    device: str = "cpu"
    batch_size: int = 32


@dataclass
class ChunkingConfig:
    chunk_size: int = 512
    chunk_overlap: int = 64
    separators: list = field(default_factory=lambda: ["\n\n", "\n", " ", ""])


@dataclass
class RetrievalConfig:
    top_k: int = 5
    rrf_k: int = 60
    rerank_top_k: int = 3
    sql_max_retries: int = 3
    sql_timeout: int = 10


@dataclass
class RAGConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    milvus: MilvusConfig = field(default_factory=MilvusConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    data_dir: str = field(default_factory=lambda: os.getenv("DATA_DIR", "data"))
    raw_data_dir: str = field(default_factory=lambda: os.getenv("RAW_DATA_DIR", "data/raw"))
    processed_data_dir: str = field(default_factory=lambda: os.getenv("PROCESSED_DATA_DIR", "data/processed"))
    eval_data_dir: str = field(default_factory=lambda: os.getenv("EVAL_DATA_DIR", "data/eval"))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
