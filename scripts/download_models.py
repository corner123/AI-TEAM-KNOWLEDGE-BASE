"""下载嵌入模型脚本"""
import sys
sys.path.insert(0, ".")

from sentence_transformers import SentenceTransformer
from config import EmbeddingConfig


def main():
    config = EmbeddingConfig()

    print(f"Downloading dense embedding model: {config.model_path}")
    SentenceTransformer(config.model_path)
    print("Dense model downloaded")

    print(f"Downloading sparse embedding model: {config.sparse_model_path}")
    SentenceTransformer(config.sparse_model_path)
    print("Sparse model downloaded")

    print("All models downloaded")


if __name__ == "__main__":
    main()
