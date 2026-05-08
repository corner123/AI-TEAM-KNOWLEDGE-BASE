"""生成评估数据集脚本"""
import json
import sys
sys.path.insert(0, ".")

EVAL_DATA = [
    {
        "question": "LangChain是什么?",
        "ground_truth": "LangChain是一个用于构建LLM应用的框架",
        "contexts": ["LangChain是一个用于开发由大语言模型驱动的应用程序的框架"],
        "category": "general"
    },
    {
        "question": "如何安装Milvus?",
        "ground_truth": "使用Docker安装Milvus: docker pull milvusdb/milvus",
        "contexts": ["Milvus可以通过Docker快速部署"],
        "category": "technical"
    },
    {
        "question": "BERT的参数量是多少?",
        "ground_truth": "BERT-base有1.1亿参数(110M)",
        "contexts": ["BERT-base包含12层Transformer，约1.1亿参数"],
        "category": "data_query"
    },
]


def main():
    output_path = "data/eval/eval_dataset.json"
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(EVAL_DATA, f, ensure_ascii=False, indent=2)

    print(f"Generated {len(EVAL_DATA)} eval samples to {output_path}")


if __name__ == "__main__":
    main()
