import gradio as gr
from config import RAGConfig
from rag_core.engine import RAGEngine
from utils.logger import get_logger

logger = get_logger("app")

config = RAGConfig()
engine = RAGEngine(config)


def query_handler(question: str, strategy: str, top_k: int):
    if not question.strip():
        return "请输入问题", "", "", ""

    try:
        if strategy == "自动":
            result = engine.query(question, top_k=top_k)
        else:
            result = engine.query(question, strategy=strategy, top_k=top_k)

        answer = result.get("answer", "无回答")
        sources = "\n".join([f"• {s}" for s in result.get("sources", [])])
        meta = f"策略: {result.get('strategy_used', 'N/A')} | 置信度: {result.get('confidence', 'N/A')} | 耗时: {result.get('latency', 0):.2f}s"
        return answer, sources, meta, ""
    except Exception as e:
        return f"错误: {e}", "", "", ""


def build_index_handler():
    try:
        engine.initialize()
        engine.build_index()
        stats = engine.get_stats()
        return f"索引构建完成\n{stats}"
    except Exception as e:
        return f"构建失败: {e}"


def eval_handler(eval_file, strategies):
    if eval_file is None:
        return "请上传评估数据集"

    try:
        engine.initialize()
        results = engine.evaluate(eval_file.name, strategies=strategies.split(","))
        from rag_core.evaluation import ReportGenerator
        return ReportGenerator.generate_text_report(results)
    except Exception as e:
        return f"评估失败: {e}"


def create_app():
    with gr.Blocks(title="AI团队知识库", theme=gr.themes.Soft()) as app:
        gr.Markdown("# AI团队知识库 - RAG智能问答系统")

        with gr.Tabs():
            with gr.Tab("智能问答"):
                with gr.Row():
                    with gr.Column(scale=2):
                        question_input = gr.Textbox(label="输入问题", placeholder="例如: BERT的参数量是多少?", lines=2)
                        with gr.Row():
                            strategy_select = gr.Dropdown(
                                choices=["自动", "HybridSearch", "Text2SQL", "QueryRewrite", "MetadataFilter"],
                                value="自动", label="检索策略"
                            )
                            top_k_slider = gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Top-K")
                        submit_btn = gr.Button("查询", variant="primary")

                    with gr.Column(scale=3):
                        answer_output = gr.Textbox(label="回答", lines=6)
                        sources_output = gr.Textbox(label="来源", lines=3)
                        meta_output = gr.Textbox(label="元信息", lines=1)

                submit_btn.click(
                    fn=query_handler,
                    inputs=[question_input, strategy_select, top_k_slider],
                    outputs=[answer_output, sources_output, meta_output],
                )

            with gr.Tab("知识库管理"):
                build_btn = gr.Button("构建索引", variant="primary")
                build_output = gr.Textbox(label="构建结果", lines=5)
                build_btn.click(fn=build_index_handler, outputs=build_output)

            with gr.Tab("评估中心"):
                eval_file = gr.File(label="上传评估数据集 (JSON)")
                eval_strategies = gr.Textbox(label="评估策略(逗号分隔)", value="HybridSearch,Text2SQL")
                eval_btn = gr.Button("运行评估", variant="primary")
                eval_output = gr.Textbox(label="评估结果", lines=15)
                eval_btn.click(fn=eval_handler, inputs=[eval_file, eval_strategies], outputs=eval_output)

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=7860)
