# 工程知识 RAG 评测说明

## 两套评测必须分开

### 纯索引消融（suite=index）

只回答“BM25、dense、hybrid 的来源级召回有什么差异”。它：

- 仅运行 `answerable=true` 题；
- 使用题集显式标注的 oracle route；
- 关闭真实 Router、`rg`/AST/Git、Evidence Guard 和 Answerer；
- 三种策略共享同一 index build、embedding model、Top-K 与 per-arm candidate budget；
- hybrid 任一支路异常时 fail closed，不发布降级后的“hybrid”成绩；
- oracle route 不计 Routing Accuracy，报告显示 `N/A`。

### E2E 检索策略评测（suite=e2e）

运行完整的 Router → 分区检索 → live rg/AST/Git → Evidence Guard → answer/refusal。它评估来源召回、路由、实时核验和拒答策略。正式 E2E predictor 显式构造确定性 Answerer，不读取 `auto/deepseek` 在线生成配置，也不调用 DeepSeek；因此报告不能称为“LLM 答案正确率”或 “RAGAS faithfulness”。

suite 与 predictor contract 会强校验：index predictor 必须声明 oracle、无 live、无 answer、fail closed；E2E predictor 必须声明真实路由、live 与 answer。标错 suite 会直接终止。

这里的 `answer=true` 表示执行 Evidence Guard 后的确定性 answer/refusal 契约，不表示调用在线模型。Web/API 中可选的 DeepSeek 归纳层只改善人工阅读体验，不进入冻结开发集或 holdout 的正式指标；若未来评测模型回答，需要单独冻结模型版本、生成参数、prompt、成本与人工/自动评分协议，不能与当前检索报告混写。

## v2 数据契约

每行必须包含：

- `expected_route`：`design | implementation | official | comparison | out_of_scope`；
- `primary_sources`：回答成立所需的主证据；
- `supporting_sources`：补充证据；
- `relevant_sources`：必须严格等于前两者按顺序拼接；
- `relevant_symbols`：大小写敏感的真实 AST qualified names；
- `refusal_reason`：可回答题必须为 null，不可回答题必须是固定枚举；
- `label_version=engineering-eval/v2`；
- `dataset_role=development_regression | holdout`。

loader 会拒绝字符串冒充数组、重复来源、非法 route、空字段、answerability 冲突和 v1 隐式回退。

数据文件：

| 文件 | 数量 | 角色 |
| --- | ---: | --- |
| `mini_nanobot_internal.jsonl` | 60 | development regression |
| `official_engineering_specs.jsonl` | 20 | development regression |
| `engineering_holdout_v2.jsonl` | 30 | holdout |

开发集用于修正实现；holdout 只在系统冻结后运行，并保留第一次结果。题目随后公开用于审计，因此不能再作为未来迭代的未见测试集，也不能根据其失败继续调参；新的最终评测需要新的私有 holdout。

## 快照绑定

`scripts.freeze_engineering_eval` 将三者绑定：

1. 当前 ingestion manifest 的 build ID；
2. Mini-Nanobot commit、dirty 状态和被纳入语料的 worktree content hash；
3. 每个 JSONL 的 SHA-256。

结果写入 `data/eval/evaluation_snapshot.json`。正式 runner 会同时检查 snapshot ↔ dataset ↔ index；E2E factory 还会重新只读采集 Mini-Nanobot 并检查 live repo ↔ manifest。任一不一致时拒绝生成正式报告。

## 指标

对 Top-K 结果先按 canonical source 去重；同一文件的多个 chunk 只保留第一次出现。

| 指标 | 定义 |
| --- | --- |
| Primary Hit@K | 是否命中至少一个 primary source |
| Primary Recall@K | 命中的 primary source 比例 |
| Supporting Recall@K | 有 supporting label 的题上，命中比例 |
| Primary MRR | 第一个 primary source 的倒数排名 |
| graded nDCG@K | primary relevance=2，supporting=1，重复 source gain=0 |
| Source precision@K | Top-K 去重来源中，primary/supporting 所占比例 |
| Symbol Recall@K | 大小写敏感 qualified symbol 命中率 |
| Route accuracy / macro-F1 | 仅 E2E 计算 |
| Refusal P/R/F1 | 对不可回答题的显式拒答 |
| Strict reason recall | 拒答且 machine reason 与标注相同的不可回答题比例 |
| False refusal rate | answerable 题被拒答的比例 |
| P50/P95 | 每策略预热一次后的 predictor 调用延迟 |

这些是 source/file-level 指标。同一网页中命中错误段落仍可能得到 file hit，因此不能把 Source Recall 或 Source Precision 写成 passage correctness、citation entailment 或答案正确率。实现题额外报告 symbol/live 命中，缓解单纯文件命中的宽松问题。

## 运行

```powershell
conda activate all-in-rag
$env:MINI_NANOBOT_REPO = (Resolve-Path ..\Mini-Nanobot).Path
$env:ENGINEERING_INDEX_DIR = "data/indexes/engineering"
$env:ENGINEERING_MANIFEST_PATH = "data/manifests/builds/current.json"
$env:HF_HUB_OFFLINE = "1"
$env:ENGINEERING_GENERATION_PROVIDER = "deterministic"

python main.py engineering-eval `
  --suite index `
  --dataset data/eval/mini_nanobot_internal.jsonl data/eval/official_engineering_specs.jsonl `
  --snapshot data/eval/evaluation_snapshot.json `
  --output data/eval/reports/index_ablation

python main.py engineering-eval `
  --suite e2e `
  --dataset data/eval/mini_nanobot_internal.jsonl data/eval/official_engineering_specs.jsonl `
  --snapshot data/eval/evaluation_snapshot.json `
  --output data/eval/reports/e2e_development

python main.py engineering-eval `
  --suite e2e `
  --dataset data/eval/engineering_holdout_v2.jsonl `
  --snapshot data/eval/evaluation_snapshot.json `
  --output data/eval/reports/e2e_holdout_first_run
```

JSON 报告保留逐题结果、每策略 contract、build/model、dataset hash 和 snapshot；Markdown 只用于快速浏览。冷启动应单独测量，不能混入稳态 P50/P95。正式 runner 本身已经使用确定性 Answerer，上述环境变量是额外的操作防线，避免同一终端随后启动 Web/API 时意外发送评测问题与证据。
