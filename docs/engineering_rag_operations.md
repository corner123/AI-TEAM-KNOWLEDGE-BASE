# 多源工程知识 RAG：运行与数据边界

## 项目边界

`ai-team-knowledge-base` 与 `Mini-Nanobot` 始终是两个独立项目：

- Mini-Nanobot 负责 Agent 执行、工具调用，以及通过 `rg`、AST、Git 对当前工作区进行实时核验；
- 本项目负责索引内部设计文档、历史决策和精选官方规范，完成检索、引用、拒答与评估；
- `knowledge.search` 只通过只读接口调用本项目，不复制 Mini-Nanobot 的执行权限，也不允许 RAG 服务修改代码仓库。

外部规范只能证明“规范要求什么”，不能证明 Mini-Nanobot 已经实现；内部设计文档只能证明设计意图；当前实现事实必须回到当前源码实时核验。

## 数据流

```text
Mini-Nanobot README / docs / ADR ─┐
                                  ├─ source catalog ─ ingestion manifest ─ partitioned index
精选官方规范网页 ─────────────────┘                                     │
                                                                         ├─ BM25
                                                                         ├─ dense
                                                                         └─ RRF hybrid

用户问题 ─ source-intent route ─ 索引召回 ─ 当前源码 rg/AST/Git 核验 ─ 引用 / 拒答
```

真实源码不依赖预构建向量索引来证明。索引中的代码片段最多用于定位候选，最终证据应标记是否完成实时核验。

## 数据源清单

实际源配置位于 `data/sources/catalog.yaml`，当前包含：

- Mini-Nanobot 的公开 README、`docs/`、Python 源码、测试、benchmark 描述和 Dockerfile；
- MCP 2025-06-18 lifecycle 与 tools 规范；
- LangGraph persistence 官方文档；
- JSON Schema Draft 2020-12 core；
- Python 3.12 的 asyncio、pathlib、sqlite3 与 subprocess 文档；
- Docker security 与 resource constraints 文档。

目录扫描必须排除 `.env`、credential、日志、SQLite 运行库、缓存、模型权重和未授权笔记。抓取器必须使用 HTTPS 域名白名单并记录 canonical URL、内容哈希、版本和抓取时间。

## 本地环境

推荐激活仓库声明的 Conda 环境后运行：

```powershell
conda activate all-in-rag
$env:PYTHONNOUSERSITE = "1"
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:TOKENIZERS_PARALLELISM = "false"
$env:MINI_NANOBOT_REPO = (Resolve-Path ..\Mini-Nanobot).Path
python -m pip check
```

模型下载阶段需要联网；本地 Embedding 的离线测试可另外设置 `$env:HF_HUB_OFFLINE = "1"`。这个变量只影响 Hugging Face，不会阻止 DeepSeek API 请求。完全离线运行还必须设置 `ENGINEERING_GENERATION_PROVIDER=deterministic`。密钥只写入未跟踪的 `.env` 或进程环境变量，`.env.example` 只能保留占位符。

## 推荐运行顺序

1. 检查 `data/sources/catalog.yaml` 的 Mini-Nanobot 路径、官方 URL 白名单和版本；
2. 只读采集各来源，生成包含内容哈希与增删改 diff 的 manifest；
3. 从同一 manifest 构建 `internal/code`、`internal/test`、`internal/design`、`internal/history`、`official/official` 物理分区；
4. 查看健康检查，确认 build ID、分区数、文档数以及 live-code 核验根目录；
5. 先执行 `/retrieve` 或本地 retrieve 命令观察路由、证据角色和引用，再执行答案生成；
6. 分别运行 70 条 answerable 的纯索引消融与 80 条开发集 E2E 策略评测；二者不能混为同一报告；
7. 最后启动只读服务，通过根路径 Web 页面人工验收，再由 Mini-Nanobot 的 `knowledge.search` 调用。

构建后运行 `python -m scripts.freeze_engineering_eval`，把 index build、Mini commit/worktree hash 和三份题集 hash 写入 `evaluation_snapshot.json`。30 条 holdout 只在系统冻结后运行一次。

具体 CLI/API 命令以项目入口的 `--help` 与 OpenAPI schema 为准；不要让运维文档替代可执行参数校验。

## 本地 Web 页面

页面和 API 使用同一 FastAPI 进程，不需要单独的前端构建或第二个端口：

```powershell
$env:ENGINEERING_INDEX_DIR = "data/indexes/engineering"
$env:ENGINEERING_MANIFEST_PATH = "data/manifests/builds/current.json"
$env:MINI_NANOBOT_REPO = (Resolve-Path ..\Mini-Nanobot).Path
$env:RAG_API_TOKEN = python -c "import secrets; print(secrets.token_urlsafe(32))"
$env:RAG_UI_PORT = "8000"
python app.py
```

打开 `http://127.0.0.1:8000/`；若启用了鉴权，在查询面板的“连接设置”中输入本次进程使用的 token。收到 401 时该设置会自动展开。页面必须满足以下边界：

- token 仅保留在 JavaScript 内存，不写入 URL 或 Web Storage；
- 所有证据以纯文本节点渲染，不执行检索内容中的 HTML、脚本或命令；
- 只有官方白名单中的 HTTPS 来源会显示为可点击外链；本地源码路径只显示为文本；
- 页面只提交 `query` 和 `top_k`，不能覆盖仓库、索引或服务端路径；
- 正常拒答和候选证据必须与系统故障使用不同状态展示；
- 回答中的 `[E#]` 只映射到本次 API 响应声明的站内引用目标，未知编号保持纯文本；
- 不提供 source sync、index build、evaluation 或其他写操作。

`python main.py serve --host 127.0.0.1 --port 8000` 与 `python app.py` 启动的是同一应用。内置服务器仍然只允许 loopback；远程访问必须放在带 TLS 和鉴权的反向代理之后。

## DeepSeek 回答生成与隐私边界

DeepSeek 只负责把已经通过证据门控的检索结果归纳为易读答案，不参与查询路由、索引召回、`rg`/AST/Git 实时核验、证据充分性判断或拒答决策。`POST /retrieve` 始终是确定性的，且不会调用 DeepSeek；`POST /answer` 也会先完成同一条确定性检索链路，证据不足时直接拒答，不把问题发送给模型。

使用 `ENGINEERING_GENERATION_PROVIDER` 选择回答模式：

| 值 | 行为 | 适用场景 |
| --- | --- | --- |
| `auto` | 有有效 `DEEPSEEK_API_KEY` 时调用 DeepSeek，否则使用确定性摘要 | 已确认外部传输边界的交互演示 |
| `deepseek` | 要求 DeepSeek 配置完整；缺少有效密钥属于启动/首次加载配置错误 | 明确要求在线生成的受控环境 |
| `deterministic`（默认） | 不创建在线客户端，不发送任何问题或证据 | 离线测试、正式评测、敏感语料 |

模型调用失败、超时、返回空内容或使用不存在的 `[E#]` 引用时，Answerer 不改变检索与引用结果，而是返回确定性证据摘要，并通过 `generation_mode=deterministic_fallback` 和 warning 标明降级。成功的模型回答使用 `generation_mode=model`；未启用模型时为 `deterministic`。不要把 fallback 当作模型生成成功，也不要通过重试绕过 Evidence Guard 的拒答。

`GET /health` 只报告 DeepSeek 客户端是否已配置，不会为探活额外发起模型请求，也不承诺密钥有效。凭据、网络和模型可用性以实际 `/answer` 的 `generation_mode` 为准。

启用 `auto` 或 `deepseek` 是一次外部数据传输决策：用户问题和本次选中的证据片段会发送到 `DEEPSEEK_BASE_URL`，其中可能包含源码、设计文档或历史记录。发送前必须确认这些内容允许交给对应服务处理。未经授权的私有仓库应固定使用 `deterministic`，或改用经过组织批准的兼容服务。服务不会把 API 密钥放入 prompt、响应、页面或结构化日志；运维人员也不应打印环境变量值排查问题。

以下生成参数可以通过进程环境或未跟踪的 `.env` 配置：`DEEPSEEK_API_KEY`、`DEEPSEEK_BASE_URL`、`DEEPSEEK_MODEL`、`ENGINEERING_LLM_TIMEOUT_SECONDS`、`ENGINEERING_LLM_MAX_RETRIES`、`ENGINEERING_LLM_MAX_TOKENS`、`ENGINEERING_LLM_TEMPERATURE` 和 `ENGINEERING_LLM_THINKING_ENABLED`。远程 Base URL 必须使用 HTTPS；只有 loopback 本地开发端点允许 HTTP。不要把真实值写进运行手册、截图、命令历史或 Git。

## 上线前最小检查

- manifest 中每个官方 `canonical_url` 与评测题 `relevant_sources` 一致；
- Mini-Nanobot revision、dirty 状态和索引 build ID 可追溯；
- 对实现类问题，返回结果包含 `current_implementation` 或明确拒答；
- 对官方类问题，引用只来自 allowlist 中的官方域名；
- 对比较类问题，同时具备当前实现与外部规范两侧证据，否则拒答；
- `/retrieve` 不触发在线生成；`/answer` 的 `generation_mode` 与 provider 配置相符，模拟模型故障时能回退且不泄露异常正文；
- 在线生成启用前，已确认用户问题和选中证据允许发送到配置的 DeepSeek 端点；
- API 限制 query 长度和 Top-K，且不能由请求传入任意本地根目录；
- `knowledge.search` 标记为 read-only，网络超时、服务离线、非 2xx、无效 JSON 都返回受控错误；
- `.env`、token、原始抓取快照和本机索引不进入 Git。

## 更新策略

Mini-Nanobot 提交变化后：

1. 读取新 commit 和 dirty 状态，不修改其工作区；
2. 重跑采集并根据 hash 生成 added/modified/deleted diff；
3. 从新的 manifest 全量重建内部索引；当前工程索引发布流程不宣称支持增量更新；
4. 核验 60 条内部题和 30 条 holdout 的路径、AST symbol 与答案点，再通过 freeze 脚本更新 `source_revision`；
5. 跑回归评测并与同配置 baseline 比较。

官方网页变化后：

1. 保留 URL、抓取时间、内容 hash 和 source version；
2. 检查许可与来源条款，不把“可访问”自动理解为“可任意再分发”；
3. 重新核对对应的 20 条官方题；
4. 生成新 manifest 与评测报告，保留旧报告的配置说明。

## 故障定位

| 现象 | 优先检查 |
| --- | --- |
| 设计题只返回源码 | router 意图、`internal/design` 分区和 authority 元数据 |
| 实现题有答案但无实时证据 | `MINI_NANOBOT_REPO`、`rg` 可执行文件、路径边界和 verification terms |
| 官方题 Recall 为 0 | catalog URL 与题集 canonical URL 是否完全一致、抓取是否被重定向或拒绝 |
| 混合检索劣于 BM25 | RRF 权重、候选数、重复 chunk、中文与标识符 tokenization |
| 不可回答题仍生成答案 | evidence sufficiency 与 `refused` 是否由答案层显式输出 |
| `/answer` 仍是证据拼接摘要 | `ENGINEERING_GENERATION_PROVIDER`、密钥是否有效，以及响应中的 `generation_mode` 与 warnings；不要打印密钥 |
| DeepSeek 失败后页面报 500 | provider 异常是否被 Answerer 捕获并进入 `deterministic_fallback` |
| 首次请求明显更慢 | 嵌入或 reranker 冷启动；分开记录冷启动和预热后延迟 |

## 面试叙述边界

可以陈述已运行、已测试且有报告支持的能力；不要把计划能力写成完成能力。尤其不要声称：

- Milvus 或 FAISS 本身提高了语义准确率；
- 外部规范检索证明了内部实现符合规范；
- 自动指标等价于人工正确性；
- 教学型 Mini-Nanobot 已达到生产级沙箱、安全认证或 SLA；
- 没有固定配置与实测报告时，某策略是“最优策略”。
