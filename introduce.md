# 面试项目说明：多源工程知识 RAG

## 一句话定位

面向 Mini-Nanobot 工程资料构建多源知识检索系统：内部设计与历史决策为主，精选官方规范为辅；当前源码事实由 `rg`、AST 和 Git 实时核验，并通过只读 `knowledge.search` 工具供独立的 Mini-Nanobot Agent 调用。

## STAR

### Situation

工程问题混合了不同权威层级：源码能证明当前实现，却不一定记录“为什么这样设计”；ADR 和文档能说明意图，却可能落后于工作区；外部规范能说明标准要求，却不能证明项目已经实现。把所有内容简单丢进向量库，会产生旧代码、错误权威和相似近邻被当成事实的问题。

### Task

实现一个可运行、可引用、可拒答、可评测的工程知识 RAG，同时保持它与 Coding Agent 两个项目独立。系统需要回答设计、历史和官方规范问题；涉及当前实现时必须回到实时源码；证据不足时返回机器可读的拒答原因。

### Action

- 使用 YAML catalog 管理 Mini-Nanobot 本地 Git worktree 和 MCP、LangGraph、JSON Schema、Python、Docker 官方 URL allowlist；采集阶段保存 commit、dirty、内容 hash、抓取时间和版本。
- 对 Markdown 做结构化分块，对 Python 通过 AST 生成类/方法 symbol card，同时纳入测试、Dockerfile 和有限 Git history；构建可追溯 manifest。
- 将语料物理拆分为 code、test、design、history、official 五类分区；每个分区建立 BGE-small-zh-v1.5 FAISS 稠密索引和 BM25 索引，并用 RRF 做混合召回。
- 用确定性 Router 区分 implementation、design、official、comparison 与 out-of-scope；实现题要求 `rg`/AST/Git 实时证据，比较题同时要求当前实现与官方证据。
- 在近邻检索之上增加 catalog topic scope、hard-anchor、经验数据缺失和独立审计缺失检查，避免把“最相似文档”误当“足够证据”。
- 设计结构化引用 DTO，区分 `current_implementation`、`internal_design`、`internal_history`、`external_normative`，公开接口过滤本机路径、remote 凭据和 HTTP 内部字段。
- 构建严格 v2 评测：纯索引消融关闭 Router/live/Answerer；E2E 检索策略评测运行路由、实时核验和拒答；另冻结 30 条 holdout，包含真实跨语料比较与 OOD 题。
- 将 RAG 作为独立 FastAPI 服务，在 Mini-Nanobot 中仅增加可选的只读 `knowledge.search` HTTP 客户端；未配置服务时不注册，服务离线不会影响其他 Agent 工具。

### Result

冻结快照 `build_4ae47172a9869d88ce0f` 共包含 339 份文档、962 个 chunk。70 条 answerable 开发题的纯索引消融中，BM25 的 Primary Hit@5 为 0.629，高于 Dense 的 0.400 和 Hybrid RRF 的 0.457；Hybrid 的 Supporting Recall@5 最高，为 0.277。这说明技术语料中的精确标识符使关键词检索非常重要，混合检索并不会自动提升所有指标。

完整 E2E 开发集共 80 题，Primary Hit@5 为 0.886、拒答 F1 为 0.833。冻结后首次运行的 30 题 holdout 上，Route accuracy 为 0.700、Primary Hit@5 为 0.870、拒答 F1 为 0.636；7 个不可答题全部拒答，但 23 个可答题中有 8 个被误拒答。这个结果暴露出系统当前最大短板是证据门控过严，而不是缺少近邻候选。所有数字来自 `data/eval/reports_public/` 中绑定同一 snapshot 的脱敏报告，且 holdout 未用于后续调参。

## 面试时应强调的判断

1. RAG 不替代代码搜索。函数定义、调用点和未提交改动由实时工具核验。
2. 向量相似度不是答案置信度，RRF 分数也不是概率；系统另做证据范围和 anchor 检查。
3. 官方规范、内部设计和当前实现是不同证据角色，回答时不能互相冒充。
4. 消融实验必须只改变检索策略；路由、live tool 或 Answerer 混入后，数字不能称为纯索引对比。
5. 自动 source-level 指标不能声称答案完全正确；答案点与忠实度仍需冻结 Judge 或人工抽检。

## 暂不写进简历的内容

- Milvus 主从/自动降级、HNSW 调优；
- ColBERT 已带来提升；
- RAGAS faithfulness 已达到某数值；
- 海量网页爬取、生产并发、SLA 或安全认证；
- “最优策略”或没有对照实验支持的提升比例。
