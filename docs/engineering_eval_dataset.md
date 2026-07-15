---
title: Engineering RAG v2 datasets
document_status: implemented
last_verified: 2026-07-14
---

# 工程知识 RAG 评测集

## 数据组成

- `mini_nanobot_internal.jsonl`：60 条开发回归题；15 设计、15 代码、10 安全、10 状态、10 边界题。
- `official_engineering_specs.jsonl`：20 条开发回归题；覆盖 MCP、LangGraph、JSON Schema、Python 3.12、Docker。
- `engineering_holdout_v2.jsonl`：30 条冻结题；9 条内部实现、6 条官方规范、5 条真实跨语料比较、7 条不可回答题和 3 条 false-refusal control。

开发回归集允许在修复 bug 后重复运行；holdout 仅做结构、路径、AST symbol 静态校验，系统冻结后第一次运行的结果必须保留，不能根据结果继续调参后只展示新版数字。

## 标签约束

每行都是 `engineering-eval/v2`：

| 字段 | 约束 |
| --- | --- |
| `id` | 全数据集唯一 |
| `answerable` | JSON boolean |
| `expected_route` | design / implementation / official / comparison / out_of_scope |
| `primary_sources` | answerable 必须非空；unanswerable 必须为空 |
| `supporting_sources` | 与 primary 去重且不相交 |
| `relevant_sources` | 精确等于 primary + supporting |
| `relevant_symbols` | AST 中真实存在的大小写敏感限定名 |
| `refusal_reason` | answerable 为 null；unanswerable 为固定 machine code |
| `source_revision` | official 记录规范版本；内部/比较/边界由 freeze 脚本绑定 manifest 与 worktree hash |
| `dataset_role` | development_regression 或 holdout |

`expected_answer_points` 是人工或冻结 Judge 的核验要点，不做字符串包含率评分。不同措辞可以表达同一事实，简单 substring 会制造无效数字。

## 权威与检索边界

- 当前代码事实：必须有 `current_implementation` live 证据；
- 设计问题：主证据来自 `docs/` 或 Git history，代码/测试可作为 supporting；
- 官方问题：主证据必须是 catalog allowlist 的 canonical URL；
- comparison：primary 至少包含一侧当前实现和一侧官方规范；
- unanswerable：不能因为 dense 总能找到近邻就生成答案。

同一路径在索引中可能产生多个 chunk。来源级指标先去重；实现题同时用 symbol recall 和 live verification 防止“命中同文件错误方法”被当成正确。

## 冻结流程

1. 完成 source sync，确保采集期间 Git commit/status 未变化；
2. 从该 manifest 构建并验证索引；
3. 静态校验所有相对路径和 relevant symbol；
4. 运行 `python -m scripts.freeze_engineering_eval`；
5. 检查 `evaluation_snapshot.json` 中的 build ID、Mini content hash 与三个 dataset SHA-256；
6. 开发评测完成后，最后执行一次 holdout 并保存原始报告。

本数据集不评价真实模型任务成功率、生产 SLA、远程 GitHub 状态、在线价格或安全认证。
