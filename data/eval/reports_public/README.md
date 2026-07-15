# 公开评测报告

本目录保存可随仓库公开的正式评测报告，统一对应：

- `build_id`: `build_4ae47172a9869d88ce0f`
- ingestion manifest、索引和 evaluation snapshot：同一冻结构建

原始报告位于被 Git 忽略的 `data/eval/reports/`。公开副本由原始报告脱敏生成，只替换本机绝对路径等机器相关路径；不修改指标、样本判定、排序结果、延迟或报告结论。Markdown 用于快速阅读，JSON 保留逐题结果与运行元数据。

## 报告清单

1. 纯索引消融：[Markdown](index_ablation_build_4ae47172a9869d88ce0f.md) · [JSON](index_ablation_build_4ae47172a9869d88ce0f.json)
2. 开发集 E2E：[Markdown](e2e_development_build_4ae47172a9869d88ce0f.md) · [JSON](e2e_development_build_4ae47172a9869d88ce0f.json)
3. 首次冻结 holdout：[Markdown](e2e_holdout_first_run_build_4ae47172a9869d88ce0f.md) · [JSON](e2e_holdout_first_run_build_4ae47172a9869d88ce0f.json)

## Holdout 边界

第三份报告是当前冻结 holdout 的第一次正式运行结果，已原样保留。题目在首次运行后随仓库公开，因而不能再作为未来迭代的未见测试集，也不得根据该结果调参后覆盖首次报告。后续系统迭代应继续使用开发回归集；若确需新的最终评测，应预先建立新的私有 holdout、snapshot 和 build 标识，并与本次首次报告并列保存。
