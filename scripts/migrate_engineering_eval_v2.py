"""Deterministically migrate the frozen 80-question development set to v2 labels."""

from __future__ import annotations

import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASETS = (
    ROOT / "data/eval/mini_nanobot_internal.jsonl",
    ROOT / "data/eval/official_engineering_specs.jsonl",
)

IMPLEMENTATION_INTENTS = {
    "trace_security_decision",
    "trace_state_rehydration",
    "trace_context_algorithm",
}
BOUNDARY_ROUTES = {
    "MN-BOUND-001": "out_of_scope",
    "MN-BOUND-002": "design",
    "MN-BOUND-003": "out_of_scope",
    "MN-BOUND-004": "out_of_scope",
    "MN-BOUND-005": "design",
    "MN-BOUND-006": "out_of_scope",
    "MN-BOUND-007": "design",
    "MN-BOUND-008": "out_of_scope",
    "MN-BOUND-009": "design",
    "MN-BOUND-010": "out_of_scope",
}
REFUSAL_REASONS = {
    "MN-BOUND-001": "dynamic_external_state_not_indexed",
    "MN-BOUND-002": "missing_empirical_evidence",
    "MN-BOUND-003": "future_commitment_not_documented",
    "MN-BOUND-004": "missing_production_telemetry",
    "MN-BOUND-005": "missing_independent_verification",
    "MN-BOUND-006": "underspecified_dynamic_cost",
    "MN-BOUND-007": "missing_empirical_evidence",
    "MN-BOUND-008": "future_commitment_not_documented",
    "MN-BOUND-009": "missing_empirical_evidence",
    "MN-BOUND-010": "unverifiable_private_information",
}
RELEVANT_SYMBOLS = {
    "MN-CODE-001": ["QueryEngine.submit_message", "QueryEngine._inject_dynamic_context", "query", "AgentState.add_message"],
    "MN-CODE-002": ["QueryEngine.resume", "SQLiteCheckpointStore.load", "QueryEngine.submit_message"],
    "MN-CODE-003": ["query"],
    "MN-CODE-004": ["AgentState.active_messages", "AgentState.set_context_projection", "AgentState.add_message"],
    "MN-CODE-005": ["AgentState.to_dict", "AgentState.to_json", "_json_safe_dict"],
    "MN-CODE-006": ["create_default_registry", "ToolRegistry.extend", "ToolRegistry.register", "ToolRegistry.get"],
    "MN-CODE-007": ["StreamingToolExecutor.execute_many", "StreamingToolExecutor.execute_one"],
    "MN-CODE-008": ["StreamingToolExecutor.execute_one", "ToolRegistry.get", "Tool.validate_input", "HookManager.check", "Tool.check_permissions", "Tool.run", "StreamingToolExecutor._persist_if_large", "HookManager.emit"],
    "MN-CODE-009": ["ToolContext.resolve_workspace_path", "FileReadTool.run", "RgSearchTool.run"],
    "MN-CODE-010": ["OpenAIProvider.generate", "Message", "ToolCall", "Usage"],
    "MN-CODE-011": ["SkillManager.discover", "SkillManager._parse_skill", "SkillManager.render_attachment", "SkillManager.load", "QueryEngine._inject_dynamic_context", "SkillTool.run"],
    "MN-CODE-012": ["QueryEngine._inject_dynamic_context", "LongTermMemoryStore.recall_attachment", "LongTermMemoryStore.recall", "LongTermMemoryStore.records"],
    "MN-CODE-013": ["AgentTool.run", "SubAgentRunner.run", "SubAgentRunner._run_child", "query", "SubAgentResult.to_tool_content"],
    "MN-CODE-014": ["SubAgentRunner.run", "SubAgentRunner.status", "SubAgentRunner._run_child", "BackgroundSubAgent", "AgentStatusTool.run"],
    "MN-CODE-015": ["QueryEngine.submit_message", "QueryEngine._compact_with_summary_agent", "ContextCompressor._autocompact_summary", "ContextCompressor._summarize", "LLMProvider.generate"],
    "MN-SEC-003": ["ShellTool.check_permissions", "CommandSafetyPolicy.inspect", "CommandSafetyPolicy.is_read_only_command"],
    "MN-SEC-007": ["SubAgentRunner._child_permissions", "SubAgentRunner._allowed_tools", "SubAgentRunner._child_registry"],
    "MN-SEC-008": ["CommandSafetyPolicy.inspect", "ShellTool.check_permissions"],
    "MN-STATE-003": ["AgentState.to_dict", "AgentState.from_dict", "_json_safe_dict", "QueryEngine.submit_message"],
    "MN-STATE-007": ["ContextCompressor._history_snip", "ContextCompressor._older_duplicate_tool_indices", "ContextCompressor._superseded_edit_indices"],
    "MN-STATE-008": ["ContextCompressor._microcompact", "ContextCompressor._cache_reference_is_fresh", "ContextCompressor._cache_reference_id"],
}


def route_for(row: dict) -> str:
    if row["id"] in BOUNDARY_ROUTES:
        return BOUNDARY_ROUTES[row["id"]]
    if row["source_scope"] == "official_specifications":
        return "official"
    if row["category"] == "design_architecture":
        return "design"
    if row["category"] == "code_symbol_call_chain":
        return "implementation"
    if row["category"] in {"security_permissions", "checkpoint_context_memory"}:
        return (
            "implementation"
            if row["expected_intent"] in IMPLEMENTATION_INTENTS
            else "design"
        )
    raise ValueError(f"no explicit route rule for {row['id']}")


def split_sources(row: dict, route: str) -> tuple[list[str], list[str]]:
    sources = list(dict.fromkeys(row.get("relevant_sources", [])))
    if not row["answerable"]:
        return [], []
    if route == "official":
        primary = sources[:1]
    elif route == "design":
        primary = next(
            ([item] for item in sources if item == "README.md" or item.startswith("docs/")),
            sources[:1],
        )
    elif route == "implementation":
        primary = next(
            ([item] for item in sources if item.startswith("mini_nanobot/") and item.endswith(".py")),
            sources[:1],
        )
    else:
        primary = sources[:1]
    support = [item for item in sources if item not in primary]
    return list(primary), support


def migrate(path: Path) -> None:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    output = []
    for row in rows:
        if row["id"] == "MN-CODE-013" and "mini_nanobot/core/query.py" not in row["relevant_sources"]:
            row["relevant_sources"].append("mini_nanobot/core/query.py")
        route = route_for(row)
        primary, support = split_sources(row, route)
        row["label_version"] = "engineering-eval/v2"
        row["dataset_role"] = "development_regression"
        row["expected_route"] = route
        row["primary_sources"] = primary
        row["supporting_sources"] = support
        row["relevant_sources"] = [*primary, *support]
        row["relevant_symbols"] = RELEVANT_SYMBOLS.get(row["id"], [])
        row["refusal_reason"] = REFUSAL_REASONS.get(row["id"])
        output.append(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(output) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    for dataset in DATASETS:
        migrate(dataset)
        print(f"migrated {dataset.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
