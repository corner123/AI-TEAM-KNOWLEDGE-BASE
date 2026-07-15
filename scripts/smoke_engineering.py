"""UTF-8 smoke test for routing, retrieval, citations, and live verification."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

from rag_core.engineering import load_engineering_service


@dataclass(frozen=True, slots=True)
class SmokeCase:
    question: str
    expected_intent: str
    required_roles: tuple[str, ...] = ()
    requires_live: bool = False
    expected_sufficient: bool = True


CASES = (
    SmokeCase(
        "为什么 Mini-Nanobot 使用手写 ReAct 循环？",
        "design",
        ("internal_design",),
    ),
    SmokeCase(
        "当前 QueryEngine.submit_message 在哪个文件实现？",
        "implementation",
        ("current_implementation",),
        requires_live=True,
    ),
    SmokeCase(
        "根据 MCP 2025-06-18 官方规范，tools/list 和 tools/call 是什么？",
        "official",
        ("external_normative",),
    ),
    SmokeCase(
        "对比当前 MCPToolAdapter 与 MCP 官方 tools 规范是否一致？",
        "comparison",
        ("current_implementation", "external_normative"),
        requires_live=True,
    ),
    SmokeCase(
        "今天上海天气如何？",
        "out_of_scope",
        expected_sufficient=False,
    ),
)


def _assert_case(case: SmokeCase, outcome) -> None:
    assert outcome.intent.value == case.expected_intent, (
        case.question,
        outcome.intent.value,
    )
    assert outcome.sufficient_evidence is case.expected_sufficient, (
        case.question,
        outcome.warnings,
    )
    if case.requires_live:
        assert outcome.live_verification_attempted, case.question
    if case.expected_sufficient:
        assert outcome.results, case.question
        roles = {
            str(result.metadata.get("evidence_role")) for result in outcome.results
        }
        assert set(case.required_roles) <= roles, (case.question, sorted(roles))
    else:
        assert outcome.refusal_reason, case.question

    public_payload = outcome.to_dict(include_content=True)
    serialized = json.dumps(public_payload, ensure_ascii=False).lower()
    assert "mini_nanobot_repo" not in serialized
    assert "authorization" not in serialized
    assert "d:\\\\code\\\\workspace_agent" not in serialized


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", default="data/indexes/engineering")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    service = load_engineering_service(args.index_dir)
    health = service.health()
    assert health["status"] == "ok", health
    assert health["index_fresh"] is not False, health
    assert health["partition_count"] >= 4, health
    assert health["live_code_enabled"], health
    assert health["live_ast_enabled"], health
    assert health["live_git_enabled"], health
    print(json.dumps({"health": health}, ensure_ascii=False))
    for case in CASES:
        outcome = service.retrieve(case.question, top_k=args.top_k)
        _assert_case(case, outcome)
        payload = {
            "query": case.question,
            "intent": outcome.intent.value,
            "sufficient": outcome.sufficient_evidence,
            "refusal_reason": outcome.refusal_reason,
            "live_attempted": outcome.live_verification_attempted,
            "live_terms": list(outcome.live_verification_terms),
            "revision": outcome.live_revision,
            "warnings": outcome.warnings,
            "results": [
                {
                    "role": result.metadata.get("evidence_role"),
                    "source": result.source,
                    "lines": [result.line_start, result.line_end],
                    "symbol": result.symbol,
                    "retriever": result.retriever,
                    "score": round(result.score, 6),
                }
                for result in outcome.results
            ],
        }
        print(json.dumps(payload, ensure_ascii=False))
    print(json.dumps({"smoke": "passed", "case_count": len(CASES)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
