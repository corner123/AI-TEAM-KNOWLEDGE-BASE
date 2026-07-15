> Public copy: local absolute paths were converted to repository-relative sources; metrics and per-question decisions are unchanged.

# Engineering RAG evaluation

- Dataset: `mini_nanobot_internal.jsonl + official_engineering_specs.jsonl`
- Questions: 80
- Top-K: 5
- Baseline: `hybrid_live`
- Suite: `e2e`

| Strategy | Route acc. | Route macro-F1 | Primary Hit@K | Primary Recall@K | nDCG@K | Symbol Recall@K | MRR | Refusal F1 | False refusal | P50 ms | P95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hybrid_live | 1.000 | 1.000 | 0.886 | 0.886 | 0.704 | 0.204 | 0.781 | 0.833 | 0.057 | 127.12 | 1412.40 |

## Interpretation boundaries

- Source metrics use explicit v2 primary/supporting labels and de-duplicate repeated chunks from the same source.
- Index ablation uses oracle routes, answerable questions only, and disables live rg/AST/Git plus answer generation.
- End-to-end evaluation fixes the hybrid index and evaluates routing, live verification, citations, and refusal separately.
- Symbol Recall@K uses explicit `relevant_symbols`; inspect the eligible sample count.
- Refusal is an explicit predictor decision. An empty retrieval result is not silently counted as a safe refusal.
- File/citation metrics are source-level, not passage entailment or answer correctness metrics.
- Latency is measured after one discarded warm-up call per strategy; model/index loading and report serialization are excluded.
